"""Trace lifecycle and adoption at cognition consumers, including valid no-op answers."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from agent.importance_evaluator import ImportanceEvaluator
from agent.memory import MemorySystem
from agent.memory_types import MemoryStream
from agent.need import NeedEngine
from agent.perception_emotion import parse_perception_emotion_response
from agent.reflection import ReflectionEngine
from agent.relation_evolution import RelationEvolution
from core.context import get_log_context, note_active_call_adoption, set_active_call, set_log_context
from core.interfaces.llm import LLMScene, extract_json
from core.interfaces.trace import Stage
from interaction.application import NarrativeApplication
from engine.world_pressure import WorldPressureEvaluator
from providers.trace import BufferedJsonlTraceSink, InMemoryTraceSink
from tests.unit.test_decision_m4_gaps import _make_personality, _mem
from tests.unit.test_narrative_editor import _narrative_editor, _world_time
from tests.unit.test_relation_evolution import (
    _make_memory_system, _make_relation_system, _personality, _seed_event_about, _seed_existing_relation,
)
from world.builder import WorldBuilder
from worlds.tiled import TiledWorldConfig
from tests.unit.confirmed_world import confirmed_world


@pytest.fixture
def traced(container):
    previous = get_log_context()
    sink = InMemoryTraceSink()
    container.trace_sink = sink
    container.llm_router._trace_sink = sink  # noqa: SLF001
    set_log_context(world_id="trace-world", agent_id="agent-self", step="20")
    try:
        yield container, sink
    finally:
        set_log_context(**previous)
        set_active_call(None)


def _reply(container, scene: LLMScene, payload: object) -> None:
    container.llm_router.get(scene).fixed_response = json.dumps(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload, expected, adopted",
    [
        ({"goals": []}, None, True),
        ({"goals": ["守护家国"]}, None, True),
        ({"goals": ["重整朝纲"]}, ["重整朝纲"], True),
        ({}, None, False),
        ({"goals": None}, None, False),
        ({"goals": "重整朝纲"}, None, False),
        ({"goals": {}}, None, False),
        ({"goals": [None, 123, {}, " ", "长" * 51]}, None, False),
    ],
)
async def test_long_term_goal_review_adoption(traced, payload, expected, adopted) -> None:
    container, sink = traced
    personality = _make_personality()
    personality.set_long_term_goals(["守护家国"])
    _reply(container, LLMScene.NEED_GOAL_GENERATION, payload)

    result = await NeedEngine(llm_router=container.llm_router).revise_long_term_goals(
        personality=personality, recent_memory_texts=["寻常的一天"], is_main_character=False,
    )

    assert result == expected
    assert personality.state.long_term_goals == ["守护家国"]
    call = sink.read_calls("trace-world")[-1]
    assert call.parse_ok is True
    assert call.adopted is adopted
    assert call.reject_reason == ("" if adopted else "no_usable_goals")


@pytest.mark.asyncio
async def test_failed_build_flushes_rejected_attempts_to_disk(mock_build_container, tmp_path) -> None:
    sink = BufferedJsonlTraceSink(str(tmp_path))
    mock_build_container.trace_sink = sink
    mock_build_container.llm_router._trace_sink = sink  # noqa: SLF001
    mock_build_container.llm_router.get(LLMScene.WORLD_BUILDING).fixed_response = "broken json"
    previous = get_log_context()
    with pytest.raises(ValueError, match="ThemeAnalyzer failed"):
        await WorldBuilder(mock_build_container).build(
            "a world", world_id="failed-build", world_configs=[TiledWorldConfig(template="changan_iso")],
        )
    assert get_log_context() == previous
    calls = sink.read_calls("failed-build")
    assert len(calls) == 2
    assert all(c.stage == Stage.WORLD_INIT.value and c.step is None for c in calls)
    assert all(c.ok and c.parse_ok is False and c.adopted is False for c in calls)
    assert (tmp_path / "failed-build" / "build.jsonl").exists()


@pytest.mark.asyncio
async def test_successful_build_consumes_each_builders_answer(mock_build_container, test_config) -> None:
    sink = InMemoryTraceSink()
    mock_build_container.trace_sink = sink
    mock_build_container.llm_router._trace_sink = sink  # noqa: SLF001
    app = NarrativeApplication(mock_build_container, test_config)
    world = await confirmed_world(app, "a world", template="changan_iso")
    calls = sink.read_calls(world.world_id)
    assert {c.scene for c in calls} >= {
        LLMScene.WORLD_BUILDING.value, LLMScene.CAST_DESIGN.value, LLMScene.PERSONA_GENERATION.value,
    }
    assert all(c.adopted is True for c in calls)


@pytest.mark.asyncio
async def test_flush_failure_preserves_the_build_error(mock_build_container) -> None:
    mock_build_container.trace_sink.flush = AsyncMock(side_effect=OSError("trace disk unavailable"))
    builder = WorldBuilder(mock_build_container)
    builder.theme_analyzer.analyze = AsyncMock(side_effect=ValueError("invalid theme output"))
    with pytest.raises(ValueError, match="invalid theme output"):
        await builder.build("a world", world_id="w", world_configs=[TiledWorldConfig(template="changan_iso")])
    mock_build_container.trace_sink.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_drains_request_traces_even_without_a_running_world(container, test_config) -> None:
    started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def flush(_world_id: str, _step: int | None = None) -> None:
        started.set()
        await release.wait()
        finished.set()

    container.trace_sink.flush = flush
    app = NarrativeApplication(container, test_config)
    app._flush_trace_async("paused-world", 3)  # noqa: SLF001
    await started.wait()
    closing = asyncio.create_task(app.shutdown())
    try:
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
        await closing
    assert finished.is_set()


@pytest.mark.asyncio
async def test_cancelled_provider_call_is_recorded_and_still_propagates(traced) -> None:
    container, sink = traced
    started = asyncio.Event()

    async def pending(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    container.llm_router.get(LLMScene.ACTOR_INTENT).complete = pending
    task = asyncio.create_task(container.llm_router.complete(LLMScene.ACTOR_INTENT, []))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    call, = sink.read_calls("trace-world")
    assert call.ok is False and call.error == "cancelled"
    assert call.parse_ok is None and call.adopted is None


@pytest.mark.asyncio
@pytest.mark.parametrize("recording_fails", [True, False])
async def test_unrecorded_call_does_not_annotate_a_previous_call(traced, recording_fails) -> None:
    container, sink = traced
    _reply(container, LLMScene.ACTOR_INTENT, {})
    await container.llm_router.complete(LLMScene.ACTOR_INTENT, [])
    previous, = sink.llm_calls
    if recording_fails:
        def fail(_call) -> None:
            raise OSError("trace unavailable")
        sink.record_llm_call = fail
    else:
        container.llm_router._trace_sink = None  # noqa: SLF001
    response = await container.llm_router.complete(LLMScene.ACTOR_INTENT, [])
    extract_json(response.content)
    note_active_call_adoption(False, "discarded")
    assert previous.parse_ok is None and previous.adopted is True


@pytest.mark.asyncio
@pytest.mark.parametrize("payload, adopted", [
    ({"emotion": "anger", "intensity": 0.5}, True),
    ({"emotion": "anger", "intensity": "invalid"}, False),
    ({"emotion": None, "need_activation": {}}, True),
    ([], False),
])
async def test_perception_appraisal_marks_usable_and_unusable_outputs(traced, payload, adopted) -> None:
    container, sink = traced
    _reply(container, LLMScene.AGENT_DECISION_MAIN, payload)
    response = await container.llm_router.complete(LLMScene.AGENT_DECISION_MAIN, [])
    parse_perception_emotion_response(response.content)
    assert sink.llm_calls[-1].adopted is adopted


@pytest.mark.asyncio
@pytest.mark.parametrize("payload, adopted", [({"score": 0.7}, True), ({}, False), ({"score": "bad"}, False)])
async def test_importance_marks_fallbacks(traced, payload, adopted) -> None:
    container, sink = traced
    _reply(container, LLMScene.MEMORY_IMPORTANCE, payload)
    await ImportanceEvaluator(llm_router=container.llm_router, agent_id="agent-self").evaluate(
        "Someone spoke to me.", personality=_make_personality(), related_agents=[],
    )
    assert sink.llm_calls[-1].adopted is adopted


@pytest.mark.asyncio
@pytest.mark.parametrize("payload, adopted", [({"points": ["The gate closed."]}, True), ({"points": []}, False), ({}, False)])
async def test_compression_marks_discarded_summaries(traced, payload, adopted) -> None:
    container, sink = traced
    _reply(container, LLMScene.MEMORY_SUMMARIZATION, payload)
    memory = MemorySystem(container.llm_router, container.embedding, container.vector_store,
                          world_id="trace-world", agent_id="agent-self")
    result = await memory._summarize_cluster_for_stream(  # noqa: SLF001
        [_mem("The gate closed.", MemoryStream.FACTUAL)], stream=MemoryStream.FACTUAL,
    )
    assert bool(result) is adopted
    assert sink.llm_calls[-1].adopted is adopted


@pytest.mark.asyncio
@pytest.mark.parametrize("payload, adopted", [({"goals": []}, True), ({}, False), ({"goals": [None]}, False)])
async def test_world_pressure_distinguishes_no_pressure_from_invalid_pressure(traced, payload, adopted) -> None:
    container, sink = traced
    _reply(container, LLMScene.WORLD_PRESSURE, payload)
    response = await container.llm_router.complete(LLMScene.WORLD_PRESSURE, [])
    WorldPressureEvaluator(llm_router=container.llm_router)._parse_agent_goals(  # noqa: SLF001
        response.content, signal_people=[], agent_id="agent-self",
    )
    assert sink.llm_calls[-1].adopted is adopted


@pytest.mark.asyncio
@pytest.mark.parametrize("payload, adopted", [({"inject": False}, True), ({"inject": True}, True), ({}, False)])
async def test_editor_gate_records_no_injection_as_a_valid_answer(traced, payload, adopted) -> None:
    container, sink = traced
    _reply(container, LLMScene.NARRATIVE_EDITOR, payload)
    editor = _narrative_editor(container, router=container.llm_router)
    await editor._passes_llm_check(_world_time(), "Recent events")  # noqa: SLF001
    assert sink.llm_calls[-1].adopted is adopted


@pytest.mark.asyncio
@pytest.mark.parametrize("payload, adopted", [
    ({"updates": []}, True), ({}, False),
    ({"updates": [{"target_index": 99, "labels": ["friend"]}]}, False),
    ({"updates": [{"target_index": 1, "summary": "They spoke frankly."}]}, True),
])
async def test_relation_updates_record_dropped_targets(traced, payload, adopted) -> None:
    container, sink = traced
    memory, relations = _make_memory_system(container), _make_relation_system(container)
    await _seed_event_about(memory, target_id="other", step=18, content="We spoke frankly.")
    await _seed_existing_relation(relations, target_id="other", labels=["friend"], to_name="Other")
    _reply(container, LLMScene.RELATION_LABEL_EVOLUTION, payload)
    await RelationEvolution(container.llm_router, memory, relations, _personality(), agent_id="agent-self").evaluate(20)
    call = sink.llm_calls[-1]
    assert call.scene == LLMScene.RELATION_LABEL_EVOLUTION.value and call.adopted is adopted


@pytest.mark.asyncio
@pytest.mark.parametrize("payload, adopted", [
    ({"insights": []}, True), ({}, False),
    ({"insights": [{"text": "I distrust them.", "source_indices": [99]}]}, False),
    ({"insights": [{"text": "I distrust them.", "source_indices": [1]}]}, True),
])
async def test_reflection_records_rejected_grounding(traced, payload, adopted) -> None:
    container, sink = traced
    container.llm_router.get(LLMScene.MEMORY_SUMMARIZATION).fixed_response = "I feel uneasy."
    memory = MemorySystem(container.llm_router, container.embedding, container.vector_store,
                          world_id="trace-world", agent_id="agent-self")
    await memory.write("They avoided me.", MemoryStream.EXPERIENTIAL,
                       personality=_personality(), importance=0.8, current_step=18)
    _reply(container, LLMScene.MEMORY_REFLECTION, payload)
    await ReflectionEngine(container.llm_router, memory, _personality(), agent_id="agent-self").reflect(20)
    call = sink.llm_calls[-1]
    assert call.scene == LLMScene.MEMORY_REFLECTION.value and call.adopted is adopted


@pytest.mark.asyncio
@pytest.mark.parametrize("on_disk", [True, False])
async def test_later_arbitration_can_discard_accepted_cognition(tmp_path, on_disk) -> None:
    from tests.unit.test_observability import _call

    sink = BufferedJsonlTraceSink(str(tmp_path)) if on_disk else InMemoryTraceSink()
    accepted = _call(stage=Stage.MOTIVATION.value)
    accepted.adopted = True
    already_rejected = _call(stage=Stage.MOTIVATION.value)
    already_rejected.adopted = False
    already_rejected.reject_reason = "no_usable_goals"
    sink.record_llm_call(accepted)
    sink.record_llm_call(already_rejected)
    sink.mark_call_unadopted("w", step=1, agent_id="a1", stage=Stage.MOTIVATION.value, reason="conscripted")
    assert accepted.adopted is False and accepted.reject_reason == "conscripted"
    assert already_rejected.reject_reason == "no_usable_goals"
    await sink.flush("w", 1)
    first, second = sink.read_calls("w", step=1)
    assert first.adopted is False and first.reject_reason == "conscripted"
    assert second.reject_reason == "no_usable_goals"
