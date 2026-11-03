"""Actor interpretation requests belong to the suspended character's decision trace."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.context import get_log_context, set_log_context
from core.interfaces.actor import ActorIdleAnswer, ActorIntent, ActorConfirmation
from core.interfaces.llm import LLMScene
from core.interfaces.trace import Stage
from interaction.application import NarrativeApplication
from providers.trace.in_memory import InMemoryTraceSink
from tests.unit.test_actor_runtime import HERO, _next_turn, _setup, _TALK_INDEX, _WORK_INDEX


@pytest.mark.asyncio
@pytest.mark.parametrize("response, adopted, parse_ok, reason", [
    (json.dumps({"decision": "act", "selected_index": _WORK_INDEX,
                 "action_description": "誊抄奏章"}), True, True, ""),
    (json.dumps({"decision": "unclear"}), False, True, "unclear"),
    (json.dumps({"decision": "act", "selected_index": -1}), False, True, "invalid_action"),
    (json.dumps({"decision": "act", "selected_index": _TALK_INDEX,
                 "action_description": "找人说话"}), False, True, "talk_without_target"),
    ("broken json", False, False, "no_json_object"),
    (None, None, None, ""),
])
async def test_actor_requests_record_context_and_result(
    container, test_config, response, adopted, parse_ok, reason,
) -> None:
    sink = InMemoryTraceSink()
    sink.flush = AsyncMock()
    container.trace_sink = sink
    container.llm_router._trace_sink = sink  # noqa: SLF001
    world_id = "actor-trace-world"
    runtime, _messages, agents = _setup(container, world_id, trace_sink=sink)
    app = NarrativeApplication(container, test_config)
    app._sessions[world_id] = SimpleNamespace(runtime=runtime)  # noqa: SLF001
    task = asyncio.create_task(runtime.run_step(agents))
    previous = get_log_context()
    try:
        turn = await _next_turn(runtime, task)
        provider = container.llm_router.get(LLMScene.ACTOR_INTENT)
        if response is None:
            provider.complete = AsyncMock(side_effect=RuntimeError("provider down"))
        else:
            provider.fixed_response = response
        set_log_context(world_id="other-world", agent_id="other-agent", step="99",
                        stage=Stage.DIRECTOR.value, request_id="actor-request")
        caller = get_log_context()

        said = await app.actor_intent(world_id, ActorIntent(turn_id=runtime.actor.pending_turn.turn_id, text="抄奏章"))

        assert get_log_context() == caller
        calls = [c for c in sink.read_calls(world_id) if c.scene == LLMScene.ACTOR_INTENT.value]
        assert len(calls) == 1
        call = calls[0]
        assert (call.step, call.agent_id, call.stage) == (turn.step, HERO, Stage.DECISION.value)
        assert call.extra == {"actor_words": "抄奏章", "turn_id": turn.turn_id}
        assert call.actor is True
        assert (call.ok, call.parse_ok, call.adopted, call.reject_reason) == (
            response is not None, parse_ok, adopted, reason,
        )
        assert (said.decision_summary is not None) is (adopted is True)
        if response is None:
            assert "provider down" in call.error
        # Arbitration still annotates this step, so the request must not flush it early.
        assert not task.done()
        sink.flush.assert_not_awaited()
    finally:
        set_log_context(**previous)
        if not task.done():
            runtime.answer_actor(ActorIdleAnswer())
        await task
        await asyncio.gather(*runtime._trace_flush_tasks, return_exceptions=True)  # noqa: SLF001
    sink.flush.assert_awaited_with(world_id, turn.step)


@pytest.mark.asyncio
async def test_actor_trace_flush_does_not_block_the_request(container, test_config) -> None:
    release = asyncio.Event()
    flushed = asyncio.Event()

    class SlowSink(InMemoryTraceSink):
        async def flush(self, world_id: str, step: int | None = None) -> None:
            await release.wait()
            flushed.set()

    sink = SlowSink()
    container.trace_sink = sink
    container.llm_router._trace_sink = sink  # noqa: SLF001
    runtime, _messages, agents = _setup(container, "actor-slow-trace", trace_sink=sink)
    app = NarrativeApplication(container, test_config)
    app._sessions["actor-slow-trace"] = SimpleNamespace(runtime=runtime)  # noqa: SLF001
    task = asyncio.create_task(runtime.run_step(agents))
    try:
        await _next_turn(runtime, task)
        container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = '{"decision":"unclear"}'
        await asyncio.wait_for(app.actor_intent("actor-slow-trace", ActorIntent(turn_id=runtime.actor.pending_turn.turn_id, text="唉")), timeout=1)
        assert not flushed.is_set()
    finally:
        release.set()
        await asyncio.gather(*app._trace_flush_tasks, return_exceptions=True)  # noqa: SLF001
        if not task.done():
            runtime.answer_actor(ActorIdleAnswer())
        await task
        await asyncio.gather(*runtime._trace_flush_tasks, return_exceptions=True)  # noqa: SLF001
    assert flushed.is_set()


@pytest.mark.asyncio
async def test_live_actor_trace_stays_annotatable_until_step_commit(container, test_config, tmp_path) -> None:
    from providers.trace import BufferedJsonlTraceSink

    sink = BufferedJsonlTraceSink(str(tmp_path))
    container.trace_sink = sink
    container.llm_router._trace_sink = sink  # noqa: SLF001
    world_id = "actor-live-trace"
    runtime, _messages, agents = _setup(container, world_id, trace_sink=sink)
    app = NarrativeApplication(container, test_config)
    app._sessions[world_id] = SimpleNamespace(runtime=runtime)  # noqa: SLF001
    container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = json.dumps({
        "decision": "act", "selected_index": _WORK_INDEX, "action_description": "誊抄奏章",
    })
    task = asyncio.create_task(runtime.run_step(agents))
    try:
        await _next_turn(runtime, task)
        said = await app.actor_intent(world_id, ActorIntent(turn_id=runtime.actor.pending_turn.turn_id, text="抄奏章"))
        await asyncio.sleep(0)
        calls = [c for c in sink.read_calls(world_id) if c.scene == LLMScene.ACTOR_INTENT.value]
        assert len(calls) == 1 and "verdict" not in calls[0].extra
        assert not (tmp_path / world_id / "step_000001.jsonl").exists()
        runtime.answer_actor(ActorConfirmation(said.decision_summary.decision_summary_id))
        await task
        await asyncio.gather(*runtime._trace_flush_tasks, return_exceptions=True)  # noqa: SLF001
        calls = [c for c in sink.read_calls(world_id) if c.scene == LLMScene.ACTOR_INTENT.value]
        assert len(calls) == 1 and calls[0].extra["verdict"] == "executed"
        assert not runtime.step_in_progress
        assert (tmp_path / world_id / "step_000001.jsonl").exists()
    finally:
        if not task.done():
            runtime.answer_actor(ActorIdleAnswer())
        await task


def _wired(container, world_id: str):
    sink = InMemoryTraceSink()
    container.trace_sink = sink
    container.llm_router._trace_sink = sink  # noqa: SLF001
    runtime, _messages, agents = _setup(container, world_id, trace_sink=sink)
    return sink, runtime, agents


def _readings(sink: InMemoryTraceSink, world_id: str) -> list:
    return [c for c in sink.read_calls(world_id) if c.scene == LLMScene.ACTOR_INTENT.value]


@pytest.mark.asyncio
async def test_the_confirmed_reading_is_the_decision_beat_and_the_others_are_superseded(
    container, test_config,
) -> None:
    world_id = "actor-beat-confirmed"
    sink, runtime, agents = _wired(container, world_id)
    app = NarrativeApplication(container, test_config)
    app._sessions[world_id] = SimpleNamespace(runtime=runtime)  # noqa: SLF001
    provider = container.llm_router.get(LLMScene.ACTOR_INTENT)
    task = asyncio.create_task(runtime.run_step(agents))
    try:
        turn = await _next_turn(runtime, task)
        for words in ("抄一份", "誊抄奏章"):
            provider.fixed_response = json.dumps({
                "decision": "act", "selected_index": _WORK_INDEX, "action_description": words,
                "expected_outcome": "奏章誊好",
            })
            said = await app.actor_intent(world_id, ActorIntent(turn_id=turn.turn_id, text=words))
        runtime.answer_actor(ActorConfirmation(said.decision_summary.decision_summary_id))
        await task
    finally:
        if not task.done():
            runtime.answer_actor(ActorIdleAnswer())
        await task
    first, second = _readings(sink, world_id)
    assert (first.adopted, first.reject_reason) == (False, "superseded")
    assert second.adopted is True
    [beat] = [b for b in sink.read_beats(world_id) if b.beat_kind == "decision"]
    assert (beat.step, beat.agent_id, beat.stage, beat.actor) == (turn.step, HERO, Stage.DECISION.value, True)
    assert beat.payload["action_type"] == "WORK"
    assert beat.payload["action_description"] == "誊抄奏章"
    assert beat.payload["expected_outcome"] == "奏章誊好"
    # Arbitration stamps its verdict on this step's decision, a beat as much as a call.
    assert beat.extra["verdict"] == "executed"


@pytest.mark.asyncio
async def test_answering_without_a_reading_supersedes_every_reading(container, test_config) -> None:
    world_id = "actor-beat-idle"
    sink, runtime, agents = _wired(container, world_id)
    app = NarrativeApplication(container, test_config)
    app._sessions[world_id] = SimpleNamespace(runtime=runtime)  # noqa: SLF001
    container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = json.dumps({
        "decision": "act", "selected_index": _WORK_INDEX, "action_description": "誊抄奏章",
    })
    task = asyncio.create_task(runtime.run_step(agents))
    try:
        turn = await _next_turn(runtime, task)
        await app.actor_intent(world_id, ActorIntent(turn_id=turn.turn_id, text="抄奏章"))
        runtime.answer_actor(ActorIdleAnswer())
        await task
    finally:
        if not task.done():
            runtime.answer_actor(ActorIdleAnswer())
        await task
    [reading] = _readings(sink, world_id)
    assert (reading.adopted, reading.reject_reason) == (False, "superseded")
    [beat] = [b for b in sink.read_beats(world_id) if b.beat_kind == "decision"]
    assert beat.payload == {"idle": True} and beat.actor is True


def _coordinator(sink: InMemoryTraceSink):
    from engine.actor_coordinator import ActorCoordinator
    from engine.clock import WorldTimeConfig

    return ActorCoordinator(WorldTimeConfig(start_hour=6, seconds_per_step=60), world_id="w", trace_sink=sink)


def test_an_agreed_time_falling_due_asks_the_actor_and_says_why() -> None:
    from core.interfaces.perception import PerceivedSignals
    coordinator = _coordinator(InMemoryTraceSink())

    assert coordinator.weighs_interrupt(step=5, signals=PerceivedSignals(), falling_due=True)
    assert coordinator._reason_for(5) == "约定好时间的事项快到了"  # noqa: SLF001
    assert not coordinator.weighs_interrupt(step=6, signals=PerceivedSignals(), falling_due=False)


@pytest.mark.asyncio
async def test_a_break_off_is_the_interrupt_beat() -> None:
    sink = InMemoryTraceSink()
    coordinator = _coordinator(sink)
    coordinator.channel.request_break_off()
    agent = SimpleNamespace(agent_id=HERO, personality=SimpleNamespace(
        state=SimpleNamespace(current_action="誊抄奏章")))

    assert await coordinator.evaluate_interrupt(agent, step=5) == (True, "")

    [beat] = sink.read_beats("w")
    assert (beat.step, beat.agent_id, beat.stage, beat.beat_kind, beat.actor) == (
        5, HERO, Stage.INTERRUPT.value, "interrupt", True,
    )
    assert beat.payload == {
        "interrupt": True, "thought": "", "doing": "誊抄奏章", "trigger": "", "self_initiated": True,
    }


@pytest.mark.asyncio
async def test_amendments_saved_between_steps_are_a_mind_beat_where_they_land() -> None:
    from core.interfaces.actor import ActorAmendments

    sink = InMemoryTraceSink()
    coordinator = _coordinator(sink)
    changed = {"insights": ["他在躲我"]}
    agent = SimpleNamespace(
        agent_id=HERO, check_amendments=AsyncMock(), amend_mind=AsyncMock(return_value=changed),
    )
    coordinator._agent = agent  # noqa: SLF001 — performing without a real character
    coordinator._channel.follow(HERO)  # noqa: SLF001

    assert await coordinator.amend(ActorAmendments(insights=("他在躲我",))) is False
    assert sink.read_beats("w") == []
    await coordinator.land_amendments(agent, step=7)

    [beat] = sink.read_beats("w")
    # Landed before the character perceives the step: the input of what follows.
    assert (beat.step, beat.stage, beat.beat_kind, beat.actor) == (7, Stage.PERCEPTION.value, "mind", True)
    assert beat.payload == changed
