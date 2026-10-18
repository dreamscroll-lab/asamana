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
from engine.application import NarrativeApplication
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
    runtime, _messages, agents = _setup(container, world_id)
    runtime._trace_sink = sink  # noqa: SLF001 — mirror production wiring
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
        assert (call.ok, call.parse_ok, call.adopted, call.reject_reason) == (
            response is not None, parse_ok, adopted, reason,
        )
        assert (said.action_summary is not None) is (adopted is True)
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
    runtime, _messages, agents = _setup(container, "actor-slow-trace")
    runtime._trace_sink = sink  # noqa: SLF001 — mirror production wiring
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
    runtime, _messages, agents = _setup(container, world_id)
    runtime._trace_sink = sink  # noqa: SLF001
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
        runtime.answer_actor(ActorConfirmation(said.action_summary.action_summary_id))
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
