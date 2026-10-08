"""The director's LLM calls must be traced under the right world and step.

They run inside an HTTP request, not the step loop, so nothing sets their observation context;
without setting it explicitly they record an empty world_id, no step, and stage=unknown.

They also carry diagnostic fields (what the director said, the menus shown, what the directive
became): an index pointing at the wrong person can only be diagnosed against the menu shown.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from core.interfaces.llm import LLMScene
from core.interfaces.trace import COGNITION_ORDER, Stage
from engine.application import NarrativeApplication
from providers.trace.in_memory import InMemoryTraceSink
from world import WorldCatalog


def _directive_json(**overrides) -> str:
    payload = {
        "reason": "导演要放个消息",
        "feasible": True,
        "refusal": "",
        "broadcast": {"content": "钟声大作。", "severity": "medium",
                      "location_scope": None, "phenomenon": "none"},
        "message": None,
        "mutations": [],
        "narrative_desc": "钟声大作",
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


async def _app_with_world(container, test_config):
    container.trace_sink = InMemoryTraceSink()
    container.llm_router._trace_sink = container.trace_sink  # noqa: SLF001 — test wiring
    app = NarrativeApplication(container, test_config, catalog=WorldCatalog(None))
    world = await app.build_world("宫廷权谋", template="changan_iso")
    return app, world.world_id


def _director_calls(sink: InMemoryTraceSink, world_id: str):
    return [c for c in sink.read_calls(world_id) if c.stage == Stage.DIRECTOR.value]


def test_director_is_its_own_stage_in_the_cognition_order() -> None:
    """The two authors get separate stages on purpose. They share an LLMScene, so the scene can't
    tell them apart, and whether a step's event came from the world or from a human is the first
    thing an observer wants to know."""
    assert Stage.DIRECTOR in COGNITION_ORDER
    assert Stage.EVENT in COGNITION_ORDER
    assert Stage.DIRECTOR is not Stage.EVENT


@pytest.mark.asyncio
async def test_submit_records_under_the_right_world_step_and_stage(
    mock_build_container, test_config,
) -> None:
    app, world_id = await _app_with_world(mock_build_container, test_config)
    mock_build_container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json()

    await app.submit_directive(world_id, "敲响全城的钟")

    calls = _director_calls(mock_build_container.trace_sink, world_id)
    assert len(calls) == 1
    call = calls[0]
    assert call.world_id == world_id      # otherwise it doesn't show up in this world's trace
    assert call.step is not None          # otherwise it is mixed in with world-build calls
    assert call.ok is True


@pytest.mark.asyncio
async def test_submit_annotates_what_was_said_and_what_could_be_picked(
    mock_build_container, test_config,
) -> None:
    """The menu annotation is the only way to diagnose an index that points at the wrong person; a
    bare #2 in the response tells you nothing."""
    app, world_id = await _app_with_world(mock_build_container, test_config)
    mock_build_container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json()

    await app.submit_directive(world_id, "敲响全城的钟")

    extra = _director_calls(mock_build_container.trace_sink, world_id)[0].extra
    assert extra["directive_text"] == "敲响全城的钟"
    assert extra["cast_menu"]["1"]        # index -> name, the same list the prompt showed
    assert extra["location_menu"]["1"]
    # Which channels the directive became: the first thing to check when asking whether it was
    # translated faithfully or embellished.
    assert extra["channels"] == ["broadcast"]
    assert extra["narrative_desc"] == "钟声大作"


@pytest.mark.asyncio
async def test_a_refused_directive_is_recorded_as_not_adopted(
    mock_build_container, test_config,
) -> None:
    """An infeasible directive is a rejected answer: it parses, but the consumer can't use it.
    Without recording that, the trace shows a successful call and hides that nothing was injected.
    """
    app, world_id = await _app_with_world(mock_build_container, test_config)
    mock_build_container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json(
        feasible=False, refusal="没说清是谁,请指名道姓。", broadcast=None,
    )

    result = await app.submit_directive(world_id, "让他去那边")

    assert result.accepted is False
    call = _director_calls(mock_build_container.trace_sink, world_id)[0]
    assert call.ok is True              # the LLM itself answered fine
    assert call.adopted is False        # we decided it was unusable
    assert call.reject_reason == "没说清是谁,请指名道姓。"


@pytest.mark.asyncio
async def test_the_context_does_not_leak_out_of_the_director_scope(
    mock_build_container, test_config,
) -> None:
    """world_id / stage must be reset when the scope ends. The HTTP handler is a long-lived task,
    and a leaked context would record every later unrelated call as director."""
    from core.context import get_log_context

    app, world_id = await _app_with_world(mock_build_container, test_config)
    mock_build_container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json()

    await app.submit_directive(world_id, "敲钟")

    ctx = get_log_context()
    assert ctx.get("world_id", "") == ""
    assert ctx.get("stage", "") in ("", Stage.UNKNOWN.value)


@pytest.mark.asyncio
async def test_the_scope_restores_the_callers_context_rather_than_clearing_it(
    mock_build_container, test_config,
) -> None:
    """Observation must restore the caller's context, not clear it.

    ``clear_log_context()`` clears all six fields. The HTTP handler's empty context hides that; inside
    another scope every later log and trace would go wrong.
    """
    from core.context import get_log_context, set_log_context

    app, world_id = await _app_with_world(mock_build_container, test_config)
    mock_build_container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json()
    set_log_context(world_id="someone-elses-world", request_id="req-42")

    await app.submit_directive(world_id, "敲钟")

    ctx = get_log_context()
    assert ctx.get("world_id") == "someone-elses-world"
    assert ctx.get("request_id") == "req-42"
    set_log_context()   # reset so later tests in this process start clean


@pytest.mark.asyncio
async def test_the_trace_flush_never_sits_in_the_request_path(
    mock_build_container, test_config,
) -> None:
    """Observation must not slow down the work it observes.

    The flush (needed because a paused world never reaches its end-of-step flush) runs as an
    unawaited task, like the per-step flush in ``NarrativeRuntime.run_step``.
    """
    flushed: list[tuple[str, int | None]] = []

    class _SlowSink(InMemoryTraceSink):
        async def flush(self, world_id: str, step: int | None = None) -> None:
            await asyncio.sleep(0.2)      # stands in for the disk write
            flushed.append((world_id, step))
            await super().flush(world_id, step)

    mock_build_container.trace_sink = _SlowSink()
    mock_build_container.llm_router._trace_sink = mock_build_container.trace_sink  # noqa: SLF001
    app = NarrativeApplication(mock_build_container, test_config, catalog=WorldCatalog(None))
    world = await app.build_world("宫廷权谋", template="changan_iso")
    mock_build_container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json()
    # World build awaits its own flush (step=None, a one-off outside this rule).
    # Clear it so it isn't mistaken for the director path's flush.
    flushed.clear()

    import time
    started = time.perf_counter()
    await app.submit_directive(world.world_id, "敲钟")
    elapsed = time.perf_counter() - started

    assert elapsed < 0.2, f"提交路径上等了 flush({elapsed:.3f}s)"
    assert flushed == [], "flush 在返回前就跑完了 —— 说明它是同步等的"
    # It was still dispatched and does complete, just not on the request path.
    await asyncio.gather(*app._trace_flush_tasks)  # noqa: SLF001 — wait for the background flush
    assert flushed and flushed[0][0] == world.world_id
