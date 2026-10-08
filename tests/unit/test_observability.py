"""Unit tests for the production LLM observability stack.

Covers the context stage machinery, LLMRouter trace recording at the single
chokepoint, and the three trace sinks (buffered jsonl / in-memory / null).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from core.context import (
    annotate_call,
    clear_log_context,
    get_call_annotations,
    get_log_context,
    note_active_call_parse,
    observe_stage,
    observe_step,
    set_active_call,
    set_log_context,
)
from core.interfaces.llm import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMRouter,
    LLMScene,
    extract_json,
    extract_json_object,
)
from core.interfaces.trace import LLMCallTrace, Stage, StepTrace, stage_sort_key
from providers.trace import BufferedJsonlTraceSink, InMemoryTraceSink, NullTraceSink


class _FailProvider(LLMProvider):
    async def complete(self, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ) -> LLMResponse:
        raise RuntimeError("provider exploded")

    async def stream(self, messages, temperature=0.7):
        yield "x"


class _OkProvider(LLMProvider):
    def __init__(self, content: str = "ok") -> None:
        self.content = content
        self.calls = 0

    async def complete(self, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=self.content, input_tokens=3, output_tokens=5, model="mock")

    async def stream(self, messages, temperature=0.7) -> AsyncIterator[str]:
        yield self.content


def _router(provider: LLMProvider, sink=None) -> LLMRouter:
    return LLMRouter({scene: provider for scene in LLMScene}, trace_sink=sink)


# ---------------------------------------------------------------------------
# context / observe_stage
# ---------------------------------------------------------------------------

def test_observe_stage_sets_and_restores() -> None:
    clear_log_context()
    set_log_context(world_id="w", step="3")
    assert "stage" not in get_log_context()
    with observe_stage(Stage.DECISION, agent_id="a1"):
        ctx = get_log_context()
        assert ctx["stage"] == "decision"
        assert ctx["agent_id"] == "a1"
        # nesting overrides then restores
        with observe_stage(Stage.MEMORY):
            inner = get_log_context()
            assert inner["stage"] == "memory"
            assert inner["agent_id"] == "a1"  # inherited
        assert get_log_context()["stage"] == "decision"
    restored = get_log_context()
    assert "stage" not in restored
    assert "agent_id" not in restored
    assert restored["world_id"] == "w"


def test_observe_stage_accepts_plain_string() -> None:
    clear_log_context()
    with observe_stage("custom_stage"):
        assert get_log_context()["stage"] == "custom_stage"


def test_observe_step_overrides_only_step_and_restores() -> None:
    """observe_step re-stamps the step for deferred/background work (frozen-context worker),
    stringifies it, restores on exit, and — unlike set_log_context — leaves world_id/agent_id
    /stage untouched."""
    clear_log_context()
    set_log_context(world_id="w", agent_id="a1", step="2")  # frozen worker-creation context
    with observe_step(13):  # deferred write for a later event step
        ctx = get_log_context()
        assert ctx["step"] == "13"            # int stringified
        assert ctx["world_id"] == "w"         # not clobbered
        assert ctx["agent_id"] == "a1"        # not clobbered
    assert get_log_context()["step"] == "2"   # restored to the frozen step


def test_annotate_call_merges_and_restores() -> None:
    assert get_call_annotations() == {}
    with annotate_call(dominant_need_old="social", dominant_need_new="safety"):
        assert get_call_annotations() == {"dominant_need_old": "social", "dominant_need_new": "safety"}
        # nesting merges (inner wins on key clash) then restores
        with annotate_call(dominant_need_new="esteem", note="x"):
            inner = get_call_annotations()
            assert inner == {"dominant_need_old": "social", "dominant_need_new": "esteem", "note": "x"}
        assert get_call_annotations() == {"dominant_need_old": "social", "dominant_need_new": "safety"}
    assert get_call_annotations() == {}  # fully restored


@pytest.mark.asyncio
async def test_a_background_write_runs_in_its_enqueuers_context() -> None:
    """Background write jobs run in the whole context from the moment they were enqueued.

    The worker is long-lived and ``create_task`` freezes contextvars at creation, so every later job
    would carry the first enqueuer's coordinates (``execution_id``, ``step``). This pins the general
    property: any contextvar present at enqueue travels with the job.
    """
    import contextvars
    from agent.memory_write_queue import MemoryWriteQueue

    probe: contextvars.ContextVar[str] = contextvars.ContextVar("probe", default="none")
    seen: list[tuple[dict, str, str]] = []

    async def _job() -> None:
        seen.append((get_call_annotations(), get_log_context().get("step", ""), probe.get()))

    queue = MemoryWriteQueue(agent_id="a1")
    with annotate_call(execution_id="e1"):          # this creates the worker, freezing its context
        set_log_context(step="1")
        probe.set("first")
        queue.enqueue(_job())
    with annotate_call(execution_id="e2"):
        set_log_context(step="9")
        probe.set("second")
        queue.enqueue(_job())
    clear_log_context()
    queue.enqueue(_job())   # nothing set: must see empty, not the first job's context
    await queue.close()

    assert seen == [({"execution_id": "e1"}, "1", "first"),
                    ({"execution_id": "e2"}, "9", "second"),
                    ({}, "", "second")]  # probe set outside the with; the rest restored


def test_get_call_annotations_returns_isolated_copy() -> None:
    """Caller (router) must not be able to mutate contextvar state via the snapshot."""
    with annotate_call(k="v"):
        snap = get_call_annotations()
        snap["k"] = "MUTATED"
        assert get_call_annotations() == {"k": "v"}  # contextvar untouched


# ---------------------------------------------------------------------------
# LLMRouter trace recording
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_router_records_call_with_stage_and_context() -> None:
    clear_log_context()
    set_log_context(world_id="w1", step="7")
    sink = InMemoryTraceSink()
    router = _router(_OkProvider("hello"), sink)
    with observe_stage(Stage.DECISION, agent_id="a1"):
        resp = await router.complete(LLMScene.AGENT_DECISION_MAIN, [LLMMessage(role="user", content="hi")])
    assert resp.content == "hello"
    assert len(sink.llm_calls) == 1
    call = sink.llm_calls[0]
    assert call.stage == "decision"
    assert call.world_id == "w1"
    assert call.agent_id == "a1"
    assert call.step == 7
    assert call.scene == LLMScene.AGENT_DECISION_MAIN.value
    assert call.input_tokens == 3 and call.output_tokens == 5
    assert call.prompt_messages == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_router_records_json_mode_per_call() -> None:
    """json_mode is part of the call parameters and must be recorded like temperature / max_tokens.
    The prompt debugger relies on it to replay the production call exactly (unrecorded, it can only
    default off and the rerun isn't the same call)."""
    clear_log_context()
    set_log_context(world_id="w1", step="1")
    sink = InMemoryTraceSink()
    router = _router(_OkProvider("{}"), sink)
    with observe_stage(Stage.DECISION, agent_id="a1"):
        await router.complete(LLMScene.AGENT_DECISION_MAIN, [LLMMessage(role="user", content="json")],
                              json_mode=True)
        await router.complete(LLMScene.AGENT_DECISION_MAIN, [LLMMessage(role="user", content="free")])
    assert [c.json_mode for c in sink.llm_calls] == [True, False]


@pytest.mark.asyncio
async def test_router_captures_annotations_into_call_extra() -> None:
    clear_log_context()
    set_log_context(world_id="w1", step="7")
    sink = InMemoryTraceSink()
    router = _router(_OkProvider("hi"), sink)
    with observe_stage(Stage.MOTIVATION, agent_id="a1"):
        with annotate_call(dominant_need_old="social", dominant_need_new="safety"):
            await router.complete(LLMScene.NEED_GOAL_GENERATION, [LLMMessage(role="user", content="x")])
        # A call outside the annotate_call scope carries no extra (proves scoping).
        await router.complete(LLMScene.NEED_GOAL_GENERATION, [LLMMessage(role="user", content="y")])
    assert sink.llm_calls[0].extra == {"dominant_need_old": "social", "dominant_need_new": "safety"}
    assert sink.llm_calls[1].extra == {}


@pytest.mark.asyncio
async def test_each_call_gets_a_unique_call_id() -> None:
    clear_log_context()
    set_log_context(world_id="w", step="1")
    sink = InMemoryTraceSink()
    router = _router(_OkProvider(), sink)
    with observe_stage(Stage.DECISION, agent_id="a1"):
        await router.complete(LLMScene.AGENT_DECISION_MAIN, [LLMMessage(role="user", content="x")])
        await router.complete(LLMScene.AGENT_DECISION_MAIN, [LLMMessage(role="user", content="y")])
    ids = [c.call_id for c in sink.llm_calls]
    assert all(ids) and len(set(ids)) == 2  # every call has a non-empty, unique id


@pytest.mark.asyncio
async def test_router_without_sink_does_not_record_or_raise() -> None:
    clear_log_context()
    router = _router(_OkProvider())
    resp = await router.complete(LLMScene.WORLD_BUILDING, [LLMMessage(role="user", content="x")])
    assert resp.content == "ok"  # no sink, no error


@pytest.mark.asyncio
async def test_router_missing_stage_records_unknown() -> None:
    clear_log_context()
    set_log_context(world_id="w")
    sink = InMemoryTraceSink()
    router = _router(_OkProvider(), sink)
    await router.complete(LLMScene.WORLD_BUILDING, [LLMMessage(role="user", content="x")])
    assert sink.llm_calls[0].stage == Stage.UNKNOWN.value


@pytest.mark.asyncio
async def test_record_failure_is_swallowed() -> None:
    clear_log_context()

    class _BrokenSink(NullTraceSink):
        def record_llm_call(self, trace):  # type: ignore[override]
            raise RuntimeError("disk on fire")

    router = _router(_OkProvider("still works"), _BrokenSink())
    resp = await router.complete(LLMScene.WORLD_BUILDING, [LLMMessage(role="user", content="x")])
    assert resp.content == "still works"  # observability failure never breaks the call


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------

def _call(world="w", *, stage="decision", step=1, agent="a1", scene="agent_decision_main") -> LLMCallTrace:
    return LLMCallTrace(
        world_id=world, stage=stage, scene=scene,
        prompt_messages=[{"role": "user", "content": "p"}], response_content="r",
        temperature=0.7, max_tokens=100, input_tokens=10, output_tokens=4,
        model="mock", latency_ms=12.3, timestamp="t", agent_id=agent, step=step,
    )


@pytest.mark.asyncio
async def test_null_sink_is_noop() -> None:
    sink = NullTraceSink()
    sink.record_llm_call(_call())
    sink.record_step(StepTrace("w", 1, {"label": "t1"}, 5.0, "ts"))
    await sink.flush("w", 1)
    assert sink.read_calls("w") == []
    assert sink.list_steps("w") == []
    assert sink.list_stages("w") == []


def test_in_memory_sink_three_dimension_filters() -> None:
    sink = InMemoryTraceSink()
    sink.record_llm_call(_call(stage="decision", step=1, agent="a1"))
    sink.record_llm_call(_call(stage="memory", step=1, agent="a2"))
    sink.record_llm_call(_call(stage="decision", step=2, agent="a1"))
    assert len(sink.read_calls("w")) == 3
    assert len(sink.read_calls("w", step=1)) == 2
    assert len(sink.read_calls("w", stage="decision")) == 2
    assert len(sink.read_calls("w", agent_id="a1")) == 2
    assert len(sink.read_calls("w", step=2, stage="decision", agent_id="a1")) == 1
    assert sink.list_steps("w") == [1, 2]
    assert sink.list_agents("w") == ["a1", "a2"]
    assert sink.list_stages("w") == ["decision", "memory"]  # cognition-ordered


@pytest.mark.asyncio
async def test_buffered_jsonl_per_step_files_and_roundtrip(tmp_path) -> None:
    sink = BufferedJsonlTraceSink(base_dir=str(tmp_path))
    sink.record_llm_call(_call(step=7, stage="decision"))
    sink.record_step(StepTrace("w", 7, {"label": "晨"}, 99.5, "ts", phase_ms={"plan": 60.0, "exec": 30.0}))
    # nothing on disk until flush
    assert not (tmp_path / "w" / "step_000007.jsonl").exists()
    await sink.flush("w", 7)
    assert (tmp_path / "w" / "step_000007.jsonl").exists()

    calls = sink.read_calls("w", step=7)
    assert len(calls) == 1 and calls[0].stage == "decision" and calls[0].step == 7
    summaries = sink.read_step_summaries("w")
    assert len(summaries) == 1 and summaries[0].wall_ms == 99.5
    # Per-phase breakdown round-trips through the JSONL sink.
    assert summaries[0].phase_ms == {"plan": 60.0, "exec": 30.0}
    assert sink.list_steps("w") == [7]


def test_in_memory_sink_marks_only_the_addressed_call_unadopted() -> None:
    sink = InMemoryTraceSink()
    sink.record_llm_call(_call(stage="motivation", step=1, agent="a1"))
    sink.record_llm_call(_call(stage="motivation", step=1, agent="a2"))
    sink.record_llm_call(_call(stage="decision", step=1, agent="a1"))
    sink.mark_call_unadopted("w", step=1, agent_id="a1", stage="motivation", reason="conscripted")
    marked = [(c.stage, c.agent_id, c.adopted, c.reject_reason) for c in sink.llm_calls]
    assert marked == [("motivation", "a1", False, "conscripted"),
                      ("motivation", "a2", None, ""),
                      ("decision", "a1", None, "")]
    # No match means nothing happens; like the two context helpers, a missed mark only loses one
    # diagnostic.
    sink.mark_call_unadopted("w", step=9, agent_id="a1", stage="motivation", reason="conscripted")
    assert [c.adopted for c in sink.llm_calls] == [False, None, None]


@pytest.mark.asyncio
async def test_buffered_jsonl_unadopted_mark_lands_in_the_flushed_record(tmp_path) -> None:
    """The mark must hit the buffered object before flush so it lands on disk with its step (same as
    parse_ok)."""
    sink = BufferedJsonlTraceSink(base_dir=str(tmp_path))
    sink.record_llm_call(_call(step=3, stage="motivation", agent="a1"))
    sink.mark_call_unadopted("w", step=3, agent_id="a1", stage="motivation", reason="conscripted")
    await sink.flush("w", 3)
    out = sink.read_calls("w", step=3)[0]
    assert out.adopted is False and out.reject_reason == "conscripted"


def test_in_memory_sink_annotates_only_the_addressed_call() -> None:
    sink = InMemoryTraceSink()
    sink.record_llm_call(_call(stage="decision", step=1, agent="a1"))
    sink.record_llm_call(_call(stage="decision", step=2, agent="a1"))
    sink.annotate_recorded_call("w", step=1, agent_id="a1", stage="decision", conscripted=True)
    assert [c.extra for c in sink.llm_calls] == [{"conscripted": True}, {}]
    sink.annotate_recorded_call("w", step=1, agent_id="a1", stage="decision", note="x")
    assert sink.llm_calls[0].extra == {"conscripted": True, "note": "x"}  # merged, old keys kept


@pytest.mark.asyncio
async def test_buffered_jsonl_annotation_lands_in_the_flushed_record(tmp_path) -> None:
    sink = BufferedJsonlTraceSink(base_dir=str(tmp_path))
    sink.record_llm_call(_call(step=3, stage="decision", agent="a1"))
    sink.annotate_recorded_call("w", step=3, agent_id="a1", stage="decision", conscripted=True)
    await sink.flush("w", 3)
    assert sink.read_calls("w", step=3)[0].extra == {"conscripted": True}


@pytest.mark.asyncio
async def test_flush_hands_the_writer_thread_finished_records_only(tmp_path, monkeypatch) -> None:
    """Records keep arriving and being annotated on the loop while a flush writes. The writer
    thread gets serialized records, so neither can break or change what it writes."""
    import asyncio
    import threading

    sink = BufferedJsonlTraceSink(base_dir=str(tmp_path))
    call = _call(step=1)
    sink.record_llm_call(call)
    writing, release = threading.Event(), threading.Event()
    write = sink._write_segments

    def _blocking_write(world_id, segments):
        writing.set()
        release.wait(5)
        write(world_id, segments)

    monkeypatch.setattr(sink, "_write_segments", _blocking_write)
    flushing = asyncio.create_task(sink.flush("w"))
    await asyncio.to_thread(writing.wait, 5)
    sink.record_llm_call(_call(world="other", step=1))
    call.extra.update(late=True)
    release.set()
    await flushing

    assert sink.read_calls("w", step=1)[0].extra == {}
    assert ("other", "step_000001") in sink._buffer


def test_null_sink_post_hoc_writes_are_noops() -> None:
    sink = NullTraceSink()
    sink.mark_call_unadopted("w", step=1, agent_id="a1", stage="motivation", reason="x")
    sink.annotate_recorded_call("w", step=1, agent_id="a1", stage="decision", conscripted=True)


def test_step_trace_phase_ms_defaults_empty() -> None:
    # Positional construction without phase_ms stays valid → empty breakdown.
    trace = StepTrace("w", 1, {"label": "晨"}, 5.0, "ts")
    assert trace.phase_ms == {}


@pytest.mark.asyncio
async def test_buffered_jsonl_extra_roundtrips(tmp_path) -> None:
    """The generic extra annotation channel must survive asdict→jsonl→reconstruct."""
    sink = BufferedJsonlTraceSink(base_dir=str(tmp_path))
    call = _call(step=4, stage="motivation")
    call.extra = {"dominant_need_old": "social", "dominant_need_new": "safety"}
    sink.record_llm_call(call)
    await sink.flush("w", 4)
    out = sink.read_calls("w", step=4)[0]
    assert out.extra == {"dominant_need_old": "social", "dominant_need_new": "safety"}


@pytest.mark.asyncio
async def test_buffered_jsonl_build_segment(tmp_path) -> None:
    sink = BufferedJsonlTraceSink(base_dir=str(tmp_path))
    sink.record_llm_call(_call(step=None, stage="world_init", agent=None, scene="world_building"))
    await sink.flush("w")  # no step → flush all buffered (build)
    assert (tmp_path / "w" / "build.jsonl").exists()
    build_calls = [c for c in sink.read_calls("w") if c.step is None]
    assert len(build_calls) == 1 and build_calls[0].stage == "world_init"


@pytest.mark.asyncio
async def test_buffered_jsonl_flush_all_rescues_stragglers(tmp_path) -> None:
    """Deferred background writes run from a frozen context and can land after their step's flush,
    which the runtime never repeats per step. A final flush(world_id) with no step must rescue
    them; the runtime relies on this at shutdown."""
    sink = BufferedJsonlTraceSink(base_dir=str(tmp_path))
    # step 2 flushed as the runtime advances past it
    sink.record_llm_call(_call(step=2, stage="decision"))
    await sink.flush("w", 2)
    assert len(sink.read_calls("w", step=2)) == 1

    # a straggler arrives for step 2 AFTER its per-step flush (frozen-context deferred write);
    # the runtime never re-flushes step 2 per step, so it is not yet on disk...
    sink.record_llm_call(_call(step=2, stage="memory", scene="memory_summarization"))
    assert len(sink.read_calls("w", step=2)) == 1
    # ...until the shutdown flush-all writes every remaining buffered segment.
    await sink.flush("w")  # no step → flush ALL remaining segments
    rescued = sink.read_calls("w", step=2)
    assert len(rescued) == 2
    assert {c.stage for c in rescued} == {"decision", "memory"}


@pytest.mark.asyncio
async def test_buffered_jsonl_prompt_truncation(tmp_path) -> None:
    sink = BufferedJsonlTraceSink(base_dir=str(tmp_path), prompt_max_chars=5)
    long = LLMCallTrace(
        world_id="w", stage="decision", scene="s",
        prompt_messages=[{"role": "user", "content": "0123456789"}], response_content="abcdefghij",
        temperature=0.7, max_tokens=10, input_tokens=1, output_tokens=1,
        model="m", latency_ms=1.0, timestamp="t", agent_id="a", step=1,
    )
    sink.record_llm_call(long)
    await sink.flush("w", 1)
    out = sink.read_calls("w", step=1)[0]
    assert out.prompt_messages[0]["content"].startswith("01234")
    assert "truncated" in out.prompt_messages[0]["content"]
    assert "truncated" in out.response_content


def test_in_memory_sink_delete_run_traces_keeps_build_calls() -> None:
    sink = InMemoryTraceSink()
    sink.record_llm_call(_call(step=None, stage="world_init", agent=None))
    sink.record_llm_call(_call(step=1))
    sink.record_step(StepTrace("w", 1, {"label": "晨"}, 5.0, "ts"))
    sink.record_llm_call(_call(world="other", step=1))

    sink.delete_run_traces("w")

    remaining = sink.read_calls("w")
    assert [c.step for c in remaining] == [None]  # build call survives, run calls gone
    assert sink.read_step_summaries("w") == []
    assert len(sink.read_calls("other")) == 1  # other worlds untouched


@pytest.mark.asyncio
async def test_buffered_jsonl_delete_run_traces_keeps_build_segment(tmp_path) -> None:
    sink = BufferedJsonlTraceSink(base_dir=str(tmp_path))
    sink.record_llm_call(_call(step=None, stage="world_init", agent=None))
    await sink.flush("w")
    sink.record_llm_call(_call(step=1))
    await sink.flush("w", 1)

    sink.delete_run_traces("w")

    assert (tmp_path / "w" / "build.jsonl").exists()
    assert not (tmp_path / "w" / "step_000001.jsonl").exists()
    assert [c.step for c in sink.read_calls("w")] == [None]


@pytest.mark.asyncio
async def test_delete_run_traces_stops_a_rewound_clock_merging_two_runs(tmp_path) -> None:
    """On reset the clock restarts at 1, and the sink appends. Without
    purging the previous run's step traces, run #2's step 1 lands on run #1's step 1
    and the read side — which groups by step number — silently merges them."""
    sink = BufferedJsonlTraceSink(base_dir=str(tmp_path))
    sink.record_llm_call(_call(step=1, stage="decision"))
    await sink.flush("w", 1)

    sink.delete_run_traces("w")  # what reset_session does

    sink.record_llm_call(_call(step=1, stage="motivation"))
    await sink.flush("w", 1)

    calls = sink.read_calls("w", step=1)
    assert [c.stage for c in calls] == ["motivation"]  # only the new run, not both


def test_stage_sort_key_orders_unknown_last() -> None:
    assert stage_sort_key("world_init") < stage_sort_key("decision")
    assert stage_sort_key("decision") < stage_sort_key("event")
    assert stage_sort_key("unknown") > stage_sort_key("event")


# ---------------------------------------------------------------------------
# Failure rates: call failures + parse failures
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_router_records_failed_call_then_reraises() -> None:
    clear_log_context()
    set_log_context(world_id="w1", step="3")
    sink = InMemoryTraceSink()
    router = _router(_FailProvider(), sink)
    with observe_stage(Stage.DECISION, agent_id="a1"):
        with pytest.raises(RuntimeError):
            await router.complete(LLMScene.AGENT_DECISION_MAIN, [LLMMessage(role="user", content="hi")])
    assert len(sink.llm_calls) == 1
    call = sink.llm_calls[0]
    assert call.ok is False
    assert "exploded" in call.error
    assert call.stage == "decision" and call.agent_id == "a1" and call.step == 3
    assert call.input_tokens == 0 and call.output_tokens == 0


@pytest.mark.asyncio
async def test_parse_ok_attaches_to_the_active_call() -> None:
    clear_log_context()
    set_log_context(world_id="w", step="2")
    # success parse → parse_ok True on the recorded call
    sink = InMemoryTraceSink()
    router = _router(_OkProvider('{"a": 1}'), sink)
    with observe_stage(Stage.DECISION, agent_id="a1"):
        resp = await router.complete(LLMScene.AGENT_DECISION_MAIN, [LLMMessage(role="user", content="x")])
        extract_json(resp.content)
    assert sink.llm_calls[0].parse_ok is True

    # failed parse → parse_ok False
    sink2 = InMemoryTraceSink()
    router2 = _router(_OkProvider("not json"), sink2)
    with observe_stage(Stage.DECISION, agent_id="a1"):
        resp = await router2.complete(LLMScene.AGENT_DECISION_MAIN, [LLMMessage(role="user", content="x")])
        with pytest.raises(Exception):
            extract_json(resp.content)
    assert sink2.llm_calls[0].parse_ok is False


@pytest.mark.asyncio
async def test_parse_ok_via_extract_json_object_and_no_parse_is_none() -> None:
    clear_log_context()
    set_log_context(world_id="w", step="1")
    sink = InMemoryTraceSink()
    router = _router(_OkProvider("garbage"), sink)
    with observe_stage(Stage.MEMORY):
        resp = await router.complete(LLMScene.MEMORY_IMPORTANCE, [LLMMessage(role="user", content="x")])
        assert extract_json_object(resp.content) is None  # parse failure
    assert sink.llm_calls[0].parse_ok is False

    # A call with no parse afterwards keeps parse_ok None (e.g. a prose summary).
    sink2 = InMemoryTraceSink()
    router2 = _router(_OkProvider("just prose"), sink2)
    await router2.complete(LLMScene.MEMORY_SUMMARIZATION, [LLMMessage(role="user", content="x")])
    assert sink2.llm_calls[0].parse_ok is None


def test_note_active_call_parse_is_noop_without_active_call() -> None:
    set_active_call(None)
    note_active_call_parse(False)  # must not raise

    class _Box:
        parse_ok = None

    box = _Box()
    set_active_call(box)
    note_active_call_parse(True)
    assert box.parse_ok is True
    # first parse wins — a later stray parse does not overwrite
    note_active_call_parse(False)
    assert box.parse_ok is True
    set_active_call(None)


def test_failed_call_has_no_parse_target() -> None:
    # After a failed call the router clears the active call, so a stray parse is ignored.
    set_active_call(None)
    note_active_call_parse(False)  # no active call → no-op, no error


# ---------------------------------------------------------------------------
# adopted / reject_reason: the third failure question, "did downstream use it?" Neither ok nor
# parse_ok can see it: an HTTP 200 with valid JSON can still be discarded whole (missing required
# field, hallucinated index), so that step silently vanishes while both metrics stay green.
# ---------------------------------------------------------------------------


def test_note_active_call_adoption_records_verdict_and_reason() -> None:
    from core.context import note_active_call_adoption

    set_active_call(None)
    note_active_call_adoption(False, "whatever")  # no active call → no-op, must not raise

    class _Box:
        adopted = None
        reject_reason = ""

    box = _Box()
    set_active_call(box)
    note_active_call_adoption(False, "missing_action_description")
    assert box.adopted is False
    assert box.reject_reason == "missing_action_description"

    # first verdict wins; later stray verdicts must not overwrite it
    note_active_call_adoption(True)
    assert box.adopted is False
    assert box.reject_reason == "missing_action_description"
    set_active_call(None)


def test_annotate_active_call_merges_into_extra() -> None:
    """Post-call channel: merge diagnostics only known after the call (e.g. goals actually enqueued
    after dedup) into the active call's extra."""
    from core.context import annotate_active_call

    set_active_call(None)
    annotate_active_call(short_term_goals_new=["x"])  # no active call → no-op, must not raise

    class _Box:
        extra: dict = {}

    box = _Box()
    box.extra = {"prior": 1}
    set_active_call(box)
    annotate_active_call(short_term_goals_new=["a", "b"])
    annotate_active_call(short_term_goals_new=["c"])  # later write overwrites the same key
    assert box.extra == {"prior": 1, "short_term_goals_new": ["c"]}  # merged, old keys kept
    set_active_call(None)


@pytest.mark.asyncio
async def test_discarded_call_is_ok_and_parseable_yet_not_adopted() -> None:
    """The core case: a discarded call is perfectly clean on ok/parse_ok, which is why adopted
    exists."""
    from core.context import note_active_call_adoption

    clear_log_context()
    set_log_context(world_id="w", step="12")
    sink = InMemoryTraceSink()
    router = _router(_OkProvider('{"act": true, "selected_index": 6}'), sink)
    with observe_stage(Stage.DECISION, agent_id="a1"):
        resp = await router.complete(LLMScene.AGENT_DECISION_MAIN, [LLMMessage(role="user", content="x")])
        extract_json(resp.content)  # the JSON is valid
        note_active_call_adoption(False, "missing_action_description")  # but unusable

    call = sink.llm_calls[0]
    assert call.ok is True, "provider 答了"
    assert call.parse_ok is True, "JSON 也解析出来了"
    assert call.adopted is False, "——可引擎把它整份丢了,只有这个字段能看见"
    assert call.reject_reason == "missing_action_description"


@pytest.mark.asyncio
async def test_no_verdict_stays_none() -> None:
    """Uninstrumented paths take no position → None (not False): don't record "nobody reported" as
    "discarded"."""
    clear_log_context()
    set_log_context(world_id="w", step="1")
    sink = InMemoryTraceSink()
    router = _router(_OkProvider("just prose"), sink)
    await router.complete(LLMScene.MEMORY_SUMMARIZATION, [LLMMessage(role="user", content="x")])
    assert sink.llm_calls[0].adopted is None
    assert sink.llm_calls[0].reject_reason == ""

def test_build_scene_order_names_real_scenes() -> None:
    """BUILD_SCENE_ORDER holds strings (avoiding an llm↔trace import cycle), so typos don't fail on
    their own.

    This test is that guard: every value must be a real LLMScene, and it must cover the
    scenes world building actually emits. Miss one and it silently sinks to the bottom in the
    observer UI.
    """
    from core.interfaces.llm import LLMScene
    from core.interfaces.trace import BUILD_SCENE_ORDER, build_scene_sort_key

    known = {s.value for s in LLMScene}
    assert set(BUILD_SCENE_ORDER) <= known, set(BUILD_SCENE_ORDER) - known
    # Order = call order in world/builder.py, not alphabetical (alphabetical would put template
    # selection last).
    assert build_scene_sort_key("world_template_selection") < build_scene_sort_key("cast_design")
    assert build_scene_sort_key("cast_design") < build_scene_sort_key("persona_generation")
    assert build_scene_sort_key("persona_generation") < build_scene_sort_key("need_goal_generation")
    # Unregistered scenes sink to the bottom instead of raising.
    assert build_scene_sort_key("not_a_scene") == len(BUILD_SCENE_ORDER)


# --- The read cache is a budgeted cache, not a registry --------------------------------------
#
# Each browsed world keeps all its parsed trace records resident and only yields on its own file
# changes; without a budget, resident memory grows with the number of worlds browsed.


@pytest.mark.asyncio
async def test_cold_worlds_are_evicted_once_the_cache_exceeds_its_budget(tmp_path, monkeypatch) -> None:
    import providers.trace.file as trace_file

    sink = BufferedJsonlTraceSink(base_dir=str(tmp_path))
    for world in ("w1", "w2", "w3"):
        for _ in range(3):
            sink.record_llm_call(_call(world, step=1))
        await sink.flush(world, 1)

    # A budget that fits only one world: reading a second must evict the first.
    one_world_bytes = sum(p.stat().st_size for p in (tmp_path / "w1").glob("*.jsonl"))
    monkeypatch.setattr(trace_file, "_MAX_CACHED_TRACE_BYTES", one_world_bytes)

    sink.read_calls("w1")
    sink.read_calls("w2")
    sink.read_calls("w3")
    assert list(sink._read_cache) == ["w3"]


@pytest.mark.asyncio
async def test_the_world_just_read_survives_even_when_it_alone_blows_the_budget(
    tmp_path, monkeypatch
) -> None:
    """Otherwise the world being viewed is fully reparsed on every poll, defeating the cache."""
    import providers.trace.file as trace_file

    sink = BufferedJsonlTraceSink(base_dir=str(tmp_path))
    for _ in range(3):
        sink.record_llm_call(_call("big", step=1))
    await sink.flush("big", 1)

    monkeypatch.setattr(trace_file, "_MAX_CACHED_TRACE_BYTES", 1)  # every world is over budget
    sink.read_calls("big")
    assert list(sink._read_cache) == ["big"]


@pytest.mark.asyncio
async def test_eviction_goes_by_last_read_not_by_insertion_order(tmp_path, monkeypatch) -> None:
    """Evict by most recently read, not insertion order: comparing two worlds back and forth would
    otherwise evict the one you're looking at. The budget is enforced only when inserting a new
    parse (a hit adds nothing), so a third world triggers it.
    """
    import providers.trace.file as trace_file

    sink = BufferedJsonlTraceSink(base_dir=str(tmp_path))
    for world in ("first", "second", "third"):
        for _ in range(3):
            sink.record_llm_call(_call(world, step=1))
        await sink.flush(world, 1)

    sink.read_calls("first")
    sink.read_calls("second")
    sink.read_calls("first")        # first becomes the most recently read again

    one_world = sum(p.stat().st_size for p in (tmp_path / "first").glob("*.jsonl"))
    monkeypatch.setattr(trace_file, "_MAX_CACHED_TRACE_BYTES", one_world * 2)
    sink.read_calls("third")        # miss → insert → trim to the two-world budget

    # The evicted one is second (least recently read), not first (inserted first).
    assert list(sink._read_cache) == ["first", "third"]


@pytest.mark.asyncio
async def test_a_new_flush_still_invalidates_the_cached_parse(tmp_path) -> None:
    """The budget must not break invalidation: when the same world lands another step, the next read
    must see it."""
    sink = BufferedJsonlTraceSink(base_dir=str(tmp_path))
    sink.record_llm_call(_call(step=1, stage="decision"))
    await sink.flush("w", 1)
    assert sink.list_steps("w") == [1]

    sink.record_llm_call(_call(step=2, stage="decision"))
    await sink.flush("w", 2)
    assert sink.list_steps("w") == [1, 2]
