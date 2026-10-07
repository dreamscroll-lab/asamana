"""Unit tests for the tuning trace layer (sink + traced router)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum

import pytest

from core.context import set_log_context
from core.interfaces.llm import LLMMessage, LLMProvider, LLMResponse, LLMRouter, LLMScene
from core.interfaces.trace import to_jsonable
from tuning.trace import InMemoryTraceSink, JsonlTraceSink, PhaseTrace, traced_router


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


class _FailOnceProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        return LLMResponse(content="recovered", input_tokens=1, output_tokens=1, model="mock")

    async def stream(self, messages, temperature=0.7) -> AsyncIterator[str]:
        yield "recovered"


def _router(provider: LLMProvider, sink) -> LLMRouter:
    return LLMRouter({s: provider for s in LLMScene}, trace_sink=sink)


@pytest.mark.asyncio
async def test_traced_router_is_transparent_and_records_call() -> None:
    set_log_context(world_id="w1", agent_id="a1", step="7")
    provider = _OkProvider("hello")
    sink = InMemoryTraceSink()
    router = _router(provider, sink)

    resp = await router.complete(
        LLMScene.WORLD_BUILDING,
        [LLMMessage(role="user", content="hi")],
        temperature=0.3,
        max_tokens=50,
    )

    assert resp.content == "hello"  # transparent passthrough
    assert len(sink.llm_calls) == 1
    rec = sink.llm_calls[0]
    assert rec.world_id == "w1"
    assert rec.agent_id == "a1"
    assert rec.step == 7
    assert rec.scene == "world_building"
    assert rec.prompt_messages == [{"role": "user", "content": "hi"}]
    assert rec.response_content == "hello"
    assert rec.temperature == 0.3 and rec.max_tokens == 50


@pytest.mark.asyncio
async def test_retry_path_records_both_attempts() -> None:
    """complete_with_retry delegates to complete, so both attempts are recorded: the failed one
    as ok=False with an empty response, then the recovered one."""
    set_log_context(world_id="w2")
    provider = _FailOnceProvider()
    sink = InMemoryTraceSink()
    router = _router(provider, sink)

    resp = await router.complete_with_retry(
        LLMScene.CAST_DESIGN, [LLMMessage(role="user", content="x")], retry_delay=0.0
    )

    assert resp.content == "recovered"
    assert provider.calls == 2
    assert [c.ok for c in sink.llm_calls] == [False, True]
    assert sink.llm_calls[-1].response_content == "recovered"


@pytest.mark.asyncio
async def test_repeated_completions_each_recorded() -> None:
    """Builder-style semantic retries (each a successful complete) are each captured."""
    set_log_context(world_id="w3")
    provider = _OkProvider("{}")
    sink = InMemoryTraceSink()
    router = _router(provider, sink)

    for _ in range(3):
        await router.complete(LLMScene.WORLD_BUILDING, [LLMMessage(role="user", content="x")])

    assert len(sink.llm_calls) == 3


def test_jsonl_sink_roundtrip(tmp_path) -> None:
    sink = JsonlTraceSink(str(tmp_path))
    sink.record_phase(
        PhaseTrace(world_id="wx", phase="world_build.theme_analysis", outputs={"world_name": "X"}, timestamp="t0")
    )
    (path,) = (tmp_path / "wx").glob("*.jsonl")
    assert path.stem == "build"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["kind"] == "phase"
    assert records[0]["outputs"] == {"world_name": "X"}


def test_unknown_world_partition(tmp_path) -> None:
    sink = JsonlTraceSink(str(tmp_path))
    sink.record_phase(PhaseTrace(world_id="", phase="p", outputs={}, timestamp="t"))
    assert [p.name for p in tmp_path.iterdir()] == ["_unknown"]


class _Color(Enum):
    RED = "red"


@dataclass
class _Nested:
    color: _Color
    items: list[int]


def test_to_jsonable_handles_dataclass_enum_and_nesting() -> None:
    out = to_jsonable(_Nested(color=_Color.RED, items=[1, 2]))
    assert out == {"color": "red", "items": [1, 2]}


class _HasAsDict:
    def as_dict(self) -> dict:
        return {"k": _Color.RED}


def test_to_jsonable_prefers_as_dict() -> None:
    assert to_jsonable(_HasAsDict()) == {"k": "red"}

# --------------------------------------------------------------------------- #
# Isolation: traces from tuning runs must never touch a world's real traces
# --------------------------------------------------------------------------- #

def test_tuning_bootstrap_disables_production_tracing() -> None:
    """The tuning process must install NullTraceSink so the production trace channel has nowhere to
    write.

    We want a guarantee by construction: `record_llm_call` only buffers and writes on flush, and
    relying on "nobody happens to flush" is too fragile.
    """
    import providers.trace  # noqa: F401 — importing registers the trace provider
    from config.loader import load_config
    from core.container import _build_trace_sink

    production = load_config()
    assert production.observability.enabled, "生产默认开着,否则这条测试什么也没证明"

    tuned = load_config()
    tuned.observability.enabled = False  # exactly what tuning.cli.bootstrap_application does
    sink = _build_trace_sink(tuned)
    assert type(sink).__name__ == "NullTraceSink", type(sink).__name__


def test_tuning_cli_bootstrap_turns_observability_off() -> None:
    """Pin the above to tuning.cli's own bootstrap (which deliberately doesn't reuse the production
    entry point)."""
    import inspect

    from tuning import cli

    # Function body only: the docstring mentions these names too, so position comparisons would hit
    # it.
    src = inspect.getsource(cli.bootstrap_application)
    body = src.rsplit('"""', 1)[-1]
    assert "config.observability.enabled = False" in body
    assert "Container.from_config(config)" in body
    # Order matters: it must be turned off before building the container, or the router already
    # holds the production sink.
    assert body.index("enabled = False") < body.index("Container.from_config")


def test_traced_router_does_not_carry_a_production_sink() -> None:
    """`traced_router` rebuilds from base's providers and does not inherit base's trace_sink.

    So calls through it only reach the tuning sink. This is the second guard: even if someone runs
    tuning with observability on, calls through the traced router never reach production traces.
    """
    production_sink = InMemoryTraceSink()
    tuning_sink = InMemoryTraceSink()
    base = LLMRouter({scene: object() for scene in LLMScene}, trace_sink=production_sink)
    wrapped = traced_router(base, tuning_sink)
    assert wrapped.trace_sink is tuning_sink, "包裹后的 router 不该握着生产 sink"

@pytest.mark.asyncio
async def test_null_sink_writes_nothing_even_when_flushed(tmp_path) -> None:
    """The last link in the isolation chain: with NullTraceSink installed, record, flush and delete
    write zero bytes."""
    from datetime import datetime

    from core.interfaces.trace import LLMCallTrace
    from providers.trace.null import NullTraceSink

    sink = NullTraceSink()
    sink.record_llm_call(
        LLMCallTrace(
            world_id="w-real", stage="decision", scene="agent_decision_main",
            prompt_messages=[{"role": "user", "content": "x"}], response_content="y",
            temperature=0.7, max_tokens=100, input_tokens=1, output_tokens=1,
            model="m", latency_ms=1.0, timestamp=datetime.now().isoformat(), step=3,
        )
    )
    await sink.flush("w-real", 3)
    await sink.flush("w-real")
    assert list(tmp_path.rglob("*")) == [], "空 sink 不该在任何地方留下文件"
    assert sink.read_calls("w-real") == []
    # delete is a no-op too: the tuning process can't even "delete production traces".
    sink.delete_world("w-real")
    sink.delete_run_traces("w-real")
