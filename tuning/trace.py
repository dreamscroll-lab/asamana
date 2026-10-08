"""Trace sinks for development-phase tuning.

LLM calls are recorded by the production ``LLMRouter`` into its trace sink — the same
``core.interfaces.trace.LLMCallTrace`` the runtime and the trace page use. Tuning adds one
record of its own:

- ``PhaseTrace`` — one pipeline phase's structured product (e.g. the world builder's
  ThemeAnalysis), emitted by an orchestration-layer hook.

``JsonlTraceSink`` partitions on disk as ``{base}/{world_id}/{segment}.jsonl`` — one JSON
object per line, tagged with ``kind``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from core.interfaces.llm import LLMRouter, LLMScene
from core.interfaces.trace import LLMCallTrace, to_jsonable
from core.logging import get_logger
from providers.trace.in_memory import InMemoryTraceSink as _RecordingSink

logger = get_logger(__name__)

# Tuning keeps its own trace root, separate from the production observability
# stack (./data/traces) — both partition by {world_id}/ and would otherwise
# collide on build.jsonl and on production's *.jsonl read glob.
_DEFAULT_TRACE_DIR = "./data/tuning_traces"
_UNKNOWN_WORLD = "_unknown"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PhaseTrace:
    """One pipeline phase's structured product."""

    world_id: str
    phase: str
    outputs: Any  # dict or list of jsonable products, depending on the phase
    timestamp: str
    step: int | None = None
    agent_id: str | None = None
    inputs: dict[str, Any] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------

class InMemoryTraceSink(_RecordingSink):
    """The production in-memory sink plus this run's phase records."""

    def __init__(self) -> None:
        super().__init__()
        self.phases: list[PhaseTrace] = []

    def record_phase(self, trace: PhaseTrace) -> None:
        self.phases.append(trace)


def traced_router(base: LLMRouter, sink: InMemoryTraceSink, *, max_concurrent: int = 0) -> LLMRouter:
    """A router over *base*'s providers that records every call into *sink*.

    Rebuilt from the providers alone, so it never inherits *base*'s production trace sink: a
    tuning run's calls land only in the tuning sink.
    """
    return LLMRouter(
        {scene: base.get(scene) for scene in LLMScene},
        max_concurrent=max_concurrent,
        trace_sink=sink,
    )


def dump_llm_calls(sink: InMemoryTraceSink, dest: Path) -> None:
    """Dump every LLM call from this run verbatim, for the developer tools.

    Why all of them rather than just this stage's: one stage run fires a chain of calls —
    perception, motivation, the stage itself, then the judge. When tuning a prompt, what you
    usually need is not the final call but what upstream fed it (decide's user section is the
    output of perception and motivation). Each stage's `prompt.json` only picks its own calls and
    can't show that chain.

    The shape is ``LLMCallTrace`` serialized as-is — the same record the trace page reads, so the
    page renders it with the same renderer instead of a second one.
    """
    dest.write_text(
        json.dumps([to_jsonable(c) for c in sink.llm_calls], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_json(path: Path, obj: Any) -> None:
    """Write *obj* as indented JSON at *path*, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


class JsonlTraceSink(InMemoryTraceSink):
    """Appends records to ``{base}/{world_id}/{segment}.jsonl`` as well as keeping them in memory.

    ``segment`` selects the file within a world's trace directory (default ``build``).
    Each line is ``{"kind": "llm_call" | "phase", ...}``.
    """

    def __init__(self, base_dir: str = _DEFAULT_TRACE_DIR, *, segment: str = "build") -> None:
        super().__init__()
        self._base = Path(base_dir)
        self._segment = segment

    def _path(self, world_id: str, segment: str | None = None) -> Path:
        wid = world_id or _UNKNOWN_WORLD
        seg = segment or self._segment
        return self._base / wid / f"{seg}.jsonl"

    def _append(self, world_id: str, record: dict[str, Any]) -> None:
        try:
            path = self._path(world_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001 — tuning must never break the observed pipeline
            logger.warning("trace_write_failed", extra={"world_id": world_id, "error": str(exc)})

    def truncate(self, world_id: str, segment: str | None = None) -> None:
        """Delete a world's segment file so a fresh run does not accumulate with prior ones."""
        path = self._path(world_id, segment)
        if path.exists():
            path.unlink()

    def record_llm_call(self, trace: LLMCallTrace) -> None:
        super().record_llm_call(trace)
        self._append(trace.world_id, {"kind": "llm_call", **dataclasses.asdict(trace)})

    def record_phase(self, trace: PhaseTrace) -> None:
        super().record_phase(trace)
        self._append(trace.world_id, {"kind": "phase", **dataclasses.asdict(trace)})
