"""Buffered JSONL trace sink.

Layout::

    {base}/{world_id}/step_{step:06d}.jsonl   one runtime step's records
    {base}/{world_id}/build.jsonl             world-build (step-less) records

Each line is one JSON object tagged ``{"kind": "llm_call" | "step", ...}``.

Records are buffered in memory per ``(world_id, segment)`` and written once on
``flush`` — under ``max_concurrent_llm`` fan-out, per-call synchronous appends
would block the event loop and interleave lines; a per-step flush is bounded and
makes the whole step land atomically. The runtime flushes each step; the
application flushes the ``build`` segment after world construction.
"""

from __future__ import annotations

import asyncio

import dataclasses
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, NamedTuple

from core.context import note_call_adoption
from core.factory import ComponentKind, ProviderFactory
from core.interfaces.trace import (
    LLMCallTrace,
    StepTrace,
    TraceSink,
    stage_sort_key,
)
from core.logging import get_logger

logger = get_logger(__name__)

_UNKNOWN_WORLD = "_unknown"
_BUILD_SEGMENT = "build"

# Read-cache byte budget in on-disk size (a good proxy for parsed size): enough for the world
# being viewed plus the one just switched from, not every world this process ever opened.
_MAX_CACHED_TRACE_BYTES = 64 * 1024 * 1024


class _CachedParse(NamedTuple):
    """All parsed trace records for one world, plus the signature of the files they came from.

    ``nbytes`` is the files' on-disk size, used for cache accounting (see
    _MAX_CACHED_TRACE_BYTES).
    """

    signature: tuple[tuple[str, float, int], ...]
    records: list[dict[str, Any]]
    nbytes: int


@ProviderFactory.register("jsonl", kind=ComponentKind.TRACE)
class BufferedJsonlTraceSink(TraceSink):
    """Buffer trace records per step, flush to per-step JSONL files."""

    def __init__(
        self,
        base_dir: str = "./data/traces",
        *,
        prompt_max_chars: int = 0,
    ) -> None:
        self._base = Path(base_dir)
        # 0 → no truncation. Otherwise cap each prompt message / response length
        # written to disk to guard against runaway prompts filling the volume.
        self._prompt_max_chars = prompt_max_chars
        # (world_id, segment) -> list of jsonl record dicts awaiting flush.
        # Buffer holds the trace *objects* (not dicts) so a call's parse_ok, filled in
        # by the parser after record_llm_call, is captured when the step is flushed.
        self._buffer: dict[tuple[str, str], list[Any]] = {}
        # world_id -> parsed records, invalidated when the files change (count / mtime / size).
        # LRU-ordered and budgeted by _evict_cold_parses, so memory doesn't grow with worlds browsed.
        self._read_cache: "OrderedDict[str, _CachedParse]" = OrderedDict()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _world_dir(self, world_id: str) -> Path:
        return self._base / (world_id or _UNKNOWN_WORLD)

    @staticmethod
    def _segment_for_step(step: int | None) -> str:
        return _BUILD_SEGMENT if step is None or step < 0 else f"step_{step:06d}"

    def _truncate(self, text: str) -> str:
        if self._prompt_max_chars and len(text) > self._prompt_max_chars:
            return text[: self._prompt_max_chars] + "…[truncated]"
        return text

    # ------------------------------------------------------------------
    # Write side
    # ------------------------------------------------------------------

    def record_llm_call(self, trace: LLMCallTrace) -> None:
        key = (trace.world_id, self._segment_for_step(trace.step))
        self._buffer.setdefault(key, []).append(trace)

    def record_step(self, trace: StepTrace) -> None:
        key = (trace.world_id, self._segment_for_step(trace.step))
        self._buffer.setdefault(key, []).append(trace)

    def _recorded(
        self, world_id: str, step: int, agent_id: str, stage: str
    ) -> list[LLMCallTrace]:
        """Matching calls still buffered and not yet flushed. Mutating them gets the change
        written with the step (as with parse_ok). Flushed calls are never rewritten: that would
        mean rewriting the whole segment file."""
        return [obj for obj in self._buffer.get((world_id, self._segment_for_step(step)), [])
                if isinstance(obj, LLMCallTrace)
                and obj.agent_id == agent_id and obj.stage == stage]

    def mark_call_unadopted(
        self, world_id: str, *, step: int, agent_id: str, stage: str, reason: str
    ) -> None:
        for call in self._recorded(world_id, step, agent_id, stage):
            note_call_adoption(call, False, reason)

    def annotate_recorded_call(
        self, world_id: str, *, step: int, agent_id: str, stage: str, **fields: Any
    ) -> None:
        for call in self._recorded(world_id, step, agent_id, stage):
            call.extra.update(fields)

    def _to_record(self, obj: Any) -> dict[str, Any]:
        """Serialise a buffered trace object to its jsonl record dict (at flush time,
        after parse_ok has been filled in)."""
        if isinstance(obj, LLMCallTrace):
            record = {"kind": "llm_call", **dataclasses.asdict(obj)}
            if self._prompt_max_chars:
                record["prompt_messages"] = [
                    {"role": m.get("role", ""), "content": self._truncate(m.get("content", ""))}
                    for m in obj.prompt_messages
                ]
                record["response_content"] = self._truncate(obj.response_content)
            return record
        return {"kind": "step", **dataclasses.asdict(obj)}

    async def flush(self, world_id: str, step: int | None = None) -> None:
        """Write buffered records. With a step, flush that step's segment; without,
        flush every buffered segment for the world (used for the build segment).

        Taking and serializing stay on the loop, where records are added and annotated; only the
        file writes go to a thread. A thread iterating ``_buffer`` or a call's ``extra`` would
        race those writers and lose the segment.
        """
        if step is not None:
            keys = [(world_id, self._segment_for_step(step))]
        else:
            keys = [key for key in self._buffer if key[0] == world_id]
        segments: list[tuple[str, list[dict[str, Any]]]] = []
        for key in keys:
            buffered = self._buffer.pop(key, None)
            if buffered:
                segments.append((key[1], [self._to_record(obj) for obj in buffered]))
        if segments:
            await asyncio.to_thread(self._write_segments, world_id, segments)
        self._read_cache.pop(world_id, None)

    def _write_segments(self, world_id: str, segments: list[tuple[str, list[dict[str, Any]]]]) -> None:
        for segment, records in segments:
            self._write_segment(world_id, segment, records)

    def delete_world(self, world_id: str) -> None:
        # Purge on-disk trace segments (step_*.jsonl + build.jsonl) and drop any
        # in-memory buffer / read cache for this world. Disk errors are swallowed
        # with a warning: observability must never break the run (matches
        # ``_write_segment``'s policy).
        try:
            shutil.rmtree(self._world_dir(world_id), ignore_errors=True)
        except Exception as exc:  # noqa: BLE001 — never break the caller for observability
            logger.warning(
                "trace_delete_failed",
                extra={"world_id": world_id, "error": str(exc)},
            )
        for key in [k for k in self._buffer if k[0] == world_id]:
            self._buffer.pop(key, None)
        self._read_cache.pop(world_id, None)
        logger.info("world_traces_deleted", extra={"world_id": world_id})

    def delete_run_traces(self, world_id: str) -> None:
        # Drop every step_*.jsonl (and its buffered records); build.jsonl stays.
        # Disk errors are swallowed with a warning, matching ``delete_world`` /
        # ``_write_segment``: observability must never break the caller.
        try:
            for path in self._world_dir(world_id).glob("step_*.jsonl"):
                path.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 — never break the caller for observability
            logger.warning(
                "trace_run_delete_failed",
                extra={"world_id": world_id, "error": str(exc)},
            )
        for key in [k for k in self._buffer if k[0] == world_id and k[1] != _BUILD_SEGMENT]:
            self._buffer.pop(key, None)
        self._read_cache.pop(world_id, None)
        logger.info("world_run_traces_deleted", extra={"world_id": world_id})

    def _write_segment(self, world_id: str, segment: str, records: list[dict[str, Any]]) -> None:
        try:
            path = self._world_dir(world_id) / f"{segment}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001 — observability must never break the run
            logger.warning("trace_write_failed", extra={"world_id": world_id, "segment": segment, "error": str(exc)})

    # ------------------------------------------------------------------
    # Read side (backs the web dashboard's step / stage / agent pivots)
    # ------------------------------------------------------------------

    def _all_records(self, world_id: str) -> list[dict[str, Any]]:
        """Parse every JSONL record for a world, cached on file signature."""
        world_dir = self._world_dir(world_id)
        if not world_dir.exists():
            return []
        files = sorted(world_dir.glob("*.jsonl"))
        stats = [(path, path.stat()) for path in files]
        signature = tuple((path.name, st.st_mtime, st.st_size) for path, st in stats)
        cached = self._read_cache.get(world_id)
        if cached is not None and cached.signature == signature:
            self._read_cache.move_to_end(world_id)
            return cached.records
        records: list[dict[str, Any]] = []
        for path in files:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("trace_record_unparseable", extra={"world_id": world_id, "file": path.name})
        self._read_cache[world_id] = _CachedParse(
            signature, records, sum(st.st_size for _, st in stats)
        )
        self._read_cache.move_to_end(world_id)
        self._evict_cold_parses()
        return records

    def _evict_cold_parses(self) -> None:
        """Evict the least recently read worlds until the cache is within budget.

        The most recently read world is never evicted, even if it alone exceeds the budget: the
        cache exists to avoid a full reparse, and evicting the world being viewed would make
        every poll pay for one.
        """
        total = sum(entry.nbytes for entry in self._read_cache.values())
        while total > _MAX_CACHED_TRACE_BYTES and len(self._read_cache) > 1:
            _, evicted = self._read_cache.popitem(last=False)
            total -= evicted.nbytes

    @staticmethod
    def _is_production_call(record: dict[str, Any]) -> bool:
        """A production llm_call record carries a ``stage``. Records without it are
        foreign (e.g. a legacy tuning trace left in the dir) and are skipped so the
        read side never crashes on a mixed directory."""
        return record.get("kind") == "llm_call" and "stage" in record

    @staticmethod
    def _to_call(record: dict[str, Any]) -> LLMCallTrace:
        fields = {f.name for f in dataclasses.fields(LLMCallTrace)}
        return LLMCallTrace(**{k: v for k, v in record.items() if k in fields})

    def read_calls(
        self,
        world_id: str,
        *,
        step: int | None = None,
        stage: str | None = None,
        agent_id: str | None = None,
    ) -> list[LLMCallTrace]:
        calls = [
            self._to_call(r)
            for r in self._all_records(world_id)
            if self._is_production_call(r)
        ]
        if step is not None:
            calls = [c for c in calls if c.step == step]
        if stage is not None:
            calls = [c for c in calls if c.stage == stage]
        if agent_id is not None:
            calls = [c for c in calls if c.agent_id == agent_id]
        return calls

    def read_step_summaries(self, world_id: str) -> list[StepTrace]:
        fields = {f.name for f in dataclasses.fields(StepTrace)}
        summaries = [
            StepTrace(**{k: v for k, v in r.items() if k in fields})
            for r in self._all_records(world_id)
            if r.get("kind") == "step"
        ]
        return sorted(summaries, key=lambda s: s.step)

    def list_steps(self, world_id: str) -> list[int]:
        steps = {
            r["step"]
            for r in self._all_records(world_id)
            if self._is_production_call(r) and isinstance(r.get("step"), int)
        }
        return sorted(steps)

    def list_agents(self, world_id: str) -> list[str]:
        agents = {
            r["agent_id"]
            for r in self._all_records(world_id)
            if self._is_production_call(r) and r.get("agent_id")
        }
        return sorted(agents)

    def list_stages(self, world_id: str) -> list[str]:
        stages = {
            r["stage"]
            for r in self._all_records(world_id)
            if self._is_production_call(r) and r.get("stage")
        }
        return sorted(stages, key=stage_sort_key)
