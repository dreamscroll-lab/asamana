"""LLM contracts."""

from __future__ import annotations

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4
from typing import TYPE_CHECKING, Any

from core.logging import get_logger
from core.rate_gate import classify_rate_limit

if TYPE_CHECKING:
    from core.interfaces.trace import TraceSink
    from core.rate_gate import RateGate

_logger = get_logger(__name__)

#: An index, optionally with the decoration the prompt uses (``#3`` / ``第3`` / ``3号``), nothing else around it.
_DECORATED_INDEX = re.compile(r"^[#第\s]*(\d+)[号\s.)、]*$")


def _as_index(value: Any) -> int | None:
    """Read one LLM reference value as a 1-based index; ``None`` if unreadable.

    Reject bools first: ``int(True)`` is 1, so a ``true`` written for "null = world-wide" would
    silently pin a broadcast to the first location on the menu.
    """
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    match = _DECORATED_INDEX.match(str(value).strip())
    return int(match.group(1)) if match else None


class IndexedRef:
    """Map opaque ids ↔ 1-based indices: the safe channel for the LLM to reference a known set
    (see "LLM Indexed Reference Pattern" in CLAUDE.md).

    Usage::

        ref = IndexedRef([c.id for c in candidates])
        # the prompt renders each item with its index (`#1 ... #N`); the caller picks the format
        # the LLM returns `{"indices": [1, 3]}`
        ids = ref.resolve([1, 3])    # → [candidates[0].id, candidates[2].id]
    """

    __slots__ = ("_ids",)

    def __init__(self, ids: Iterable[str]) -> None:
        self._ids: list[str] = list(ids)

    def resolve(self, indices: Iterable[Any]) -> list[str]:
        """Map LLM-returned 1-based indices back to ids, skipping out-of-range, duplicate or
        unreadable values.

        Accepts decorated indices (``"#3"`` / ``"第3"`` / ``"3号"``): models copy the prompt's ``#3``
        back, and a dropped ``location_scope`` means "broadcast to the whole world". Decoration
        is limited to ``#第号.)`` and whitespace: guessing ``"李世民3号"`` as person 3 is worse than
        dropping it.
        """
        seen: set[int] = set()
        out: list[str] = []
        for value in indices:
            n = _as_index(value)
            if n is None:
                continue
            if n in seen:
                continue
            if 1 <= n <= len(self._ids):
                seen.add(n)
                out.append(self._ids[n - 1])
        return out


_INDEX_REF_IN_TEXT = re.compile(r"#\s*\d+")


def leaks_index_ref(text: str) -> bool:
    """Has an index reference leaked into this human-readable text?

    Indices live only in reference fields; in prose ("send #1 and #2 to #20") they reach readers
    who don't have the list. Detect only, never repair: ``#1`` is ambiguous across lists, and a
    guessed name would fabricate a persistent fact. The caller decides whether to block.
    """
    return bool(_INDEX_REF_IN_TEXT.search(text or ""))


def _note_parse(ok: bool) -> None:
    """Attach a parse outcome to the active LLM call. Never propagates: observability must not
    break parsing."""
    try:
        from core.context import note_active_call_parse

        note_active_call_parse(ok)
    except Exception as exc:  # noqa: BLE001 — must never break parsing
        _logger.warning("parse_note_failed", extra={"error": str(exc)})


_FALSE_TOKENS = frozenset({"false", "no", "0", "否", "假", "不", "null", "none"})


def coerce_bool(value: object, default: bool) -> bool:
    """Coerce an LLM boolean field to a real bool — silent field-level coercion, no logging.

    Don't use bare ``bool(...)``: models write ``"false"`` / ``"否"``, and ``bool("false")`` is
    ``True``, laundering a failure verdict into success. A structural guarantee, independent of
    how the prompt writes the schema.

    Strings in ``_FALSE_TOKENS`` are false, other non-empty strings true; numbers by 0/non-0;
    ``None``, missing or empty string return ``default``.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        token = value.strip().lower()
        if not token:
            return default
        return token not in _FALSE_TOKENS
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def extract_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response, tolerating markdown fences.

    Records a parse-outcome event (for the parse-failure rate) and re-raises on
    failure so existing callers' try/except fallbacks are unaffected.
    """
    try:
        result = _extract_json_impl(text)
    except Exception:
        _note_parse(False)
        raise
    _note_parse(True)
    return result


def _extract_json_impl(text: str) -> dict:
    """The first bracket-balanced object in the response, parsed; the whole text when there is
    none. Raises when that candidate is not valid JSON, repair included."""
    candidate = next(_top_level_objects(text), None)
    return _loads_or_repair(candidate if candidate is not None else text.strip())


_STRING_TERMINATORS = frozenset(",}]:")
_CONTROL_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}


def _repair_json(text: str) -> str:
    """Escape unescaped ``"`` and control characters inside string values.

    Models use ASCII quotes as title marks in Chinese prose, and one pair ruins the payload. A
    ``"`` inside a string closes it only when followed by ``,}]:`` or end of text.

    Call only after ``json.loads`` has failed: this lossy guess would turn structural errors like
    a missing comma into a longer string.
    """
    out: list[str] = []
    in_string = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if not in_string:
            out.append(ch)
            in_string = ch == '"'
            i += 1
        elif ch == "\\":
            out.append(text[i : i + 2])
            i += 2
        elif ch == '"':
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            in_string = not (j >= n or text[j] in _STRING_TERMINATORS)
            out.append('\\"' if in_string else ch)
            i += 1
        else:
            out.append(_CONTROL_ESCAPES.get(ch) or (f"\\u{ord(ch):04x}" if ch < " " else ch))
            i += 1
    return "".join(out)


def _loads_or_repair(candidate: str) -> Any:
    """``json.loads``; on failure retry once with escapes repaired; if that fails, raise the
    original error (it points at what the model actually wrote)."""
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        try:
            parsed = json.loads(_repair_json(candidate))
        except json.JSONDecodeError:
            raise exc from None
        _logger.warning("json_repaired", extra={"error": str(exc)})
        return parsed


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict | None:
    """Extract the first JSON *object* from an LLM response, never raising.

    Records a parse-outcome event (None result ⇒ parse failure) for the
    parse-failure rate.
    """
    result = _extract_json_object_impl(text)
    _note_parse(result is not None)
    return result


def _extract_json_object_impl(text: str) -> dict | None:
    """Unlike :func:`extract_json` (which raises and may return any JSON value), skips balanced
    spans that don't parse, accepts only ``dict`` payloads, and returns ``None`` on failure.
    """
    if not text:
        return None

    match = _JSON_FENCE_RE.search(text)
    if match:
        try:
            return _loads_or_repair(match.group(1))
        except json.JSONDecodeError:
            pass

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return _loads_or_repair(stripped)
        except json.JSONDecodeError:
            pass

    for candidate in _top_level_objects(text):
        try:
            parsed = _loads_or_repair(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _top_level_objects(text: str) -> Iterator[str]:
    """Each bracket-balanced ``{...}`` span, in order — the one scanner both extractors share.

    Braces inside JSON strings (escapes included) do not count, so trailing prose containing
    ``}`` cannot corrupt a span.
    """
    depth = 0
    in_string = False
    escaped = False
    start = -1
    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start : i + 1]


#: ``max_tokens`` as a multiple of the answer estimate. Thinking and the answer share the budget,
#: and an always-thinking endpoint now and then thinks ~3× the answer itself. Billing is on actual
#: output, so a higher multiple costs only tail latency; a lower one truncates into unparseable JSON.
MAX_TOKENS_MULTIPLIER = 6


def output_budget(estimate: int) -> int:
    """``max_tokens`` for an answer estimated at ``estimate`` tokens, rounded up to the next ten.

    The estimate comes from the prompt's declared length and count limits (CLAUDE.md, "max_tokens
    needs headroom"), never from observed output.
    """
    return -(-estimate * MAX_TOKENS_MULTIPLIER // 10) * 10


class LLMScene(str, Enum):
    """Supported LLM routing scenes."""

    AGENT_DECISION_MAIN = "agent_decision_main"
    AGENT_ACTION_NARRATION = "agent_action_narration"
    MEMORY_SUMMARIZATION = "memory_summarization"
    NEED_GOAL_GENERATION = "need_goal_generation"
    EVENT_TIMING = "event_timing"
    # A human director's sentence → injection plan. Translation, not authoring: worth a different
    # tier than the editor's pacing gate / event authoring, so each is tuned separately.
    DIRECTIVE = "directive"
    PERSONA_GENERATION = "persona_generation"
    WORLD_BUILDING = "world_building"
    # Pick an installed map for the theme. Separate from WORLD_BUILDING because the two are worth
    # different tiers: theme analysis needs the strongest model, map picking outputs one index and
    # can run cheap; it also keeps map selection out of theme-analysis traces.
    WORLD_TEMPLATE_SELECTION = "world_template_selection"
    CAST_DESIGN = "cast_design"
    WORLD_PRESSURE = "world_pressure"
    AGENT_INTERRUPT_DECISION = "agent_interrupt_decision"
    # MEMORY_IMPORTANCE: scores a memory's importance [0,1] at write time; drives decay/retrieval/compression.
    # MEMORY_REFLECTION: periodically condenses high-emotion experiential memories into insights.
    MEMORY_IMPORTANCE = "memory_importance"
    MEMORY_REFLECTION = "memory_reflection"
    # Periodic relation-label evolution: re-assess a relation's labels from recent memory; outputs a
    # full replacement list.
    RELATION_LABEL_EVOLUTION = "relation_label_evolution"


@dataclass
class LLMMessage:
    """Chat message passed to providers."""

    role: str
    content: str


@dataclass
class LLMResponse:
    """Normalized LLM response."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    # The endpoint's reasoning_content. Observability only: the engine never reads it.
    thinking: str = ""
    # The only basis for calibrating max_tokens, which thinking shares with the answer
    # (see MAX_TOKENS_MULTIPLIER).
    thinking_tokens: int = 0


class LLMProvider(ABC):
    """Abstract LLM provider."""

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        *,
        json_mode: bool = False,
    ) -> LLMResponse:
        r"""Run one complete model call.

        ``json_mode`` constrains DECODING to valid JSON, so an unparseable payload is never
        generated. It is a HINT: how it is expressed (``response_format`` on OpenAI-compatible
        endpoints) stays inside the provider; one that cannot ignores it.

        PRECONDITION: OpenAI-compatible endpoints reject the request (HTTP 400) unless the
        prompt contains the word "json". The provider enforces this rather than let a silent
        400 masquerade as an LLM failure.
        """


class LLMRouter:
    """Route scene-specific calls to configured providers."""

    def __init__(
        self,
        scene_providers: dict[LLMScene, LLMProvider],
        *,
        max_concurrent: int = 0,
        trace_sink: "TraceSink | None" = None,
        rate_gates: "dict[LLMScene, RateGate] | None" = None,
        capture_thinking: bool = False,
    ) -> None:
        missing_scenes = set(LLMScene) - set(scene_providers)
        if missing_scenes:
            missing = ", ".join(sorted(scene.value for scene in missing_scenes))
            raise ValueError(f"LLMRouter is missing providers for scenes: {missing}")
        self._scene_providers = dict(scene_providers)
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        )
        # One gate per endpoint (see core.container._build_rate_gates), looked up by scene.
        # Acquired *before* the semaphore so a call under cooldown doesn't hold an in-flight slot.
        self._rate_gates = dict(rate_gates or {})
        # None → no recording. The router is the single LLM chokepoint, so this sees every call.
        self._trace_sink = trace_sink
        # Decided here, not in the provider: how much to record is an observability decision.
        self._capture_thinking = capture_thinking

    def get(self, scene: LLMScene) -> LLMProvider:
        return self._scene_providers[scene]

    async def complete(
        self,
        scene: LLMScene,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        *,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Complete through the provider configured for a scene. See LLMProvider.complete."""

        _t = time.perf_counter()
        # Shared by the runtime log and the trace record, to locate a call across both.
        call_id = str(uuid4())
        gate = self._semaphore if self._semaphore is not None else nullcontext()
        rate_gate = self._rate_gates.get(scene)
        if rate_gate is not None:
            await rate_gate.acquire()
        try:
            async with gate:
                result = await self._scene_providers[scene].complete(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
        except Exception as exc:
            # A rate-limit / overload arms the shared cooldown so sibling callers back off
            # together. Re-raise: callers own the fallback (CLAUDE.md Rule 1).
            if rate_gate is not None:
                limited, retry_after = classify_rate_limit(exc)
                if limited:
                    rate_gate.penalize(retry_after)
            latency_ms = round((time.perf_counter() - _t) * 1000, 1)
            if self._trace_sink is not None:
                self._record_trace(scene, messages, None, temperature, max_tokens, latency_ms, call_id=call_id, error=str(exc), json_mode=json_mode)
            raise
        if rate_gate is not None:
            rate_gate.note_success()
            # Actual token cost, not an estimate.
            rate_gate.note_usage(result.input_tokens, result.output_tokens)
        latency_ms = round((time.perf_counter() - _t) * 1000, 1)
        _logger.info(
            "llm_complete",
            extra={
                "call_id": call_id,
                "scene": scene.value,
                "elapsed_ms": latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.input_tokens + result.output_tokens,
            },
        )
        if self._trace_sink is not None:
            self._record_trace(scene, messages, result, temperature, max_tokens, latency_ms, call_id=call_id, json_mode=json_mode)
        return result

    @property
    def trace_sink(self) -> "TraceSink | None":
        return self._trace_sink

    def _record_trace(
        self,
        scene: LLMScene,
        messages: list["LLMMessage"],
        response: "LLMResponse | None",
        temperature: float,
        max_tokens: int,
        latency_ms: float,
        *,
        call_id: str = "",
        error: str = "",
        json_mode: bool = False,
    ) -> None:
        """Mirror one completion into the trace sink; ``response is None`` records a failure.
        Recording must never break the observed call."""
        from datetime import datetime

        from core.context import get_call_annotations, get_log_context, set_active_call
        from core.interfaces.trace import LLMCallTrace, Stage

        try:
            ctx = get_log_context()
            step_raw = ctx.get("step")
            call = LLMCallTrace(
                world_id=ctx.get("world_id", ""),
                stage=ctx.get("stage", Stage.UNKNOWN.value),
                scene=scene.value,
                prompt_messages=[{"role": m.role, "content": m.content} for m in messages],
                response_content=response.content if response is not None else "",
                thinking=(
                    response.thinking if (response is not None and self._capture_thinking) else ""
                ),
                thinking_tokens=response.thinking_tokens if response is not None else 0,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                input_tokens=response.input_tokens if response is not None else 0,
                output_tokens=response.output_tokens if response is not None else 0,
                model=response.model if response is not None else "",
                latency_ms=latency_ms,
                timestamp=datetime.now().isoformat(),
                agent_id=ctx.get("agent_id") or None,
                step=int(step_raw) if step_raw and step_raw.isdigit() else None,
                ok=response is not None,
                error=error,
                call_id=call_id,
                extra=get_call_annotations(),
            )
            self._trace_sink.record_llm_call(call)  # type: ignore[union-attr]
            # The shared parser fills in this call's parse_ok next.
            set_active_call(call if response is not None else None)
        except Exception as exc:  # noqa: BLE001 — recording must never break the call
            _logger.warning("llm_trace_record_failed", extra={"scene": scene.value, "error": str(exc)})

    async def complete_with_retry(
        self,
        scene: LLMScene,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        *,
        retry_delay: float = 1.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Complete with one application-level retry on any exception.

        Used exclusively for main-character decision/interrupt/emotion paths.
        Transport-level retries (rate limit, overload) are handled inside the provider.
        On the second failure the exception propagates — callers must handle it.
        """
        try:
            return await self.complete(scene, messages, temperature, max_tokens, json_mode=json_mode)
        except Exception:
            _logger.warning(
                "llm_retry",
                extra={"scene": scene.value, "retry_delay": retry_delay},
            )
            await asyncio.sleep(retry_delay)
            return await self.complete(scene, messages, temperature, max_tokens, json_mode=json_mode)
