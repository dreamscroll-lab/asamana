"""Request-scoped context helpers for Asamana logging and observability."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from core.interfaces.trace import Stage

_world_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "world_id",
    default="",
)
_agent_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "agent_id",
    default="",
)
_step_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "step",
    default="",
)
_event_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "event_id",
    default="",
)
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id",
    default="",
)
# Read by LLMRouter when recording a call's trace; set via observe_stage().
_stage_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "stage",
    default="",
)
# The LLM call most recently recorded in the current task, so the shared JSON parsers can
# fill in its parse_ok. Per-task (asyncio copies context), so concurrent agents never
# cross-attribute.
_active_call_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "active_call",
    default=None,
)
# Read by LLMRouter into LLMCallTrace.extra; set via annotate_call(). The shared default dict
# is safe because every write sets a fresh merged dict, never mutating in place.
_call_annotations_var: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "call_annotations",
    default={},
)


def set_active_call(call: Any) -> None:
    """Mark *call* (an LLMCallTrace, or None) as the parse target for this task."""
    _active_call_var.set(call)


def note_active_call_parse(ok: bool) -> None:
    """Record a parse outcome on the active call. First parse after a call wins
    (subsequent strays are ignored); no-op when there is no active call."""
    call = _active_call_var.get()
    if call is not None and getattr(call, "parse_ok", "x") is None:
        call.parse_ok = ok


def note_call_adoption(call: Any, adopted: bool, reason: str = "") -> None:
    """Record whether the consumer **used** *call*'s output (``reject_reason`` when not).
    First verdict wins; no-op when *call* is None.

    Observability only. Never read back into logic, a prompt, or an embedding.
    """
    if call is not None and getattr(call, "adopted", "x") is None:
        call.adopted = adopted
        if not adopted:
            call.reject_reason = reason


def note_active_call_adoption(adopted: bool, reason: str = "") -> None:
    """``note_call_adoption`` on the active call.

    Report it at the seam that judges the payload, where a parseable response would otherwise
    be discarded without trace; retries then file correctly (each attempt rebinds the pointer).
    """
    note_call_adoption(_active_call_var.get(), adopted, reason)


def annotate_active_call(**fields: Any) -> None:
    """Merge post-call diagnostic *fields* into the active call's ``extra``.

    The post-call counterpart to ``annotate_call``, for values only known after the response
    (e.g. which generated goals survived dedup). No-op without an active call; later keys win.
    Observability only: JSON-serializable, never read back into logic, a prompt, or an embedding.
    """
    call = _active_call_var.get()
    if call is not None and getattr(call, "extra", None) is not None:
        call.extra.update(fields)


def get_call_annotations() -> dict[str, Any]:
    """A fresh copy, so LLMRouter can store it without aliasing the contextvar state."""
    return dict(_call_annotations_var.get())


def set_log_context(
    *,
    world_id: str = "",
    agent_id: str = "",
    step: str = "",
    event_id: str = "",
    request_id: str = "",
    stage: str = "",
) -> None:
    _world_id_var.set(world_id)
    _agent_id_var.set(agent_id)
    _step_var.set(step)
    _event_id_var.set(event_id)
    _request_id_var.set(request_id)
    _stage_var.set(stage)


def get_log_context() -> dict[str, str]:
    context = {
        "world_id": _world_id_var.get(),
        "agent_id": _agent_id_var.get(),
        "step": _step_var.get(),
        "event_id": _event_id_var.get(),
        "request_id": _request_id_var.get(),
        "stage": _stage_var.get(),
    }
    return {key: value for key, value in context.items() if value}


def clear_log_context() -> None:
    set_log_context()


@contextmanager
def observe_stage(stage: "Stage | str", *, agent_id: str | None = None) -> Iterator[None]:
    """Mark the enclosed block as running in *stage* (and optionally for *agent_id*).

    Token-based reset, so nested stages restore correctly. Under ``asyncio.gather`` each task
    copies the context, so enter inside the per-agent coroutine (not before gather) to keep ids
    from bleeding across concurrent tasks.
    """
    # hasattr instead of importing Stage at runtime: that would be an import cycle.
    stage_value = stage.value if hasattr(stage, "value") else str(stage)
    stage_token = _stage_var.set(stage_value)
    agent_token = _agent_id_var.set(agent_id) if agent_id is not None else None
    try:
        yield
    finally:
        _stage_var.reset(stage_token)
        if agent_token is not None:
            _agent_id_var.reset(agent_token)


@contextmanager
def observe_step(step: "int | str") -> Iterator[None]:
    """Override ONLY the step in the logging/trace context for the enclosed block.

    For long-lived background workers: ``asyncio.create_task`` freezes contextvars at creation,
    so every deferred call would be attributed to that one step. Wrap each job to re-stamp it.
    """
    token = _step_var.set(str(step))
    try:
        yield
    finally:
        _step_var.reset(token)


class GivenFacts(list):
    """The facts actually put in front of this LLM call, each prefixed with its channel, passed
    via ``annotate_call(given_facts=...)`` so a later audit can judge whether the output invented
    facts. ``add`` takes items or sequences and skips empty values.

    Pure observation; business logic never reads it. Keep it in sync with the prompt: an input
    missing here makes the audit flag grounded statements as fabricated.
    """

    def add(self, channel: str, *values: Any) -> "GivenFacts":
        for value in values:
            for item in value if isinstance(value, (list, tuple)) else (value,):
                if text := str(item).strip():
                    self.append(f"{channel}：{text}")
        return self


@contextmanager
def annotate_call(**fields: Any) -> Iterator[None]:
    """Attach code-layer diagnostic *fields* to every LLM call recorded within this block.

    The router snapshots them into ``LLMCallTrace.extra``, so a new diagnostic field needs no
    schema change. Merges onto annotations in scope (inner key wins). Pure observability:
    JSON-serializable, never read back into logic or a prompt. Like ``observe_stage``, enter
    inside the per-agent coroutine, not before gather.
    """
    merged = {**_call_annotations_var.get(), **fields}
    token = _call_annotations_var.set(merged)
    try:
        yield
    finally:
        _call_annotations_var.reset(token)
