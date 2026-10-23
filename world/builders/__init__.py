"""The world-building stages ``world.builder.WorldBuilder`` runs — one module per LLM stage."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Mapping, TypeVar

from core.context import note_active_call_adoption
from core.interfaces.llm import LLMResponse
from core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Rule 2: a world-building LLM call gets one retry on an unusable reply, then raises.
BUILD_ATTEMPTS = 2
BUILD_RETRY_DELAY_SECONDS = 1.0


class BuildReplyRejected(ValueError):
    """A reply a stage's parse can't use; ``adoption_reason`` labels the call's trace."""

    def __init__(self, message: str, adoption_reason: str) -> None:
        super().__init__(message)
        self.adoption_reason = adoption_reason


async def complete_with_build_retry(
    call: Callable[[], Awaitable[LLMResponse]],
    parse: Callable[[str], T],
    *,
    attempt_failed_event: str,
    log_extra: Mapping[str, object],
    failure: str,
) -> T:
    """Run a stage's LLM call until ``parse`` accepts the reply, then raise (Rule 2).

    ``parse`` owns the stage's validation and raises on rejection: ``BuildReplyRejected`` names
    the reason; any other exception (the call itself included) is filed as ``unusable_payload``.
    The final ``ValueError`` reads ``"{failure} after N attempts"``.
    """
    last_exc: Exception | None = None
    for attempt in range(1, BUILD_ATTEMPTS + 1):
        try:
            return parse((await call()).content)
        except BuildReplyRejected as exc:
            note_active_call_adoption(False, exc.adoption_reason)
            last_exc = exc
        except Exception as exc:
            note_active_call_adoption(False, "unusable_payload")
            last_exc = exc
        logger.warning(
            attempt_failed_event,
            extra={**log_extra, "attempt": attempt, "reason": str(last_exc)},
        )
        if attempt < BUILD_ATTEMPTS:
            await asyncio.sleep(BUILD_RETRY_DELAY_SECONDS)
    raise ValueError(f"{failure} after {BUILD_ATTEMPTS} attempts") from last_exc
