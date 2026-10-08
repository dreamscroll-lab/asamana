"""Endpoint-scoped rate governance shared by all LLM + embedding calls to one endpoint.

Rate limits belong to an *endpoint*, not a world or scene: many worlds share one process and
must not each retry independently. One gate per ``llm.providers`` entry in use (see
``core.container._build_rate_gates``), plus the embedding's own (``_embedding_gate``).

Two layers, both checked in :meth:`acquire`:

**Reactive shared cooldown** (always on). After a rate-limit / overload (:meth:`penalize`),
every caller waits out one *shared* cooldown: uncoordinated retries pile load on exactly when
the endpoint is shedding. Honors ``Retry-After`` up to a ceiling, else a base cooldown that
doubles per round still failing (bounded) and resets on success.

**Proactive governor** (opt-in; ``rpm_limit`` / ``tpm_limit`` > 0). Keeps load just under a
*known* quota so requests queue locally instead of being rejected:

- **RPM is strict admission**: the slot is reserved atomically before the call (no ``await``
  between check and append).
- **TPM is feedback over actual tokens** (:meth:`note_usage`), since output cost is unknown
  before the call. The in-flight cost isn't reserved; the router's semaphore bounds the lag
  to ``max_concurrent_llm × max_tokens`` and the cooldown catches any residual 429.

Embedding calls pass ``reserve_rate=False`` (separate quota, no token counts): cooldown only.

No dependency on openai/httpx: call sites classify errors with :func:`classify_rate_limit`
and call :meth:`penalize`.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable

from core.logging import get_logger

logger = get_logger(__name__)

# Statuses that mean "the account is shedding load, back off": 429 rate-limit,
# 503/529 overload. Anything else is a normal error the caller's fallback owns.
_BACKPRESSURE_STATUSES = frozenset({429, 503, 529})


def classify_rate_limit(exc: BaseException) -> tuple[bool, float | None]:
    """Duck-type an exception as an account-shedding signal.

    Returns ``(is_backpressure, retry_after_seconds)``. Recognizes the openai SDK
    (``status_code``) and httpx (``response.status_code``) without importing either.
    """

    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    if status not in _BACKPRESSURE_STATUSES:
        return False, None

    retry_after: float | None = None
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            raw = headers.get("retry-after")
        except Exception:  # noqa: BLE001 — headers may be a non-mapping; treat as absent
            raw = None
        if raw is not None:
            try:
                retry_after = float(raw)
            except (TypeError, ValueError):
                retry_after = None
    return True, retry_after


class RateGate:
    """Shared adaptive cooldown + optional RPM/TPM governor for one endpoint.

    Single-event-loop safe: no ``await`` inside the admission or escalation critical sections,
    and :meth:`acquire` re-reads all deadlines after each sleep.
    """

    def __init__(
        self,
        *,
        base_cooldown: float = 2.0,
        max_cooldown: float = 30.0,
        max_retry_after: float = 120.0,
        jitter: float = 0.5,
        rpm_limit: int = 0,
        tpm_limit: int = 0,
        window_seconds: float = 60.0,
        now: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if base_cooldown <= 0 or max_cooldown < base_cooldown:
            raise ValueError("require 0 < base_cooldown <= max_cooldown")
        if max_retry_after <= 0:
            raise ValueError("max_retry_after must be > 0")
        if rpm_limit < 0 or tpm_limit < 0:
            raise ValueError("rpm_limit / tpm_limit must be >= 0 (0 = disabled)")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self._base = base_cooldown
        self._max = max_cooldown
        self._max_retry_after = max_retry_after
        self._jitter = max(0.0, jitter)
        self._rpm_limit = rpm_limit
        self._tpm_limit = tpm_limit
        self._window = window_seconds
        # Injectable clock/sleep keep tests deterministic without real waits.
        self._now = now or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._cooldown_until: float = 0.0
        # Last cooldown armed since the most recent success; 0 = none, next hit uses base.
        self._current_delay: float = 0.0
        # Sliding windows, unused when the respective limit is 0.
        self._admissions: deque[float] = deque()  # RPM: admission timestamps
        self._usage: deque[tuple[float, int]] = deque()  # TPM: (timestamp, actual tokens)

    async def acquire(self, *, reserve_rate: bool = True) -> None:
        """Block until the endpoint will accept another request.

        Waits out the shared cooldown and, when ``reserve_rate``, the RPM and observed-TPM
        windows, then reserves an RPM slot. Jitter on each wait keeps waiters from stampeding
        the instant a slot frees.
        """

        while True:
            now = self._now()
            wait = self._required_wait(now, reserve_rate)
            if wait <= 0:
                if reserve_rate and self._rpm_limit > 0:
                    self._admissions.append(now)  # reserve atomically (no await since check)
                return
            await self._sleep(wait + random.uniform(0.0, self._jitter))

    def _required_wait(self, now: float, reserve_rate: bool) -> float:
        """Seconds until this request may proceed; ≤ 0 means now."""

        wait = self._cooldown_until - now  # cooldown applies to every caller
        if not reserve_rate:
            return wait
        if self._rpm_limit > 0:
            self._evict(self._admissions, now)
            if len(self._admissions) >= self._rpm_limit:
                wait = max(wait, self._admissions[0] + self._window - now)
        if self._tpm_limit > 0:
            self._evict(self._usage, now)
            if self._usage and sum(tok for _, tok in self._usage) >= self._tpm_limit:
                # Wait for the oldest usage to age out, then re-check (converges within a window).
                wait = max(wait, self._usage[0][0] + self._window - now)
        return wait

    def _evict(self, window: deque, now: float) -> None:
        """Entry may be a float or a tuple."""

        cutoff = now - self._window
        while window:
            head = window[0]
            ts = head[0] if isinstance(head, tuple) else head
            if ts <= cutoff:
                window.popleft()
            else:
                break

    def penalize(self, retry_after: float | None = None) -> None:
        """Record an observed backpressure response and (re)arm the shared cooldown.

        ``Retry-After`` is honored up to ``max_retry_after``: a spent quota may ask for hours, and
        waiting that out would stall every world on the endpoint silently instead of letting calls
        fail into their fallbacks. Otherwise escalation counts *rounds*: a hit after a cooldown
        elapsed with no success doubles the last delay (capped); a hit after a success starts at
        ``base_cooldown``. The deadline only ever moves later.
        """

        now = self._now()
        cooling = now < self._cooldown_until
        if retry_after is not None and retry_after > 0:
            delay = min(retry_after, self._max_retry_after)
        elif cooling:
            # An echo from a request released before the cooldown. Don't stack it: a burst of
            # concurrent 429s would push a single limit all the way to max_cooldown.
            logger.warning("llm_endpoint_backpressure_echo")
            return
        elif self._current_delay > 0:
            delay = min(self._max, self._current_delay * 2)
        else:
            delay = self._base
        self._current_delay = delay
        self._cooldown_until = max(self._cooldown_until, now + delay)
        logger.warning(
            "llm_endpoint_backpressure",
            extra={"cooldown_s": round(delay, 2), "retry_after": retry_after},
        )

    def note_success(self) -> None:
        """The next hit starts over at the base cooldown; an active deadline still stands."""

        self._current_delay = 0.0

    def note_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Debit a completed call's *actual* token cost into the rolling TPM window."""

        if self._tpm_limit <= 0:
            return
        now = self._now()
        self._usage.append((now, max(0, input_tokens) + max(0, output_tokens)))
        self._evict(self._usage, now)
