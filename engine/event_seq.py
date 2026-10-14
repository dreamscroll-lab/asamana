"""The world's emission timeline: a monotonic ``seq`` stamped on every per-step event."""

from __future__ import annotations

from typing import Any, Dict, List


class EventSequencer:
    """Owns the per-world event-seq counter and the intra-step emission order."""

    def __init__(self, *, start_seq: int = 0) -> None:
        # Each event's canonical position on the world's emission timeline: an engine property,
        # not a render concept. Integer, not wall-clock, so replay is faithful. Never resets;
        # persisted as ``EngineState.event_seq`` and restored via ``start_seq``.
        self._event_seq: int = start_seq

    @property
    def next_seq(self) -> int:
        """The next ordinal to assign; a snapshot persists it so a restored run resumes from it."""
        return self._event_seq

    def _next_seq(self) -> int:
        """Return the next event ordinal, then advance.

        The read-then-increment is safe only because ``order_events`` is synchronous and runs
        after the adjudication gather. Don't mint seqs inside gathered tasks, add an ``await``
        mid-assignment, or run it across threads without making the counter atomic.

        A code-layer ordinal: it must never appear in narrative text, prompts or memory.
        """
        seq = self._event_seq
        self._event_seq += 1
        return seq

    def order_events(
        self,
        *,
        messages: List[Dict[str, Any]],
        broadcasts: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
        injections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """One step's events (``engine.step_record``) in emission order, each stamped with its
        seq: delivered messages → broadcasts → actions → injections. Arrivals were emitted last
        step, so they precede this step's actions; a death broadcast lands on step+1, so its seq
        trails the victim's terminal action. The single owner of intra-step ordering; minted at
        assembly, the only point with a deterministic total order.
        """
        ordered = [*messages, *broadcasts, *actions, *injections]
        for event in ordered:
            event["seq"] = self._next_seq()
        return ordered
