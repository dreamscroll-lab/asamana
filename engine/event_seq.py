"""The world's emission timeline: a monotonic ``seq`` stamped on every per-step narrative event."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from core.interfaces.directory import WorldDirectory

if TYPE_CHECKING:
    from engine.message_system import MessageDelivery


class EventSequencer:
    """Owns the per-world event-seq counter and the intra-step emission order."""

    def __init__(self, *, directory: WorldDirectory, start_seq: int = 0) -> None:
        self._directory = directory
        # Each event's canonical position on the world's emission timeline: an engine property,
        # not a render concept. Integer, not wall-clock, so replay is faithful. Never resets;
        # persisted as ``metadata["event_seq"]`` and restored via ``start_seq``.
        self._event_seq: int = start_seq

    @property
    def next_seq(self) -> int:
        """The next ordinal to assign; a snapshot persists it so a restored run resumes from it."""
        return self._event_seq

    def _next_seq(self) -> int:
        """Return the next event ordinal, then advance.

        The read-then-increment is safe only because ``assign_event_seqs`` is synchronous and
        runs after the adjudication gather. Don't mint seqs inside gathered tasks, add an
        ``await`` mid-assignment, or run it across threads without making the counter atomic.

        A code-layer ordinal: it must never appear in narrative text, prompts or memory.
        """
        seq = self._event_seq
        self._event_seq += 1
        return seq

    def assign_event_seqs(
        self,
        *,
        action_records: List[Dict[str, Any]],
        deliveries: "MessageDelivery",
        broadcasts: List[Any],
        events: List[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Stamp seqs on one step's events in emission order: delivered messages → broadcasts →
        actions → world events (arrivals were emitted last step, so they precede this step's
        actions). The single owner of intra-step ordering; minted at assembly, the only point
        with a deterministic total order.

        ``action_records`` and ``events`` are enriched in place. Message and broadcast payloads
        are built here and returned, so snapshot and live step_event carry identical ordinals.
        """
        messages_payload = deliveries.as_dict()
        for message_record in messages_payload.get("delivered", []):
            message_record["seq"] = self._next_seq()
            # Resolve the name here (like ``sender_name``) so no consumer translates the id itself.
            scope = message_record.get("location_scope")
            message_record["location_name"] = self._directory.location_name(str(scope)) if scope else ""
        # A death broadcast lands on step+1, so its seq trails the victim's terminal action.
        broadcast_records: List[Dict[str, Any]] = [
            {
                "content": b.content,
                "source": b.source,
                # JSON-native: a live str-Enum reaches the live observer as "BroadcastType.WORLD_EVENT".
                "broadcast_type": str(getattr(b.broadcast_type, "value", b.broadcast_type or "")),
                "location_scope": b.location_scope,
                "location_name": self._directory.location_name(b.location_scope) if b.location_scope else "",
                "severity": b.severity.value,
                # What the change looks like (fire/rain/…), orthogonal to severity. A render-neutral
                # fact on the broadcast because a broadcast already is a location-scoped change.
                "phenomenon": b.phenomenon.value,
                "seq": self._next_seq(),
            }
            for b in broadcasts
        ]
        for record in action_records:
            record["seq"] = self._next_seq()
        for event_record in events:
            event_record["seq"] = self._next_seq()
        return messages_payload, broadcast_records
