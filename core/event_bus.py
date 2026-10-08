"""Runtime narrative event bus shared across subsystems."""

from __future__ import annotations

import asyncio
from typing import Any

# Observation is a transient feed (snapshots are the durable record), so a subscriber that stops
# reading loses its oldest frames rather than growing the bus without bound.
_SUBSCRIBER_QUEUE_SIZE = 256


class NarrativeEventBus:
    """In-process push bus: events go only to the queues subscribed when they are published.

    Nothing is kept for later: a reader subscribes first, then reads. Subscribers are filtered by
    ``world_id`` (missing → ``""``) on the publish side. Don't feed subscribers everything to pick
    from: queues are bounded and drop their head, so one world's event storm would push out
    another world's frames, and live never backfills.
    """

    def __init__(self) -> None:
        # (wanted world, queue); None receives everything, for tests and process-wide observers.
        self._subscribers: list[tuple[str | None, asyncio.Queue[dict[str, Any] | None]]] = []

    def publish(self, event: dict[str, Any]) -> None:
        key = str(event.get("world_id", ""))
        for wanted, subscriber in list(self._subscribers):
            if wanted is not None and wanted != key:
                continue
            # Drop the oldest when full: losing a frame beats unbounded growth.
            if subscriber.full():
                try:
                    subscriber.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            subscriber.put_nowait(event)

    def subscribe(self, world_id: str | None = None) -> asyncio.Queue[dict[str, Any] | None]:
        """Return a queue receiving events published after subscription.

        ``world_id`` selects which world's events to receive (see the class docstring: this is
        the isolation boundary, not a convenience). ``None`` receives everything.
        """

        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE)
        self._subscribers.append((world_id, queue))
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        self._subscribers = [entry for entry in self._subscribers if entry[1] is not queue]
