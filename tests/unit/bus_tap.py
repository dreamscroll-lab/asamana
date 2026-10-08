"""Read what a NarrativeEventBus publishes, the way production observers do: by subscribing."""

from __future__ import annotations

from typing import Any, Callable

from core.event_bus import NarrativeEventBus


def tap(bus: NarrativeEventBus, world_id: str | None = None) -> Callable[[], list[dict[str, Any]]]:
    """Subscribe now; each call returns the events published since the previous call."""
    queue = bus.subscribe(world_id)

    def take() -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while not queue.empty():
            event = queue.get_nowait()
            if event is not None:
                events.append(event)
        return events

    return take
