"""World-state bulletin board: a delivery queue shaped like MessageSystem."""

from __future__ import annotations

from typing import List, Mapping

from core.interfaces.perception import Broadcast


class BroadcastChannel:
    """Delivery queue for senderless world-state changes (weather, resources, deaths, ...).

    Same shape as MessageSystem: each broadcast is collected exactly once, by the perception
    pass of the step it falls due. Event injections are due the current step; death notices
    (published after this step's perception) the next, which is why there's no per-step
    clear: it would lose them unperceived.

    Communication with a sender_id goes through MessageSystem instead.
    """

    def __init__(self) -> None:
        self._pending: List[Broadcast] = []

    def publish(self, broadcast: Broadcast) -> None:
        self._pending.append(broadcast)

    def collect(self, *, step: int) -> List[Broadcast]:
        """Take and remove every broadcast with deliver_step <= step; once per step, before
        perception."""
        due = [b for b in self._pending if b.deliver_step <= step]
        if due:
            self._pending = [b for b in self._pending if b.deliver_step > step]
        return due

    def peek_pending(self) -> List[Broadcast]:
        return list(self._pending)

    @staticmethod
    def reaches(broadcast: Broadcast, location_id: str) -> bool:
        """Does this broadcast reach that location? Global ones (location_scope=None) reach
        everywhere; scoped ones reach only their own location.

        The single reachability rule, for both ``for_location`` and ``audience``. Don't write it
        twice: a receipt would report who heard it by a stale rule.
        """
        return broadcast.location_scope is None or broadcast.location_scope == location_id

    @classmethod
    def for_location(cls, broadcasts: List[Broadcast], location_id: str) -> List[Broadcast]:
        """Filter collected broadcasts down to those perceivable at a location."""
        return [b for b in broadcasts if cls.reaches(b, location_id)]

    @classmethod
    def audience(
        cls, broadcast: Broadcast, agent_locations: Mapping[str, str],
    ) -> List[str]:
        """Whose ears did this broadcast reach? The inverse of ``for_location``, same rule.

        The caller passes only those who count (the living): this knows only reachability.
        """
        return [aid for aid, location in agent_locations.items() if cls.reaches(broadcast, location)]
