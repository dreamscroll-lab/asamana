"""A place in the world's space: the type the ``WorldConfig`` contract hands out."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class Place:
    """A node in space: a room, a gate, a square.

    Fixed at build time; nothing at runtime can change it. ``frozen`` enforces that contract. With
    fixed topology, shortest paths are constant for the world's lifetime (so ``SpaceManager`` can
    compute them all once without invalidation), and the indexed "I can go to" list stays stable
    across steps; in a world that reorders every step, agents would have to relearn it. To change a
    place's capacity or access, change its build-time value, not an assignment at runtime.

    Who is standing here isn't stored here. Occupancy changes every step while this is an identity
    object; two separately maintained copies will drift, and this one would also be baked into the
    persisted world asset and come back as ghost occupants on restore. Ask
    ``EnvironmentSystem.bodies_at``.

    It has no placement. A location isn't inside another location, can't be taken and can't be
    destroyed; those belong to placed things (``WorldEntity``). In a single type, "destroy a
    palace" would be a legal call and the palace would still sit in the graph, open for business.
    """

    place_id:    str
    name:        str
    description: str = ""
    # Destination → seconds to walk. A map fact, independent of the world's clock: steps come from a
    # whole route's time, never per edge (see ``engine/executors/movement.py``).
    connections: dict[str, int] = field(default_factory=dict)
    capacity:    int = 50
    # Whether this place is open to outsiders: a fact visible on sight, not an access check. Who gets
    # in depends on who they are and their business; that's a judgment, not a threshold (see ``core.interfaces.perception.LocationView``).
    is_public:   bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Place":
        """Rebuild a place from its serialized form. Unknown keys dropped, defaults fill in."""
        known = {f.name for f in fields(cls)}
        payload = {k: v for k, v in dict(data).items() if k in known}
        raw_connections = payload.get("connections") or {}
        payload["connections"] = {str(k): int(v) for k, v in dict(raw_connections).items()}
        return cls(**payload)


@dataclass(frozen=True)
class SubPlace:
    """A named part of a place, within it: a building, facility, venue or area the story touches
    (a mansion in a ward, a hall in a palace). It belongs to exactly one place.

    Narrative only, never a node: it isn't in the graph, nobody stands "in" it, and whoever is at
    ``location_id`` is in every one of its sub-places, seen and heard by all there. Its job is to
    give the story's places a name the map lacks, so going there means going to ``location_id``.

    Not a ``Place`` (no connections, capacity or occupants), not a landmark (a landmark is a thing,
    with state, that can be acted on), not an alias (an alias names the whole place). The theme
    supplies it at build time, so it lives in the world's setup, never in the theme-agnostic map.
    """

    location_id: str
    name: str

    def as_dict(self) -> dict[str, str]:
        return {"location_id": self.location_id, "name": self.name}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubPlace":
        return cls(location_id=str(data["location_id"]), name=str(data["name"]))
