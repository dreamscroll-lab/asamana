"""The world's places and the graph between them — fixed at build, read-only at runtime."""

from __future__ import annotations

from heapq import heappop, heappush

from core.interfaces.perception import LocationView
from core.interfaces.place import Place


class SpaceManager:
    """The roster of places, and the graph over it.

    Places are supplied once by ``WorldConfig`` and never change afterwards, so each source's
    Dijkstra tree is computed once and never invalidated, and the numbered "我可前往" (where I can
    go) list is stable across steps.

    It doesn't know who is standing on it (ask ``EnvironmentSystem.bodies_at``), and doesn't judge
    whether someone can get in: that is judgment, not a threshold.

    ``IN_TRANSIT`` / ``UNPLACED`` belong to the carrying layer, not here. Unknown places return
    ``None`` / empty and the caller picks the fallback: in transit reads "途中" (on the way), an
    unknown place "某地" (somewhere).
    """

    def __init__(self) -> None:
        # ``register_place`` is the only way in: a second entry point couldn't share its guard.
        self._places: dict[str, Place] = {}
        # source_id → {node: (shortest_dist, predecessor_on_path)}
        self._path_cache: dict[str, dict[str, tuple[int, str | None]]] = {}

    # ---- roster -------------------------------------------------------------

    def register_place(self, place: Place) -> None:
        """The single build-time entry point. Don't call at runtime: the "我可前往" index contract
        would drift. Clearing the cache only serves batch registration during build.
        """
        self._places[place.place_id] = place
        self._path_cache.clear()

    def has(self, place_id: str) -> bool:
        return place_id in self._places

    def get(self, place_id: str) -> Place | None:
        return self._places.get(place_id)

    def all_places(self) -> list[Place]:
        """Every place, ordered by id: callers number this list for the LLM, so order must be
        stable."""
        return [self._places[pid] for pid in sorted(self._places)]

    def all_place_ids(self) -> list[str]:
        return sorted(self._places)

    def resolve(self, raw: str) -> str | None:
        """Resolve place text (id or name) to a place id; ``None`` if unrecognized."""
        if not raw:
            return None
        if raw in self._places:
            return raw
        for place_id, place in self._places.items():
            if place.name == raw:
                return place_id
        return None

    def name_of(self, place_id: str) -> str | None:
        """This place's name; ``None`` if unknown (the caller picks the fallback referent)."""
        place = self._places.get(place_id)
        return place.name if place is not None else None

    def view_of(self, place_id: str) -> LocationView | None:
        """This place's narrative-layer view (name + description + whether open to outsiders);
        ``None`` if unknown.

        No name prepended to the description: ``core.prompts.render_location`` already writes
        "name——description".
        """
        place = self._places.get(place_id)
        if place is None:
            return None
        return LocationView(
            name=place.name, description=place.description, is_public=place.is_public,
        )

    # ---- graph --------------------------------------------------------------

    def _shortest_tree(self, source: str) -> dict[str, tuple[int, str | None]]:
        """Cached Dijkstra tree from ``source``: ``{node: (dist, predecessor)}``.

        Edge weights are positive travel seconds. Empty if ``source`` is unknown.
        """
        cached = self._path_cache.get(source)
        if cached is not None:
            return cached
        tree: dict[str, tuple[int, str | None]] = {}
        if source in self._places:
            tree[source] = (0, None)
            frontier: list[tuple[int, str]] = [(0, source)]
            while frontier:
                dist, node = heappop(frontier)
                if dist > tree[node][0]:
                    continue
                place = self._places.get(node)
                if place is None:
                    continue
                for neighbour, weight in place.connections.items():
                    if neighbour not in self._places:
                        continue
                    nd = dist + weight
                    if neighbour not in tree or nd < tree[neighbour][0]:
                        tree[neighbour] = (nd, node)
                        heappush(frontier, (nd, neighbour))
        self._path_cache[source] = tree
        return tree

    def reachable_from(self, from_id: str) -> dict[str, int]:
        """Shortest-path travel seconds from ``from_id`` to every reachable place.

        Multi-hop, NOT just direct neighbours; excludes ``from_id`` itself.
        """
        return {node: dist for node, (dist, _) in self._shortest_tree(from_id).items() if node != from_id}

    def shortest_path(self, from_id: str, to_id: str) -> list[str] | None:
        """The ordered waypoint sequence [from_id, …, to_id] along the shortest path.

        ``None`` if unreachable.
        """
        tree = self._shortest_tree(from_id)
        if to_id not in tree:
            return None
        path = [to_id]
        while path[-1] != from_id:
            predecessor = tree[path[-1]][1]
            if predecessor is None:
                break
            path.append(predecessor)
        path.reverse()
        return path

    def edge_seconds(self, from_id: str, to_id: str) -> int:
        """Travel seconds of the DIRECT edge ``from_id``→``to_id`` (must be an existing edge)."""
        return self._places[from_id].connections[to_id]
