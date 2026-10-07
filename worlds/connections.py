"""Derive a template's ``connections`` graph from the ground it is drawn on.

WHY THIS EXISTS. ``connections`` and the map art are two independent statements
about the same thing — which places are next to each other — and nothing else
checks that they agree. When they drift, the drift is invisible: declaring "东宫"
adjacent to "太极宫" across a palace wall makes the sim price the move as a short walk
while the renderer has to walk 45 cells to honour it. Hand-authored edges come
from a picture of the city in someone's head, not from the city as drawn.

THE RULE, and it is a route question, not a distance one:

    A and B are connected  ⟺  the shortest walkable route between them
                              enters no THIRD location's footprint.

If the route does pass through C, then what the map is saying is A—C—B, and that
is what the graph should say too.

Locations are AREAS, not points, and that is load-bearing. "东宫" is a 3×11 tile
rectangle; an anchor-to-anchor test measures a walk between two ward centres that
nobody ever takes and lands 45 cells where the real edge-to-edge walk is 19.
Don't use a nearest-anchor (Voronoi) partition either: it looks equivalent and is
not — it severs a long through-street whenever some unrelated ward's anchor sits
nearer its middle (on Chang'an it would delete "朱雀大街" itself).

WHAT IT DOES NOT DECIDE. Whether a wall has a door in it. If the art shows solid
stone, no derivation can invent a passage, and if the art shows open ground the
derivation will always call it walkable — a locked gate has to be drawn as one.
So this answers "does the ground permit it"; the author still owns what the
world is like.
"""

from __future__ import annotations

import json
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable

from worlds.ground import GroundGrids, derive_ground
from worlds.tiled import MAP_FILENAME, read_template_asset, template_dir, tiled_properties

Cell = tuple[int, int]

__all__ = ["Place", "derive_connections", "declared_connections", "places_of", "apply_to_template"]


class Place:
    """A location as the ground sees it: the cells it covers, and the ones you can stand on."""

    __slots__ = ("location_id", "name", "area", "footing")

    def __init__(self, location_id: str, name: str, area: set[Cell], footing: set[Cell]) -> None:
        self.location_id = location_id
        self.name = name
        self.area = area  # every cell of the object rectangle
        self.footing = footing  # …of those, the ones on the walkable network

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return f"Place({self.location_id!r}, {len(self.footing)} walkable cells)"


def _objects(layers: Any) -> Iterable[dict[str, Any]]:
    for layer in layers or ():
        if layer.get("type") == "objectgroup":
            yield from layer.get("objects", ())
        if layer.get("layers"):
            yield from _objects(layer["layers"])



def main_component(ground: GroundGrids) -> set[Cell]:
    """The largest 4-connected walkable region — the network the renderer routes on.

    Mirrors ``pathfinding.computeMainComponent``; a route derived on any other set
    would not be the route that ships.
    """
    seen = [[False] * ground.cols for _ in range(ground.rows)]
    best: set[Cell] = set()
    for y0 in range(ground.rows):
        for x0 in range(ground.cols):
            if not ground.walkable[y0][x0] or seen[y0][x0]:
                continue
            seen[y0][x0] = True
            stack = [(x0, y0)]
            comp = {(x0, y0)}
            while stack:
                cx, cy = stack.pop()
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if (
                        0 <= nx < ground.cols
                        and 0 <= ny < ground.rows
                        and ground.walkable[ny][nx]
                        and not seen[ny][nx]
                    ):
                        seen[ny][nx] = True
                        comp.add((nx, ny))
                        stack.append((nx, ny))
            if len(comp) > len(best):
                best = comp
    return best


def places_of(doc: dict[str, Any], network: set[Cell]) -> dict[str, Place]:
    """Every location object as a `Place`, footed on the walkable network.

    A gate is often a one-tile object sitting ON its own wall, so its rectangle holds
    no walkable cell at all. Those take the nearest network cells instead — the same
    give ``pathfinding.snap`` and ``_check_roads`` allow, for the same reason.
    """
    unit = float(doc.get("tileheight") or 32) or 32.0
    out: dict[str, Place] = {}
    for obj in _objects(doc.get("layers")):
        props = tiled_properties(obj)
        location_id = str(props.get("location_id") or "").strip()
        if not location_id:
            continue
        x0 = int(float(obj.get("x", 0)) / unit)
        y0 = int(float(obj.get("y", 0)) / unit)
        x1 = x0 + max(1, int(float(obj.get("width", 0)) / unit))
        y1 = y0 + max(1, int(float(obj.get("height", 0)) / unit))
        area = {(x, y) for x in range(x0, x1) for y in range(y0, y1)}
        footing = area & network
        if not footing:
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            for r in range(1, 16):
                footing = {
                    (cx + dx, cy + dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
                } & network
                if footing:
                    break
        out[location_id] = Place(location_id, str(obj.get("name") or location_id), area, footing)
    return out


def _route(a: Place, b: Place, network: set[Cell]) -> list[Cell] | None:
    """Shortest walkable route from anywhere in `a` to anywhere in `b`, or None.

    Multi-source BFS, which on an unweighted grid IS the shortest path — and being
    multi-source at both ends is the point: a figure leaves a ward by its near edge
    and arrives at the far one's, never by way of either centre.
    """
    if not a.footing or not b.footing:
        return None
    prev: dict[Cell, Cell | None] = {c: None for c in a.footing}
    queue = deque(a.footing)
    while queue:
        cell = queue.popleft()
        if cell in b.footing:
            path = []
            node: Cell | None = cell
            while node is not None:
                path.append(node)
                node = prev[node]
            return path[::-1]
        cx, cy = cell
        for nxt in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if nxt in network and nxt not in prev:
                prev[nxt] = cell
                queue.append(nxt)
    return None


def _passes_through(path: list[Cell], a: str, b: str, places: dict[str, Place]) -> set[str]:
    """Which other locations this route walks into."""
    return {
        lid
        for lid, place in places.items()
        if lid not in (a, b) and any(cell in place.area for cell in path)
    }


def derive_connections(
    doc: dict[str, Any], root: Path
) -> tuple[dict[str, list[str]], dict[tuple[str, str], tuple[int, set[str]]]]:
    """The graph the map implies, plus every pair's route length and who it passes through.

    Returns ``(graph, pairs)`` where ``graph[id]`` is the sorted neighbour list and
    ``pairs[(a, b)]`` (a < b) is ``(walk_cells, locations_the_route_enters)``. Pairs
    with no route at all are absent from both.
    """

    ground = derive_ground(doc, lambda image: read_template_asset(root, image))
    network = main_component(ground)
    places = places_of(doc, network)

    graph: dict[str, list[str]] = {lid: [] for lid in places}
    pairs: dict[tuple[str, str], tuple[int, set[str]]] = {}
    ids = sorted(places)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            path = _route(places[a], places[b], network)
            if path is None:
                continue
            through = _passes_through(path, a, b, places)
            pairs[(a, b)] = (len(path) - 1, through)
            if not through:
                graph[a].append(b)
                graph[b].append(a)
    return {lid: sorted(nbrs) for lid, nbrs in graph.items()}, pairs


def declared_connections(doc: dict[str, Any]) -> dict[str, list[str]]:
    """The graph the template currently claims."""
    out: dict[str, list[str]] = {}
    for obj in _objects(doc.get("layers")):
        props = tiled_properties(obj)
        location_id = str(props.get("location_id") or "").strip()
        if not location_id:
            continue
        out[location_id] = [
            part.split(":")[0].strip()
            for part in str(props.get("connections", "")).split(",")
            if part.split(":")[0].strip()
        ]
    return out


_PROPS_BLOCK = re.compile(r'"properties":\[.*?\}\]', re.S)
_LOCATION_ID = re.compile(r'"name":\s*"location_id",\s*"type":\s*"string",\s*"value":\s*"([^"]+)"')
_CONNECTIONS = re.compile(r'("name":\s*"connections",\s*"type":\s*"string",\s*"value":\s*")[^"]*(")')
# The whole property OBJECT that carries location_id, with the newline and indent in
# front of it — the template a missing `connections` is cloned from, so an inserted
# property lands in the author's own layout rather than one this module invented.
_LOCATION_ID_OBJECT = re.compile(
    r'(\n[ \t]*)(\{[^{}]*"name":\s*"location_id"[^{}]*\})', re.S
)


def apply_to_template(path: Path, graph: dict[str, list[str]]) -> int:
    """Write `graph` into a .tmj IN PLACE, touching only the connections values.

    Deliberately a text edit rather than a JSON round-trip: Tiled writes its own
    layout, and re-serialising reflows all 2400 lines of the file, which destroys
    the reviewable diff and hands the author back something they did not write.

    A location declaring no `connections` at all gets one inserted. The field is
    derived, not authored — nobody should have to type an empty property just so
    this can fill it in.
    """
    text = path.read_text(encoding="utf-8")
    written = 0

    def fix(block: str) -> str:
        nonlocal written
        found = _LOCATION_ID.search(block)
        if not found or found.group(1) not in graph:
            return block
        value = ",".join(graph[found.group(1)])
        patched, replaced = _CONNECTIONS.subn(
            lambda m: m.group(1) + value + m.group(2), block, count=1
        )
        if replaced:
            written += replaced
            return patched

        def insert(m: re.Match[str]) -> str:
            # Cloned from the location_id property rather than composed here, so the
            # inserted one carries the author's indentation and quoting, not ours.
            nonlocal written
            clone = (
                m.group(2)
                .replace('"location_id"', '"connections"', 1)
                .replace(f'"{found.group(1)}"', f'"{value}"', 1)
            )
            written += 1
            return f"{m.group(1)}{clone}, {m.group(1)}{m.group(2)}"

        return _LOCATION_ID_OBJECT.sub(insert, block, count=1)

    path.write_text(_PROPS_BLOCK.sub(lambda m: fix(m.group(0)), text), encoding="utf-8")
    return written


def _main(argv: list[str]) -> int:

    if not argv or argv[0] in {"-h", "--help"}:
        print(f"usage: python -m worlds.connections <template> [--write]\n\n{__doc__}")
        return 0
    template, write = argv[0], "--write" in argv[1:]
    root = template_dir(template)
    path = root / MAP_FILENAME
    doc = json.loads(path.read_text(encoding="utf-8"))

    graph, pairs = derive_connections(doc, root)
    names = _names(doc)
    declared = declared_connections(doc)

    def label(lid: str) -> str:
        return names.get(lid, lid)

    derived_edges = {(a, b) for a, nbrs in graph.items() for b in nbrs if a < b}
    declared_edges = {
        tuple(sorted((a, b))) for a, nbrs in declared.items() for b in nbrs if b in graph
    }

    print(f"{template}: 声明 {len(declared_edges)} 条 → 推导 {len(derived_edges)} 条")
    wrong = sorted(declared_edges - derived_edges)
    if wrong:
        print(f"\n地图不支持的声明边 {len(wrong)} 条（应改写成经由中间地点）：")
        for a, b in wrong:
            walk, through = pairs.get((a, b), (None, set()))
            via = "/".join(sorted(label(t) for t in through)) or "无路可走"
            print(f"   − {label(a)} — {label(b)}   实走 {walk} 格，经 {via}")
    missing = sorted(derived_edges - declared_edges)
    if missing:
        print(f"\n地图支持但未声明 {len(missing)} 条：")
        for a, b in missing:
            print(f"   + {label(a)} — {label(b)}   实走 {pairs[(a, b)][0]} 格")
    if not wrong and not missing:
        print("\n声明与地图一致。")

    if write:
        n = apply_to_template(path, graph)
        print(f"\n已写回 {n} 个地点的 connections。")
    elif wrong or missing:
        print("\n（加 --write 按推导结果写回）")
    return 0


def _names(doc: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for obj in _objects(doc.get("layers")):
        lid = str(tiled_properties(obj).get("location_id") or "").strip()
        if lid:
            out[lid] = str(obj.get("name") or lid)
    return out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv[1:]))
