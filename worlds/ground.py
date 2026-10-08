"""Where a body may stand, and where a route may run — read off the map's own art.

A location is a rectangle drawn over a whole ward, and a ward is mostly roofs,
canopies and water. Placing anyone inside that rectangle without asking what is
drawn there puts figures on rooftops and up trees; routing anyone across it does
the same thing while they walk. Both questions come down to one: which cells does
each tile's art actually take?

The obvious answer — have the author tag every solid tile — cannot work on its
own, because **a tile's occupied cells are not derivable from the tile's size**.
Tiled anchors a tile image by its bottom-left corner onto the cell's bottom-left,
so a tall image grows UP out of its cell; a 512×256 wall on a 128×64 grid is four
cells wide only if it is flat, and it isn't — most of that canvas is height.
Nothing in the document says how much. So a tag would still need a guessed
footprint, and the guess is what puts a figure through a wall (measured on a real
city map: the guess invents 347 road cells that no road tile ever painted).

The art already knows. Anchored the way Tiled anchors it, a tile's own pixels say
exactly which cells' ground it covers: project each candidate cell's center back
into the image and look. Three rules follow.

1. GEOMETRY, free — a tile TALLER than the map's cell is a thing standing up, and
   it takes the cells its pixels cover. A tile exactly one cell high IS the ground
   (paving, grass) and takes nothing. The author draws a house; the ground under
   and behind it stops being available, with nothing to tag.
2. ``collides``, tagged — ground you can SEE but cannot stand on: open water, and
   whatever a later map means by it. No image tells water from ice, so this one is
   the author's word. It is a 1×1 ground tile, so it needs no footprint guess.
3. ``road``, already tagged — never hides anything, whatever its canvas size, and
   opens the cells its art covers. The mark declares the tile IS a surface.

Rule 3 exists because rule 1 is a PROXY, and this is where it leaks: canvas height
cannot distinguish a thing standing up from a large flat slab (a 2×2-cell paving
patch is 256×128 on a 128×64 grid — taller than a cell, still the floor). Nor can
the pixels settle it. The cover mask absorbs height by construction — it asks which
ground is HIDDEN, and a fixed camera cannot tell "hidden by something standing
there" from "that is the surface" — so measuring how far a tile's art rises above
its own footprint reports a palace as flat. Both tests were tried on the real art
and both failed. What is left is what the map already says.

WHY THE TWO GRIDS ARE SEPARATE. Standing and walking are different questions and a
single grid answers neither well. Intersecting them shatters the route network
(Chang'an's roads fall into 21 pieces, the metro map's into 51) because a street
legitimately passes behind a wall for a beat. So ``walkable`` stays the author's
declared network and ``standable`` is the stricter one; the renderer bridges them
by making hidden ground EXPENSIVE to route over rather than impassable.

WHY THIS LIVES IN ``worlds/`` AND NOT THE RENDERER. It has three consumers — the
renderer's staging, its A* network, and ``template_check`` — and the maths is the
map FORMAT's, not Phaser's: the anchoring rule is Tiled's own (the renderer's
``mapSource`` exists precisely to bend Phaser back onto it) and the layer order is
the document's. One implementation, checkable offline, testable.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from PIL import Image

# Alpha at or below this is see-through enough that the ground still shows.
OPAQUE_ALPHA = 40
# Where inside a cell's diamond to look, as fractions of the cell: its center and
# four inset points. Sampling the center alone lets a canopy with a gap at its
# middle read as clear; sampling the whole diamond makes every clipped corner cost
# a whole cell.
CELL_SAMPLES: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (-0.28, 0.0),
    (0.28, 0.0),
    (0.0, -0.28),
    (0.0, 0.28),
)
# How many of those must be opaque before the cell counts as covered. At 1 a tree
# trunk clipping a corner costs the whole cell; at 3 cells a building visibly covers
# stay standable.
COVER_HITS = 2

# Tiled packs flip/rotate flags into a gid's top bits.
GID_MASK = 0x1FFFFFFF

# What a template's own art directory is read through when no world froze it.
ReadAsset = Callable[[str], bytes | None]


@dataclass(frozen=True)
class GroundGrids:
    """The two per-cell answers, both indexed ``grid[y][x]``."""

    cols: int
    rows: int
    walkable: list[list[bool]]  # a route may run here
    standable: list[list[bool]]  # a body may be seen at rest here

    def to_payload(self) -> dict[str, Any]:
        """Wire form: one string of ``0``/``1`` per row, which is compact enough
        (a 48×56 map is 2.7 KB a grid) that no encoding cleverness earns its keep."""
        rows = lambda grid: ["".join("1" if cell else "0" for cell in row) for row in grid]
        return {
            "cols": self.cols,
            "rows": self.rows,
            "walkable": rows(self.walkable),
            "standable": rows(self.standable),
        }

    def standable_in(self, cells: Iterable[tuple[int, int]]) -> int:
        return sum(
            1
            for x, y in cells
            if 0 <= x < self.cols and 0 <= y < self.rows and self.standable[y][x]
        )


def _num(value: Any, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _tile_layers(layers: Any) -> list[dict[str, Any]]:
    """Every tile layer, groups walked through — Tiled lets an author file layers
    into groups, and a map organized that way nests them a level down."""
    out: list[dict[str, Any]] = []

    def walk(items: Any) -> None:
        for layer in items if isinstance(items, list) else []:
            if layer.get("type") == "group":
                walk(layer.get("layers", []))
            elif layer.get("type") == "tilelayer" and isinstance(layer.get("data"), list):
                out.append(layer)

    walk(layers)
    return out


def _marks(tile: dict[str, Any]) -> set[str]:
    return {
        str(p.get("name"))
        for p in tile.get("properties", []) or []
        if p.get("value") is True
    }


class _Sheets:
    """One decoded tileset image at a time, by the path the map writes."""

    def __init__(self, read_asset: ReadAsset) -> None:
        self._read = read_asset
        self._cache: dict[str, Any] = {}

    def pixels(self, image_path: str) -> Any:
        """The sheet's pixel accessor, or None if it cannot be read. A sheet that
        fails to load means "this tileset covers nothing" — the map still validates
        and still renders, it is simply less careful about that one tileset."""
        if image_path in self._cache:
            return self._cache[image_path]
        access = None
        raw = self._read(image_path)
        if raw is not None:
            try:
                image = Image.open(io.BytesIO(raw)).convert("RGBA")
                access = (image.load(), image.width, image.height)
            except (OSError, ValueError):
                access = None
        self._cache[image_path] = access
        return access


def _cover_mask(
    sheet: tuple[Any, int, int],
    tileset: dict[str, Any],
    local_id: int,
    map_w: float,
    map_h: float,
) -> list[tuple[int, int]]:
    """The cells whose ground this tile's pixels cover, as (dx, dy) from the cell
    it is placed on."""
    pixels, sheet_w, sheet_h = sheet
    tile_w = int(_num(tileset.get("tilewidth"), map_w))
    tile_h = int(_num(tileset.get("tileheight"), map_h))
    margin = int(_num(tileset.get("margin")))
    spacing = int(_num(tileset.get("spacing")))
    columns = int(_num(tileset.get("columns"))) or max(
        1, (sheet_w - margin + spacing) // (tile_w + spacing)
    )
    ox = margin + (local_id % columns) * (tile_w + spacing)
    oy = margin + (local_id // columns) * (tile_h + spacing)

    span_x = math.ceil(tile_w / map_w) + 1
    span_y = math.ceil(tile_h / map_h) + 1
    out: list[tuple[int, int]] = []
    # A tall tile only ever covers cells at or BEHIND its own (dy <= 1): it grows
    # upward out of its cell, and up-screen is north.
    for dy in range(-span_y, 2):
        for dx in range(-span_x, span_x + 1):
            # The cell's center in the tile image's own pixels, given Tiled's
            # anchoring (image bottom-left onto the cell's bottom-left).
            cx = (dx - dy) * map_w / 2 + map_w / 2
            cy = (dx + dy) * map_h / 2 - map_h / 2 + tile_h
            hits = 0
            for u, v in CELL_SAMPLES:
                px = round(cx + u * map_w)
                py = round(cy + v * map_h)
                if px < 0 or py < 0 or px >= tile_w or py >= tile_h:
                    continue
                # Also clamp to the atlas bounds: a coordinate valid inside the tile doesn't mean ox/oy lands on
                # the image (an atlas whose real size disagrees with its declared tilecount overruns). This path
                # serves the map API, where an out-of-bounds pixel read becomes a 500. Treat an unreadable pixel
                # as transparent: hiding slightly less beats failing to produce the image at all.
                if ox + px >= sheet_w or oy + py >= sheet_h:
                    continue
                if pixels[ox + px, oy + py][3] > OPAQUE_ALPHA:
                    hits += 1
            if hits >= COVER_HITS:
                out.append((dx, dy))
    return out


def derive_ground(doc: dict[str, Any], read_asset: ReadAsset) -> GroundGrids:
    """Both grids for a map document. ``read_asset`` is handed each tileset image
    path exactly as the map writes it, and may return None for any of them."""
    cols = int(_num(doc.get("width")))
    rows = int(_num(doc.get("height")))
    map_w = _num(doc.get("tilewidth"), 32) or 32.0
    map_h = _num(doc.get("tileheight"), 32) or 32.0

    tilesets = sorted(
        (
            ts
            for ts in doc.get("tilesets", [])
            if isinstance(ts, dict) and "source" not in ts and ts.get("image")
        ),
        key=lambda ts: _num(ts.get("firstgid")),
    )

    blocking: set[int] = set()  # gids tagged collides — no body here at all
    surface: set[int] = set()  # gids tagged road — declared floor
    for tileset in tilesets:
        first = int(_num(tileset.get("firstgid")))
        for tile in tileset.get("tiles", []) or []:
            marks = _marks(tile)
            gid = first + int(_num(tile.get("id")))
            if "collides" in marks:
                blocking.add(gid)
            elif "road" in marks:
                surface.add(gid)
    # One `road` tile anywhere makes the map a WHITELIST: it declares its streets,
    # so anything unmarked carries no traffic. A map with none marks its obstacles
    # instead, and everything else is open. Never both readings at once — an absent
    # `road` on a map that has no roads would otherwise block the world.
    has_roads = bool(surface)

    covered = [[False] * cols for _ in range(rows)]
    opened = [[False] * cols for _ in range(rows)]
    shut = [[False] * cols for _ in range(rows)]

    sheets = _Sheets(read_asset)
    masks: dict[int, list[tuple[int, int]]] = {}

    def gid_tileset(gid: int) -> dict[str, Any] | None:
        """Which sheet this gid's art lives on, or None when it lives on none of them.

        ``tilesets`` holds only the sheet-backed ones, so "the last sheet with firstgid ≤ gid" would
        assign a gid from an excluded sheet (an external source reference, or an object sheet built
        from an image collection with no single image) to the sheet before it, where the number
        doesn't belong. Bounds must be checked against tilecount: a misattributed gid reads pixels
        from someone else's atlas that aren't its own, or runs off the edge.
        """
        found = None
        for tileset in tilesets:
            if gid >= _num(tileset.get("firstgid")):
                found = tileset
        if found is None:
            return None
        count = int(_num(found.get("tilecount")))
        if count and gid - int(_num(found.get("firstgid"))) >= count:
            return None
        return found

    def mask_of(gid: int, tileset: dict[str, Any]) -> list[tuple[int, int]]:
        if gid not in masks:
            sheet = sheets.pixels(str(tileset.get("image") or ""))
            masks[gid] = (
                []
                if sheet is None
                else _cover_mask(
                    sheet, tileset, gid - int(_num(tileset.get("firstgid"))), map_w, map_h
                )
            )
        return masks[gid]

    def stamp(grid: list[list[bool]], x: int, y: int, cells: list[tuple[int, int]]) -> None:
        for dx, dy in cells:
            cx, cy = x + dx, y + dy
            if 0 <= cx < cols and 0 <= cy < rows:
                grid[cy][cx] = True

    for layer in _tile_layers(doc.get("layers", [])):
        for index, raw in enumerate(layer.get("data", [])):
            gid = int(raw) & GID_MASK
            if not gid:
                continue
            x, y = index % cols, index // cols
            if x >= cols or y >= rows:
                continue
            if gid in blocking:
                shut[y][x] = True
            tileset = gid_tileset(gid)
            if tileset is None:
                continue
            if gid in surface:
                # Rule 3: a declared floor. Opens the cells its ART covers — not the
                # cells a size-based guess would claim — and hides nothing.
                stamp(opened, x, y, mask_of(gid, tileset) or [(0, 0)])
                continue
            if _num(tileset.get("tileheight"), map_h) <= map_h:
                continue  # rule 1: ground art covers nothing
            stamp(covered, x, y, mask_of(gid, tileset))

    standable = [
        [not covered[y][x] and not shut[y][x] for x in range(cols)] for y in range(rows)
    ]
    walkable = [
        [(opened[y][x] if has_roads else True) and not shut[y][x] for x in range(cols)]
        for y in range(rows)
    ]
    return GroundGrids(cols=cols, rows=rows, walkable=walkable, standable=standable)
