"""The ground derivation: which cells a map's art actually takes.

Every case here is built from PIXELS, not from a fixture map, because the whole
point of ``worlds.ground`` is that the answer comes off the art rather than off a
declared footprint. A tile is drawn as a diamond (flat) or a diamond with a tower
on top of it (standing), and the test asserts which cells that costs.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from PIL import Image, ImageDraw

from worlds.ground import derive_ground

TW, TH = 128, 64  # the grid every case below is on


def diamond(draw: ImageDraw.ImageDraw, cx: float, cy: float, color: tuple[int, int, int, int]) -> None:
    """One cell's ground plate, centered at (cx, cy) in the image."""
    draw.polygon(
        [(cx, cy - TH / 2), (cx + TW / 2, cy), (cx, cy + TH / 2), (cx - TW / 2, cy)],
        fill=color,
    )


def sheet(width: int, height: int, paint) -> bytes:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    paint(ImageDraw.Draw(image))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def flat_tile() -> bytes:
    """A 1×1 ground plate — exactly one cell, nothing above it."""
    return sheet(TW, TH, lambda d: diamond(d, TW / 2, TH / 2, (90, 90, 90, 255)))


def tower_tile() -> bytes:
    """A 512×256 canvas whose base is ONE cell and whose art rises far above it —
    the shape a wall or a house actually has, and the one a size-based footprint
    guess gets wrong (it would claim a 4×4 block)."""

    def paint(d: ImageDraw.ImageDraw) -> None:
        # base plate in the bottom-most cell of the canvas
        diamond(d, TW / 2, 256 - TH / 2, (60, 40, 30, 255))
        # the body, straight up out of that cell
        d.rectangle([TW / 2 - 30, 20, TW / 2 + 30, 256 - TH / 2], fill=(60, 40, 30, 255))

    return sheet(512, 256, paint)


def slab_tile() -> bytes:
    """A 256×128 canvas fully painted as four flat cell plates — a large paving
    slab. Taller than a cell and still the floor, which is exactly where the
    height heuristic leaks and why `road` has the last word."""

    def paint(d: ImageDraw.ImageDraw) -> None:
        for cx, cy in ((TW, TH / 2), (TW / 2, TH), (TW * 1.5, TH), (TW, TH * 1.5)):
            diamond(d, cx, cy, (120, 120, 110, 255))

    return sheet(256, 128, paint)


def doc(
    *,
    tilesets: list[dict[str, Any]],
    layers: list[list[int]],
    cols: int = 4,
    rows: int = 4,
) -> dict[str, Any]:
    return {
        "width": cols,
        "height": rows,
        "tilewidth": TW,
        "tileheight": TH,
        "orientation": "isometric",
        "tilesets": tilesets,
        "layers": [
            {"type": "tilelayer", "name": f"L{i}", "data": data}
            for i, data in enumerate(layers)
        ],
    }


def tileset(
    name: str,
    firstgid: int,
    w: int,
    h: int,
    *,
    tiles: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "firstgid": firstgid,
        "image": f"{name}.png",
        "tilewidth": w,
        "tileheight": h,
        "columns": 1,
        "tilecount": 1,
        "margin": 0,
        "spacing": 0,
        "tiles": tiles or [],
    }


def reader(**art: bytes):
    return lambda path: art.get(path)


def cell(grid, x: int, y: int) -> bool:
    return grid[y][x]


def test_flat_ground_takes_nothing() -> None:
    """A tile exactly one cell high IS the ground — every cell stays standable."""
    ground = derive_ground(
        doc(tilesets=[tileset("floor", 1, TW, TH)], layers=[[1] * 16]),
        reader(**{"floor.png": flat_tile()}),
    )
    assert all(all(row) for row in ground.standable)


def test_a_standing_tile_takes_the_cells_its_pixels_cover_not_its_canvas() -> None:
    """The whole reason the pixels are read: a 512×256 tower occupies the ONE cell
    it stands on plus what its body hides, never the 4×4 block its canvas implies."""
    layers = [[1] * 16, [0] * 16]
    layers[1][2 * 4 + 2] = 2  # tower at (2, 2)
    ground = derive_ground(
        doc(tilesets=[tileset("floor", 1, TW, TH), tileset("tower", 2, 512, 256)], layers=layers),
        reader(**{"floor.png": flat_tile(), "tower.png": tower_tile()}),
    )
    hidden = {(x, y) for y in range(4) for x in range(4) if not ground.standable[y][x]}
    # Straight up out of its cell is the diagonal dx == dy == -k in grid space, so a
    # tower four cells tall takes exactly that column and nothing else. A footprint
    # guessed from the 512×256 canvas would instead claim a 4×4 block reaching SOUTH
    # — the opposite direction, and the whole grid.
    assert hidden == {(2, 2), (1, 1), (0, 0)}
    assert cell(ground.standable, 3, 0) and cell(ground.standable, 0, 3), "off the column, untouched"


def test_a_tower_hides_the_ground_behind_it_never_in_front() -> None:
    """It grows upward out of its cell, and up-screen is north — so the cells it
    hides are the ones BEHIND it. Standing in front of a house is fine."""
    layers = [[1] * 16, [0] * 16]
    layers[1][3 * 4 + 3] = 2  # tower at (3, 3), the south corner
    ground = derive_ground(
        doc(tilesets=[tileset("floor", 1, TW, TH), tileset("tower", 2, 512, 256)], layers=layers),
        reader(**{"floor.png": flat_tile(), "tower.png": tower_tile()}),
    )
    hidden = {(x, y) for y in range(4) for x in range(4) if not ground.standable[y][x]}
    assert (3, 3) in hidden
    assert all(x + y <= 6 for x, y in hidden), "nothing in FRONT of it was taken"


def test_collides_takes_ground_that_is_visible_but_unstandable() -> None:
    """Water is a one-cell plate: geometry sees nothing wrong with it, so the mark
    is the only thing that can say so."""
    layers = [[1] * 16]
    layers[0][1 * 4 + 1] = 2
    water = tileset("water", 2, TW, TH, tiles=[{"id": 0, "properties": [{"name": "collides", "value": True}]}])
    ground = derive_ground(
        doc(tilesets=[tileset("floor", 1, TW, TH), water], layers=layers),
        reader(**{"floor.png": flat_tile(), "water.png": flat_tile()}),
    )
    assert not cell(ground.standable, 1, 1)
    assert not cell(ground.walkable, 1, 1)
    assert cell(ground.standable, 0, 0)


def test_a_road_slab_stays_floor_however_big_its_canvas() -> None:
    """The height heuristic alone would read a 256×128 paving slab as a thing
    standing up and shut its own cells. `road` says otherwise, and wins."""
    layers = [[1] * 16, [0] * 16]
    layers[1][2 * 4 + 2] = 2
    slab = tileset("slab", 2, 256, 128, tiles=[{"id": 0, "properties": [{"name": "road", "value": True}]}])
    ground = derive_ground(
        doc(tilesets=[tileset("floor", 1, TW, TH), slab], layers=layers),
        reader(**{"floor.png": flat_tile(), "slab.png": slab_tile()}),
    )
    assert all(all(row) for row in ground.standable), "a declared floor hides nothing"
    assert cell(ground.walkable, 2, 2), "and it opens the ground it paints"


def test_one_road_mark_makes_the_map_a_whitelist() -> None:
    """A map that declares its streets says nothing carries traffic elsewhere."""
    layers = [[1] * 16, [0] * 16]
    layers[1][2 * 4 + 2] = 2
    slab = tileset("slab", 2, 256, 128, tiles=[{"id": 0, "properties": [{"name": "road", "value": True}]}])
    ground = derive_ground(
        doc(tilesets=[tileset("floor", 1, TW, TH), slab], layers=layers),
        reader(**{"floor.png": flat_tile(), "slab.png": slab_tile()}),
    )
    assert not cell(ground.walkable, 0, 0), "unmarked ground carries no traffic"


def test_a_map_with_no_roads_is_open_except_where_it_says_otherwise() -> None:
    """The other convention: mark the obstacles, everything else is open."""
    layers = [[1] * 16]
    layers[0][1 * 4 + 1] = 2
    wall = tileset("wall", 2, TW, TH, tiles=[{"id": 0, "properties": [{"name": "collides", "value": True}]}])
    ground = derive_ground(
        doc(tilesets=[tileset("floor", 1, TW, TH), wall], layers=layers),
        reader(**{"floor.png": flat_tile(), "wall.png": flat_tile()}),
    )
    assert cell(ground.walkable, 0, 0)
    assert not cell(ground.walkable, 1, 1)


def test_unreadable_art_degrades_to_hiding_nothing() -> None:
    """A sheet that will not load must not blank the map or crash the check —
    that tileset simply covers nothing and everything else still works."""
    layers = [[1] * 16, [0] * 16]
    layers[1][2 * 4 + 2] = 2
    ground = derive_ground(
        doc(tilesets=[tileset("floor", 1, TW, TH), tileset("tower", 2, 512, 256)], layers=layers),
        reader(**{"floor.png": flat_tile()}),  # tower.png missing
    )
    assert all(all(row) for row in ground.standable)


def test_payload_is_one_string_per_row() -> None:
    ground = derive_ground(
        doc(tilesets=[tileset("floor", 1, TW, TH)], layers=[[1] * 16], cols=4, rows=4),
        reader(**{"floor.png": flat_tile()}),
    )
    payload = ground.to_payload()
    assert payload["cols"] == 4 and payload["rows"] == 4
    assert payload["standable"] == ["1111"] * 4
    assert len(payload["walkable"]) == 4


@pytest.mark.parametrize("template", ["changan_iso", "metro"])
def test_every_shipped_map_gives_every_location_ground_to_stand_on(template: str) -> None:
    """The case the whole thing exists for: a location rectangle drawn over
    solid rooftops leaves the people the sim puts there nowhere to be."""
    import json
    from worlds.template_check import MIN_STANDING_CELLS
    from worlds.tiled import MAP_FILENAME, template_dir

    root = template_dir(template)
    document = json.loads((root / MAP_FILENAME).read_text(encoding="utf-8"))
    ground = derive_ground(
        document,
        lambda p: (root / p).read_bytes() if (root / p).is_file() else None,
    )
    unit = float(document["tileheight"])
    thin: list[tuple[str, int]] = []
    for layer in document["layers"]:
        for group in [layer] if layer["type"] != "group" else layer["layers"]:
            if group.get("type") != "objectgroup":
                continue
            for obj in group["objects"]:
                if obj.get("type") != "location":
                    continue
                gx0, gy0 = obj["x"] / unit, obj["y"] / unit
                gw, gh = obj["width"] / unit, obj["height"] / unit
                cells = [
                    (x, y)
                    for y in range(ground.rows)
                    for x in range(ground.cols)
                    if gx0 <= x + 0.5 <= gx0 + gw and gy0 <= y + 0.5 <= gy0 + gh
                ]
                room = ground.standable_in(cells)
                if room < MIN_STANDING_CELLS:
                    name = next(p["value"] for p in obj["properties"] if p["name"] == "location_id")
                    thin.append((name, room))
    assert not thin, f"{template} has locations with no room to stand: {thin}"


def test_a_gid_from_an_image_collection_tileset_borrows_nobody_else_s_art() -> None:
    """An image-collection tileset (no single image; the usual way to hold prop art in Tiled)
    isn't in tilesets, so its gids must not be attributed to the sheet before it.

    "The last sheet with firstgid <= gid" would hand it to the previous sheet and sample pixels
    with a gid that doesn't belong to that image. Landing outside the image just crashes; landing
    inside is worse: that ground cell is computed from unrelated art, silently.
    """
    # The image actually holds two tiles (512x256 each, stacked), but the tileset declares only 1;
    # the second tile belongs to someone else.
    def paint(d: ImageDraw.ImageDraw) -> None:
        diamond(d, TW / 2, 256 - TH / 2, (60, 40, 30, 255))
        d.rectangle([0, 256, 511, 511], fill=(200, 0, 0, 255))   # second tile: fully opaque

    two_tile_sheet = sheet(512, 512, paint)
    grids = derive_ground(
        doc(
            tilesets=[
                tileset("tower", 1, 512, 256),                    # tilecount=1
                {"name": "props", "firstgid": 2, "tilewidth": TW, "tileheight": 256,
                 "tilecount": 50, "tiles": [{"id": 0, "image": "prop.png"}]},
            ],
            layers=[[0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # gid 2 belongs to props
        ),
        reader(**{"tower.png": two_tile_sheet}),
    )
    # Unidentified art blocks nothing. Don't derive ground from the red second tile of tower.
    assert all(all(row) for row in grids.standable)


def test_a_tileset_that_overstates_its_tilecount_does_not_crash_the_map() -> None:
    """When a sheet doesn't match its declared tilecount, sample points fall outside the image.

    A valid in-tile coordinate doesn't mean ox/oy lands on the image. This path also serves the
    map API, so an unsampleable point counts as transparent: blocking a bit less beats failing
    to load the whole map.
    """
    grids = derive_ground(
        doc(
            tilesets=[{**tileset("tower", 1, 512, 256), "tilecount": 999, "columns": 1}],
            layers=[[0, 0, 0, 0, 0, 400, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        ),
        reader(**{"tower.png": tower_tile()}),
    )
    assert all(all(row) for row in grids.standable)
