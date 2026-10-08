"""Deriving ``connections`` from the ground a map is drawn on.

Every case is built from a hand-laid grid rather than from a shipped template, so
what is asserted is the RULE — "A and B are connected iff the shortest walkable
route between them enters no third location" — and not the current state of
Chang'an. The three cases below are the three easy ways to get the rule wrong: a
wall between two places declared adjacent, a through-street that a
nearest-anchor partition wrongly severs, and a location treated as a point when
it is an area.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from worlds.connections import apply_to_template, declared_connections, derive_connections

TW, TH = 128, 64


def _sheet() -> bytes:
    """Two cells side by side: gid 1 is open ground, gid 2 is the same art marked
    ``collides``. Walkability starts open on a map that declares no road network and
    is SUBTRACTED by collision, so a wall has to be a tile — leaving a cell empty
    leaves it walkable (worlds/ground.py)."""
    image = Image.new("RGBA", (TW * 2, TH), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for i in range(2):
        ox = i * TW
        draw.polygon(
            [(ox + TW / 2, 0), (ox + TW, TH / 2), (ox + TW / 2, TH), (ox, TH / 2)],
            fill=(90, 90, 90, 255),
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _map(cols: int, rows: int, floor: list[str], places: list[dict[str, Any]]) -> dict[str, Any]:
    """A tiny isometric map: ``floor`` is one string per row, '.' walkable, ' ' void."""
    data = [1 if floor[y][x] == "." else 2 for y in range(rows) for x in range(cols)]
    return {
        "orientation": "isometric",
        "width": cols,
        "height": rows,
        "tilewidth": TW,
        "tileheight": TH,
        "tilesets": [
            {
                "firstgid": 1,
                "name": "ground",
                "image": "ground.png",
                "imagewidth": TW * 2,
                "imageheight": TH,
                "tilewidth": TW,
                "tileheight": TH,
                "tilecount": 2,
                "columns": 2,
                "tiles": [
                    {"id": 1, "properties": [{"name": "collides", "type": "bool", "value": True}]}
                ],
            }
        ],
        "layers": [
            {"type": "tilelayer", "name": "floor", "width": cols, "height": rows, "data": data},
            {"type": "objectgroup", "name": "place", "objects": places},
        ],
    }


def _place(lid: str, gx: int, gy: int, w: int = 1, h: int = 1, conns: str = "") -> dict[str, Any]:
    return {
        "name": lid,
        "type": "location",
        "x": gx * TH,
        "y": gy * TH,
        "width": w * TH,
        "height": h * TH,
        "properties": [
            {"name": "location_id", "type": "string", "value": lid},
            {"name": "connections", "type": "string", "value": conns},
        ],
    }


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "ground.png").write_bytes(_sheet())
    return tmp_path


def test_a_wall_between_two_places_is_not_a_connection(root: Path) -> None:
    """The core case: an edge declared across solid ground.

    A and B sit either side of a wall with the only way round passing through C, so
    the graph must say A—C—B. Declaring A—B prices at one step a journey the map
    cannot make in one.
    """
    doc = _map(
        5,
        3,
        [
            "..#..",
            "..#..",
            ".....",  # the only way round is along the bottom, through C
        ],
        [_place("a", 0, 0), _place("b", 4, 0), _place("c", 2, 2)],
    )
    graph, pairs = derive_connections(doc, root)
    assert graph["a"] == ["c"]
    assert graph["b"] == ["c"]
    assert pairs[("a", "b")][1] == {"c"}  # …and it names what stands in the way


def test_a_through_street_survives_a_bystander(root: Path) -> None:
    """A place BESIDE the road does not break the road.

    Don't partition the map by nearest location: that makes the middle of a long
    street belong to whatever happens to sit next to it, severing the street. What
    matters is whether the route ENTERS the bystander, and here it does not.
    """
    doc = _map(
        5,
        3,
        ["....."] * 3,  # open ground; the shortest a→b run is along the top row
        [_place("a", 0, 0), _place("b", 4, 0), _place("c", 2, 2)],
    )
    graph, pairs = derive_connections(doc, root)
    assert "b" in graph["a"], "a bystander beside the road must not sever it"
    assert pairs[("a", "b")][1] == set(), "the route never enters c"
    # c being a neighbor of both is fine and true — it is reachable from either
    # without passing the other. What must not happen is a—b being severed by it.
    assert "c" in graph["a"] and "c" in graph["b"]


def test_a_place_is_an_area_not_its_center(root: Path) -> None:
    """Routes run edge-to-edge, because that is where people walk from.

    A 1×3 block measured center-to-center reads two cells longer than the walk anyone
    actually takes, and on a real ward (Chang'an's "东宫" is 3×11) the error is large
    enough to make a neighbor look like a trek.
    """
    doc = _map(5, 3, ["....."] * 3, [_place("a", 0, 0, 1, 3), _place("b", 4, 0, 1, 3)])
    graph, pairs = derive_connections(doc, root)
    assert graph["a"] == ["b"]
    assert pairs[("a", "b")][0] == 4, "edge-to-edge, not center-to-center"


def test_declared_and_derived_are_compared_on_the_same_shape(root: Path) -> None:
    doc = _map(
        5,
        3,
        ["..#..", "..#..", "....."],
        [_place("a", 0, 0, conns="b"), _place("b", 4, 0, conns="a"), _place("c", 2, 2)],
    )
    assert declared_connections(doc) == {"a": ["b"], "b": ["a"], "c": []}
    graph, _ = derive_connections(doc, root)
    assert graph["a"] == ["c"]


def test_apply_to_template_touches_only_the_connections_values(tmp_path: Path) -> None:
    """Written as a text edit so a Tiled file comes back the shape its author left it."""
    path = tmp_path / "map.tmj"
    original = (
        '{"layers":[{"objects":[{"properties":[\n'
        '  {"name":"connections","type":"string","value":"old"},\n'
        '  {"name":"location_id","type":"string","value":"a"}]}]}]}\n'
    )
    path.write_text(original, encoding="utf-8")
    assert apply_to_template(path, {"a": ["x", "y"]}) == 1
    written = path.read_text(encoding="utf-8")
    assert '"value":"x,y"' in written
    assert written.count("\n") == original.count("\n"), "layout preserved"
