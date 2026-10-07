"""Unit tests for cast-level identity-colour resolution."""

from __future__ import annotations

import dataclasses

from world.identity_color import (
    _PALETTE,
    _PALETTE_BY_NAME,
    MINDLESS_BODY_COLOR,
    resolve_identity_colors,
)


@dataclasses.dataclass(frozen=True)
class _Soul:
    color: str = ""


@dataclasses.dataclass(frozen=True)
class _Tier:
    name: str


@dataclasses.dataclass(frozen=True)
class _Def:
    soul: _Soul
    tier: _Tier
    metadata: dict


def _mk(color_name: str, *, main: bool = False) -> _Def:
    return _Def(_Soul(), _Tier("MAIN" if main else "BACKGROUND"), {"signature_color": color_name})


def test_assigns_distinct_colors_within_palette():
    defs = [_mk(n) for n in ["gold", "teal", "crimson", "violet", "green"]]
    out = resolve_identity_colors(defs)
    colors = [d.soul.color for d in out]
    assert all(c in _PALETTE for c in colors)
    assert len(set(colors)) == len(colors)  # all distinct


def test_honours_authored_preference():
    out = resolve_identity_colors([_mk("gold")])
    assert out[0].soul.color == _PALETTE_BY_NAME["gold"]


def test_main_character_wins_a_contested_colour():
    # Both want indigo; the main character should land on it, the other elsewhere.
    defs = [_mk("indigo"), _mk("indigo", main=True)]
    out = resolve_identity_colors(defs)
    main = next(d for d in out if d.tier.name == "MAIN")
    other = next(d for d in out if d.tier.name != "MAIN")
    assert main.soul.color == _PALETTE_BY_NAME["indigo"]
    assert other.soul.color != main.soul.color


def test_unknown_name_still_gets_a_distinct_colour():
    defs = [_mk("gold"), _mk("not-a-colour"), _mk("")]
    out = resolve_identity_colors(defs)
    colors = [d.soul.color for d in out]
    assert all(c in _PALETTE for c in colors)
    assert len(set(colors)) == 3


def test_cast_larger_than_palette_does_not_crash():
    defs = [_mk("gold") for _ in range(len(_PALETTE) + 4)]
    out = resolve_identity_colors(defs)
    assert len(out) == len(defs)
    assert all(d.soul.color in _PALETTE for d in out)


def test_empty_cast():
    assert resolve_identity_colors([]) == []


def _saturation(hex_color: str) -> float:
    raw = hex_color.lstrip("#")
    channels = [int(raw[i : i + 2], 16) for i in (0, 2, 4)]
    top = max(channels)
    return 0.0 if top == 0 else (top - min(channels)) / top


def _distance(a: str, b: str) -> float:
    lhs = [int(a.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
    rhs = [int(b.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
    return sum((x - y) ** 2 for x, y in zip(lhs, rhs)) ** 0.5


def test_no_cast_colour_can_pass_for_a_mindless_body():
    """The marker only works if no cast member can be given it; it is the map's only basis for
    "grey means no cognition"."""
    assert _saturation(MINDLESS_BODY_COLOR) < 0.25  # it is grey, not a hue
    for color in _PALETTE:
        assert _saturation(color) > 0.30, f"{color} is grey enough to pass for a mindless body"
        assert _distance(color, MINDLESS_BODY_COLOR) > 25


def test_an_achromatic_choice_lands_on_a_saturated_swatch():
    out = resolve_identity_colors([_mk("slate"), _mk("charcoal")])
    for definition in out:
        assert definition.soul.color in _PALETTE
        assert _distance(definition.soul.color, MINDLESS_BODY_COLOR) > 25
