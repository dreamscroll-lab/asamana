"""Resolve each character's narrative color name into a fixed identity hex.

The LLM (persona generation) authors a narrative color NAME per figure (e.g.
"朱红" / "玄色" / "靛蓝") — it works in narrative terms, never in hex, mirroring the
``parse_emotion_type`` label→value pattern. This module owns the ENCODING: it maps
each name to a reference color, then assigns every character a distinct swatch from
a curated, dark-map-legible palette.

Distinctness is a SET-level constraint (the whole cast must be mutually
distinguishable), so this runs ONCE over the full cast after all personas are
generated — an isolated per-figure call cannot satisfy it. Main characters get first
pick so they land closest to their authored color; the rest fill the remaining
palette to maximize spread. The result is frozen onto ``SoulLayer.color``.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

from core.logging import get_logger

logger = get_logger(__name__)

# Curated palette: (name, hex). Mutually distinct, mid/high saturation, legible on
# the dark (#0a0e1a) map. The NAMES are the LLM's vocabulary — the persona prompt
# lists exactly these and the LLM picks one (IndexedRef-style constrained choice, far
# more stable than free-text).
#
# NO NEAR-NEUTRAL ENTRY MAY BE ADDED HERE, however well an achromatic swatch would
# serve a somber character. Gray is the map's mark for a body with no mind, and the
# mark only reads because nobody in the cast can wear it — a cast member dealt a slate
# becomes indistinguishable from every mindless body on screen, which is the one
# distinction the map cannot afford to lose. An achromatic authored color lands on
# the nearest saturated swatch instead (``_NAME_REF`` still maps the gray words, as
# reference points rather than as choices). Guarded by a test.
_PALETTE_ENTRIES: tuple[tuple[str, str], ...] = (
    ("rose", "#e0607a"),
    ("crimson", "#c05a5a"),
    ("coral", "#e8785e"),
    ("vermilion", "#e07a4a"),
    ("apricot", "#e0a35a"),
    ("gold", "#d9a441"),
    ("olive", "#9aa84a"),
    ("chartreuse", "#cdd24a"),
    ("green", "#6fae7a"),
    ("mint", "#4fc79a"),
    ("teal", "#3fb98f"),
    ("cyan", "#45c0cd"),
    ("sky", "#59b0e6"),
    ("cobalt", "#5a7be0"),
    ("indigo", "#7c83ff"),
    ("periwinkle", "#9aa0f0"),
    ("violet", "#a06fe0"),
    ("lavender", "#b79ae8"),
    ("plum", "#b866c0"),
    ("magenta", "#d98cf0"),
    ("pink", "#e06fb0"),
    ("tan", "#b0895a"),
    ("steel", "#5f8fb0"),
)
_PALETTE: tuple[str, ...] = tuple(hex_ for _, hex_ in _PALETTE_ENTRIES)

# The one color worn by every mindless body, declared here because it is the palette's
# complement: "no cast member is gray" and "the mindless are gray" are one rule, and split
# across modules nobody can test it. One value on purpose: they take no part in the spread.
MINDLESS_BODY_COLOR = "#8a93a6"
_PALETTE_BY_NAME: dict[str, str] = {name: hex_ for name, hex_ in _PALETTE_ENTRIES}

# The exact option list injected into the persona prompt.
PALETTE_CHOICES: str = "、".join(name for name, _ in _PALETTE_ENTRIES)

# Narrative color word → a reference hex. The LLM emits an English color word;
# matched case-insensitively by substring (e.g. "deep crimson" matches "crimson"),
# longest word wins. Unknown → no preference (spread-assigned instead).
_NAME_REF: dict[str, str] = {
    "rose": "#e0607a", "pink": "#e06fb0", "magenta": "#d98cf0", "fuchsia": "#e06fb0",
    "red": "#c05a5a", "crimson": "#e0607a", "scarlet": "#e0607a", "cinnabar": "#e0607a",
    "vermilion": "#e07a4a", "orange": "#e07a4a", "coral": "#e0607a",
    "amber": "#d9a441", "gold": "#d9a441", "golden": "#d9a441", "yellow": "#d9a441",
    "ochre": "#b0895a", "tan": "#b0895a", "brown": "#b0895a", "sienna": "#b0895a",
    "bronze": "#b0895a", "chestnut": "#b0895a", "khaki": "#cdd24a",
    "chartreuse": "#cdd24a", "lime": "#cdd24a", "olive": "#cdd24a",
    "green": "#6fae7a", "emerald": "#6fae7a", "jade": "#6fae7a", "forest": "#6fae7a",
    "teal": "#3fb98f", "cyan": "#3fb98f", "turquoise": "#3fb98f", "aqua": "#3fb98f",
    "sky": "#59b0e6", "azure": "#59b0e6", "cerulean": "#59b0e6", "blue": "#59b0e6",
    "cobalt": "#7c83ff", "navy": "#7c83ff", "indigo": "#7c83ff", "sapphire": "#7c83ff",
    "violet": "#a06fe0", "purple": "#a06fe0", "amethyst": "#a06fe0", "lavender": "#a06fe0",
    "plum": "#d98cf0", "lilac": "#a06fe0",
    "steel": "#5f8fb0", "slate": "#8a94a8", "gray": "#8a94a8", "grey": "#8a94a8",
    "silver": "#8a94a8", "ash": "#8a94a8", "pale": "#8a94a8", "white": "#8a94a8",
    "black": "#6a7690", "ink": "#6a7690", "onyx": "#6a7690", "obsidian": "#6a7690",
    "charcoal": "#6a7690", "dark": "#6a7690",
}


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _dist(a: str, b: str) -> float:
    (r1, g1, b1), (r2, g2, b2) = _rgb(a), _rgb(b)
    return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5


def _reference_for(name: str) -> str | None:
    """The palette-space reference hex for an LLM color choice, or None if unknown.

    The LLM is asked to pick a palette NAME, so an exact palette match is the common
    path; the synonym table is a safety net for when it emits a near-word anyway.
    """
    if not name:
        return None
    lower = name.strip().lower()
    if lower in _PALETTE_BY_NAME:
        return _PALETTE_BY_NAME[lower]
    best_word = ""
    for word in _NAME_REF:
        if word in lower and len(word) > len(best_word):
            best_word = word
    return _NAME_REF.get(best_word) if best_word else None


def resolve_identity_colors(definitions: Iterable["object"]) -> list["object"]:
    """Assign each AgentDefinition a distinct identity hex on ``soul.color``.

    Reads the transient ``metadata['signature_color']`` name, resolves it against the
    palette, and returns definitions with ``soul`` replaced to carry the final hex.
    Main characters are served first (closest to their authored color); the rest are
    packed to maximize mutual spread. Never assigns the same swatch twice while the
    palette has room.
    """
    defs = list(definitions)
    if not defs:
        return defs

    # Mains first so they land nearest their authored color; stable within a tier.
    def _is_main(d: "object") -> bool:
        tier = getattr(d, "tier", None)
        return getattr(tier, "name", "") == "MAIN"

    order = sorted(range(len(defs)), key=lambda i: (not _is_main(defs[i]), i))

    used: set[str] = set()
    assigned: dict[int, str] = {}
    deferred: list[int] = []

    # Pass 1 — honor authored preferences: nearest UNUSED palette swatch.
    for i in order:
        ref = _reference_for(str(getattr(defs[i], "metadata", {}).get("signature_color", "")))
        if ref is None:
            deferred.append(i)
            continue
        free = [c for c in _PALETTE if c not in used]
        if not free:
            deferred.append(i)
            continue
        pick = min(free, key=lambda c: _dist(c, ref))
        used.add(pick)
        assigned[i] = pick

    # Pass 2 — no preference (or palette exhausted): maximize spread from assigned.
    for i in deferred:
        free = [c for c in _PALETTE if c not in used]
        if not free:
            # Palette smaller than the cast: reuse the swatch most distant from
            # this agent's neighbors to keep collisions maximally far apart.
            free = list(_PALETTE)
            pick = max(free, key=lambda c: min((_dist(c, a) for a in assigned.values()), default=0.0))
        elif assigned:
            pick = max(free, key=lambda c: min(_dist(c, a) for a in assigned.values()))
        else:
            pick = free[0]
        used.add(pick)
        assigned[i] = pick

    out: list["object"] = []
    for i, d in enumerate(defs):
        color = assigned.get(i, _PALETTE[0])
        soul = dataclasses.replace(d.soul, color=color)
        out.append(dataclasses.replace(d, soul=soul))
    logger.info(
        "identity_colors_assigned",
        extra={"count": len(out), "distinct": len(set(assigned.values()))},
    )
    return out
