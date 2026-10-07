"""Phenomenon — classification of the visible phenomena accompanying a world change.

Motivation
==========

A ``Broadcast`` says what happened; its ``severity`` says how big it is. Neither says what it
looks like: "fire in the East Market" and "grim news from the East Market" can share a severity
and a channel, yet only one should show flames.

``phenomenon`` tags the visible natural phenomenon accompanying a change. Render-neutral, like
``engine.executors.base.action_semantics``: facts ("it is raining here"), never render
instructions. A closed enum, not free text, which would force the renderer into keyword matching
on theme words (CLAUDE.md Rule 7).

It is a classification, not a scale: no ``level``, no ordering, and no ``*_SCALE_DESCRIPTION``.
Intensity is carried by ``severity`` on the same Broadcast; don't build a second ruler here.

Sited vs ambient
================

- Sited (fire/smoke/quake): always happens somewhere specific. "Fire everywhere" doesn't mean
  the whole map is burning; it means nobody said where.
- Ambient (rain/snow/wind/dark): covers the whole sky, so ``location_scope=None`` is natural.

Two sides hold this line, in opposite directions:

- The prompt makes it well-formed: both authors (``EventSystem`` / ``DirectorChannel``) share
  ``core.prompts.PHENOMENON_DEFINITION``, which says fire needs a location.
- The code drops malformed ones: ``parse_broadcast_spec`` (``engine/injection.py``) turns sited +
  no location into ``NONE`` and logs a warning.

Filter only at parse time. Don't add a check in ``Broadcast.__post_init__``: every construction
path is already filtered or passes no phenomenon, so it would only be a second source of truth.

Subclasses ``str`` → JSON/snapshots serialize directly as "rain" etc.
"""

from __future__ import annotations

from enum import Enum

from core.coerce import coerce_enum


class Phenomenon(str, Enum):
    """Visible phenomena accompanying a world change; a closed vocabulary (unordered).

    ``NONE`` is an explicit "nothing visible", not a default placeholder: giving the LLM a clear
    "nothing to see here" option beats forcing it to pick one of the phenomena.
    """

    NONE  = "none"
    RAIN  = "rain"
    SNOW  = "snow"
    WIND  = "wind"
    FIRE  = "fire"
    SMOKE = "smoke"
    QUAKE = "quake"
    DARK  = "dark"

    @classmethod
    def prompt_choices(cls) -> str:
        """Prompt-side choices: pipe-separated ("none|rain|snow|..."), generated from the enum.
        ``none`` leads to show it's a valid, common answer."""
        return "|".join(p.value for p in cls)

    @classmethod
    def sited_choices(cls) -> str:
        """Phenomena that must have a location ("fire|smoke|quake"). Generated from ``_SITED``,
        never hand-written: a hand-written list would silently drift from what the code filters."""
        return "|".join(p.value for p in cls if p.is_sited)

    @classmethod
    def ambient_choices(cls) -> str:
        """Phenomena that can cover the whole world ("rain|snow|wind|dark"). Excludes ``none`` (not a phenomenon)."""
        return "|".join(p.value for p in cls if not p.is_sited and p is not cls.NONE)

    @property
    def is_sited(self) -> bool:
        """Whether this phenomenon must have a location (see "Sited vs ambient" above)."""
        return self in _SITED


#: Sited phenomena: always originate somewhere; meaningless without a location.
_SITED: frozenset[Phenomenon] = frozenset(
    {Phenomenon.FIRE, Phenomenon.SMOKE, Phenomenon.QUAKE}
)


def parse_phenomenon(raw: object, default: Phenomenon = Phenomenon.NONE) -> Phenomenon:
    """Lenient parse: any input → Phenomenon; unrecognized input returns ``default``.

    Every LLM output value goes through this. Fall back to ``NONE`` instead of guessing a close
    phenomenon: the renderer drawing nothing always beats drawing weather that never happened.
    """
    return coerce_enum(Phenomenon, raw, default=default)
