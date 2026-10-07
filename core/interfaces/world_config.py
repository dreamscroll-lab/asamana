"""World configuration contract, owned by core so ``core/`` never imports ``worlds/``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from core.interfaces.place import Place


class WorldConfig(ABC):
    """A narrative-agnostic spatial substrate (locations, connections, buildings) that can host
    any theme. No character names, era references or theme-specific objects: ThemeAnalyzer
    produces those.

    Each build creates a read-only build-time template (``NarrativeApplication._world_config_for``):
    ``WorldInitializer`` deep-copies it per world and persists the copy (``world/stored_config.py``),
    so implementations must be ``deepcopy``-able and the four data facets (places, aliases,
    description, runtime context) JSON-serializable. Never mutate the template at runtime.
    """

    @abstractmethod
    def get_places(self) -> dict[str, "Place"]:
        """Return the world's space — every place keyed by id.

        Places only. Things come from ``ThemeAnalysis.world_entity_seeds`` (landed via
        ``register_entity``) and ``spawn_entity``; the two paths never cross.
        """

    def get_location_aliases(self) -> dict[str, str]:
        """Return alternate location labels keyed to canonical location ids."""
        return {}

    def resolve_location_id(self, raw: str) -> str | None:
        """Resolve user/LLM-facing location text to a canonical location id."""
        candidate = str(raw).strip()
        if not candidate:
            return None

        places = self.get_places()

        if candidate in places:
            return candidate

        aliases = self.get_location_aliases()
        aliased = aliases.get(candidate) or aliases.get(candidate.lower())
        if aliased and aliased in places:
            return aliased

        lowered = candidate.lower()
        for place_id, place in places.items():
            if str(place_id).lower() == lowered:
                return str(place_id)
            place_name = str(getattr(place, "name", "")).strip()
            if place_name == candidate or place_name.lower() == lowered:
                return str(place_id)

        # An LLM may extend a known name with a sub-area suffix. Match only when exactly one
        # location's name or id is a substring; ambiguity yields None, not a guess.
        substring_matches: set[str] = set()
        for place_id, place in places.items():
            place_name = str(getattr(place, "name", "")).strip()
            for token in (place_name, str(place_id)):
                if len(token) >= 2 and token in candidate:
                    substring_matches.add(str(place_id))
        if len(substring_matches) == 1:
            return next(iter(substring_matches))

        return None

    @abstractmethod
    def get_world_description(self) -> str:
        """Return a description of the world as a spatial substrate.

        Must describe the setting's permanent characteristics (geography,
        political space structure, architectural features). Must NOT reference
        specific characters, era names, or narrative events — those vary per theme.
        """

    def render_map(self) -> dict[str, object] | None:
        """Return the raw render-map document (e.g. a Tiled ``.tmj``) for the 2D
        renderer, or None if this config carries no map art.

        Purely the renderer's geometry, not simulated on. ``WorldInitializer`` freezes it per
        world so later template edits never shift a built world's map.
        """
        return None

    def render_characters(self) -> dict[str, object] | None:
        """Return the cast's art manifest for the 2D renderer, or None if this
        config carries none.

        Figures are dressed for the map's period, so they live beside the map in the template
        and are frozen with it. Art shipped with the renderer would restyle every past
        world's replay and make a new period cost a frontend change.

        The manifest maps each (gender, age bracket) to a body and names its poses, so the
        renderer compiles in no frame names or age split. Paths are relative to
        ``render_assets_dir``.
        """
        return None

    def render_assets_dir(self) -> Path | None:
        """Directory the render map's image paths are relative to, or None.

        ``WorldInitializer`` freezes these pixels together with ``render_map`` /
        ``render_characters``; freezing one without the other is a half-freeze.
        """
        return None

    @abstractmethod
    def to_runtime_context(self) -> dict[str, object]:
        """Return a runtime-friendly representation.

        era_name should be a neutral period label; ThemeAnalyzer sets the specific era. No
        year: that is the theme's to decide (the same city hosts 618 and 907), so
        ``start_year`` comes from ThemeAnalysis alone.
        """
