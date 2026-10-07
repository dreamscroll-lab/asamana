"""Per-world persisted world configuration.

The template ``NarrativeApplication._world_config_for`` creates is read-only and used only at
build time. ``WorldInitializer.initialize`` serializes the template's data into the
world's own persisted asset (via ``SnapshotProvider.save_world_config``);
``restore`` rebuilds the config from that asset alone. Template code edits
after a world is built therefore never change an existing world's map.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from core.interfaces.world_config import WorldConfig
from core.interfaces.place import Place


def serialize_world_config(config: WorldConfig) -> dict[str, Any]:
    """Capture a WorldConfig's four data facets as a JSON-serializable dict.

    This asset must be a clean base: step-0 occupancy baked in here would come back as ghost
    occupants on restore. Nothing needs guarding against that here: ``Place`` has no occupancy
    field at all, and occupancy lives only in the environment, which doesn't go through this
    serialization. Entities don't either: they come from seeds and runtime spawns and belong to the
    snapshot's ``entity_states``.
    """

    places: dict[str, dict[str, Any]] = {}
    for place_id, place in config.get_places().items():
        places[str(place_id)] = dataclasses.asdict(place)
    return {
        "places": places,
        "location_aliases": dict(config.get_location_aliases()),
        "world_description": config.get_world_description(),
        "runtime_context": dict(config.to_runtime_context()),
    }


class StoredWorldConfig(WorldConfig):
    """WorldConfig rebuilt from a world's persisted config asset.

    Covers the full contract from data alone: ``resolve_location_id`` is the
    generic base-class implementation over places + aliases. Known boundary:
    a template that overrides ``resolve_location_id`` falls back to the generic
    implementation after a persist/restore round-trip.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        places = data.get("places")
        if not isinstance(places, dict) or not places:
            raise ValueError("Stored world config has no places.")
        self._places: dict[str, Place] = {
            str(place_id): Place.from_dict(payload)
            for place_id, payload in places.items()
        }
        self._location_aliases: dict[str, str] = {
            str(k): str(v) for k, v in dict(data.get("location_aliases", {})).items()
        }
        self._world_description: str = str(data.get("world_description", ""))
        self._runtime_context: dict[str, Any] = dict(data.get("runtime_context", {}))

    def get_places(self) -> dict[str, Place]:
        return self._places

    def get_location_aliases(self) -> dict[str, str]:
        return dict(self._location_aliases)

    def get_world_description(self) -> str:
        return self._world_description

    def to_runtime_context(self) -> dict[str, Any]:
        return dict(self._runtime_context)
