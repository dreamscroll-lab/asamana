"""A world's setup, read and revised until the world is confirmed.

Reads and writes the setup (``world.setup``) field for field; the rules it must keep are the
setup's own (``WorldSetup.validate``). A character's name is given when it is added and is not
revisable.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from agent.need import NeedState, NeedType
from config.models import MAX_CAST_SIZE
from core.prompts import canonical_relation_label
from agent.personality import EmotionState, EmotionType
from world.setup import (
    CharacterRevision, EntityRevision, MapPlaces, NpcRevision, SetupHistoricalEvent, SetupRejected, SetupRelation,
    WorldSetup,
)

from interaction.api.app import ApiServices
from interaction.api.serialization import to_wire_json
from interaction.models import (
    CharacterSetupView,
    EntitySetupView,
    NpcSetupView,
    InitialEmotionView,
    SetupHistoricalEventView,
    InnateNeedView,
    SetupRelationView,
    SetupView,
    agent_profile,
)

_CHARACTER_TEXT_FIELDS = (
    "role", "gender", "appearance", "background", "self_image", "life_goal", "secret", "initial_location",
)
_CHARACTER_LIST_FIELDS = ("core_traits", "core_values", "hard_constraints", "long_term_goals")
_CHARACTER_FIELDS = frozenset({
    *_CHARACTER_TEXT_FIELDS, *_CHARACTER_LIST_FIELDS, "age", "is_main_character", "initial_emotion", "needs",
})


def setup_view(setup: WorldSetup) -> SetupView:
    return SetupView(
        seconds_per_step=setup.lore.seconds_per_step,
        entities=[EntitySetupView(**asdict(seed)) for seed in setup.entity_seeds],
        npcs=[NpcSetupView(**asdict(seed)) for seed in setup.npc_seeds],
        characters=[
            CharacterSetupView(
                **asdict(agent_profile(d)),
                initial_location=d.initial_location,
                initial_emotion=InitialEmotionView(
                    emotion=d.initial_emotion.primary.value,
                    intensity=d.initial_emotion.intensity,
                    valence=d.initial_emotion.valence,
                ),
                long_term_goals=list(d.long_term_goals),
                hard_constraints=list(d.soul.hard_constraints),
                needs=[
                    InnateNeedView(
                        type=n.type.value, label=n.label, intensity=n.intensity, weight=n.weight,
                        is_hidden=n.is_hidden,
                    )
                    for n in d.soul.innate_needs
                ],
            )
            for d in setup.characters
        ],
        historical_events=[
            SetupHistoricalEventView(
                event=e.event,
                hours_before_start=e.hours_before_start,
                figure_ids=list(e.figure_ids),
                importance=e.importance,
            )
            for e in setup.historical_events
        ],
        relations=[
            SetupRelationView(
                source_id=r.source_id,
                target_id=r.target_id,
                trust=r.trust,
                affection=r.affection,
                labels=list(r.labels),
                reverse_trust=r.reverse_trust,
                reverse_affection=r.reverse_affection,
                reverse_labels=list(r.reverse_labels),
            )
            for r in setup.relations
        ],
    )


def character_revision(payload: dict[str, Any]) -> CharacterRevision:
    """Read a character revision off a request body; ``ValueError`` on a field it can't take."""
    if "name" in payload:
        raise ValueError("A character's name is not revisable")
    unknown = sorted(set(payload) - _CHARACTER_FIELDS)
    if unknown:
        raise ValueError(f"Unknown character fields: {unknown}")
    changes: dict[str, Any] = {}
    for key in _CHARACTER_TEXT_FIELDS:
        if key in payload:
            changes[key] = _text(payload[key], key)
    for key in _CHARACTER_LIST_FIELDS:
        if key in payload:
            changes[key] = tuple(_texts(payload[key], key))
    if "age" in payload:
        changes["age"] = _integer(payload["age"], "age")
    if "is_main_character" in payload:
        changes["is_main_character"] = _boolean(payload["is_main_character"], "is_main_character")
    if "initial_emotion" in payload:
        emotion = payload["initial_emotion"]
        changes["initial_emotion"] = EmotionState(
            primary=EmotionType(_text(_field(emotion, "emotion"), "emotion")),
            intensity=_number(_field(emotion, "intensity"), "intensity"),
            valence=_number(_field(emotion, "valence"), "valence"),
        )
    if "needs" in payload:
        changes["needs"] = tuple(
            NeedState(
                type=NeedType(_text(_field(need, "type"), "type")),
                label=_text(_field(need, "label"), "label"),
                intensity=_number(_field(need, "intensity"), "intensity"),
                weight=_number(_field(need, "weight"), "weight"),
                is_hidden=_boolean(_field(need, "is_hidden"), "is_hidden"),
            )
            for need in _list(payload, "needs")
        )
    return CharacterRevision(**changes)


def entity_revision(payload: dict[str, Any]) -> EntityRevision:
    allowed = {"entity_type", "description", "initial_state", "location", "content"}
    if unknown := sorted(set(payload) - allowed):
        raise ValueError(f"Unknown entity fields: {unknown}")
    return EntityRevision(**{key: _text(value, key) for key, value in payload.items()})


def npc_revision(payload: dict[str, Any]) -> NpcRevision:
    allowed = {"gender", "age", "description", "location"}
    if unknown := sorted(set(payload) - allowed):
        raise ValueError(f"Unknown NPC fields: {unknown}")
    return NpcRevision(**{
        key: _integer(value, key) if key == "age" else _text(value, key)
        for key, value in payload.items()
    })


def historical_events(payload: Any) -> list[SetupHistoricalEvent]:
    return [
        SetupHistoricalEvent(
            event=_text(_field(item, "event"), "event"),
            hours_before_start=_integer(_field(item, "hours_before_start"), "hours_before_start"),
            figure_ids=tuple(_texts(_field(item, "figure_ids"), "figure_ids")),
            importance=_number(_field(item, "importance"), "importance"),
        )
        for item in _list(payload, "historical_events")
    ]


def relations(payload: Any) -> list[SetupRelation]:
    return [
        SetupRelation(
            source_id=_text(_field(item, "source_id"), "source_id"),
            target_id=_text(_field(item, "target_id"), "target_id"),
            trust=_number(_field(item, "trust"), "trust"),
            affection=_number(_field(item, "affection"), "affection"),
            labels=tuple(canonical_relation_label(t) for t in _texts(_field(item, "labels"), "labels")),
            reverse_trust=_number(_field(item, "reverse_trust"), "reverse_trust"),
            reverse_affection=_number(_field(item, "reverse_affection"), "reverse_affection"),
            reverse_labels=tuple(
                canonical_relation_label(t) for t in _texts(_field(item, "reverse_labels"), "reverse_labels")
            ),
        )
        for item in _list(payload, "relations")
    ]


def build_setup_router(services: ApiServices) -> Any:
    from fastapi import APIRouter, Body, HTTPException

    router = APIRouter(prefix="/api/worlds/{world_id}/setup")
    application = services.application
    manager = services.manager

    async def _readable_setup(world_id: str) -> WorldSetup:
        if await manager.get_world(world_id) is None:
            raise HTTPException(status_code=404, detail=f"World not found: {world_id}")
        # The setup holds every character's secrets and every relation: while an actor performs,
        # the world is seen through one character, who knows none of that.
        if application.performing_character(world_id) is not None:
            raise HTTPException(status_code=403, detail="The setup is closed while an actor performs")
        setup = await manager.world_setup(world_id)
        if setup is None:
            raise HTTPException(status_code=404, detail=f"World {world_id!r} has no setup.")
        return setup

    async def _revise(world_id: str, revise: Any) -> WorldSetup:
        await _readable_setup(world_id)
        try:
            setup = await application.revise_setup(world_id, revise)
        except SetupRejected as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return setup

    def _parsed(parse: Any, payload: Any) -> Any:
        try:
            return parse(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("")
    async def get_setup(world_id: str) -> dict[str, Any]:
        return to_wire_json(setup_view(await _readable_setup(world_id)))

    @router.post("/characters")
    async def add_character(world_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        """Add a character written by hand. Answers with the new character's id and the setup."""
        name = _parsed(lambda p: _text(_field(p, "name"), "name"), payload)
        revision = _parsed(character_revision, {k: v for k, v in payload.items() if k != "name"})
        added: list[str] = []

        def add(s: WorldSetup, places: MapPlaces) -> WorldSetup:
            # Not ``world.max_agents``: that bounds what a build generates, for the build's own
            # output budget, and a character written by hand costs the build nothing.
            setup, agent_id = s.with_character_added(
                name, revision, places=places, cast_limit=MAX_CAST_SIZE,
            )
            added.append(agent_id)
            return setup

        setup = await _revise(world_id, add)
        return {"agent_id": added[0], "setup": to_wire_json(setup_view(setup))}

    @router.patch("/characters/{agent_id}")
    async def revise_character(
        world_id: str, agent_id: str, payload: dict = Body(...),
    ) -> dict[str, Any]:
        revision = _parsed(character_revision, payload)
        setup = await _revise(
            world_id, lambda s, places: s.with_character(agent_id, revision, places=places),
        )
        return to_wire_json(setup_view(setup))

    @router.delete("/characters/{agent_id}")
    async def remove_character(world_id: str, agent_id: str) -> dict[str, Any]:
        setup = await _revise(world_id, lambda s, places: s.without_character(agent_id, places=places))
        return to_wire_json(setup_view(setup))

    @router.post("/entities")
    async def add_entity(world_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        """Add a thing written by hand. Answers with its id and the setup."""
        name = _parsed(lambda p: _text(_field(p, "name"), "name"), payload)
        revision = _parsed(entity_revision, {k: v for k, v in payload.items() if k != "name"})
        added: list[str] = []

        def add(s: WorldSetup, places: MapPlaces) -> WorldSetup:
            setup, entity_id = s.with_entity_added(name, revision, places=places)
            added.append(entity_id)
            return setup

        setup = await _revise(world_id, add)
        return {"entity_id": added[0], "setup": to_wire_json(setup_view(setup))}

    @router.delete("/entities/{entity_id}")
    async def remove_entity(world_id: str, entity_id: str) -> dict[str, Any]:
        setup = await _revise(world_id, lambda s, places: s.without_entity(entity_id, places=places))
        return to_wire_json(setup_view(setup))

    @router.post("/npcs")
    async def add_npc(world_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        """Add an NPC written by hand. Answers with its id and the setup."""
        name = _parsed(lambda p: _text(_field(p, "name"), "name"), payload)
        revision = _parsed(npc_revision, {k: v for k, v in payload.items() if k != "name"})
        added: list[str] = []

        def add(s: WorldSetup, places: MapPlaces) -> WorldSetup:
            setup, npc_id = s.with_npc_added(name, revision, places=places, npc_limit=services.config.engine.max_npcs)
            added.append(npc_id)
            return setup

        setup = await _revise(world_id, add)
        return {"npc_id": added[0], "setup": to_wire_json(setup_view(setup))}

    @router.delete("/npcs/{npc_id}")
    async def remove_npc(world_id: str, npc_id: str) -> dict[str, Any]:
        setup = await _revise(world_id, lambda s, places: s.without_npc(npc_id, places=places))
        return to_wire_json(setup_view(setup))

    @router.patch("/entities/{entity_id}")
    async def revise_entity(world_id: str, entity_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        revision = _parsed(entity_revision, payload)
        setup = await _revise(world_id, lambda s, places: s.with_entity(entity_id, revision, places=places))
        return to_wire_json(setup_view(setup))

    @router.patch("/npcs/{npc_id}")
    async def revise_npc(world_id: str, npc_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        revision = _parsed(npc_revision, payload)
        setup = await _revise(world_id, lambda s, places: s.with_npc(npc_id, revision, places=places))
        return to_wire_json(setup_view(setup))

    @router.put("/history")
    async def replace_history(world_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        events = _parsed(historical_events, payload)
        setup = await _revise(world_id, lambda s, places: s.with_history(events, places=places))
        return to_wire_json(setup_view(setup))

    @router.put("/seconds-per-step")
    async def revise_seconds_per_step(world_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        seconds = _parsed(lambda p: _integer(_field(p, "seconds_per_step"), "seconds_per_step"), payload)
        setup = await _revise(
            world_id, lambda s, places: s.with_seconds_per_step(seconds, places=places),
        )
        return to_wire_json(setup_view(setup))

    @router.put("/relations")
    async def replace_relations(world_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        seeds = _parsed(relations, payload)
        setup = await _revise(world_id, lambda s, places: s.with_relations(seeds, places=places))
        return to_wire_json(setup_view(setup))

    return router


def _field(item: Any, key: str) -> Any:
    if not isinstance(item, dict) or key not in item:
        raise ValueError(f"{key} is required")
    return item[key]


def _list(payload: Any, key: str) -> list[Any]:
    value = _field(payload, key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value.strip()


def _texts(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of strings")
    return [_text(item, name) for item in value]


def _integer(value: Any, name: str) -> int:
    # bool subclasses int: ``true`` is not an age.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)
