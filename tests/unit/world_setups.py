"""A minimal world setup for tests that only need the cast's identities."""

from __future__ import annotations

from typing import Any

from agent.need import build_innate_needs
from agent.personality import EmotionState, EmotionType, SoulLayer
from world.models import AgentDefinition, AgentTier, NpcSeed, unique_npc_id
from world.setup import WorldLore, WorldSetup


def setup_of(profiles: dict[str, dict[str, Any]]) -> WorldSetup:
    """``profiles``: agent_id → soul fields, plus ``is_main_character`` for the tier. Everyone
    starts in ``palace``."""
    characters = []
    for agent_id, profile in profiles.items():
        fields = dict(profile)
        main = bool(fields.pop("is_main_character", False))
        soul = SoulLayer(**{
            "name": "某人", **fields, "agent_id": agent_id, "innate_needs": build_innate_needs([], []),
        })
        characters.append(AgentDefinition(
            agent_id=agent_id,
            tier=AgentTier.MAIN if main else AgentTier.BACKGROUND,
            soul=soul,
            initial_location="palace",
            initial_emotion=EmotionState(primary=EmotionType.NEUTRAL),
        ))
    lore = WorldLore(
        world_name="测试", era_description="", core_tension="", narrative_theme="",
        narrative_pitch="", world_time_config={"seconds_per_step": 3600},
    )
    return WorldSetup(theme="测试", lore=lore, characters=tuple(characters), relations=(), historical_events=())


def setup_dict(profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The same setup as a build persists it."""
    return setup_of(profiles).as_dict()


def npc_seed(name: str, **fields: Any) -> NpcSeed:
    """An NPC seed with the id its setup would give it, for tests that spawn one directly."""
    return NpcSeed(name=name, npc_id=unique_npc_id(name, set()), **fields)
