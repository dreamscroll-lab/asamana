"""Live world directory: id → identity info backed by souls + environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Mapping

from agent.personality import SoulLayer
from core.interfaces.directory import DirectoryEntry, WorldDirectory
from core.interfaces.perception import NpcIdentity, PerceivedIdentity
from core.logging import get_logger

if TYPE_CHECKING:
    from agent.agent import Agent
    from engine.environment import EnvironmentSystem

logger = get_logger(__name__)


class LiveWorldDirectory(WorldDirectory):
    """Per-world identity directory assembled from agents and environment.

    Agent identities are copied at construction (SoulLayer is frozen and the
    agent roster never grows mid-run). Locations and items are resolved by
    live delegation to EnvironmentSystem, so entities registered or restored
    mid-run are visible without invalidation.
    """

    def __init__(
        self,
        *,
        souls: Mapping[str, SoulLayer],
        environment: "EnvironmentSystem",
    ) -> None:
        self._souls: dict[str, SoulLayer] = dict(souls)
        self._environment = environment

    @classmethod
    def from_agents(
        cls,
        agents: Mapping[str, "Agent"],
        environment: "EnvironmentSystem",
    ) -> "LiveWorldDirectory":
        return cls(
            souls={agent_id: agent.personality.soul for agent_id, agent in agents.items()},
            environment=environment,
        )

    def agent_name(self, agent_id: str) -> str:
        soul = self._souls.get(agent_id)
        if soul is None or not soul.name:
            logger.debug("directory_miss", extra={"lookup_id": agent_id, "kind": "agent"})
            return "某人"
        return soul.name

    def agent_identity_map(self, agent_ids: Iterable[str]) -> dict[str, PerceivedIdentity]:
        result: dict[str, PerceivedIdentity] = {}
        for agent_id in agent_ids:
            soul = self._souls.get(agent_id)
            if soul is not None and soul.name:
                result[agent_id] = PerceivedIdentity(name=soul.name, gender=soul.gender)
        return result

    def npc_identity_map(self, npc_ids: Iterable[str]) -> dict[str, NpcIdentity]:
        """Live delegation, unlike the souls copied at construction: an Npc roster may
        grow after this directory was built, and a body nobody can name renders as 「某人」
        in every prompt it appears in."""
        result: dict[str, NpcIdentity] = {}
        for npc_id in npc_ids:
            npc = self._environment.get_npc(npc_id)
            if npc is not None:
                result[npc_id] = NpcIdentity(
                    name=npc.name, gender=npc.gender, age=npc.age, description=npc.description,
                )
        return result

    def all_agent_names(self) -> dict[str, str]:
        return {
            agent_id: soul.name
            for agent_id, soul in self._souls.items()
            if soul.name
        }

    def location_name(self, location_id: str) -> str:
        return self._environment.narrative_location_name(location_id)

    def entity_name(self, entity_id: str) -> str:
        entity = self._environment.get_entity(entity_id)
        if entity is None or not entity.name:
            logger.debug("directory_miss", extra={"lookup_id": entity_id, "kind": "entity"})
            return "某物"
        return entity.name

    def describe(self, any_id: str) -> DirectoryEntry | None:
        soul = self._souls.get(any_id)
        if soul is not None:
            return DirectoryEntry(
                entry_id=any_id,
                name=soul.name or "某人",
                kind="agent",
                role=soul.role,
                gender=soul.gender,
            )
        npc = self._environment.get_npc(any_id)
        if npc is not None:
            return DirectoryEntry(
                entry_id=any_id,
                name=npc.name or "某人",
                kind="npc",
                description=npc.description,
                gender=npc.gender,
            )
        place = self._environment.space.get(any_id)
        if place is not None:
            return DirectoryEntry(
                entry_id=any_id,
                name=place.name or "某地",
                kind="location",
                description=place.description,
            )
        entity = self._environment.get_entity(any_id)
        if entity is not None:
            return DirectoryEntry(
                entry_id=any_id,
                name=entity.name or "某物",
                kind=entity.entity_type.value,
                description=entity.description,
                is_takeable=entity.is_takeable,
            )
        logger.debug("directory_miss", extra={"lookup_id": any_id, "kind": "unknown"})
        return None
