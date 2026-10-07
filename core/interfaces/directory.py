"""World directory contract: per-world read-only id → identity info."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from core.interfaces.perception import NpcIdentity, PerceivedIdentity


@dataclass(frozen=True)
class DirectoryEntry:
    """Immutable identity info for one world object."""

    entry_id: str
    name: str
    kind: str                  # body / place / thing — values grow with ``WorldEntityType``; don't copy them here
    description: str = ""      # entities + npcs; "" for agents
    role: str = ""             # agents only
    gender: str = ""           # agents only; free-text label (set by the world theme), "" when unset
    is_takeable: bool = False  # entities only


class WorldDirectory(ABC):
    """Per-world read-only identity facade: id → immutable display info.

    Identity ONLY; mutable state stays on EnvironmentSystem / Agent, never a second source of
    truth here. Serves only god-view layers (engine / world / interaction / tuning): names and
    genders in agent/ cognition prompts must come from perception, to keep information asymmetry.
    """

    @abstractmethod
    def agent_name(self, agent_id: str) -> str:
        """Display name; "某人" on miss, never the raw id (that would leak into narrative text)."""

    @abstractmethod
    def agent_identity_map(self, agent_ids: Iterable[str]) -> dict[str, PerceivedIdentity]:
        """Return id → perceived identity (name + gender); unknown ids are omitted.

        The god-view side of the perception channel: fills the identity half of
        ``SpatialPerception.visible_agents`` each step.
        """

    @abstractmethod
    def npc_identity_map(self, npc_ids: Iterable[str]) -> dict[str, "NpcIdentity"]:
        """Return id → immutable Npc identity (name/gender/age/description); misses omitted.

        Identity only: busy / held down lives on the live ``Npc``; ``engine.presence`` joins
        the two halves.
        """

    @abstractmethod
    def all_agent_names(self) -> dict[str, str]:
        """Return the full id → name map for all agents in the world."""

    @abstractmethod
    def location_name(self, location_id: str) -> str:
        """Return the location's display name; descriptive referent ("某地") on miss."""

    @abstractmethod
    def entity_name(self, entity_id: str) -> str:
        """Return a non-agent entity's display name; descriptive ("某物") on miss."""

    @abstractmethod
    def describe(self, any_id: str) -> DirectoryEntry | None:
        """Resolve an id of unknown kind (agent, then npc, then entity); None on total miss.

        On a partial hit (object exists but is unnamed) ``DirectoryEntry.name`` is
        a descriptive referent, never the raw id.
        """
