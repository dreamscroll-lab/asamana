"""In-memory agent store."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from core.factory import ComponentKind, ProviderFactory
from core.interfaces.agent_store import AgentRelation, AgentState, AgentStoreProvider


@ProviderFactory.register("in_memory", kind=ComponentKind.AGENT_STORE)
class InMemoryAgentStore(AgentStoreProvider):
    """Simple in-memory agent persistence."""

    def __init__(self) -> None:
        self._states: Dict[Tuple[str, str], AgentState] = {}
        self._relations: Dict[Tuple[str, str, str], AgentRelation] = {}
        self._initial_states: Dict[Tuple[str, str], AgentState] = {}
        self._initial_relations: Dict[Tuple[str, str, str], AgentRelation] = {}

    async def save_agent_state(self, world_id: str, agent_id: str, state: AgentState) -> None:
        self._states[(world_id, agent_id)] = state

    async def load_agent_state(self, world_id: str, agent_id: str) -> Optional[AgentState]:
        return self._states.get((world_id, agent_id))

    async def save_relation(self, relation: AgentRelation) -> None:
        key = (relation.world_id, relation.from_id, relation.to_id)
        self._relations[key] = relation

    async def load_relation(
        self,
        world_id: str,
        from_id: str,
        to_id: str,
    ) -> Optional[AgentRelation]:
        return self._relations.get((world_id, from_id, to_id))

    async def load_all_relations(self, world_id: str, agent_id: str) -> List[AgentRelation]:
        return [
            relation
            for (stored_world_id, from_id, _), relation in self._relations.items()
            if stored_world_id == world_id and from_id == agent_id
        ]

    async def save_initial_agent_state(self, world_id: str, agent_id: str, state: AgentState) -> None:
        self._initial_states[(world_id, agent_id)] = state

    async def load_initial_agent_state(self, world_id: str, agent_id: str) -> Optional[AgentState]:
        return self._initial_states.get((world_id, agent_id))

    async def save_initial_relation(self, relation: AgentRelation) -> None:
        self._initial_relations[(relation.world_id, relation.from_id, relation.to_id)] = relation

    async def load_all_initial_relations(self, world_id: str, agent_id: str) -> List[AgentRelation]:
        return [
            relation
            for (stored_world_id, from_id, _), relation in self._initial_relations.items()
            if stored_world_id == world_id and from_id == agent_id
        ]

    async def list_agent_ids(self, world_id: str) -> List[str]:
        return [
            agent_id
            for (stored_world_id, agent_id) in self._states
            if stored_world_id == world_id
        ]

    async def clear_relations(self, world_id: str) -> None:
        self._relations = {
            key: relation
            for key, relation in self._relations.items()
            if key[0] != world_id
        }

    async def delete_world(self, world_id: str) -> None:
        self._states = {k: v for k, v in self._states.items() if k[0] != world_id}
        self._relations = {k: v for k, v in self._relations.items() if k[0] != world_id}
        self._initial_states = {
            k: v for k, v in self._initial_states.items() if k[0] != world_id
        }
        self._initial_relations = {
            k: v for k, v in self._initial_relations.items() if k[0] != world_id
        }
