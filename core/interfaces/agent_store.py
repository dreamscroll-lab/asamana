"""Agent store contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Iterable
from typing import Any, List


@dataclass
class AgentState:
    """Persisted agent state."""

    world_id: str
    agent_id: str
    updated_step: int
    current_emotion: str
    emotion_intensity: float
    emotion_valence: float
    emotion_triggered_by: str | None
    active_needs: list[str]
    dominant_need: str | None
    long_term_goals: list[str]
    short_term_goals: list[str]
    current_location: str
    activity_status: str
    activity_target: str | None
    action_status: str
    current_action: str | None
    action_remaining_steps: int
    last_action: str | None
    last_action_result: str | None
    last_action_succeeded: bool | None
    short_term_goal_entities: list[dict[str, object]] = field(default_factory=list)
    need_intensities: dict[str, float] = field(default_factory=dict)
    vitality: float = 1.0
    # A dict, not a BodyCondition, so the file and in-memory stores agree (see
    # condition_to_dict); _apply_stored_state rebuilds it.
    condition: dict[str, object] | None = None
    # 0 = never decided; the scheduler's cadence gate reads that as starved and admits it, so a
    # fresh world's step 1 isn't empty. Must keep a default: the file store rehydrates by keyword,
    # so a required field would raise TypeError on every existing save file.
    last_decision_step: int = 0


@dataclass
class AgentRelation:
    """Persisted relation state: `from_id`'s directed view of `to_id`; `B→A` is a separate,
    independent record.

    Persists only the slow-changing objective trust/affection baseline. The perceived view
    (distorted by emotion and recent memory) is computed by `RelationSystem.perceive()` and
    never persisted.

    `labels` say what kind of relationship this is, from `from_id`'s viewpoint: ``"kind:role"``
    (e.g. ``"父子:儿子"``, role = `from_id`'s position) or a bare ``"kind"`` (e.g. ``"见过"``).
    Several may coexist. Updated only at narrative milestones; structural bonds (blood /
    marriage / etc.) are immutable once set.
    """

    world_id: str
    from_id: str  # relation holder (whose viewpoint)
    to_id: str  # relation target
    trust_objective: float  # 0.0-1.0, default 0.5
    affection_objective: float  # -1.0-1.0, negative = dislike, default 0.0
    updated_step: int
    labels: List[str] = field(default_factory=list)
    history_summary: str = ""  # rolling summary of recent interactions (last 3, " | "-joined)
    interaction_count: int = 0
    to_name: str = ""  # cached display name of `to_id`; lets absent-but-known relations render by name (not opaque id)
    to_gender: str = ""  # cached gender of `to_id`; travels with to_name so an absent-but-known target still renders a full referent

    def has_substance(self) -> bool:
        """See ``relation_has_substance``."""
        return relation_has_substance(
            labels=self.labels,
            history_summary=self.history_summary,
            interaction_count=self.interaction_count,
        )


def relation_has_substance(
    *, labels: "list[str] | None", history_summary: str, interaction_count: int
) -> bool:
    """Does a relation carry any substance, or is it a bare baseline record?

    ``get_or_create`` persists a neutral record as soon as two agents share a memory, even an
    overheard one. Those aren't relationships: they clutter the graph and ask the evolution
    judge about someone never dealt with. Any one of these is substance:
    - ``labels`` — seeded bonds live here with ``interaction_count == 0``, so count alone
      would drop the protagonist's own brother.
    - ``history_summary``.
    - ``interaction_count > 0`` — ``apply_interaction``, the sole writer of trust/affection,
      bumps it on every write, so the float values need no check of their own.

    The one rule for both the object form and the graph endpoint's snapshot dicts.
    """
    return bool(labels or history_summary or interaction_count > 0)


def relation_to_snapshot_dict(relation: AgentRelation) -> dict[str, Any]:
    """Serialize a relation into the dict form persisted in world snapshots."""

    return {
        "world_id": relation.world_id,
        "from_id": relation.from_id,
        "to_id": relation.to_id,
        "trust_objective": relation.trust_objective,
        "affection_objective": relation.affection_objective,
        "labels": list(relation.labels),
        "updated_step": relation.updated_step,
        "history_summary": relation.history_summary,
        "interaction_count": relation.interaction_count,
        "to_name": relation.to_name,
        "to_gender": relation.to_gender,
    }


class AgentStoreProvider(ABC):
    """Abstract agent store."""

    @abstractmethod
    async def save_agent_state(self, world_id: str, agent_id: str, state: AgentState) -> None:
        """Save or replace a persisted agent state."""

    @abstractmethod
    async def load_agent_state(self, world_id: str, agent_id: str) -> AgentState | None:
        """Read a persisted agent state."""

    @abstractmethod
    async def save_relation(self, relation: AgentRelation) -> None:
        """Save or replace a relation."""

    @abstractmethod
    async def load_relation(
        self,
        world_id: str,
        from_id: str,
        to_id: str,
    ) -> AgentRelation | None:
        """Read a relation."""

    @abstractmethod
    async def load_all_relations(self, world_id: str, agent_id: str) -> list[AgentRelation]:
        """Load all outgoing relations for an agent."""

    @abstractmethod
    async def save_initial_agent_state(self, world_id: str, agent_id: str, state: AgentState) -> None:
        """Persist the step-0 agent state as a reset baseline."""

    @abstractmethod
    async def load_initial_agent_state(self, world_id: str, agent_id: str) -> AgentState | None:
        """Load the step-0 baseline agent state."""

    @abstractmethod
    async def save_initial_relation(self, relation: AgentRelation) -> None:
        """Persist the step-0 relation as a reset baseline."""

    @abstractmethod
    async def load_all_initial_relations(self, world_id: str, agent_id: str) -> list[AgentRelation]:
        """Load all step-0 baseline outgoing relations for an agent."""

    @abstractmethod
    async def list_agent_ids(self, world_id: str) -> list[str]:
        """List all agent IDs that have persisted state for a world."""

    @abstractmethod
    async def clear_relations(self, world_id: str) -> None:
        """Delete all current relations for a world; the step-0 baseline is kept.

        Used on reset, or runtime-created pairs would survive it.
        """

    @abstractmethod
    async def delete_world(self, world_id: str) -> None:
        """Purge every persisted agent state (including step-0 baselines) and
        relation for *world_id*.

        Called by whole-world deletion; irreversible. Concrete providers must
        also drop any in-memory state they keep for the world.
        """


async def relations_snapshot(
    store: "AgentStoreProvider", world_id: str, agent_ids: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Every listed agent's outgoing relations, keyed ``"from->to"``, as world snapshots hold them."""
    relations: dict[str, dict[str, Any]] = {}
    for agent_id in agent_ids:
        for relation in await store.load_all_relations(world_id, agent_id):
            relations[f"{relation.from_id}->{relation.to_id}"] = relation_to_snapshot_dict(relation)
    return relations
