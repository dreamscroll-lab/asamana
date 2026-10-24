"""Relation contract: one agent's directed view of another."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class AgentRelation:
    """`from_id`'s directed view of `to_id`; `B→A` is a separate, independent record. Held by
    `from_id`'s ``RelationSystem`` and committed with each step's snapshot (``WorldState.relations``).

    Holds only the slow-changing objective trust/affection baseline. The perceived view
    (distorted by emotion and recent memory) is computed by `RelationSystem.perceive()` and
    never persisted.

    `labels` say who `to_id` is to `from_id`: ``"kind:role"`` (e.g. ``"父子:父亲"``, role =
    `to_id`'s position relative to `from_id`) or a bare ``"kind"`` (e.g. ``"见过"``).
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

    ``get_or_create`` creates a neutral record as soon as two agents share a memory, even an
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


def relation_from_snapshot_dict(payload: dict[str, Any]) -> AgentRelation:
    """The inverse of ``relation_to_snapshot_dict``."""

    return AgentRelation(**payload)

