"""The character's mind as the actor sees it at any time: its own feeling, goals and relations,
with the amendments the actor saved that haven't landed yet."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from core.interfaces.actor import ActorAmendments, ActorMind, BriefRelation

if TYPE_CHECKING:
    from agent.personality import PersonalityLayer
    from core.interfaces.agent_store import AgentRelation


def build_actor_mind(
    personality: "PersonalityLayer",
    relations: Iterable["AgentRelation"],
    *,
    dead_ids: "frozenset[str]",
    pending: ActorAmendments | None,
) -> ActorMind:
    state = personality.state
    return ActorMind(
        emotion=state.emotion.primary.value,
        emotion_intensity=state.emotion.intensity,
        emotion_valence=state.emotion.valence,
        long_term_goals=tuple(state.long_term_goals),
        short_term_goals=tuple(state.short_term_goals),
        relations=tuple(
            BriefRelation(
                agent_id=r.to_id,
                name=r.to_name or "某人",
                gender=r.to_gender,
                trust=r.trust_objective,
                affection=r.affection_objective,
                labels=tuple(r.labels),
                summary=r.history_summary,
                deceased=r.to_id in dead_ids,
            )
            for r in relations
        ),
        pending=pending,
    )
