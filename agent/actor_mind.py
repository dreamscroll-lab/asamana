"""The character's mind as the actor sees it at any time: its own feeling, goals and relations,
with the amendments the actor saved that haven't landed yet."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from agent.goals import goal_age_hint, goal_due_hint, is_due, is_live_goal, is_past_allowance
from core.interfaces.actor import ActorAmendments, ActorMind, BriefGoal, BriefRelation

if TYPE_CHECKING:
    from agent.personality import PersonalityLayer
    from core.interfaces.relation import AgentRelation


def build_actor_mind(
    personality: "PersonalityLayer",
    relations: Iterable["AgentRelation"],
    *,
    dead_ids: "frozenset[str]",
    pending: ActorAmendments | None,
    step: int,
    seconds_per_step: int,
) -> ActorMind:
    state = personality.state
    return ActorMind(
        emotion=state.emotion.primary.value,
        emotion_intensity=state.emotion.intensity,
        emotion_valence=state.emotion.valence,
        long_term_goals=tuple(state.long_term_goals),
        short_term_goals=tuple(
            BriefGoal(
                id=g.id, text=g.text, status=g.status.value, origin=g.origin.value,
                created_step=g.created_step, due_step=g.due_step,
                progress_summary=g.progress_summary,
                time_hint=(
                    goal_due_hint(g, step, seconds_per_step) or goal_age_hint(g, step, seconds_per_step)
                ) if is_live_goal(g) else "",
                attention=is_live_goal(g) and (is_due(g, step) or is_past_allowance(g, step)),
            )
            for g in state.short_term_goal_entities
        ),
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
