"""The one codec between an agent's in-memory ``StateLayer`` and its stored ``AgentState``.

Every write (each step's commit and the world's initial save) goes through ``encode_agent_state``
and every restore through ``decode_agent_state``, so a field added to one side can't be missed
by the other.
"""

from __future__ import annotations

from agent.goals import GoalEntity, GoalOrigin, GoalStatus, text_to_long_term_goal_entity
from agent.need import NeedType
from agent.personality import (
    ActionStatus,
    AgentActivityStatus,
    EmotionState,
    StateLayer,
    parse_emotion_type,
)
from core.interfaces.agent_store import AgentState
from core.interfaces.condition import condition_from_dict, condition_to_dict


def encode_agent_state(world_id: str, agent_id: str, state: StateLayer) -> AgentState:
    return AgentState(
        world_id=world_id,
        agent_id=agent_id,
        updated_step=state.step,
        current_emotion=state.emotion.primary,
        emotion_intensity=state.emotion.intensity,
        emotion_valence=state.emotion.valence,
        emotion_triggered_by=state.emotion.triggered_by,
        active_needs=list(state.active_needs),
        dominant_need=state.dominant_need,
        long_term_goals=list(state.long_term_goals),
        short_term_goals=list(state.short_term_goals),
        current_location=state.current_location,
        activity_status=state.activity_status.value,
        activity_target=state.activity_target,
        action_status=state.action_status.value,
        current_action=state.current_action,
        action_remaining_steps=state.action_remaining_steps,
        last_action=state.last_action,
        last_action_result=state.last_action_result,
        last_action_succeeded=state.last_action_succeeded,
        short_term_goal_entities=[_goal_entity_to_dict(g) for g in state.short_term_goal_entities],
        need_intensities=dict(state.need_intensities),
        vitality=state.vitality,
        condition=condition_to_dict(state.condition),
        # 0 at world init = "never decided": the scheduler's cadence gate treats every agent as
        # starved on step 1, so a fresh world does not open on a step where nobody thought.
        last_decision_step=state.last_decision_step,
        acted_upon=list(state.acted_upon),
    )


def decode_agent_state(agent_id: str, stored: AgentState, *, orphaned: bool) -> StateLayer:
    """Rebuild the ``StateLayer`` a stored state was encoded from. ``orphaned``: an act in
    progress lost the execution that carries it, so the character comes back idle (nothing would
    ever tick or end that act)."""
    return StateLayer(
        agent_id=agent_id,
        step=stored.updated_step,
        emotion=EmotionState(
            primary=parse_emotion_type(stored.current_emotion),
            intensity=stored.emotion_intensity,
            valence=stored.emotion_valence,
            triggered_by=stored.emotion_triggered_by,
        ),
        active_needs=list(stored.active_needs),
        dominant_need=stored.dominant_need,
        long_term_goals=list(stored.long_term_goals),
        short_term_goals=list(stored.short_term_goals),
        current_location=stored.current_location,
        activity_status=(
            AgentActivityStatus.IDLE if orphaned else AgentActivityStatus(stored.activity_status)
        ),
        activity_target=None if orphaned else stored.activity_target,
        action_status=ActionStatus.IDLE if orphaned else ActionStatus(stored.action_status),
        current_action=None if orphaned else stored.current_action,
        action_remaining_steps=0 if orphaned else stored.action_remaining_steps,
        last_action=stored.last_action,
        last_action_result=stored.last_action_result,
        last_action_succeeded=stored.last_action_succeeded,
        # Every lifecycle field must come back: dropping one silently resets fail-retry counts,
        # goal age or need linkage on every restore.
        short_term_goal_entities=[
            goal_entity_from_dict(ent, fallback_id=f"restored-stg-{i}")
            for i, ent in enumerate(stored.short_term_goal_entities)
        ],
        # Long-term goals persist as text; restore rebuilds the entities from text (related_need
        # left None, as at build).
        long_term_goal_entities=[text_to_long_term_goal_entity(t) for t in stored.long_term_goals],
        need_intensities=dict(stored.need_intensities),
        vitality=stored.vitality,
        # Stored as a dict (see the AgentState.condition comment), rebuilt into a BodyCondition here.
        condition=condition_from_dict(stored.condition),
        # Without it everyone's starvation resets to zero after restore and they all flood the
        # decision loop next step (the cadence gate is bypassed for a beat).
        last_decision_step=stored.last_decision_step,
        acted_upon=list(stored.acted_upon),
    )


def _goal_entity_to_dict(goal: GoalEntity) -> dict[str, object]:
    return {
        "id": goal.id,
        "text": goal.text,
        "goal_type": goal.goal_type,
        "status": goal.status if isinstance(goal.status, str) else goal.status.value,
        "related_need": goal.related_need.value if goal.related_need is not None else None,
        "created_step": goal.created_step,
        "due_step": goal.due_step,
        "last_evaluated_step": goal.last_evaluated_step,
        "progress_summary": goal.progress_summary,
        "origin": goal.origin if isinstance(goal.origin, str) else goal.origin.value,
    }


def goal_entity_from_dict(ent: dict, *, fallback_id: str) -> GoalEntity:
    """Decode a stored short-term goal. ``related_need`` is coerced silently: an unknown or empty
    value degrades to None rather than raising (field-level coercion)."""
    raw_need = ent.get("related_need")
    try:
        related_need = NeedType(raw_need) if raw_need else None
    except ValueError:
        related_need = None
    last_evaluated = ent.get("last_evaluated_step")
    return GoalEntity(
        id=ent.get("id", fallback_id),
        text=ent.get("text", ""),
        goal_type=ent.get("goal_type", "short_term"),
        status=GoalStatus(ent["status"]) if ent.get("status") else GoalStatus.ACTIVE,
        related_need=related_need,
        created_step=int(ent.get("created_step", 0) or 0),
        due_step=_coerce_due_step(ent.get("due_step")),
        last_evaluated_step=int(last_evaluated) if last_evaluated is not None else None,
        progress_summary=str(ent.get("progress_summary", "")),
        origin=_coerce_origin(ent.get("origin")),
    )


def _coerce_origin(raw: object) -> GoalOrigin:
    """Missing or unrecognized → COGNITIVE, the agent's own goal; one bad field must not abort the
    whole world's restore (field-level coercion, as ``related_need``)."""
    try:
        return GoalOrigin(raw)
    except ValueError:
        return GoalOrigin.COGNITIVE


def _coerce_due_step(raw: object) -> int | None:
    """The agreed time: missing, null or unrecognized all become None. No deadline is the normal
    case, not a parse failure."""
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
