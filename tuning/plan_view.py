"""Shape an AgentStepPlan (plan_step output) into a JSON-safe, browsable view.

Lives in tuning. plan_step runs the full perception → motivation → decision
assembly and returns an AgentStepPlan without persisting; this view extracts the
three layers (what the agent perceived, what it wants, what it decided) for the
per-step trace inspector. The LLM calls behind these layers (perception emotion,
need-goal generation, decision) are captured separately by the traced router.
"""

from __future__ import annotations

from typing import Any

from agent.agent import Agent, AgentStepPlan
from core.interfaces.action import (
    KIND_AGENT, KIND_LOCATION, KIND_NPC, KIND_OBJECT, ActionTarget, Ref,
)
from world.models import WorldEntityType
from agent.decision import _entity_owner_name


def _action_type(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def entity_view(entity, spatial, self_id: str) -> str:
    """One entity in the tuning view, shown together with whoever holds it.

    With only the name, "in someone's hands" looks the same as "lying on the ground", and that is
    exactly the half of the information a decision should weigh.
    """
    owner = _entity_owner_name(entity, self_id=self_id, spatial=spatial)
    return f"{entity.name}（{owner}）" if owner else entity.name


def target_view(target: "ActionTarget | None", names: dict[str, str] | None = None) -> dict[str, Any]:
    """Flatten the three relation axes into JSON the audit side can read, each with its kind.

    A bare id list can't tell "take him along" from "attack him". And if only acts_on carried a kind
    while the other two were bare, readers would have to know from elsewhere that "claims always
    holds people" — knowledge that goes stale the first time an action hits a nearby entity.
    When ``names`` is given, ids are replaced by names (the judge prompt reads prose, not coordinates).
    """
    tgt = target if target is not None else ActionTarget()

    def refs(group: "list[Ref]") -> list[dict[str, str]]:
        return [{"kind": r.kind, "id": (names or {}).get(r.id, r.id)} for r in group]

    return {"acts_on": refs(tgt.acts_on), "claims": refs(tgt.claims), "reaches": refs(tgt.reaches)}


# How the three target axes read in a judge prompt. Kept next to the code that writes this shape:
# if each judge kept its own copy they would drift, and when a flattened field changes the stale
# copy would silently render as "无显式目标".
_TARGET_KIND_LABEL = {
    KIND_AGENT: "对象", KIND_NPC: "对象", KIND_LOCATION: "目的地", KIND_OBJECT: "物",
    **{t.value: "物" for t in WorldEntityType},
}


def render_target(target: dict[str, Any] | None) -> str:
    """Three-axis target → one line of prose. An empty target reads as "无显式目标"."""
    tgt = target or {}
    parts: list[str] = []
    aimed = {str(ref.get("id")) for ref in tgt.get("acts_on") or []}
    for ref in tgt.get("acts_on") or []:
        label = _TARGET_KIND_LABEL.get(ref.get("kind"), "指向")
        parts.append(f"{label}={ref.get('id')}")
    # Don't report someone as "带着" when they're already the object: in a conversation the other
    # party is both, but readers only need it once, and a repeated name looks like two people.
    claimed = [str(r.get("id")) for r in tgt.get("claims") or [] if str(r.get("id")) not in aimed]
    if claimed:
        parts.append("带着=" + "、".join(claimed))
    reached = [str(r.get("id")) for r in tgt.get("reaches") or []]
    if reached:
        parts.append("波及=" + "、".join(reached))
    return "；".join(parts) or "无显式目标"


def plan_step_view(agent: Agent, plan: AgentStepPlan) -> dict[str, Any]:
    """Return a comprehensive, JSON-safe view of one agent's plan_step."""
    spatial = plan.spatial
    ne = plan.need_evaluation
    action = plan.action
    return {
        "agent_id": plan.agent_id,
        "agent_name": agent.personality.soul.name,
        "is_main_character": agent.is_main_character,
        "step": plan.step,
        "perception": {
            "location": spatial.location_view.name or spatial.location_id,
            "world_time": spatial.world_time_label,
            "visible_agents": [
                ((p.identity.name if (p := spatial.visible_agents.get(a)) else "") or "某人")
                for a in spatial.visible_agent_ids
            ],
            "visible_entities": [
                entity_view(e, spatial, plan.agent_id) for e in spatial.visible_entities
            ],
            "ambient_events": [e.content for e in spatial.ambient_events],
        },
        "motivation": {
            "dominant_need": ne.dominant_need.value if ne.dominant_need else None,
            "active_needs": [
                {"type": n.type.value, "intensity": round(n.intensity, 3), "weight": round(n.weight, 3)}
                for n in ne.active_needs
            ],
            "short_term_goals": list(ne.short_term_goals),
            "long_term_goals": list(ne.long_term_goals),
        },
        # action is None when the decision LLM was unavailable this step (no decision
        # made → the runtime skips the agent). Surface that explicitly rather than crash.
        "decision": None if action is None else {
            "action_type": _action_type(action.action_type),
            "action_description": action.action_description,
            # Three relation axes, not a flat id list (see target_view).
            "target": target_view(action.target),
            "reason": action.reason,
            "inner_monologue": action.inner_monologue,
            "expected_outcome": action.expected_outcome,
            "estimated_steps": action.estimated_steps,
            "llm_model": action.llm_model,
        },
    }
