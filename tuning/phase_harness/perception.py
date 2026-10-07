"""Perception stage dry-run (``perceive_world``)."""

from __future__ import annotations

from typing import Any

from agent.perception_layer import PerceptionMemoryLayer, PerceptionTuning, _strength_to_importance
from core.logging import get_logger
from world import World

from tuning.phase_harness.scenario import (
    apply_ambient_from_scenario, broadcasts_from_scenario, inboxes_from_scenario,
    presence_name, spatials_for,
)

logger = get_logger(__name__)


def _item_view(item, tuning: PerceptionTuning) -> dict:
    """Serialize a PerceptionItem with its tuning-resolved importance bucket."""
    return {
        "source": item.source,
        "content": item.content,
        "signal_strength": round(item.signal_strength, 3),
        "importance": _strength_to_importance(
            item.signal_strength, tuning.importance_high_cutoff, tuning.importance_medium_cutoff
        ).name,
        "related_agents": list(item.related_agents),
    }


async def perceive_world(
    world: World, world_id: str, run_step: int, world_time_label: str,
    *, scenario: dict | None, tuning: PerceptionTuning,
) -> dict[str, Any]:
    """Compute per-agent perception selection for one knob set against a restored world.

    Rebuilds fresh spatials each call (scenario injectors mutate them), so this is safe
    to invoke repeatedly with different tunings for A/B comparison. Read-only: uses the
    layer's ``preview`` (no memory writes, no relation updates).
    """
    name_by_id = world.directory.all_agent_names()
    spatials = spatials_for(world, run_step, world_time_label)
    active = {aid: world.agents[aid] for aid in spatials}
    broadcasts = broadcasts_from_scenario(scenario, run_step)
    inboxes = inboxes_from_scenario(scenario, world_id, run_step, active, name_by_id)
    apply_ambient_from_scenario(scenario, spatials, active, name_by_id)
    # External pressure isn't written to memory (it's a live signal only), so the perception harness doesn't inject external_goals.

    agents_out: list[dict[str, Any]] = []
    for aid, agent in active.items():
        sp = spatials[aid]
        # Mirror runtime _perceive_all_agents: an agent only perceives broadcasts global or scoped to its location.
        agent_bcs = [b for b in broadcasts if b.location_scope is None or b.location_scope == sp.location_id]
        inbox = inboxes.get(aid, [])
        # tuning reuses the production layer's selection logic under an injected knob set,
        # reading its internals directly (read-only — no record/write) to avoid adding a
        # tuning-only public method to the business class.
        layer = PerceptionMemoryLayer(
            memory_system=agent.memory_system,
            relation_system=agent.relation_system,
            agent_id=aid,
            is_main_character=agent.is_main_character,
            tuning=tuning,
        )
        collected = await layer._collect_items(sp, inbox, agent_bcs)
        selected = layer._select_items(collected)
        agents_out.append({
            "agent_id": aid,
            "agent_name": name_by_id.get(aid, aid),
            "is_main_character": agent.is_main_character,
            "location": sp.location_view.name or sp.location_id,
            "location_id": sp.location_id,
            "inputs": {
                "injected_broadcasts": [
                    {"content": b.content, "severity": b.severity, "location_scope": b.location_scope}
                    for b in agent_bcs
                ],
                "injected_messages": [
                    {"from": m.sender_name or m.sender_id, "content": m.content, "urgency": m.urgency.value}
                    for m in inbox
                ],
                "injected_ambient": [{"content": ev.content, "strength": ev.strength} for ev in sp.ambient_events],
                "visible_agents": [presence_name(sp, v) for v in sp.visible_agent_ids],
            },
            "collected": [_item_view(i, tuning) for i in collected],
            "selected": [_item_view(i, tuning) for i in selected],
        })
    agents_out.sort(key=lambda a: (not a["is_main_character"], a["agent_name"]))
    return {"step": run_step, "agents": agents_out}
