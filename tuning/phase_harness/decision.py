"""Decision stage dry-run (``run_decision``)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agent.memory_types import Memory, MemoryKind, MemoryStream
from agent.perception import PerceptionPacket
from agent.relation import describe_relations
from config.models import Config
from core.container import Container
from core.context import clear_log_context, set_log_context
from core.logging import get_logger
from core.interfaces.perception import VisibleEntity
from engine.presence import attach_presence, reset_visible

from tuning.fixtures import load_fixture, save_fixture
from tuning.phase_harness.common import emotion_view, restore_traced
from tuning.phase_harness.scenario import (
    apply_ambient_from_scenario, broadcasts_from_scenario, external_goals_from_scenario,
    inboxes_from_scenario, presence_name, resolve_ref, spatials_for,
)
from tuning.plan_view import entity_view, target_view
from tuning.trace import InMemoryTraceSink, JsonlTraceSink, PhaseTrace

logger = get_logger(__name__)


def _apply_present_from_scenario(
    scenario: dict | None, spatials: dict, name_by_id: dict,
    directory=None, agents: dict | None = None, environment=None,
) -> None:
    """scenario.present = [{"agent", "sees": [<name/id>...]}] — overrides who the agent sees present.

    Decisions need present people as bindable targets (TALK/COVERT/PHYSICAL), but agents in the
    baseline world are spread out and rarely together, so the scenario places them explicitly
    (harness test scaffolding, not a production change).
    """
    id_by_name = {v: k for k, v in name_by_id.items()}
    all_ids = list(name_by_id.keys())
    for p in (scenario or {}).get("present", []):
        aid = resolve_ref(p.get("agent") or "", id_by_name, list(spatials.keys()))
        if aid not in spatials:
            continue
        seen: list[str] = []
        for ref in p.get("sees", []):
            sid = resolve_ref(ref, id_by_name, all_ids)
            if sid and sid != aid and sid not in seen:
                seen.append(sid)
        sp = spatials[aid]
        # Replace the whole present set, then fill in identity and condition. attach_presence must follow
        # reset_visible immediately, or the inserted people have only an id, with no name and no condition.
        reset_visible(sp, seen)
        if directory is not None and agents is not None and environment is not None:
            attach_presence(
                sp, directory=directory, agents=agents, environment=environment,
            )


def _apply_time_from_scenario(scenario: dict | None, spatials: dict) -> None:
    """scenario.time = {"hour": 0-23, "label"?} — overrides this step's world time.

    hour drives the need stage's day/night weighting (late night → physiological boosted). Ambient
    text saying it's late isn't enough; world_time_hour has to actually be set to night (otherwise
    the restored world's current time is used).
    """
    t = (scenario or {}).get("time")
    if not t:
        return
    hour = t.get("hour")
    label = t.get("label")
    for sp in spatials.values():
        if isinstance(hour, int) and 0 <= hour <= 23:
            sp.world_time_hour = hour
        if label:
            sp.world_time_label = str(label)


def _apply_entities_from_scenario(scenario: dict | None, spatials: dict, name_by_id: dict) -> None:
    """scenario.entities = [{"agent"?/"location"?, "items": [{"name","state"?,"desc"?,"takeable"?,"type"?}]}].

    Puts visible things (tokens, bows, secret letters, …) into the matching agent's
    spatial.visible_entities (where decide gets its bindable entities). No agent/location filter →
    injected for every present agent. Harness test scaffolding.
    """
    id_by_name = {v: k for k, v in name_by_id.items()}
    for idx, e in enumerate((scenario or {}).get("entities", [])):
        agent_ref = e.get("agent")
        target_aid = resolve_ref(agent_ref, id_by_name, list(spatials.keys())) if agent_ref else None
        loc = e.get("location")
        items = e.get("items", [])
        for aid, sp in spatials.items():
            if (target_aid and aid == target_aid) or (loc and sp.location_id == loc) or (not agent_ref and not loc):
                for j, it in enumerate(items):
                    # holder: put the thing in someone's hands (self or another present agent). Being in the same
                    # place is enough to perceive it; holding changes what you can do with it, so it still goes into the bindable list, just with its holder.
                    holder = it.get("holder")
                    holder_id = (
                        aid if holder in ("self", "me") else id_by_name.get(str(holder), holder)
                    ) if holder else None
                    sp.visible_entities.append(VisibleEntity(
                        entity_id=f"inj-ent-{idx}-{j}",
                        name=str(it.get("name", "")),
                        entity_type=str(it.get("type", "object")),
                        state=str(it.get("state", "intact")),
                        description=str(it.get("desc", "")),
                        is_takeable=bool(it.get("takeable", False)),
                        holder_id=holder_id,
                    ))


def _memories_from_scenario(scenario, agent_ids, name_by_id: dict, run_step: int):
    """scenario.memories = [{"agent"/"to", "factual":[...], "experiential":[...], "insights":[...]}]
    or {"<agent>": {factual/experiential/insights}}. Built into Memory objects, grouped by agent.

    Lets the decision suite inject memories deterministically (retrieval is empty on a fresh restore
    and in-memory embedding is unreliable); the harness puts them straight into internal_context.
    Test scaffolding, not a production change.
    """
    id_by_name = {v: k for k, v in name_by_id.items()}
    out: dict[str, dict[str, list[Memory]]] = {
        aid: {"factual": [], "experiential": [], "insights": []} for aid in agent_ids
    }
    entries = (scenario or {}).get("memories", [])
    if isinstance(entries, dict):
        entries = [{"agent": k, **v} for k, v in entries.items()]
    for i, e in enumerate(entries):
        aid = resolve_ref(e.get("agent") or e.get("to") or "", id_by_name, agent_ids)
        if aid not in out:
            continue
        for j, c in enumerate(e.get("factual", []) or []):
            out[aid]["factual"].append(Memory(
                id=f"seed-f-{aid}-{i}-{j}", stream=MemoryStream.FACTUAL, agent_id=aid,
                stored_content=str(c), raw_content=str(c), created_step=run_step,
                kind=MemoryKind.EVENT, emotion_label="objective"))
        for j, c in enumerate(e.get("experiential", []) or []):
            out[aid]["experiential"].append(Memory(
                id=f"seed-e-{aid}-{i}-{j}", stream=MemoryStream.EXPERIENTIAL, agent_id=aid,
                stored_content=str(c), raw_content=str(c), created_step=run_step, kind=MemoryKind.EVENT))
        for j, c in enumerate(e.get("insights", []) or []):
            out[aid]["insights"].append(Memory(
                id=f"seed-i-{aid}-{i}-{j}", stream=MemoryStream.EXPERIENTIAL, agent_id=aid,
                stored_content=str(c), raw_content=str(c), created_step=run_step, kind=MemoryKind.INSIGHT))
    return out


def _action_view(action) -> dict[str, Any]:
    """Serialize an AgentAction for the decision trace / judge.

    ``action`` is None when the decision LLM was unavailable (no decision made) —
    surface that explicitly rather than crash the dry-run.
    """
    if action is None:
        return {"action_type": None, "action_description": "", "inner_monologue": "",
                "expected_outcome": "", "estimated_steps": 0, "urgency": None,
                "target": target_view(None)}
    at = action.action_type
    return {
        "action_type": at.value if hasattr(at, "value") else str(at),
        "action_description": action.action_description,
        "inner_monologue": action.inner_monologue,
        "expected_outcome": action.expected_outcome,
        "estimated_steps": action.estimated_steps,
        "urgency": action.urgency.value if hasattr(action.urgency, "value") else str(action.urgency),
        # Three relation axes, not a flat id list (see plan_view.target_view).
        "target": target_view(action.target),
    }


async def run_decision(
    container: Container,
    config: Config,
    world_id: str,
    *,
    scenario: dict | None = None,
    sink: InMemoryTraceSink | None = None,
    trace_dir: str = "./data/tuning_traces",
) -> int:
    """Isolated dry-run of the decision stage (perception→motivation→**decision**) for a restored world.

    Chains the real production path: ``_build_internal_context`` (appraisal + need + memory + relations)
    → inject the scenario's declared memories (deterministic) → ``decision_engine.decide``. Calls
    production methods directly with zero production changes (only memory injection is test
    scaffolding). Restores per scenario (the decide chain mutates need state); nothing persisted.
    """
    trace_sink = sink if sink is not None else JsonlTraceSink(trace_dir, segment="decision")
    if isinstance(trace_sink, JsonlTraceSink):
        trace_sink.truncate(world_id)
    if scenario is None:
        scenario = load_fixture(trace_dir, world_id, "decision_scenario")
    _, world, run_step, world_time_label = await restore_traced(container, config, world_id, trace_sink)
    spatials = spatials_for(world, run_step, world_time_label)
    name_by_id = world.directory.all_agent_names()
    active = {aid: world.agents[aid] for aid in spatials}
    broadcasts = broadcasts_from_scenario(scenario, run_step)
    inboxes = inboxes_from_scenario(scenario, world_id, run_step, active, name_by_id)
    apply_ambient_from_scenario(scenario, spatials, active, name_by_id)
    _apply_present_from_scenario(
        scenario, spatials, name_by_id, world.directory, world.agents, world.environment,
    )
    _apply_entities_from_scenario(scenario, spatials, name_by_id)
    _apply_time_from_scenario(scenario, spatials)
    ext_goals = external_goals_from_scenario(scenario, active, name_by_id)
    mems = _memories_from_scenario(scenario, active, name_by_id, run_step)

    for aid, agent in active.items():
        sp = spatials[aid]
        agent_bcs = [b for b in broadcasts if b.location_scope is None or b.location_scope == sp.location_id]
        inbox = inboxes.get(aid, [])
        goals = ext_goals.get(aid, [])
        set_log_context(world_id=world_id, agent_id=aid, step=str(run_step))
        try:
            agent.pending_external_goals = list(goals)
            internal = await agent._build_internal_context(
                spatial=sp, inbox=inbox, broadcasts=agent_bcs,
                current_step=run_step, consumed_external_goals=goals,
            )
            m = mems.get(aid, {})
            if m.get("factual"):
                internal.factual_memories = m["factual"]
            if m.get("experiential"):
                internal.experiential_memories = m["experiential"]
            if m.get("insights"):
                internal.insights = m["insights"]
            packet = PerceptionPacket(
                agent_id=aid, step=run_step, spatial=sp, inbox=inbox,
                broadcasts=agent_bcs, internal_context=internal,
            )
            decision = await agent.decision_engine.decide(personality=agent.personality, packet=packet)
            action = decision.action  # None for NO_ACTION / FAILED (handled by _action_view)
        except Exception as exc:  # noqa: BLE001 — one agent's failure must not abort the run
            logger.warning("decision_dryrun_failed",
                           extra={"world_id": world_id, "agent_id": aid, "error": str(exc)})
            continue
        soul = agent.personality.soul
        ev = internal.need_evaluation
        trace_sink.record_phase(PhaseTrace(
            world_id=world_id,
            phase="stage.decision",
            step=run_step,
            agent_id=aid,
            inputs={
                "agent_name": name_by_id.get(aid, aid),
                "is_main_character": agent.is_main_character,
                "location": sp.location_view.name or sp.location_id,
                "role": getattr(soul, "role", ""),
                # Judgment lens
                "core_traits": list(getattr(soul, "core_traits", [])),
                "core_values": list(getattr(soul, "core_values", [])),
                "hard_constraints": list(getattr(soul, "hard_constraints", [])),
                "emotion": emotion_view(internal.emotion),
                # Direction
                "dominant_need": ev.dominant_need.value if ev.dominant_need else None,
                "short_term_goals": list(ev.short_term_goals),
                # Soft reference (including injected memories)
                "relation_context": describe_relations(internal.relevant_relations),
                "factual_memories": [mm.stored_content for mm in internal.factual_memories],
                "experiential_memories": [mm.stored_content for mm in internal.experiential_memories],
                "insights": [mm.stored_content for mm in internal.insights],
                "injected_messages": [
                    {"from": msg.sender_name or msg.sender_id, "content": msg.content, "urgency": msg.urgency.value}
                    for msg in inbox
                ],
                "injected_broadcasts": [{"content": b.content, "severity": b.severity} for b in agent_bcs],
                "injected_ambient": [{"content": ev2.content} for ev2 in sp.ambient_events],
                "external_goals": [
                    {"text": g.text, "urgency": g.urgency.value, "drive_type": g.drive_type.value,
                     "related_need": g.related_need.value if g.related_need else None}
                    for g in goals
                ],
                # Reality (bindable options)
                "visible_agents": [presence_name(sp, v) for v in sp.visible_agent_ids],
                "reachable_locations": [rl.location_id for rl in sp.reachable_locations],
                "visible_entities": [entity_view(e, sp, aid) for e in sp.visible_entities],
            },
            outputs={"agent_name": name_by_id.get(aid, aid),
                     "decision_status": decision.status.value, **_action_view(action)},
            timestamp=datetime.now().isoformat(),
        ))
    clear_log_context()
    if scenario:
        save_fixture(trace_dir, world_id, "decision_scenario", scenario)
    logger.info(
        "tuning_decision_dry_run_complete",
        extra={"world_id": world_id, "step": run_step, "agent_count": len(active)},
    )
    return run_step
