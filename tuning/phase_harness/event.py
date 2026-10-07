"""EventSystem narrative event injection dry-run

kind=gate → _passes_llm_check (pacing gate: should an event be injected now; reason-first JSON)
kind=plan → _generate_event_plan (event generation: entity spawn/alter/destroy + broadcast/message channels + content)
Each is one EVENT_TIMING LLM call, captured by the traced router. The briefing
(_build_narrative_summary) is driven by seeded InMemory snapshots that follow the narrative;
the premise (core_tension/narrative_theme) comes from the restored world.analysis. Zero
production intrusion: a standalone EventSystem + InMemory snapshot; baseline ./data is never written."""

from __future__ import annotations

import random
from collections import Counter
from datetime import datetime
from typing import Any

from config.models import Config
from core.container import Container
from core.context import clear_log_context, set_log_context
from core.logging import get_logger
from core.interfaces.snapshot import WorldSnapshot
from providers.snapshot.in_memory import InMemorySnapshotProvider
from engine.broadcast import BroadcastChannel
from engine.clock import GlobalClock
from engine.event import EventSettings, EventSystem, _menu_places
from engine.execution_processor import ExecutionProcessor
from engine.executors.registry import ActionExecutorRegistry
from engine.injection import Author, InjectionDispatcher, WorldEvent
from engine.message_system import MessageSystem
from engine.world_mutation import WorldMutationChannel

from tuning.phase_harness.common import restore_traced
from tuning.phase_harness.scenario import resolve_ref
from tuning.trace import InMemoryTraceSink, PhaseTrace

logger = get_logger(__name__)


def _event_relations_block(
    scenario: dict, id_by_name: dict, valid_ids, name_by_id: dict
) -> dict[str, Any]:
    """scenario.relations → agent_relations on the latest snapshot frame (dict form, matching the production snapshot schema)."""
    out: dict[str, Any] = {}
    for r in scenario.get("relations", []) or []:
        fid = resolve_ref(str(r.get("from", "")), id_by_name, valid_ids)
        tid = resolve_ref(str(r.get("to", "")), id_by_name, valid_ids)
        out[f"{fid}->{tid}"] = {
            "from_id": fid,
            "to_id": tid,
            "to_name": name_by_id.get(tid, tid),
            "labels": [str(x) for x in (r.get("labels") or [])],
            "history_summary": str(r.get("history_summary", "")),
            "trust_objective": float(r.get("trust", 0.5)),
            "affection_objective": float(r.get("affection", 0.0)),
        }
    return out


def _event_seed_snapshots(
    world_id: str, current_step: int, scenario: dict, world,
    id_by_name: dict, valid_ids, name_by_id: dict,
) -> list[WorldSnapshot]:
    """scenario.recent (per step) → a list of WorldSnapshots placed some steps before current_step.

    The latest frame carries relations (for 【当前关系】); each frame holds actions (main character
    actions → recent outcomes), metadata.broadcasts (world broadcasts) and
    metadata.environment.step_annotations (background/environment traces).
    """
    recent = scenario.get("recent", []) or []
    base = current_step - len(recent)
    snaps: list[WorldSnapshot] = []
    for i, entry in enumerate(recent):
        step = base + i
        actions: list[dict[str, Any]] = []
        for a in entry.get("actions", []) or []:
            aid = resolve_ref(str(a.get("actor", "")), id_by_name, valid_ids)
            agent = world.agents.get(aid)
            actions.append({
                "agent_id": aid,
                "agent_name": name_by_id.get(aid, aid),
                "is_main_character": bool(agent.is_main_character) if agent else False,
                "location_id": str(a.get("location", "")),
                "outcome": str(a.get("outcome", "")),
            })
        broadcasts = [{
            "content": str(b.get("content", "")),
            "source": "system",
            "broadcast_type": "world_event",
            "location_scope": b.get("location"),
            "severity": str(b.get("severity", "medium")),
        } for b in entry.get("broadcasts", []) or []]
        annotations: dict[str, list[dict[str, Any]]] = {}
        for ann in entry.get("annotations", []) or []:
            scope = str(ann.get("location", ""))
            actor = ann.get("actor")
            annotations.setdefault(scope, []).append({
                "content": str(ann.get("content", "")),
                "actor_id": resolve_ref(str(actor), id_by_name, valid_ids) if actor else None,
            })
        agent_relations = (
            _event_relations_block(scenario, id_by_name, valid_ids, name_by_id)
            if i == len(recent) - 1 else {}
        )
        snaps.append(WorldSnapshot(
            world_id=world_id, step=step, timestamp=datetime.now(),
            world_time=str(entry.get("time", "")),
            agent_relations=agent_relations,
            actions_this_step=actions,
            metadata={"broadcasts": broadcasts, "environment": {"step_annotations": annotations}},
        ))
    return snaps


def _event_prior_events(
    scenario: dict, directory, id_by_name: dict, valid_ids,
) -> list[WorldEvent]:
    """scenario.prior_events → already-injected WorldEvents (narrative references already resolved to names / place names)."""
    out: list[WorldEvent] = []
    for ev in scenario.get("prior_events", []) or []:
        affected_ids = [resolve_ref(str(n), id_by_name, valid_ids) for n in ev.get("affected", []) or []]
        loc = ev.get("location")
        out.append(WorldEvent(
            id=f"prior-{ev.get('step', 0)}",
            triggered_step=int(ev.get("step", 0)),
            narrative_desc=str(ev.get("narrative_desc", "")),
            is_positive=ev.get("is_positive"),
            dispatched_to=list(ev.get("dispatched_to") or ["broadcast"]),
            affected_names=[directory.agent_name(aid) for aid in affected_ids],
            location_label=directory.location_name(str(loc)) if loc else None,
            authored_by=Author.SYSTEM,
        ))
    return out


def _event_plan_view(plan, directory) -> dict[str, Any] | None:
    """Render an _EventPlan as readable outputs (recipients/location turned into names for the web view)."""
    if plan is None:
        return None
    view: dict[str, Any] = {
        "narrative_desc": plan.narrative_desc,
        "is_positive": plan.is_positive,
        "broadcast": None,
        "message": None,
    }
    if plan.broadcast is not None:
        view["broadcast"] = {
            "content": plan.broadcast.content,
            "severity": plan.broadcast.severity,
            "location_id": plan.broadcast.location_scope,
            "location_name": directory.location_name(plan.broadcast.location_scope) if plan.broadcast.location_scope else None,
        }
    if plan.message is not None:
        view["message"] = {
            "content": plan.message.content,
            "urgency": plan.message.urgency.value,
            "recipient_ids": list(plan.message.recipients),
            "recipient_names": [directory.agent_name(r) for r in plan.message.recipients],
        }
    view["spawn"] = None if plan.spawn is None else {
        "location_name": directory.location_name(plan.spawn.location_id),
        "entity_type": plan.spawn.entity_type,
        "name": plan.spawn.name,
        "description": plan.spawn.description,
        "content": plan.spawn.content,
        "observation": plan.spawn.observation,
    }
    view["alter"] = None if plan.alter is None else {
        "entity_name": directory.entity_name(plan.alter.entity_id),
        "state": plan.alter.new_state,
        "description": plan.alter.new_description,
        "content": plan.alter.new_content,
        "observation": plan.alter.observation,
    }
    view["destroy"] = None if plan.destroy is None else {
        "entity_name": directory.entity_name(plan.destroy.entity_id),
        "observation": plan.destroy.observation,
    }
    return view


async def run_event(
    container: Container,
    config: Config,
    world_id: str,
    *,
    scenario: dict | None = None,
    sink: InMemoryTraceSink | None = None,
    trace_dir: str = "./data/tuning_traces",
) -> int:
    """Isolated dry-run of the EventSystem narrative-injection stage for a restored world.

    See the module docstring; the briefing is seeded from scenario.recent / relations / prior_events.
    """
    trace_sink = sink if sink is not None else InMemoryTraceSink()
    traced_container, world, _, _ = await restore_traced(container, config, world_id, trace_sink)
    name_by_id = world.directory.all_agent_names()
    id_by_name = {v: k for k, v in name_by_id.items()}
    valid_ids = list(world.agents.keys())
    scenario = scenario or {}
    kind = scenario.get("kind", "plan")
    current_step = int(scenario.get("step", 60))

    # Main characters keep their real is_main_character; only position is overridden per scenario (to match its presence layout).
    for nm, loc in (scenario.get("locations", {}) or {}).items():
        aid = resolve_ref(str(nm), id_by_name, valid_ids)
        if aid in world.agents:
            world.agents[aid].personality.update_location(location=str(loc))

    snapshots = _event_seed_snapshots(
        world_id, current_step, scenario, world, id_by_name, valid_ids, name_by_id)
    snapshot_provider = InMemorySnapshotProvider()
    for snap in snapshots:
        await snapshot_provider.save(world_id, snap.step, snap)

    # The dry-run only generates and never commits; the dispatcher is assembled in production shape just to satisfy the constructor contract.
    message_system = MessageSystem(traced_container.message_provider, world_id=world_id)
    event_system = EventSystem(
        llm_router=traced_container.llm_router,
        snapshot_provider=snapshot_provider,
        dispatcher=InjectionDispatcher(
            broadcast_channel=BroadcastChannel(),
            message_system=message_system,
            mutation_channel=WorldMutationChannel(
                environment=world.environment,
                processor=ExecutionProcessor(
                    executor_registry=ActionExecutorRegistry(),
                    environment=world.environment,
                    message_system=message_system,
                    directory=world.directory,
                ),
                seconds_per_step=world.clock_config.seconds_per_step,
            ),
            directory=world.directory,
        ),
        directory=world.directory,
        settings=EventSettings(
            core_tension=world.analysis.core_tension,
            narrative_theme=world.analysis.narrative_theme,
        ),
    )
    for prior in _event_prior_events(scenario, world.directory, id_by_name, valid_ids):
        event_system._ledger.record(prior)  # noqa: SLF001 — dry-run seed

    world_time = GlobalClock(world.clock_config, start_step=current_step).current
    set_log_context(world_id=world_id, agent_id=None, step=str(current_step))
    brief = await event_system._build_narrative_summary(  # noqa: SLF001
        world_id, current_step, world.agents)

    error = None
    inject = None
    plan_view = None
    try:
        if kind == "gate":
            inject = await event_system._passes_llm_check(world_time, brief)  # noqa: SLF001
        else:
            plan = await event_system._generate_event_plan(  # noqa: SLF001
                current_step=current_step, world_time=world_time,
                narrative_summary=brief, all_agents=world.agents,
                locations=world.environment.space.all_places(),
                entities=world.environment.all_live_entities())
            plan_view = _event_plan_view(plan, world.directory)
    except Exception as exc:  # noqa: BLE001 — capture, never abort the suite
        error = str(exc)
    clear_log_context()

    # The plan's candidate lists (for deterministic checks that recipients/location indices resolve, and for the web view).
    candidate_agents = [name_by_id.get(aid, aid) for aid in world.agents.keys()]
    candidate_location_names = [loc.name for loc in _menu_places(
        world.environment.space.all_places(),
        Counter(a.personality.state.current_location for a in world.agents.values() if a.is_active),
        Counter(e.location_id for e in world.environment.all_live_entities() if e.location_id is not None),
        random.Random(current_step),
    )]

    outputs = {
        "kind": kind,
        "brief": brief,
        "inject": inject,
        "plan": plan_view,
        "candidate_agents": candidate_agents,
        "candidate_locations": candidate_location_names,
        "prior_event_descs": [e.narrative_desc for e in event_system.list_events()],
        "error": error,
    }
    inputs = {
        "kind": kind, "step": current_step,
        "world_time": world_time.time_label,
        "core_tension": world.analysis.core_tension,
        "narrative_theme": world.analysis.narrative_theme,
        "locations": scenario.get("locations", {}),
        "relations": scenario.get("relations", []),
        "recent": scenario.get("recent", []),
        "prior_events": scenario.get("prior_events", []),
    }
    trace_sink.record_phase(PhaseTrace(
        world_id=world_id, phase="stage.event", step=current_step, agent_id=None,
        inputs=inputs, outputs=outputs, timestamp=datetime.now().isoformat()))
    logger.info("tuning_event_dry_run_complete",
                extra={"world_id": world_id, "step": current_step, "kind": kind})
    return current_step
