"""World-pressure stage dry-run (``run_world_pressure``)."""

from __future__ import annotations

from datetime import datetime

from config.models import Config
from core.container import Container
from core.context import clear_log_context, set_log_context
from core.interfaces.trace import to_jsonable
from core.logging import get_logger
from engine.world_pressure import WorldPressureEvaluator

from tuning.fixtures import load_fixture, save_fixture
from tuning.phase_harness.common import restore_traced
from tuning.phase_harness.scenario import (
    apply_ambient_from_scenario, broadcasts_from_scenario, inboxes_from_scenario,
    presence_name, spatials_for,
)
from tuning.trace import InMemoryTraceSink, JsonlTraceSink, PhaseTrace

logger = get_logger(__name__)


async def run_world_pressure(
    container: Container,
    config: Config,
    world_id: str,
    *,
    scenario: dict | None = None,
    sink: InMemoryTraceSink | None = None,
    trace_dir: str = "./data/tuning_traces",
) -> int:
    """Isolated dry-run of WorldPressureEvaluator for a restored world.

    World-level: one LLM call evaluates external pressure on every agent. Pressure
    is signal-gated — a fresh step-1 world has no inbox/broadcast/ambient signal,
    so a ``scenario`` (injected broadcasts) is what makes this stage tunable. The
    scenario is frozen as the ``scenario`` fixture so the same input drives every
    prompt-tuning re-run; the resulting external_goals are frozen as the
    ``external_goals`` fixture for downstream stages.
    """
    trace_sink = sink if sink is not None else JsonlTraceSink(trace_dir, segment="world_pressure")
    if isinstance(trace_sink, JsonlTraceSink):
        trace_sink.truncate(world_id)
    # Resolve scenario: explicit arg wins; else reuse the saved fixture (stable re-runs).
    if scenario is None:
        scenario = load_fixture(trace_dir, world_id, "scenario")
    traced_container, world, run_step, world_time_label = await restore_traced(
        container, config, world_id, trace_sink
    )
    spatials = spatials_for(world, run_step, world_time_label)
    name_by_id = world.directory.all_agent_names()
    active = {aid: world.agents[aid] for aid in spatials}
    injected = broadcasts_from_scenario(scenario, run_step)
    inboxes = inboxes_from_scenario(scenario, world_id, run_step, active, name_by_id)
    apply_ambient_from_scenario(scenario, spatials, active, name_by_id)  # mutates spatials in place

    evaluator = WorldPressureEvaluator(llm_router=traced_container.llm_router)
    set_log_context(world_id=world_id, step=str(run_step))  # world-level, no single agent
    try:
        pressure_map = await evaluator.evaluate(
            agents=active,
            agent_inboxes=inboxes,
            broadcasts=injected,
            world_time_label=world_time_label,
            agent_spatials=spatials,
        )
    except Exception as exc:  # noqa: BLE001
        clear_log_context()
        logger.warning("world_pressure_dryrun_failed", extra={"world_id": world_id, "error": str(exc)})
        raise
    clear_log_context()

    bc_view = [{"content": b.content, "severity": b.severity, "location_scope": b.location_scope} for b in injected]
    goals_json = {aid: [to_jsonable(g) for g in goals] for aid, goals in pressure_map.items()}

    # Per-agent view of co-located others (role + this agent's own relation) — surfaced
    # so the judge / deterministic checks can spot ally-as-threat.
    co_located_by_aid: dict[str, list[dict]] = {}
    for aid in active:
        sp = spatials[aid]
        rels = []
        for vid in sp.visible_agent_ids:
            rel = None
            try:
                rel = await world.agents[aid].agent_store.load_relation(world_id, aid, vid)
            except Exception:  # noqa: BLE001
                rel = None
            other = world.agents.get(vid)
            osoul = other.personality.soul if other is not None else None
            rels.append({
                "name": presence_name(sp, vid),
                "role": osoul.role if osoul is not None else "",
                "traits": list(osoul.core_traits) if osoul is not None else [],
                "background": (osoul.background or "")[:150] if osoul is not None else "",
                "trust": round(rel.trust_objective, 2) if rel is not None else None,
                "affection": round(rel.affection_objective, 2) if rel is not None else None,
                "labels": list(rel.labels) if rel is not None else [],
            })
        co_located_by_aid[aid] = rels

    for aid in active:
        sp = spatials[aid]
        msg_view = [{"from": m.sender_name or m.sender_id, "content": m.content, "urgency": m.urgency.value}
                    for m in inboxes.get(aid, [])]
        amb_view = [{"content": ev.content, "strength": ev.strength} for ev in sp.ambient_events]
        trace_sink.record_phase(
            PhaseTrace(
                world_id=world_id,
                phase="stage.world_pressure",
                step=run_step,
                agent_id=aid,
                inputs={
                    "agent_name": name_by_id.get(aid, aid),
                    "location": sp.location_view.name or sp.location_id,
                    "location_id": sp.location_id,
                    "visible_agents": [presence_name(sp, v) for v in sp.visible_agent_ids],
                    "world_time": world_time_label,
                    "injected_broadcasts": bc_view,
                    "injected_messages": msg_view,
                    "injected_ambient": amb_view,
                    "co_located": co_located_by_aid.get(aid, []),
                },
                outputs={"agent_name": name_by_id.get(aid, aid), "external_goals": goals_json.get(aid, [])},
                timestamp=datetime.now().isoformat(),
            )
        )
    # Freeze the scenario (stable re-runs) and external_goals (downstream input).
    if scenario:
        save_fixture(trace_dir, world_id, "scenario", scenario)
    save_fixture(trace_dir, world_id, "external_goals", {"step": run_step, "goals": goals_json})
    logger.info(
        "tuning_world_pressure_dry_run_complete",
        extra={"world_id": world_id, "step": run_step, "agent_count": len(active)},
    )
    return run_step
