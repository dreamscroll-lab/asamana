"""Perception-emotion stage dry-run (``run_perception_emotion``)."""

from __future__ import annotations

from datetime import datetime

from config.models import Config
from core.container import Container
from core.context import clear_log_context, set_log_context
from core.logging import get_logger

from tuning.fixtures import load_fixture, save_fixture
from tuning.phase_harness.common import emotion_view, restore_traced
from tuning.phase_harness.scenario import (
    apply_ambient_from_scenario, broadcasts_from_scenario, external_goals_from_scenario,
    inboxes_from_scenario, presence_name, spatials_for,
)
from tuning.trace import InMemoryTraceSink, JsonlTraceSink, PhaseTrace

logger = get_logger(__name__)


async def run_perception_emotion(
    container: Container,
    config: Config,
    world_id: str,
    *,
    scenario: dict | None = None,
    sink: InMemoryTraceSink | None = None,
    trace_dir: str = "./data/tuning_traces",
) -> int:
    """Isolated dry-run of ``Agent._assess_perception_emotion`` for a restored world.

    The pre-decision instinctive emotion (every agent takes the LLM path). Calls the production
    method directly on the restored (traced) agent, replicating ``perceive_step``'s relation
    computation so it sees the same ``perceived_relations`` as at runtime. Nothing is persisted.
    """
    trace_sink = sink if sink is not None else JsonlTraceSink(trace_dir, segment="perception_emotion")
    if isinstance(trace_sink, JsonlTraceSink):
        trace_sink.truncate(world_id)
    if scenario is None:
        scenario = load_fixture(trace_dir, world_id, "perception_emotion_scenario")
    _, world, run_step, world_time_label = await restore_traced(container, config, world_id, trace_sink)
    spatials = spatials_for(world, run_step, world_time_label)
    name_by_id = world.directory.all_agent_names()
    active = {aid: world.agents[aid] for aid in spatials}
    broadcasts = broadcasts_from_scenario(scenario, run_step)
    inboxes = inboxes_from_scenario(scenario, world_id, run_step, active, name_by_id)
    apply_ambient_from_scenario(scenario, spatials, active, name_by_id)
    ext_goals = external_goals_from_scenario(scenario, active, name_by_id)

    for aid, agent in active.items():
        sp = spatials[aid]
        agent_bcs = [b for b in broadcasts if b.location_scope is None or b.location_scope == sp.location_id]
        inbox = inboxes.get(aid, [])
        goals = ext_goals.get(aid, [])
        # Reuse the production relation computation (visible + senders); no state mutated.
        perceived_relations = await agent._perceive_relevant_relations(sp, inbox, current_step=run_step)
        set_log_context(world_id=world_id, agent_id=aid, step=str(run_step))
        try:
            appraisal = await agent._assess_perception_emotion(
                sp, inbox, agent_bcs, goals, perceived_relations
            )
            emotion = appraisal.emotion
            need_activation = {nt.value: round(v, 3) for nt, v in appraisal.need_activation.items()}
        except Exception as exc:  # noqa: BLE001 — one agent's failure must not abort the run
            logger.warning("perception_emotion_dryrun_failed",
                           extra={"world_id": world_id, "agent_id": aid, "error": str(exc)})
            emotion = None
            need_activation = {}
        soul = agent.personality.soul
        trace_sink.record_phase(PhaseTrace(
            world_id=world_id,
            phase="stage.perception_emotion",
            step=run_step,
            agent_id=aid,
            inputs={
                "agent_name": name_by_id.get(aid, aid),
                "is_main_character": agent.is_main_character,
                # Pre-perception emotion = the "current mood" baseline fed into the prompt (only included when intensity > 0.3).
                "current_emotion": emotion_view(agent.personality.state.emotion),
                "location": sp.location_view.name or sp.location_id,
                "role": getattr(soul, "role", ""),
                "core_traits": list(getattr(soul, "core_traits", [])),
                "background": (getattr(soul, "background", "") or "")[:150],
                "injected_broadcasts": [
                    {"content": b.content, "severity": b.severity, "location_scope": b.location_scope}
                    for b in agent_bcs
                ],
                "injected_messages": [
                    {"from": m.sender_name or m.sender_id, "content": m.content, "urgency": m.urgency.value}
                    for m in inbox
                ],
                "injected_ambient": [{"content": ev.content, "strength": ev.strength} for ev in sp.ambient_events],
                "external_goals": [
                    {"text": g.text, "urgency": g.urgency.value, "drive_type": g.drive_type.value} for g in goals
                ],
                "visible_agents": [presence_name(sp, v) for v in sp.visible_agent_ids],
                "relations": [
                    {
                        "name": r.target_agent_name or r.target_agent_id,
                        "labels": list(r.labels),
                        "trust": round(r.trust, 2),
                        "affection": round(r.affection, 2),
                        "history": r.history_summary or "",
                    }
                    for r in perceived_relations
                ],
            },
            outputs={
                "agent_name": name_by_id.get(aid, aid),
                "emotion": emotion_view(emotion),
                "need_activation": need_activation,
            },
            timestamp=datetime.now().isoformat(),
        ))
    clear_log_context()
    if scenario:
        save_fixture(trace_dir, world_id, "perception_emotion_scenario", scenario)
    logger.info(
        "tuning_perception_emotion_dry_run_complete",
        extra={"world_id": world_id, "step": run_step, "agent_count": len(active)},
    )
    return run_step
