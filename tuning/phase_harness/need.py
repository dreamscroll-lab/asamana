"""Need stage dry-run (``run_need_engine``)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from config.models import Config
from core.container import Container
from core.context import clear_log_context, set_log_context
from core.logging import get_logger
from core.prompts import render_perceived_signals

from tuning.fixtures import load_fixture, save_fixture
from tuning.phase_harness.common import emotion_view, restore_traced
from tuning.phase_harness.scenario import (
    apply_ambient_from_scenario, broadcasts_from_scenario, external_goals_from_scenario,
    inboxes_from_scenario, presence_name, spatials_for,
)
from tuning.trace import InMemoryTraceSink, JsonlTraceSink, PhaseTrace

logger = get_logger(__name__)


def _need_eval_view(ev) -> dict[str, Any]:
    """JSON-safe view of a NeedEvaluation (dominant / scores / goals / external)."""
    def _nv(n) -> str | None:
        return n.value if n is not None and hasattr(n, "value") else (str(n) if n is not None else None)
    return {
        "dominant_need": _nv(ev.dominant_need),
        "scores": {_nv(nt): round(v, 3) for nt, v in ev.scores.items()},
        "short_term_goals": list(ev.short_term_goals),
        "goal_entities": [
            {
                "text": g.text,
                "related_need": _nv(g.related_need),
                "status": g.status.value if hasattr(g.status, "value") else str(g.status),
                # Emit a boolean, not due_step: this view is rendered into the judge prompt, and a step integer
                # there is a layer leak. "Has a time been set" is all the check needs to ask.
                "has_due": g.due_step is not None,
            }
            for g in ev.short_term_goal_entities
        ],
        "external_goals": [
            {
                "text": g.text,
                "urgency": g.urgency.value,
                "drive_type": g.drive_type.value,
                "related_need": _nv(g.related_need),
            }
            for g in ev.external_goals
        ],
    }


async def run_need_engine(
    container: Container,
    config: Config,
    world_id: str,
    *,
    scenario: dict | None = None,
    sink: InMemoryTraceSink | None = None,
    trace_dir: str = "./data/tuning_traces",
) -> int:
    """Isolated dry-run of ``NeedEngine.run`` (motivation) for a restored world.

    Chains the real upstream appraisal (``_assess_perception_emotion`` → emotion +
    need_activation) into ``need_engine.run`` with ``force_goal_update=True`` so each scenario
    re-generates short-term goals. Calls the production methods directly — **zero production
    change**. ``run`` mutates the agent's goal entities / need intensities in-memory, so each
    invocation restores its own world; nothing is persisted.
    """
    trace_sink = sink if sink is not None else JsonlTraceSink(trace_dir, segment="need")
    if isinstance(trace_sink, JsonlTraceSink):
        trace_sink.truncate(world_id)
    if scenario is None:
        scenario = load_fixture(trace_dir, world_id, "need_scenario")
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
            need_relevance = appraisal.need_activation
            # Mirror _build_internal_context: channel-labeled signal lines + perceived_relations.
            perceived_signal_texts = render_perceived_signals(spatial=sp, inbox=inbox, broadcasts=agent_bcs)
            ev = await agent.need_engine.run(
                current_step=run_step,
                personality=agent.personality,
                visible_agents=sp.visible_agent_ids,
                pending_messages=len(inbox),
                emotion=emotion,
                force_goal_update=True,
                world_time_hour=sp.world_time_hour,
                perceived_signal_texts=perceived_signal_texts,
                external_goals=goals,
                need_relevance=need_relevance,
                perceived_relations=perceived_relations,
            )
        except Exception as exc:  # noqa: BLE001 — one agent's failure must not abort the run
            logger.warning("need_engine_dryrun_failed",
                           extra={"world_id": world_id, "agent_id": aid, "error": str(exc)})
            continue
        soul = agent.personality.soul
        trace_sink.record_phase(PhaseTrace(
            world_id=world_id,
            phase="stage.need",
            step=run_step,
            agent_id=aid,
            inputs={
                "agent_name": name_by_id.get(aid, aid),
                "is_main_character": agent.is_main_character,
                "location": sp.location_view.name or sp.location_id,
                "role": getattr(soul, "role", ""),
                "core_traits": list(getattr(soul, "core_traits", [])),
                "long_term_goals": list(agent.personality.state.long_term_goals),
                "emotion": emotion_view(emotion),
                "need_activation": {nt.value: round(v, 3) for nt, v in need_relevance.items()},
                "injected_broadcasts": [
                    {"content": b.content, "severity": b.severity, "location_scope": b.location_scope}
                    for b in agent_bcs
                ],
                "injected_messages": [
                    {"from": m.sender_name or m.sender_id, "content": m.content, "urgency": m.urgency.value}
                    for m in inbox
                ],
                "injected_ambient": [{"content": ev2.content, "strength": ev2.strength} for ev2 in sp.ambient_events],
                "external_goals": [
                    {"text": g.text, "urgency": g.urgency.value, "drive_type": g.drive_type.value,
                     "related_need": g.related_need.value if g.related_need else None}
                    for g in goals
                ],
                "visible_agents": [presence_name(sp, v) for v in sp.visible_agent_ids],
            },
            outputs={"agent_name": name_by_id.get(aid, aid), **_need_eval_view(ev)},
            timestamp=datetime.now().isoformat(),
        ))
    clear_log_context()
    if scenario:
        save_fixture(trace_dir, world_id, "need_scenario", scenario)
    logger.info(
        "tuning_need_engine_dry_run_complete",
        extra={"world_id": world_id, "step": run_step, "agent_count": len(active)},
    )
    return run_step
