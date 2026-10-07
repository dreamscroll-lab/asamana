"""Feedback stage (completion-cell landing + target effects) dry-run"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agent.agent import AgentStepPlan
from agent.goals import GoalEntity
from agent.need import NeedEvaluation, NeedType
from config.models import Config
from core.container import Container
from core.context import clear_log_context, set_log_context
from core.logging import get_logger
from core.interfaces.action import (
    ActionResult, ActionTarget, ActionType, AgentAction, Ref, TargetAgentEffect,
)
from core.interfaces.condition import BodyCondition

from tuning.phase_harness.common import emotion_view, isolate_agent, restore_traced
from tuning.phase_harness.scenario import apply_action_scene, resolve_ref
from tuning.trace import InMemoryTraceSink, PhaseTrace

logger = get_logger(__name__)


def _need_intensities(agent) -> dict[str, float]:
    # The source of truth for drives is personality.state.need_intensities. profile.active_needs[].intensity
    # is only the build-time seed and isn't updated at runtime, so reading it gives stale values.
    return {k: round(v, 3) for k, v in agent.personality.state.need_intensities.items()}


def _goals_view(agent) -> list[dict[str, Any]]:
    return [
        {"text": g.text, "status": g.status.value if hasattr(g.status, "value") else str(g.status)}
        for g in agent.personality.state.short_term_goal_entities
    ]


async def _relations_view(agent, target_ids: list[str], name_by_id: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tid in target_ids:
        if not tid:
            continue
        rel = await agent.relation_system.get_or_create(tid)
        out[name_by_id.get(tid, tid)] = {
            "trust": round(rel.trust_objective, 3),
            "affection": round(rel.affection_objective, 3),
            "labels": list(rel.labels),
        }
    return out


def _agent_state_view(agent, target_ids: list[str], name_by_id: dict) -> dict[str, Any]:
    st = agent.personality.state
    return {
        "emotion": emotion_view(st.emotion),
        "needs": _need_intensities(agent),
        "vitality": round(st.vitality, 3),
        "goals": _goals_view(agent),
        "is_active": agent.is_active,
    }


def _build_result_from_spec(actor_id: str, spec: dict, run_step: int, id_by_name: dict, valid_ids) -> ActionResult:
    """Construct an executor-style ActionResult from a scenario spec (the feedback INPUT)."""
    atype = ActionType(spec.get("type", "work"))
    target = ActionTarget()
    rel_updates: list[tuple[str, float, float]] = []
    for u in spec.get("relation_updates", []) or []:
        tid = resolve_ref(str(u.get("target", "")), id_by_name, valid_ids)
        rel_updates.append((tid, float(u.get("trust_delta", 0.0)), float(u.get("affection_delta", 0.0))))
        target = ActionTarget(acts_on=[Ref.agent(tid)])
    effects: list[TargetAgentEffect] = []
    for e in spec.get("target_effects", []) or []:
        effects.append(_build_effect_from_spec(e, id_by_name, valid_ids))
    action = AgentAction(
        agent_id=actor_id, step=run_step, action_type=atype,
        action_description=spec.get("description", ""),
        expected_outcome=spec.get("expected_outcome", ""),
        estimated_steps=1, target=target,
    )
    return ActionResult(
        action=action,
        expected_outcome=spec.get("expected_outcome", ""),
        outcome=spec.get("outcome", ""),
        succeeded=bool(spec.get("succeeded", True)),
        failure_reason=spec.get("failure_reason", ""),
        factual_memory=spec.get("factual_memory", ""),
        relation_updates=rel_updates,
        target_effects=effects,
        vitality_damage=float(spec.get("vitality_damage", 0.0)),
    )


def _build_effect_from_spec(spec: dict, id_by_name: dict, valid_ids) -> TargetAgentEffect:
    rel = None
    if spec.get("relation_toward_actor"):
        r = spec["relation_toward_actor"]
        rel = (resolve_ref(str(r.get("actor", "")), id_by_name, valid_ids),
               float(r.get("trust_delta", 0.0)), float(r.get("affection_delta", 0.0)))
    return TargetAgentEffect(
        agent_id=resolve_ref(str(spec.get("agent", "")), id_by_name, valid_ids),
        factual_memory=spec.get("factual_memory", ""),
        emotion_type=spec.get("emotion_type"),
        emotion_intensity=float(spec.get("emotion_intensity", 0.0)),
        emotion_valence=float(spec.get("emotion_valence", 0.0)),
        relation_toward_actor=rel,
        vitality_damage=float(spec.get("vitality_damage", 0.0)),
        # Restore the three states per the contract: condition given = apply, condition_cleared = clear, neither = leave alone.
        condition_set=(
            BodyCondition(
                description=str(spec["condition"]),
                source_agent_id=resolve_ref(str(spec.get("condition_from", "")), id_by_name, valid_ids),
                since_step=int(spec.get("condition_since_step", 0)),
                until_step=spec.get("condition_until_step"),
            )
            if spec.get("condition") else None
        ),
        condition_cleared=bool(spec.get("condition_cleared", False)),
    )


async def run_feedback(
    container: Container,
    config: Config,
    world_id: str,
    *,
    scenario: dict | None = None,
    sink: InMemoryTraceSink | None = None,
    trace_dir: str = "./data/tuning_traces",
) -> int:
    """Isolated dry-run of the feedback stage (landing) for a restored world.

    Drives the real ``begin_ongoing_step`` → ``finalize_ongoing_action`` (kind=self) or ``agent.apply_target_effect``
    (kind=target); persistence is shadowed by _DryRunAgentStore + InMemoryVectorStore so
    the baseline world's ./data is never written. Captures before/after agent state.
    """
    trace_sink = sink if sink is not None else InMemoryTraceSink()
    traced_container, world, run_step, _ = await restore_traced(container, config, world_id, trace_sink)
    name_by_id = world.directory.all_agent_names()
    id_by_name = {v: k for k, v in name_by_id.items()}
    valid_ids = list(world.agents.keys())
    scenario = scenario or {}

    await apply_action_scene(scenario, world, run_step, id_by_name)

    kind = scenario.get("kind", "self")
    set_log_context(world_id=world_id, step=str(run_step))
    if kind == "target":
        await _run_feedback_target(scenario, world, run_step, name_by_id, id_by_name, valid_ids, trace_sink, world_id)
    else:
        await _run_feedback_self(scenario, world, run_step, name_by_id, id_by_name, valid_ids, trace_sink, world_id)
    clear_log_context()
    logger.info("tuning_feedback_dry_run_complete", extra={"world_id": world_id, "step": run_step})
    return run_step


def _rel_targets_of_result(result: ActionResult) -> list[str]:
    ids = [t for (t, _td, _ad) in result.relation_updates]
    ids += [e.agent_id for e in result.target_effects]
    return list(dict.fromkeys(i for i in ids if i))


async def _run_feedback_self(
    scenario, world, run_step, name_by_id, id_by_name, valid_ids, sink, world_id,
) -> None:
    actor_id = resolve_ref(scenario["actor"], id_by_name, valid_ids)
    agent = world.agents[actor_id]
    isolate_agent(agent, name_by_id)
    # Optional: plant explicit short-term goals so a scenario can probe goal-progress
    # judgment (a multi-action goal should stay active; a single-action one should complete).
    goal_texts = (scenario.get("scene", {}) or {}).get("goals")
    if goal_texts:
        # created_step=run_step, NOT 0: planted at step 0 into a step-N world they'd be N steps old,
        # trip the stall fallback (_SHORT_TERM_GOAL_STALL_STEPS) and be force-marked FAILED,
        # clobbering the judge verdict these scenarios probe.
        agent.personality.set_short_term_goal_entities([
            GoalEntity(id=f"stg-fb-{i}", text=str(t), goal_type="short_term", created_step=run_step)
            for i, t in enumerate(goal_texts)
        ])
    spec = scenario["result"]
    dom = spec.get("dominant_need")
    dominant_need = NeedType(dom) if dom else None
    result = _build_result_from_spec(actor_id, spec, run_step, id_by_name, valid_ids)
    rel_targets = _rel_targets_of_result(result)

    before = _agent_state_view(agent, rel_targets, name_by_id)
    before["relations"] = await _relations_view(agent, rel_targets, name_by_id)
    before_mem = set(agent.memory_system._entries.keys())  # noqa: SLF001

    active_needs = [n for n in agent.personality.current_needs() if not n.is_hidden]
    sp = world.environment.spatial_for(agent_id=actor_id, step=run_step)
    state = agent.personality.state  # reflects any short-term goals planted above
    plan = AgentStepPlan(
        agent_id=actor_id, step=run_step, spatial=sp, inbox=[], broadcasts=[],
        need_evaluation=NeedEvaluation(
            dominant_need=dominant_need, scores={},
            active_needs=active_needs,
            short_term_goals=[g.text for g in state.short_term_goal_entities],
            long_term_goals=list(state.long_term_goals), prompt_context="",
            short_term_goal_entities=list(state.short_term_goal_entities)),
        action=result.action,
    )
    try:
        # A born-zero (duration-1) self action begins and finalizes on the same step; finalize
        # lands all feedback (emotion/need/relation/memory/goal-progress).
        await agent.begin_ongoing_step(plan=plan, estimated_steps=1)
        await agent.finalize_ongoing_action(result=result, step=run_step)
        error = None
    except Exception as exc:  # noqa: BLE001
        logger.warning("feedback_self_failed", extra={"world_id": world_id, "actor": actor_id, "error": str(exc)})
        error = str(exc)

    after = _agent_state_view(agent, rel_targets, name_by_id)
    after["relations"] = await _relations_view(agent, rel_targets, name_by_id)
    new_memories = [
        {"stream": m.stream.value, "content": m.stored_content}
        for mid, m in agent.memory_system._entries.items() if mid not in before_mem  # noqa: SLF001
    ]
    inputs = _feedback_input_view(agent, spec, dominant_need, name_by_id, kind="self")
    sink.record_phase(PhaseTrace(
        world_id=world_id, phase="stage.feedback", step=run_step, agent_id=actor_id,
        inputs=inputs,
        outputs={"kind": "self", "before": before, "after": after,
                 "deltas": _compute_deltas(before, after), "new_memories": new_memories,
                 "error": error},
        timestamp=datetime.now().isoformat(),
    ))


async def _run_feedback_target(
    scenario, world, run_step, name_by_id, id_by_name, valid_ids, sink, world_id,
) -> None:
    spec = scenario["effect"]
    effect = _build_effect_from_spec(spec, id_by_name, valid_ids)
    target_id = effect.agent_id
    from_id = resolve_ref(str(scenario.get("from_agent", "")), id_by_name, valid_ids)
    agent = world.agents.get(target_id)
    if agent is None:
        sink.record_phase(PhaseTrace(
            world_id=world_id, phase="stage.feedback", step=run_step, agent_id=target_id,
            inputs={"kind": "target", "error": f"target {target_id} not in world"},
            outputs={"kind": "target", "error": "target missing"}, timestamp=datetime.now().isoformat()))
        return
    isolate_agent(agent, name_by_id)
    rel_targets = [from_id] if from_id else []

    before = _agent_state_view(agent, rel_targets, name_by_id)
    before["relations"] = await _relations_view(agent, rel_targets, name_by_id)
    before_mem = set(agent.memory_system._entries.keys())  # noqa: SLF001

    try:
        await agent.apply_target_effect(effect, from_agent_id=from_id, step=run_step)
        error = None
    except Exception as exc:  # noqa: BLE001
        logger.warning("feedback_target_failed", extra={"world_id": world_id, "target": target_id, "error": str(exc)})
        error = str(exc)

    after = _agent_state_view(agent, rel_targets, name_by_id)
    after["relations"] = await _relations_view(agent, rel_targets, name_by_id)
    new_memories = [
        {"stream": m.stream.value, "content": m.stored_content}
        for mid, m in agent.memory_system._entries.items() if mid not in before_mem  # noqa: SLF001
    ]
    sink.record_phase(PhaseTrace(
        world_id=world_id, phase="stage.feedback", step=run_step, agent_id=target_id,
        inputs=_feedback_input_view(agent, spec, None, name_by_id, kind="target",
                                    from_name=name_by_id.get(from_id, from_id)),
        outputs={"kind": "target", "before": before, "after": after,
                 "deltas": _compute_deltas(before, after), "new_memories": new_memories,
                 "error": error},
        timestamp=datetime.now().isoformat(),
    ))


def _feedback_input_view(agent, spec: dict, dominant_need, name_by_id: dict, *, kind: str, from_name: str = "") -> dict[str, Any]:
    soul = agent.personality.soul
    base = {
        "kind": kind,
        "actor_name": name_by_id.get(agent.agent_id, agent.agent_id),
        "is_main_character": agent.is_main_character,
        "core_traits": list(getattr(soul, "core_traits", [])),
        "core_values": list(getattr(soul, "core_values", [])),
        "spec": spec,
    }
    if kind == "self":
        base["dominant_need"] = dominant_need.value if dominant_need else None
    else:
        base["from_name"] = from_name
    return base


def _compute_deltas(before: dict, after: dict) -> dict[str, Any]:
    d: dict[str, Any] = {}
    be, ae = before.get("emotion") or {}, after.get("emotion") or {}
    if be.get("primary") != ae.get("primary") or be.get("intensity") != ae.get("intensity") or be.get("valence") != ae.get("valence"):
        d["emotion"] = {"from": be, "to": ae}
    need_d = {k: round(after["needs"][k] - before["needs"].get(k, 0.0), 3)
              for k in after.get("needs", {}) if abs(after["needs"][k] - before["needs"].get(k, 0.0)) > 1e-6}
    if need_d:
        d["need_shift"] = need_d
    if abs(after.get("vitality", 0.0) - before.get("vitality", 0.0)) > 1e-6:
        d["vitality"] = {"from": before.get("vitality"), "to": after.get("vitality")}
    if before.get("is_active") != after.get("is_active"):
        d["is_active"] = {"from": before.get("is_active"), "to": after.get("is_active")}
    bg = {g["text"]: g["status"] for g in before.get("goals", [])}
    transitions = [{"goal": g["text"], "from": bg.get(g["text"]), "to": g["status"]}
                   for g in after.get("goals", []) if bg.get(g["text"]) != g["status"]]
    if transitions:
        d["goal_transitions"] = transitions
    rel_d = {}
    for name, ra in after.get("relations", {}).items():
        rb = before.get("relations", {}).get(name, {})
        td, ad = round(ra["trust"] - rb.get("trust", 0.5), 3), round(ra["affection"] - rb.get("affection", 0.0), 3)
        if abs(td) > 1e-6 or abs(ad) > 1e-6:
            rel_d[name] = {"trust_delta": td, "affection_delta": ad}
    if rel_d:
        d["relation_deltas"] = rel_d
    return d
