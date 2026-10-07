"""Action stage (executor lifecycle + arbitration) dry-run harness"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agent.agent import AgentStepPlan
from agent.need import NeedEvaluation, NeedState, NeedType
from agent.decision import ActionIntent
from config.models import Config
from core.container import Container
from core.context import clear_log_context, set_log_context
from core.logging import get_logger
from core.interfaces.action import (
    ErrandOrder, ActionResult, ActionTarget, ActionType, AgentAction, Ref,
)
from core.prompts import render_condition
from engine.executors import build_default_registry
from engine.presence import attach_presence
from engine.executors.base import ActionExecutionState
from engine.message_system import MessageSystem
from world import World

from tuning.phase_harness.common import emotion_view, make_action_runtime, restore_traced
from tuning.phase_harness.scenario import apply_action_scene, resolve_location_ref, resolve_ref
from tuning.plan_view import target_view
from tuning.trace import InMemoryTraceSink, PhaseTrace

logger = get_logger(__name__)


_INTERACTIVE_TYPES = {ActionType.TALK, ActionType.SEND_MESSAGE, ActionType.COVERT}


def _result_view(r: ActionResult, name_by_id: dict | None = None) -> dict[str, Any]:
    """Serialize an ActionResult for trace/judge (narrative-layer fields)."""
    nm = name_by_id or {}
    return {
        "actor": nm.get(r.action.agent_id, r.action.agent_id),
        # Some checks branch on type (e.g. "accepted or not" only applies to errands); without it they'd have to guess.
        "action_type": getattr(r.action.action_type, "value", str(r.action.action_type)),
        "outcome": r.outcome,
        "observation": r.observation,
        "succeeded": r.succeeded,
        "failure_reason": r.failure_reason,
        "detected": r.detected,
        "factual_memory": r.factual_memory,
        "dialogue": list(r.dialogue),
        "vitality_damage": r.vitality_damage,
        "relation_updates": [
            {"target": t, "target_name": nm.get(t, t), "trust_delta": td, "affection_delta": ad}
            for (t, td, ad) in r.relation_updates
        ],
        "target_effects": [
            {
                "agent_id": e.agent_id, "agent_name": nm.get(e.agent_id, e.agent_id),
                "factual_memory": e.factual_memory,
                "emotion_type": e.emotion_type, "emotion_intensity": e.emotion_intensity,
                "emotion_valence": e.emotion_valence,
                "relation_toward_actor": list(e.relation_toward_actor) if e.relation_toward_actor else None,
                "vitality_damage": e.vitality_damage,
                # The condition is narrative text. If it isn't included here it never shows up in the trace, never
                # reaches the judge, and skips validation's id/step leak scan, as if the field didn't exist in tuning.
                "condition": e.condition_set.description if e.condition_set else "",
                "condition_until_step": e.condition_set.until_step if e.condition_set else None,
                "condition_cleared": e.condition_cleared,
            }
            for e in r.target_effects
        ],
        "entity_state_changes": [
            {"entity_id": c.entity_id, "new_state": c.new_state, "owner_id": c.owner_id,
             "location_id": c.location_id, "perception": c.perception}
            for c in r.entity_state_changes
        ],
        # Made things: their name / description go into other agents' reachable lists and get embedded,
        # so for the same reason they must be included; otherwise they skip the judge and the id/step leak scan.
        "entity_spawns": [
            {"name": s.name, "description": s.description, "entity_type": s.entity_type,
             "state": s.state, "holder_id": s.holder_id, "is_public": s.is_public,
             "perception": s.perception}
            for s in r.entity_spawns
        ],
        # Delegated errands exist only when accepted; when declined this must be empty. The message they
        # carry is privileged content: including it gives the judge evidence and puts it under the id/step leak scan (same reason as above).
        "errand_orders": [
            {
                "bearer": nm.get(o.npc_id, o.npc_id),
                "destination_id": o.destination_id,
                "item_id": o.item_id,
                "recipient": nm.get(o.recipient_id, o.recipient_id),
                "message": o.message,
            }
            for o in r.errand_orders
        ],
        "npc_effects": [
            {"npc_id": e.npc_id,
             "condition": e.condition_set.description if e.condition_set else "",
             "condition_cleared": e.condition_cleared}
            for e in r.npc_effects
        ],
    }


def _state_view(s: ActionExecutionState) -> dict[str, Any]:
    return {
        "action_type": s.action_type.value,
        "initiator_id": s.initiator_id,
        "participant_ids": list(s.participant_ids),
        "estimated_steps": s.estimated_steps,
        "purpose": s.purpose,
        "expected_outcome": s.expected_outcome,
    }


def _build_agent_action(
    actor_id: str, spec: dict, *, env, run_step: int, id_by_name: dict, valid_ids,
) -> AgentAction:
    atype = ActionType(spec["type"])
    tgt = spec.get("target")
    target = ActionTarget()
    if atype in _INTERACTIVE_TYPES:
        if tgt:
            other = Ref.agent(resolve_ref(tgt, id_by_name, valid_ids))
            # A conversation also takes the other party's turn (arbitration absorbs it as INVITE); messages and covert actions don't occupy anyone else's body.
            target = ActionTarget(
                acts_on=[other], claims=[other] if atype is ActionType.TALK else [],
            )
    elif atype == ActionType.MOVE:
        if str(tgt or "") == "$ADJACENT":
            loc = env.space.get(env.get_body_location(actor_id))
            dest = next(iter(loc.connections.keys()), None) if loc else None
            target = ActionTarget(acts_on=[Ref.place(dest)] if dest else [])
        else:
            target = ActionTarget(acts_on=[Ref.place(resolve_location_ref(str(tgt or ""), env))])
    elif atype == ActionType.PHYSICAL:
        it = spec.get("item_type", "agent")
        if it == "agent":
            target = ActionTarget(
                acts_on=[Ref.agent(resolve_ref(str(tgt or ""), id_by_name, valid_ids))]
            )
        else:
            # recipient: who receives a delivery (only meaningful on the entity channel); pairs with the scenario's holder.
            rcpt = spec.get("recipient")
            target = ActionTarget(
                acts_on=[Ref.entity(str(tgt or ""), it)],
                reaches=(
                    [Ref.agent(resolve_ref(str(rcpt), id_by_name, valid_ids))] if rcpt else []
                ),
            )
    intent = None
    if atype == ActionType.ERRAND:
        # The person given the errand only goes into acts_on: in claims, arbitration would reject the whole action as "no response".
        bearer = next(
            (n for n in env.all_npcs() if n.name == str(tgt or "")), None,
        ) or next(iter(env.all_npcs()), None)
        to = spec.get("to")
        if bearer is not None:
            target = ActionTarget(acts_on=[Ref.npc(bearer.npc_id)])
            intent = ActionIntent(
                purpose=spec.get("description", ""),
                errand=ErrandOrder(
                    npc_id=bearer.npc_id,
                    destination_id=resolve_location_ref(str(spec.get("destination", "")), env),
                    item_id=str(spec.get("carry", "")),
                    recipient_id=(
                        resolve_ref(str(to), id_by_name, valid_ids) if to else ""
                    ),
                    message=str(spec.get("say", "")),
                ),
            )
    return AgentAction(
        agent_id=actor_id, step=run_step, action_type=atype,
        action_description=spec.get("description", ""),
        expected_outcome=spec.get("expected_outcome", ""),
        estimated_steps=int(spec.get("estimated_steps", 1)),
        target=target,
        intent=intent,
    )


def _feasibility_view(action: AgentAction, env) -> dict[str, Any]:
    """Pre-action: run the type's feasibility check (where applicable)."""
    at = action.action_type
    if at == ActionType.TALK:
        r = env.check_talk_feasibility(action.agent_id, action.target.acted_on_agents)
    elif at == ActionType.MOVE:
        r = env.check_move_feasibility(action.agent_id, action.target.acted_on_place)
    elif at == ActionType.PHYSICAL:
        aimed = action.target.acts_on[0] if action.target.acts_on else None
        r = env.check_physical_feasibility(
            action.agent_id, aimed.id if aimed else None, action.target.acted_on_kind)
    else:
        return {"ok": True, "reason": "", "checked": False}
    return {"ok": r.ok, "reason": r.reason, "checked": True}


def _actor_input_view(agent, action: AgentAction, env, name_by_id: dict, seeded: list[str]) -> dict[str, Any]:
    soul = agent.personality.soul
    st = agent.personality.state
    loc = env.get_body_location(agent.agent_id)
    return {
        "actor_name": name_by_id.get(agent.agent_id, agent.agent_id),
        "is_main_character": agent.is_main_character,
        "location": env.narrative_location_name(loc),
        "core_traits": list(getattr(soul, "core_traits", [])),
        "core_values": list(getattr(soul, "core_values", [])),
        "vitality": st.vitality,
        "condition": render_condition(st.condition),
        "emotion": emotion_view(st.emotion),
        "action_type": action.action_type.value,
        "action_description": action.action_description,
        "expected_outcome": action.expected_outcome,
        "estimated_steps": action.estimated_steps,
        "target": target_view(action.target, name_by_id),
        "co_present": [name_by_id.get(a, a) for a in env.agents_at(loc) if a != agent.agent_id],
        "seeded_memories": seeded,
    }


async def _run_lifecycle(
    scenario: dict, world: World, registry, message_system, run_step: int,
    name_by_id: dict, id_by_name: dict, sink: InMemoryTraceSink, world_id: str,
) -> None:
    env = world.environment
    valid_ids = list(world.agents.keys())
    actor_id = resolve_ref(scenario["actor"], id_by_name, valid_ids)
    agent = world.agents[actor_id]
    spec = scenario["action"]
    action = _build_agent_action(actor_id, spec, env=env, run_step=run_step, id_by_name=id_by_name, valid_ids=valid_ids)
    seeded = [str(c) for e in (scenario.get("scene", {}) or {}).get("memories", []) for c in (e.get("factual") or [])]
    executor = registry.get_executor(action.action_type)

    phases: dict[str, Any] = {"pre": _feasibility_view(action, env)}
    set_log_context(world_id=world_id, agent_id=actor_id, step=str(run_step))
    try:
        result = await executor.start(
            action, run_step, agents=world.agents, environment=env, message_system=message_system)
    except Exception as exc:  # noqa: BLE001
        logger.warning("action_start_failed", extra={"world_id": world_id, "actor": actor_id, "error": str(exc)})
        clear_log_context()
        phases["start"] = {"error": str(exc)}
        _record_action_phase(sink, world_id, run_step, agent, action, env, name_by_id, seeded, phases)
        return

    if isinstance(result, ActionResult):
        phases["start"] = {"kind": "immediate", "result": _result_view(result, name_by_id)}
        phases["complete"] = {"kind": "immediate(同 start)", "results": [_result_view(result, name_by_id)]}
    else:
        registry.add_active(result)
        phases["start"] = {"kind": "multi_step", "state": _state_view(result)}
        ticks: list[dict] = []
        total = result.estimated_steps
        interrupt_at = scenario.get("interrupt_at")
        # advance ticks until completion or the scheduled interrupt point
        tick_limit = total - 1 if interrupt_at is None else min(int(interrupt_at), total - 1)
        for _ in range(max(0, tick_limit)):
            result.remaining_steps -= 1
            trs = await executor.tick(result, run_step, agents=world.agents, environment=env, message_system=message_system)
            ticks.append({"elapsed": total - result.remaining_steps,
                          "narratives": [{"agent": name_by_id.get(t.agent_id, t.agent_id), "narrative": t.outcome} for t in trs]})
        phases["tick"] = ticks
        if interrupt_at is not None:
            finals = await executor.interrupt(
                result, scenario.get("interrupt_reason", "突发变故"), run_step,
                agents=world.agents, environment=env,
                interrupted_agent_id=actor_id, reaction=scenario.get("interrupt_reaction", ""))
            phases["complete"] = {"kind": "interrupt", "results": [_result_view(r, name_by_id) for r in finals]}
        else:
            result.remaining_steps = 0
            finals = await executor.complete(result, run_step, agents=world.agents, environment=env, message_system=message_system)
            phases["complete"] = {"kind": "complete", "results": [_result_view(r, name_by_id) for r in finals]}
    clear_log_context()
    _record_action_phase(sink, world_id, run_step, agent, action, env, name_by_id, seeded, phases)


def _record_action_phase(sink, world_id, run_step, agent, action, env, name_by_id, seeded, phases) -> None:
    sink.record_phase(PhaseTrace(
        world_id=world_id, phase="stage.action", step=run_step, agent_id=agent.agent_id,
        inputs=_actor_input_view(agent, action, env, name_by_id, seeded),
        outputs={"kind": "lifecycle", "phases": phases},
        timestamp=datetime.now().isoformat(),
    ))


async def _run_arbitration(
    scenario: dict, container, config, world: World, registry, message_system,
    run_step: int, name_by_id: dict, id_by_name: dict, sink: InMemoryTraceSink, world_id: str,
) -> None:
    env = world.environment
    valid_ids = list(world.agents.keys())
    runtime = make_action_runtime(container, config, world_id, world, registry, message_system)

    planned: list = []
    actions_in: list[dict] = []
    for entry in scenario.get("actions", []):
        aid = resolve_ref(entry["actor"], id_by_name, valid_ids)
        if aid not in world.agents:
            continue
        action = _build_agent_action(aid, entry["action"], env=env, run_step=run_step,
                                     id_by_name=id_by_name, valid_ids=valid_ids)
        sp = env.spatial_for(agent_id=aid, step=run_step)
        attach_presence(
            sp, directory=world.directory, agents=world.agents,
            environment=env,
        )
        need = NeedEvaluation(
            dominant_need=NeedType.SAFETY, scores={NeedType.SAFETY: 1.0},
            active_needs=[NeedState(NeedType.SAFETY, "safety", 1.0)],
            short_term_goals=[], long_term_goals=[], prompt_context="")
        plan = AgentStepPlan(agent_id=aid, step=run_step, spatial=sp, inbox=[], broadcasts=[],
                             need_evaluation=need, action=action)
        phase = "main" if world.agents[aid].is_main_character else "bg"
        planned.append((phase, world.agents[aid], plan))
        actions_in.append({
            "actor": name_by_id.get(aid, aid),
            "action_type": action.action_type.value,
            "target": [name_by_id.get(r.id, r.id) for r in action.target.acts_on],
            "estimated_steps": action.estimated_steps,
        })

    # Initiative order: main-character batch first by default (same as the scheduler). A scenario can
    # set order=[names...] to force a specific initiative layout (test scaffolding for building a contention order deterministically).
    order = scenario.get("order")
    if order:
        rank = {resolve_ref(n, id_by_name, valid_ids): i for i, n in enumerate(order)}
        planned.sort(key=lambda t: rank.get(t[2].agent_id, 999))
    else:
        planned.sort(key=lambda t: 0 if t[0] == "main" else 1)
    arb = await runtime._arbitrate_execution(planned, world.agents, run_step)  # noqa: SLF001
    active = {es.execution_id: es for es in runtime._executor_registry.all_active()}  # noqa: SLF001

    verdicts = []
    for _phase, agent, plan in planned:
        aa = arb.get(plan.agent_id)
        if aa is None:
            continue
        # A rejection is still an execution (create_failed), so read the shape from the execution: a
        # rejected one carries a preset not_executed result; a multi-step one still has steps left after its first beat.
        exec_state = active.get(aa.ongoing_execution_id)
        preset = exec_state.extra.get("completed_result") if exec_state else None
        rejected = preset is not None and preset.not_executed
        ongoing = exec_state is not None and exec_state.remaining_steps > 0
        if aa.is_passive_join:
            kind = "被动并入"
        elif rejected:
            kind = "被拒"
        elif ongoing:
            kind = "起始(多步)"
        else:
            kind = "即时准入"
        verdicts.append({
            "actor": name_by_id.get(plan.agent_id, plan.agent_id),
            "kind": kind,
            "succeeded": not rejected and aa.action_result.succeeded,
            "is_passive_join": aa.is_passive_join,
            "ongoing": ongoing,
            "outcome": preset.outcome if rejected else aa.action_result.outcome,
        })

    sink.record_phase(PhaseTrace(
        world_id=world_id, phase="stage.action", step=run_step, agent_id=None,
        inputs={"kind": "arbitration", "actions": actions_in},
        outputs={"kind": "arbitration", "verdicts": verdicts},
        timestamp=datetime.now().isoformat(),
    ))


async def run_action(
    container: Container,
    config: Config,
    world_id: str,
    *,
    scenario: dict | None = None,
    sink: InMemoryTraceSink | None = None,
    trace_dir: str = "./data/tuning_traces",
) -> int:
    """Isolated dry-run of the action stage (executor lifecycle / arbitration) for a restored world.

    Drives the real production executors / ``runtime._arbitrate_execution`` directly. Restores
    per scenario (execution mutates env/registry); nothing persisted. Zero production changes (only
    scene/memory injection is test scaffolding).
    """
    trace_sink = sink if sink is not None else InMemoryTraceSink()
    # traced_container.llm_router is the traced router — build executors/runtime with it so
    # executor narration LLM calls are captured in the sink.
    traced_container, world, run_step, _ = await restore_traced(container, config, world_id, trace_sink)
    name_by_id = world.directory.all_agent_names()
    id_by_name = {v: k for k, v in name_by_id.items()}
    registry = build_default_registry(
        traced_container.llm_router, world.directory, seconds_per_step=world.clock_config.seconds_per_step)
    message_system = MessageSystem(traced_container.message_provider, world_id=world_id)

    await apply_action_scene(scenario or {}, world, run_step, id_by_name)

    if (scenario or {}).get("kind") == "arbitration":
        await _run_arbitration(scenario, traced_container, config, world, registry, message_system,
                               run_step, name_by_id, id_by_name, trace_sink, world_id)
    else:
        await _run_lifecycle(scenario, world, registry, message_system, run_step,
                             name_by_id, id_by_name, trace_sink, world_id)
    logger.info("tuning_action_dry_run_complete", extra={"world_id": world_id, "step": run_step})
    return run_step
