"""Interrupt decision (Agent.evaluate_interrupt) dry-run"""

from __future__ import annotations

from datetime import datetime

from agent.personality import parse_emotion_type
from config.models import Config
from core.container import Container
from core.context import clear_log_context, set_log_context
from core.logging import get_logger
from core.interfaces.action import ActionType
from core.interfaces.condition import BodyCondition
from core.prompts import render_condition
from engine.executors import build_default_registry
from engine.executors.base import ActionExecutionState
from engine.message_system import MessageSystem

from tuning.phase_harness.common import (
    emotion_view, isolate_agent, make_action_runtime, restore_traced,
)
from tuning.phase_harness.scenario import resolve_ref
from tuning.trace import InMemoryTraceSink, PhaseTrace

logger = get_logger(__name__)


def _parse_ongoing_action_type(value: str) -> ActionType:
    """Scenario ongoing.action_type string → ActionType; invalid values fall back to REST (neutral, needs no target)."""
    try:
        return ActionType(str(value or "rest"))
    except ValueError:
        return ActionType.REST


async def run_interrupt(
    container: Container,
    config: Config,
    world_id: str,
    *,
    scenario: dict | None = None,
    sink: InMemoryTraceSink | None = None,
    trace_dir: str = "./data/tuning_traces",
) -> int:
    """Isolated dry-run of the interrupt-decision stage for a restored world.

    Drives the real production path ``runtime._decide_interrupt``: it turns the ongoing action's
    progress into natural duration/qualitative text through ``_render_progress_hint`` (no step leak),
    then calls ``Agent.evaluate_interrupt`` (every agent, LLM, first person, think before deciding).
    This stage validates only the interrupt decision itself and doesn't touch executor.interrupt
    (validated separately).

    The injected personality state (emotion / short-term goals) and the ongoing action are test
    scaffolding; persistence is isolated by isolate_agent's shadow store, baseline ./data is never
    written. Restores per scenario.
    """
    trace_sink = sink if sink is not None else InMemoryTraceSink()
    traced_container, world, run_step, _ = await restore_traced(container, config, world_id, trace_sink)
    name_by_id = world.directory.all_agent_names()
    id_by_name = {v: k for k, v in name_by_id.items()}
    valid_ids = set(world.agents.keys())
    scenario = scenario or {}

    # Pick the character making the interrupt decision: the explicit agent, else the main character.
    ref = scenario.get("agent")
    if ref:
        aid = resolve_ref(str(ref), id_by_name, valid_ids)
    else:
        aid = next((a for a, ag in world.agents.items() if ag.is_main_character), next(iter(valid_ids)))
    agent = world.agents[aid]
    isolate_agent(agent, name_by_id)

    # Inject personality state (inputs to to_prompt_context): emotion + short-term goals. state is a
    # defensive copy, so changes must go through update_emotion / restore_state to reach the live _state.
    emo = scenario.get("emotion")
    if isinstance(emo, dict):
        agent.personality.update_emotion(
            primary=parse_emotion_type(str(emo.get("type", "neutral"))),
            intensity=float(emo.get("intensity", 0.4)),
            valence=float(emo.get("valence", 0.0)),
            triggered_by="scene",
        )
    goals = scenario.get("goals")
    if goals is not None:
        st = agent.personality.state
        st.short_term_goals = [str(g) for g in goals]
        agent.personality.restore_state(st)

    # Pre-set them as restrained: someone who can't move weighs "should I drop what I'm doing" differently from someone free.
    cond = scenario.get("condition")
    if cond:
        agent.personality.set_condition(BodyCondition(description=str(cond), since_step=0))

    # Build the ongoing action's execution state (progress comes from estimated/remaining).
    ongoing = scenario.get("ongoing", {}) or {}
    action_type = _parse_ongoing_action_type(ongoing.get("action_type", "rest"))
    estimated = max(1, int(ongoing.get("estimated_steps", 8)))
    remaining = max(0, min(estimated, int(ongoing.get("remaining_steps", max(1, estimated // 2)))))
    purpose = str(ongoing.get("purpose", ""))
    exec_state = ActionExecutionState(
        execution_id=f"tuning-interrupt-{aid}-{run_step}",
        action_type=action_type,
        initiator_id=aid,
        participant_ids=[aid],
        started_step=run_step,
        estimated_steps=estimated,
        remaining_steps=remaining,
        purpose=purpose,
    )
    reason = str(scenario.get("reason", ""))

    # Real runtime: _decide_interrupt computes progress_hint internally and calls evaluate_interrupt.
    registry = build_default_registry(
        traced_container.llm_router, world.directory, seconds_per_step=world.clock_config.seconds_per_step)
    message_system = MessageSystem(traced_container.message_provider, world_id=world_id)
    runtime = make_action_runtime(traced_container, config, world_id, world, registry, message_system)
    progress_hint = runtime._interrupts._render_progress_hint(exec_state)  # noqa: SLF001 — reuse the real translation, capture it for display

    set_log_context(world_id=world_id, agent_id=aid, step=str(run_step))
    error = None
    should_interrupt: bool | None = None
    thought = ""
    try:
        should_interrupt, thought = await runtime._interrupts._decide_interrupt(agent, exec_state, reason)  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001 — one scenario's failure must not abort the run
        logger.warning("interrupt_dryrun_failed",
                       extra={"world_id": world_id, "agent_id": aid, "error": str(exc)})
        error = str(exc)
    clear_log_context()

    soul = agent.personality.soul
    trace_sink.record_phase(PhaseTrace(
        world_id=world_id,
        phase="stage.interrupt",
        step=run_step,
        agent_id=aid,
        inputs={
            "agent_name": name_by_id.get(aid, aid),
            "is_main_character": agent.is_main_character,
            "role": getattr(soul, "role", ""),
            "core_traits": list(getattr(soul, "core_traits", [])),
            "core_values": list(getattr(soul, "core_values", [])),
            "emotion": emotion_view(agent.personality.state.emotion),
            # The judge uses this to check whether the weighing fits the agent's own situation; leaving it out of inputs means it isn't tested.
            "condition": render_condition(agent.personality.state.condition),
            "short_term_goals": list(agent.personality.state.short_term_goals),
            "ongoing_action_type": action_type.value,
            "ongoing_purpose": purpose,
            "progress_hint": progress_hint,
            "source": scenario.get("source", ""),
            "reason": reason,
        },
        outputs={
            "agent_name": name_by_id.get(aid, aid),
            "should_interrupt": should_interrupt,
            "thought": thought,
            "error": error,
        },
        timestamp=datetime.now().isoformat(),
    ))
    logger.info("tuning_interrupt_dry_run_complete",
                extra={"world_id": world_id, "step": run_step, "agent_id": aid})
    return run_step
