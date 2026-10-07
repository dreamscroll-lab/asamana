"""Interrupt evaluation & application for the runtime step.

An interrupt is three parts within one step: break off the current action, apply interrupt
feedback, then think and act anew. Without the third it pays the teardown cost and buys nothing.
The first two are ``_apply_interrupt``; the third lands in ``NarrativeRuntime.run_step``
(removing the torn-down agent from in_progress_at_step_start) and ``AgentScheduler``
(``CadenceReason.INTERRUPTED``).

Signals: urgent messages, high-severity broadcasts, external THREAT goals. Teardown lands through
the ExecutionProcessor; this never calls back into the runtime.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

from agent.motivation import ExternalDriveType
from agent.personality import ActionStatus
from core.context import annotate_call, observe_stage
from core.interfaces.directory import WorldDirectory
from core.interfaces.severity import Severity
from core.interfaces.trace import Stage
from core.interfaces.urgency import URGENCY_INTERRUPT_THRESHOLD
from core.logging import get_logger
from core.duration import describe_duration
from engine.clock import GlobalClock
from engine.environment import EnvironmentSystem
from engine.execution_processor import ExecutionProcessor
from engine.executors.base import (
    ActionExecutionState, execution_annotations, participant_action_desc, participant_intent,
)
from engine.executors.registry import ActionExecutorRegistry

if TYPE_CHECKING:
    from agent.agent import Agent
    from core.interfaces.perception import Broadcast
    from engine.message_system import MessageDelivery


logger = get_logger(__name__)


@dataclass
class _InterruptCandidate:
    """One agent's interrupt signals this step, merged so a single decision sees them all.

    Each source is named narratively inside its reason part ("收到X的急讯：…"), not as an id.
    """

    agent: "Agent"
    exec_state: ActionExecutionState
    reason_parts: List[str] = field(default_factory=list)

class InterruptCoordinator:
    """Evaluates + applies this step's interrupts (see module docstring)."""

    def __init__(
        self,
        *,
        executor_registry: ActionExecutorRegistry,
        environment: EnvironmentSystem,
        directory: WorldDirectory,
        clock: GlobalClock,
        processor: ExecutionProcessor,
    ) -> None:
        self._executor_registry = executor_registry
        self._environment = environment
        self._directory = directory
        self._clock = clock
        self._processor = processor
        # Edge-trigger memory for Path 3: standing THREAT goals recur every step, so without it
        # an agent that decided "finish first" would be re-asked each step and oscillate.
        # Accepted coarseness: the key omits goal.text (a new threat from the same source+drive
        # won't re-trigger), and it is not snapshotted (a restore re-asks once).
        self._interrupt_seen_goals: Dict[str, set[tuple[str, ExternalDriveType]]] = {}

    async def evaluate_interrupts(
        self,
        deliveries: "MessageDelivery",
        broadcasts: "List[Broadcast]",
        agents: "Dict[str, Agent]",
        step: int,
    ) -> List[Dict[str, Any]]:
        """Evaluate interrupt triggers from messages, broadcasts, and external goals.

        Collect merges each agent's signals; Decide runs per-agent decisions concurrently; Apply
        runs serially, so a shared multi-participant action is torn down only once.
        """
        candidates: Dict[str, _InterruptCandidate] = {}

        def _register(agent_id: str, reason: str) -> None:
            agent = agents.get(agent_id)
            if agent is None or agent.personality.state.action_status != ActionStatus.IN_PROGRESS:
                return
            candidate = candidates.get(agent_id)
            if candidate is None:
                exec_state = self._executor_registry.get_active_for_agent(agent_id)
                if exec_state is None:
                    return
                candidate = _InterruptCandidate(agent=agent, exec_state=exec_state)
                candidates[agent_id] = candidate
            candidate.reason_parts.append(reason)

        # Path 1: messages the sender marked urgency >= HIGH (a hint; the agent still decides).
        # Must iterate each agent's inbox_for(), not delivered_messages: the latter includes
        # undelivered messages, and recipients=None would bypass location_scope.
        for agent_id in agents:
            for message in deliveries.inbox_for(agent_id):
                if message.urgency < URGENCY_INTERRUPT_THRESHOLD:
                    continue
                if agent_id == message.sender_id:
                    continue
                # The sender is always an agent; name it, never the id.
                sender_name = self._directory.agent_name(message.sender_id)
                _register(agent_id, f"收到{sender_name}的急讯：{message.content}")

        # Path 2: high-severity broadcasts
        for bc in broadcasts:
            if bc.severity != Severity.HIGH:
                continue
            for agent_id in agents:
                if bc.location_scope is not None:
                    if bc.location_scope != self._environment.get_body_location(agent_id):
                        continue
                _register(agent_id, f"世界广播：{bc.content}")

        # Path 3: ExternalGoal THREAT, urgency >= HIGH, edge-triggered
        for agent_id, agent in agents.items():
            if agent.personality.state.action_status != ActionStatus.IN_PROGRESS:
                self._interrupt_seen_goals.pop(agent_id, None)
                continue
            seen = self._interrupt_seen_goals.get(agent_id, set())
            qualifying: set[tuple[str, ExternalDriveType]] = set()
            for goal in agent.pending_external_goals:
                if goal.drive_type != ExternalDriveType.THREAT or goal.urgency < URGENCY_INTERRUPT_THRESHOLD:
                    continue
                key = (goal.source_id, goal.drive_type)
                qualifying.add(key)
                if key not in seen:
                    # Name the source only when it is an agent: it may be an event_id, which the
                    # directory would render as "某人".
                    if goal.source_id in agents:
                        threat_reason = f"我感受到的来自{self._directory.agent_name(goal.source_id)}的外部压力：{goal.text}"
                    else:
                        threat_reason = f"我感受到的外部压力：{goal.text}"
                    _register(agent_id, threat_reason)
            self._interrupt_seen_goals[agent_id] = qualifying

        if not candidates:
            return []

        ordered = list(candidates.values())
        decisions = await asyncio.gather(
            *(
                self._decide_interrupt(c.agent, c.exec_state, "\n".join(c.reason_parts), step)
                for c in ordered
            ),
            return_exceptions=True,
        )

        interrupt_records: List[Dict[str, Any]] = []
        for candidate, decision in zip(ordered, decisions):
            if isinstance(decision, BaseException):
                logger.warning(
                    "interrupt_decision_failed",
                    extra={"agent_id": candidate.agent.agent_id, "error": str(decision)},
                )
                # Don't interrupt: it is the destructive side, and the signal is already
                # perceived for the next re-plan.
                should_interrupt, thought = False, ""
            else:
                should_interrupt, thought = decision
            if not should_interrupt:
                logger.debug(
                    "interrupt_declined",
                    extra={
                        "agent_id": candidate.agent.agent_id,
                        "step": step,
                        "action_type": candidate.exec_state.action_type.value,
                    },
                )
                continue
            # info, not debug: rare, and it explains an IN_PROGRESS→IDLE transition.
            logger.info(
                "interrupt_fired",
                extra={
                    "agent_id": candidate.agent.agent_id,
                    "step": step,
                    "action_type": candidate.exec_state.action_type.value,
                    "execution_id": candidate.exec_state.execution_id,
                },
            )
            # Only the THOUGHT travels on, not the signals: reason_parts carries ExternalGoal
            # text, which must never reach memory, and every downstream consumer writes memory.
            # See ActionExecutor.interrupt.
            interrupt_records.extend(await self._apply_interrupt(
                agent_id=candidate.agent.agent_id,
                thought=thought,
                step=step,
                agents=agents,
            ))
        return interrupt_records

    async def _decide_interrupt(
        self,
        agent: "Agent | None",
        exec_state: ActionExecutionState,
        reason: str,
        step: int,
    ) -> tuple[bool, str]:
        """Read-only, so safe to run concurrently. Missing agent → don't interrupt (the
        conservative side)."""
        if agent is None:
            return False, ""
        # The weighing belongs to this execution, not the cognition round that may follow;
        # otherwise a review reads "doing X" after "I just got out of X" as a contradiction.
        with annotate_call(execution_id=exec_state.execution_id):
            return await agent.evaluate_interrupt(
                step=step,
                reason=reason,
                current_action_desc=participant_action_desc(
                    self._directory, agent.agent_id, exec_state),
                intent=participant_intent(agent.agent_id, exec_state),
                progress_hint=self._render_progress_hint(exec_state),
            )

    def _render_progress_hint(self, exec_state: ActionExecutionState) -> str:
        """Render an ongoing action's progress as natural duration plus a qualitative phase,
        never "3/5 步" (step is a code-layer coordinate)."""
        elapsed = max(0, exec_state.estimated_steps - exec_state.remaining_steps)
        ratio = elapsed / max(exec_state.estimated_steps, 1)
        if ratio < 0.25:
            phase = "才刚开始"
        elif ratio < 0.5:
            phase = "进行了一会儿"
        elif ratio < 0.75:
            phase = "已过了大半"
        else:
            phase = "就快收尾了"
        if elapsed <= 0:
            return phase
        duration = describe_duration(elapsed, self._clock.config.seconds_per_step)
        return f"已进行了{duration}，{phase}"

    async def _apply_interrupt(
        self,
        *,
        agent_id: str,
        thought: str,
        step: int,
        agents: Dict[str, "Agent"],
    ) -> List[Dict[str, Any]]:
        """Run the executor's interrupt path and write back results. Must run serially: it
        re-fetches the executor state, so a shared action already torn down this pass is skipped.

        Returns one display record per participant result (phase="interrupt"); empty when
        nothing was interrupted."""
        exec_state = self._executor_registry.get_active_for_agent(agent_id)
        if exec_state is None:
            return []
        executor = self._executor_registry.get_executor(exec_state.action_type)
        if executor is None:
            return []
        try:
            with observe_stage(Stage.ACTION, agent_id=agent_id), \
                    annotate_call(**execution_annotations(self._directory, exec_state)):
                results = await executor.interrupt(
                    exec_state,
                    step,
                    agents=agents,
                    environment=self._environment,
                    # Lets a multi-person TALK tell the interrupter from those who only saw the
                    # other leave (see social.interrupt).
                    interrupted_agent_id=agent_id,
                    thought=thought,
                )
        except Exception as exc:  # noqa: BLE001 — Rule 1/5: an executor failure must never kill the step
            # Conservatively DON'T interrupt: leave the action active and write nothing back.
            logger.warning(
                "interrupt_apply_failed",
                extra={"agent_id": agent_id, "step": step, "error": str(exc)},
            )
            return []
        records: List[Dict[str, Any]] = []
        for result in results:
            # Same landing path as same-tick completion, but serial, not the concurrent batch.
            with annotate_call(execution_id=exec_state.execution_id):
                await self._processor.land_result_feedback(result, agents, step)
            # Display only: a terminal record for the timeline. No bystander channel is produced
            # (see ActionExecutor.interrupt) and carry ambient isn't fed.
            record = self._processor.completion_record(exec_state, result, agents, phase="interrupt")
            # Only this layer knows WHO broke it off; without it a cut-short TALK can't say
            # which side walked out.
            record["interrupted_by"] = agent_id
            # The REASON rides in ``outcome`` (attributed by the executor). Don't borrow
            # ``inner_monologue``: that is the thought behind the ABANDONED action, not the
            # abandoning.
            records.append(record)
            # A COVERT exposed on the beat it was interrupted is not perceived by bystanders
            # (no carry ambient here). If that's needed, add that channel; don't produce a string
            # in interrupt that nobody reads.
        self._executor_registry.remove_active(exec_state.execution_id)
        return records
