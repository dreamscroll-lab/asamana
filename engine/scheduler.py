"""Agent scheduling for the runtime loop."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Collection, List, Optional, Sequence

from agent.need import NEED_URGENT_INTENSITY
from agent.personality import EMOTION_STRONG_INTENSITY
from core.interfaces.urgency import Urgency
from core.logging import get_logger

if TYPE_CHECKING:
    from agent.agent import Agent


logger = get_logger(__name__)


class CadenceReason(str, Enum):
    """Why the gate admitted an agent to the decision loop; reported per step so it is
    measurable which conditions actually carry the world and which are inert."""

    NEVER_DECIDED = "never_decided"  # cold start — has not thought once yet
    INTERRUPTED = "interrupted"      # it just broke off its own action to attend to something
    PRESSURE = "pressure"            # the pressure evaluator gave it an external goal
    NEED = "need"                    # a need is driving it
    EMOTION = "emotion"              # an emotion has it
    STARVED = "starved"              # nothing pressing, but it has gone too long


# Cadence defaults, a world-time argument with no other anchor: at one world hour per step, a
# background character with nothing happening still gets a few beats a waking day, and a main
# character never goes long without an inner life. These are the knobs to tune; the signal
# thresholds are not.
DEFAULT_MAIN_MAX_IDLE_STEPS = 2
DEFAULT_BACKGROUND_MAX_IDLE_STEPS = 6

# The minimum pressure urgency that alone forces a decision: the Urgency scale's own
# actionability boundary (LOW is 「可留意(背景信号)」, NORMAL the first 「值得响应」 band), not a
# tuned knob. A LOW goal is not discarded: it still colours motivation if the agent decides for
# another reason.
PRESSURE_ADMIT_URGENCY = Urgency.NORMAL


@dataclass(frozen=True)
class SchedulerBatch:
    """A scheduling phase with a shared runtime rule."""

    phase: str
    concurrency_rule: str
    agents: List[Agent] = field(default_factory=list)

    def agent_ids(self) -> List[str]:
        return [agent.agent_id for agent in self.agents]


@dataclass(frozen=True)
class StepExecutionPlan:
    """Explicit runtime plan for a single world step."""

    step: int
    batches: List[SchedulerBatch] = field(default_factory=list)

    def ordered_agents(self) -> List[Agent]:
        ordered: List[Agent] = []
        for batch in self.batches:
            ordered.extend(batch.agents)
        return ordered

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "batches": [
                {
                    "phase": batch.phase,
                    "concurrency_rule": batch.concurrency_rule,
                    "agent_ids": batch.agent_ids(),
                }
                for batch in self.batches
            ],
        }


class AgentScheduler:
    """Decide who the engine spends cognition on this step.

    Two separate questions:

    - **Who CAN act**: an agent whose action continues or completes this step does not
      re-decide (``in_progress_at_step_start``). An agent torn down by an interrupt is NOT in
      that set: its body is free and the interrupting signal wants an answer now.
    - **Who is ASKED to think**: the cadence gate. Admitting everyone wastes LLM calls on
      characters with nothing to react to and keeps every body busy, so conversations collide.

    The gate is a **scheduling** decision, never "what do you decide": an admitted agent runs the
    same LLM cognition path (CLAUDE.md §5). Standing down after thinking stays with the LLM
    (``act=false`` → ``DecisionStatus.NO_ACTION``).

    A skipped agent gets a stub plan (``action=None``, ``DecisionStatus.NOT_SCHEDULED``, zero
    LLM calls) and stays a free body that can still be conscripted into someone else's action.
    """

    def __init__(
        self,
        *,
        seed: Optional[str] = None,
        main_max_idle_steps: int = DEFAULT_MAIN_MAX_IDLE_STEPS,
        background_max_idle_steps: int = DEFAULT_BACKGROUND_MAX_IDLE_STEPS,
    ) -> None:
        if main_max_idle_steps <= 0 or background_max_idle_steps <= 0:
            raise ValueError("max_idle_steps must be positive")
        # Initiative-fairness seed: each batch is shuffled so no agent permanently
        # out-initiatives a same-tier peer (a fixed agent_id order starves conscription).
        # Seeded per (seed, step) so order reproduces across restarts without persisting RNG
        # state. None → deterministic agent_id sort (tests / unseeded callers).
        self._seed = seed
        self._main_max_idle_steps = main_max_idle_steps
        self._background_max_idle_steps = background_max_idle_steps

    def plan(
        self,
        agents: Sequence[Agent],
        *,
        step: int,
        in_progress_at_step_start: Collection[str],
        interrupters: Collection[str] = (),
    ) -> StepExecutionPlan:
        """Build an explicit execution plan for the given step.

        ``in_progress_at_step_start`` holds agents whose own action occupies this step, captured
        before any status-mutating phase, minus the interrupt-torn-down participants (see
        ``NarrativeRuntime.run_step``). Those agents don't decide this step even if their action
        completes on its final tick: they first perceive the feedback, then decide next step.

        The snapshot is the single source of truth: nothing goes IDLE→IN_PROGRESS before this
        call (begin_action runs later, in commit), so no live-status check is needed.

        Pressure is read off ``pending_external_goals``, already written by the pressure phase,
        so this stays a pure policy over agent state.
        """
        candidates = [
            agent for agent in agents
            if agent.is_active and agent.agent_id not in in_progress_at_step_start
        ]
        interrupter_ids = set(interrupters)
        verdicts = {
            agent.agent_id: self._should_decide(
                agent, step=step, interrupted=agent.agent_id in interrupter_ids,
            )
            for agent in candidates
        }
        deciders = [agent for agent in candidates if verdicts[agent.agent_id] is not None]
        self._log_cadence(
            step=step,
            agents=agents,
            candidates=candidates,
            verdicts=verdicts,
            occupied=len(in_progress_at_step_start),
        )

        # Sort-then-shuffle: shuffle reproducibility depends on a deterministic starting order.
        rng = random.Random(f"{self._seed}:{step}") if self._seed is not None else None

        def _order(group: "list[Agent]") -> "List[Agent]":
            ordered = sorted(group, key=lambda agent: agent.agent_id)
            if rng is not None:
                rng.shuffle(ordered)
            return ordered

        # The main/background split is orthogonal to the gate: it carries the model tier and
        # the arbiter's initiative order.
        main_agents = _order([a for a in deciders if a.is_main_character])
        background_agents = _order([a for a in deciders if not a.is_main_character])

        batches: List[SchedulerBatch] = []
        if main_agents:
            batches.append(
                SchedulerBatch(
                    phase="main",
                    concurrency_rule="shared-read snapshot; commit after main batch",
                    agents=main_agents,
                )
            )
        if background_agents:
            batches.append(
                SchedulerBatch(
                    phase="background",
                    concurrency_rule="shared-read snapshot; commit after background batch",
                    agents=background_agents,
                )
            )

        return StepExecutionPlan(step=step, batches=batches)

    def _should_decide(
        self, agent: "Agent", *, step: int, interrupted: bool = False,
    ) -> CadenceReason | None:
        """Is this idle agent worth a decision loop this step? Returns why, or None to skip.

        Conditions are OR'd; order only decides the reported label (pressure outranks starved).
        Each is an objective threshold on structured state (CLAUDE.md §5 puts these on the rule
        side); the pressure condition reads the pressure LLM's verdict, it doesn't re-judge.

        Internal drives (need, emotion) are required: a gate keyed only on external pressure
        makes everyone purely reactive, and the world deadlocks since nobody acts to create
        pressure.

        A threshold must be able to be FALSE in the distribution the simulation produces:
        ``NEED_URGENT_INTENSITY`` admits a minority of agent-steps. Don't key the gate on the
        「明显」 emotion band (0.45) instead of ``EMOTION_STRONG_INTENSITY``: LLM emotion sits above
        it most of the time, so it would admit nearly everyone.
        """
        state = agent.personality.state

        # Without this, a fresh world's step 1 would find every agent short of the starvation
        # bound and think nobody at all.
        if state.last_decision_step == 0:
            return CadenceReason.NEVER_DECIDED

        # The agent itself decided to break off its action, a stronger statement than any
        # signal below; overruling it would tear the action down and buy nothing. A
        # co-participant torn down as collateral never made that claim and takes the gate below.
        if interrupted:
            return CadenceReason.INTERRUPTED

        # Gate on the pressure evaluator's verdict, NOT raw perception: re-testing the raw
        # signal re-admits the chatter it dismissed and wakes every co-located agent. Read its
        # urgency, not mere non-emptiness (PRESSURE_ADMIT_URGENCY).
        if any(g.urgency >= PRESSURE_ADMIT_URGENCY for g in agent.pending_external_goals):
            return CadenceReason.PRESSURE

        if self._max_need(agent) >= NEED_URGENT_INTENSITY:
            return CadenceReason.NEED
        if state.emotion.intensity >= EMOTION_STRONG_INTENSITY:
            return CadenceReason.EMOTION

        if step - state.last_decision_step >= self._max_idle_steps(agent):
            return CadenceReason.STARVED

        return None

    def _max_idle_steps(self, agent: "Agent") -> int:
        return (
            self._main_max_idle_steps if agent.is_main_character
            else self._background_max_idle_steps
        )

    @staticmethod
    def _max_need(agent: "Agent") -> float:
        intensities = agent.personality.state.need_intensities
        return max(intensities.values()) if intensities else 0.0

    def _log_cadence(
        self,
        *,
        step: int,
        agents: Sequence["Agent"],
        candidates: List["Agent"],
        verdicts: dict[str, CadenceReason | None],
        occupied: int,
    ) -> None:
        """One aggregate record per step, not a line per agent (per-agent logs in the engine loop
        flood output).

        ``skipped_margins`` records how far each turned-away agent was from each threshold, so
        "what would a threshold of X have admitted?" can be answered from a real run without
        re-running it.
        """
        by_reason = Counter(r.value for r in verdicts.values() if r is not None)
        decided = sum(by_reason.values())
        skipped = [a for a in candidates if verdicts[a.agent_id] is None]
        logger.info(
            "cadence",
            extra={
                "step": step,
                "agents": len(agents),
                "occupied": occupied,          # mid-action: never asked, by design
                "candidates": len(candidates),  # idle bodies the gate actually judged
                "decided": decided,             # ← these cost LLM calls
                "skipped": len(skipped),        # ← these are free, and still conscriptable
                "admit_rate": round(decided / len(candidates), 3) if candidates else 0.0,
                "by_reason": dict(by_reason),
                "skipped_margins": [
                    {
                        "agent_id": a.agent_id,
                        "main": a.is_main_character,
                        "need": round(self._max_need(a), 3),
                        "emotion": round(a.personality.state.emotion.intensity, 3),
                        "idle": step - a.personality.state.last_decision_step,
                        "idle_bound": self._max_idle_steps(a),
                    }
                    for a in skipped
                ],
            },
        )
