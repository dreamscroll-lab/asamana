"""World concurrency arbiter for the runtime step.

Decides, for one step's parallel per-agent plans, *which* intents are admitted onto the shared
world and *in what form* (start / passive-join / rejected); mechanics go to executors and all
writeback to the feedback layer. It never mutates agent-internal state and holds no per-step
state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

from agent.decision import DecisionStatus
from core.context import observe_stage
from core.interfaces.action import ActionResult, ActionTarget, ActionType, AgentAction
from core.interfaces.directory import WorldDirectory
from core.interfaces.trace import Stage
from core.interfaces.urgency import URGENCY_PREEMPT_THRESHOLD
from core.logging import get_logger
from core.duration import describe_duration
from core.prompts import strip_end_punct
from engine.clock import GlobalClock
from engine.execution_processor import ExecutionProcessor
from engine.executors.base import (
    ActionExecutionState, ActionExecutor, Conscription, participant_action_desc, participant_target,
)
from engine.executors.registry import ActionExecutorRegistry
from engine.message_system import MessageSystem

if TYPE_CHECKING:
    from agent.agent import Agent, AgentStepPlan
    from engine.environment import EnvironmentSystem


logger = get_logger(__name__)

# An intent foiled on this many recent beats gets first pick. The agent keeps the count; the
# owner of the ordering holds the criterion (same split as URGENCY_PREEMPT_THRESHOLD).
_FOILED_MISS_PRIORITY_THRESHOLD = 3


# What the committed body is visibly doing, as the rejected agent sees it face to face. COVERT
# isn't listed: naming it would expose it, so it falls through to the vague phrase. TALK is
# built in _busy_reason (it names the partner).
_BUSY_WITH: dict[ActionType, str] = {
    ActionType.REST: "正在歇息",
    ActionType.WORK: "正埋头忙着手上的事",
    ActionType.MOVE: "正要动身离开",
    ActionType.PHYSICAL: "正在动手做别的事",
    ActionType.SEND_MESSAGE: "正忙着传讯",
    ActionType.ERRAND: "正在吩咐人办事",
}
_BUSY_UNSEEN = "正忙于他事"


@dataclass(frozen=True)
class ArbitratedAction:
    """Resolved arbitration outcome for one agent action within a runtime step."""

    agent_id: str
    action_result: ActionResult
    location_id: str
    visible_agent_ids: List[str] = field(default_factory=list)
    message_ids: List[str] = field(default_factory=list)
    ongoing_execution_id: str | None = None  # None means immediate completion
    is_passive_join: bool = False


@dataclass
class _Admission:
    """One admitted action, not started yet: landing happens after all adjudication."""

    plan: AgentStepPlan
    action: AgentAction
    executor: ActionExecutor
    co_participants: list[str]
    # Bodies in another execution at step start, whose old task is torn down at landing.
    seized: list[str]


class ExecutionArbiter:
    """Admits/forms this step's intents onto the shared world (see module docstring)."""

    def __init__(
        self,
        *,
        executor_registry: ActionExecutorRegistry,
        environment: "EnvironmentSystem",
        message_system: MessageSystem,
        processor: ExecutionProcessor,
        directory: WorldDirectory,
        clock: GlobalClock,
    ) -> None:
        self._executor_registry = executor_registry
        self._environment = environment
        # Compulsory conscription tears down the conscript's task: a lifecycle capability.
        self._processor = processor
        self._message_system = message_system
        self._directory = directory
        self._clock = clock

    def _initiative_order(
        self, planned_steps: List[tuple[str, "Agent", "AgentStepPlan"]],
    ) -> List[tuple[str, "Agent", "AgentStepPlan"]]:
        """Initiative order. Four tiers:

        1. Starvation (``foiled_misses`` ≥ ``_FOILED_MISS_PRIORITY_THRESHOLD``) picks first. The
           only tier that crosses phases: starving agents are those that always sort late, so a
           within-phase fix would do nothing. Starvation is transient (``Agent._clear_foiled``),
           unlike pressure. Ties break by depth, or one of two starving agents always wins.
        2. Narrative tier (phase): main before background, in first-appearance order (lexical
           would put "background" first).
        3. Stakes (``consumed_external_goals`` ≥ ``URGENCY_PREEMPT_THRESHOLD``). Must be the
           third-party signal, not self-reported ``AgentAction.urgency``, which clusters high
           and would let everyone jump the queue.
        4. Constraint (``ActionExecutor.conscription``, not the action type): two-body actions
           first, or TALK keeps missing; a displaced solo action is deferred
           (``Agent.defer_decided_intent``). A MOVE leaving alone sits after COMPEL and before
           INVITE: enlisting someone who decided to leave would make INVITE a soft COMPEL.

        Full ties keep the scheduler's seeded shuffle (stable sort). Re-queued agents go to the
        back in ``_resolve``. Picking first isn't getting what you pick, and tier 1 can't free a
        target already inside an earlier execution.
        """
        phase_rank: dict[str, int] = {}
        for phase, _agent, _plan in planned_steps:
            phase_rank.setdefault(phase, len(phase_rank))

        preempting = {
            plan.agent_id for _phase, _agent, plan in planned_steps
            if plan.action is not None
            and any(g.urgency >= URGENCY_PREEMPT_THRESHOLD for g in plan.consumed_external_goals)
        }

        def order_key(item: tuple[str, "Agent", "AgentStepPlan"]) -> tuple[int, ...]:
            plan = item[2]
            starved = plan.foiled_misses >= _FOILED_MISS_PRIORITY_THRESHOLD
            mode = self._conscription_of(plan.action)
            if mode is Conscription.COMPEL:
                claim_rank = 0
            elif self._leaves(plan.action):
                claim_rank = 1
            elif mode is Conscription.INVITE:
                claim_rank = 2
            else:
                claim_rank = 3
            return (
                0 if starved else 1,
                # Must be zero below the threshold (counts 1 and 2); otherwise it would quietly
                # reorder agents it shouldn't touch.
                -plan.foiled_misses if starved else 0,
                phase_rank[item[0]],
                0 if plan.agent_id in preempting else 1,
                claim_rank,
            )

        ordered = sorted(planned_steps, key=order_key)
        logger.debug(
            "initiative_order",
            extra={
                "step": ordered[0][2].step if ordered else None,
                "order": [
                    {
                        "agent_id": plan.agent_id,
                        "phase": phase,
                        "foiled_misses": plan.foiled_misses,
                        "preempting": plan.agent_id in preempting,
                        "conscripts": self._conscription_of(plan.action) is not None,
                        "leaves": self._leaves(plan.action),
                    }
                    for phase, _agent, plan in ordered if plan.action is not None
                ],
            },
        )
        return ordered

    def _leaves(self, action: "AgentAction | None") -> bool:
        """Leaving alone: a MOVE carrying nobody."""
        return (
            action is not None and action.action_type is ActionType.MOVE
            and self._conscription_of(action) is None
        )

    def _mode_of_execution(
        self, exec_state: "ActionExecutionState | None",
    ) -> "Conscription | None":
        """``_conscription_of`` for an established execution; the executor's declared mode is
        the authoritative answer, no separate ledger."""
        if exec_state is None:
            return None
        executor = self._executor_registry.get_executor(exec_state.action_type)
        return getattr(executor, "conscription", None)

    def _conscription_of(self, action: "AgentAction | None") -> "Conscription | None":
        """The mode by which this act takes others' bodies; ``None`` = it takes nobody this time.

        Requires both a declared mode and actual ``claims`` (not people in the target): a MOVE
        travelling alone takes nobody, a five-recipient message spends nobody's turn.
        """
        if action is None or not action.target.claims:
            return None
        executor = self._executor_registry.get_executor(action.action_type)
        return getattr(executor, "conscription", None)

    async def arbitrate(
        self,
        planned_steps: List[tuple[str, "Agent", AgentStepPlan]],
        agents_dict: Dict[str, "Agent"],
        step: int,
        in_progress_at_step_start: set[str],
    ) -> tuple[dict[str, ArbitratedAction], list[dict[str, Any]]]:
        """Collapse this step's mutually-unaware decisions onto the shared world.

        Two phases: ``_resolve`` only keeps the books (no start, no teardown, no world), then
        ``_enact`` lands admissions in adjudication order. So an admission revoked by a later
        compulsory conscription is simply struck: it never happened, nothing to restore.

        It writes no agent state itself. ``Conscription.COMPEL`` on a body already in an
        execution does end that execution with writeback, through
        ``ExecutionProcessor.teardown_with_writeback``, i.e. in the feedback layer.
        """
        plans_by_id = {plan.agent_id: plan for _phase, _agent, plan in planned_steps}
        admissions, claimed_by, denied = self._resolve(
            planned_steps, plans_by_id, agents_dict, step, in_progress_at_step_start,
        )
        return await self._enact(admissions, claimed_by, denied, plans_by_id, agents_dict, step)

    def _resolve(
        self,
        planned_steps: List[tuple[str, "Agent", AgentStepPlan]],
        plans_by_id: dict[str, AgentStepPlan],
        agents_dict: Dict[str, "Agent"],
        step: int,
        in_progress_at_step_start: set[str],
    ) -> tuple[dict[str, _Admission], dict[str, str], dict[str, str]]:
        """Adjudication: who is admitted, whom they enlist, who is rejected. Pure bookkeeping;
        the world and registry it reads are as of step start.

        Returns ``(admissions, claimed_by, denied)``: admissions in adjudication order (which is
        landing order); ``claimed_by`` maps enlisted → initiator; ``denied`` maps rejected →
        reason.

        Walks ``planned_steps`` in initiative order. Two gates:

        - ``consumed``: bodies already spent this step, seeded with
          ``in_progress_at_step_start``. Such a body can't be **invited**, only taken by
          ``Conscription.COMPEL`` (tearing down its task); that is the whole difference between
          the modes. The seed matters because ``is_agent_active`` misses executions that
          completed on the step-start tick. Interrupted bodies are not seeded (free again);
          denied bodies are never added (the denial changed nothing). A body belongs to at most
          one admission.
        - ``claimed_by``: a body enrolled as a co-participant of someone else's action.

        Being the *target* of an immediate action (PHYSICAL/COVERT/SEND) consumes nothing; only
        enrollment does. Contention branches on conscription mode, never action type; who is
        claimed comes from the read-only ``ActionExecutor.claim_bodies``.

        When COMPEL hits an action admitted earlier this step, that admission is revoked: its
        initiator is rejected, its other bodies freed and re-queued at the back, along with
        those it had turned away. Each agent is re-queued at most once, so the walk ends.

        Not handled: item/location contention among independent immediate actions, and joint
        resolution of symmetric mutual actions (they need a resource ledger with no consumer yet).
        """
        admissions: dict[str, _Admission] = {}
        claimed_by: dict[str, str] = {}
        denied: dict[str, str] = {}
        # Rejected-as-busy → the bodies that blocked them; revoking those makes the reason false.
        blocked_by: dict[str, set[str]] = {}
        consumed: set[str] = set(in_progress_at_step_start)
        planned_ids = set(plans_by_id)
        queue: deque[AgentStepPlan] = deque(plan for _p, _a, plan in self._initiative_order(planned_steps))
        reached: set[str] = set()
        requeued: set[str] = set()

        def revoke(owner: str, *, seized: set[str], by: str) -> list[str]:
            """Revoke the action owner admitted this step; return the bodies it would have torn
            down, which the seizer now tears down instead."""
            adm = admissions.pop(owner)
            requeued_before = set(requeued)
            taken = "、".join(self._directory.agent_name(pid) for pid in (owner, *adm.co_participants) if pid in seized)
            if owner not in seized:
                consumed.discard(owner)
                denied[owner] = f"{taken}被{self._directory.agent_name(by)}强行拉走"
            for pid in adm.co_participants:
                if pid in seized:
                    continue
                claimed_by.pop(pid, None)
                # Still taken by an execution from before this step, which wasn't torn down.
                if pid not in in_progress_at_step_start:
                    consumed.discard(pid)
                pid_plan = plans_by_id.get(pid)
                if (
                    pid in reached and pid not in requeued and pid not in denied
                    and pid_plan is not None and pid_plan.action is not None
                ):
                    requeued.add(pid)
                    queue.append(pid_plan)
            # Re-adjudicate those turned away by this action: it never happened, and a stale
            # "he is talking with someone" would enter their first-person memory.
            members = {owner, *adm.co_participants}
            for pid, blockers in list(blocked_by.items()):
                if pid in denied and pid not in requeued and blockers & members:
                    del denied[pid], blocked_by[pid]
                    requeued.add(pid)
                    queue.append(plans_by_id[pid])
            logger.info(
                "admission_revoked",
                extra={
                    "step": step, "agent_id": owner, "seized_by": by, "seized": sorted(seized),
                    "requeued": sorted(requeued - requeued_before),
                },
            )
            return [pid for pid in adm.seized if pid in seized]

        while queue:
            plan = queue.popleft()
            agent_id = plan.agent_id
            reached.add(agent_id)

            # Enrollment doesn't depend on the agent's own decision, so this precedes the
            # no-action check: a decision-failed agent can still passive-join.
            if agent_id in claimed_by:
                continue

            # No action → no entry, zero state change (Rule 1 tier 1), but the body stays FREE:
            # still conscriptable this step. Three causes differ only in log level:
            #   NOT_SCHEDULED — the cadence gate didn't ask (debug: common and healthy).
            #   NO_ACTION     — deliberately stood down (info).
            #   FAILED/unknown — the decision LLM was unavailable (warning).
            if plan.action is None:
                if plan.decision_status is DecisionStatus.NOT_SCHEDULED:
                    logger.debug(
                        "agent_step_not_scheduled",
                        extra={"agent_id": agent_id, "step": step},
                    )
                elif plan.decision_status is DecisionStatus.NO_ACTION:
                    logger.info(
                        "agent_step_no_action",
                        extra={"agent_id": agent_id, "step": step},
                    )
                else:
                    logger.warning(
                        "agent_step_skipped_no_decision",
                        extra={"agent_id": agent_id, "step": step},
                    )
                continue

            executor = self._executor_registry.get_executor(plan.action.action_type)
            if executor is None:
                # Invariant guard: build_default_registry asserts completeness, so this is a config
                # bug. Skip rather than fabricate feedback for a missing type.
                logger.error("no_executor_registered", extra={"action_type": plan.action.action_type.value})
                continue

            # Ask unconditionally: mode only answers what to do on hitting a committed body.
            mode = self._conscription_of(plan.action)
            claimed = executor.claim_bodies(
                plan.action, agents=agents_dict, environment=self._environment,
            )
            co_participants = [pid for pid in claimed if pid != agent_id]

            # Under either mode, nobody conscripts a corpse (still visible until end-of-step
            # cleanup) or someone with no plan entry and no execution to book against.
            unresponsive = [
                pid for pid in co_participants
                if pid not in agents_dict
                or not agents_dict[pid].is_active
                or (
                    pid not in planned_ids
                    and pid not in consumed
                    and not self._executor_registry.is_agent_active(pid)
                )
            ]
            # Committed bodies: INVITE rejects the whole action ("he's talking with someone" is a
            # valid world fact); COMPEL tears down what he's doing and enlists him.
            committed = [
                pid for pid in co_participants
                if pid not in unresponsive
                and (pid in consumed or self._executor_registry.is_agent_active(pid))
            ]
            # Decided once and shared by the rejection and the teardown below. Don't infer
            # "no rejection ⟹ COMPEL": a new rejection reason or mode would then silently tear down
            # for non-compulsory actions.
            seize = bool(committed) and mode is Conscription.COMPEL
            # Rejected agents don't enter consumed: looking "busy" would give later agents a false
            # reason and turn one contention failure into two.
            if unresponsive or (committed and not seize):
                if unresponsive:
                    names = "、".join(self._directory.agent_name(pid) for pid in unresponsive)
                    denied[agent_id] = f"{names}毫无回应"
                else:
                    denied[agent_id] = "，".join(
                        self._busy_reason(pid, admissions, claimed_by) for pid in committed
                    )
                    blocked_by[agent_id] = set(committed)
                logger.info(
                    "admission_denied",
                    extra={
                        "step": step, "agent_id": agent_id,
                        "action_type": plan.action.action_type.value,
                        "unresponsive": unresponsive,
                        # None = already in another execution at step start.
                        "blocked_by": {
                            pid: claimed_by.get(pid) or (pid if pid in admissions else None)
                            for pid in committed
                        },
                    },
                )
                continue

            # Admitted earlier this step → revoke. In an execution from before this step → tear it
            # down at landing (it really happened; each side remembers it).
            to_tear_down: list[str] = []
            if seize:
                seized = set(committed)
                for pid in committed:
                    owner = claimed_by.get(pid) or (pid if pid in admissions else None)
                    if owner is not None and owner in admissions:
                        to_tear_down.extend(revoke(owner, seized=seized, by=agent_id))
                    elif owner is None:
                        to_tear_down.append(pid)

            admissions[agent_id] = _Admission(
                plan=plan, action=plan.action, executor=executor,
                co_participants=co_participants, seized=to_tear_down,
            )
            consumed.add(agent_id)
            for pid in co_participants:
                claimed_by[pid] = agent_id
                consumed.add(pid)

        return admissions, claimed_by, denied

    async def _enact(
        self,
        admissions: dict[str, _Admission],
        claimed_by: dict[str, str],
        denied: dict[str, str],
        plans_by_id: dict[str, AgentStepPlan],
        agents_dict: Dict[str, "Agent"],
        step: int,
    ) -> tuple[dict[str, ArbitratedAction], list[dict[str, Any]]]:
        """Land admissions in adjudication order: tear down the seized old tasks, then start,
        then emit records; rejections are booked last.

        Feasibility checks inside start() see the world as changed by earlier admissions.
        """
        arbitration: dict[str, ArbitratedAction] = {}
        # phase="interrupt" records for torn-down tasks, or they would vanish from the feed.
        seize_records: list[dict[str, Any]] = []

        for agent_id, adm in admissions.items():
            plan = adm.plan
            # Tear down before the seizer's start(), while he still stands where he was, or the
            # end is recorded at the place he is dragged to. It may end a third party's
            # execution too (a conversation loses a side).
            #
            # cause says only what onlookers see: who took him. Not why: that is the seizer's
            # intent and would enter the seized agent's first-person memory.
            cause = f"被{self._directory.agent_name(agent_id)}强行拉走"
            for pid in adm.seized:
                seize_records.extend(await self._processor.teardown_with_writeback(
                    agents_dict[pid], agents_dict, step, cause=cause, trigger="conscription",
                ))

            # start() always returns an ActionExecutionState (a feasibility failure is a
            # create_failed one), so every admission takes one registration path.
            # Starting can move people (MOVE's first waypoint), so record positions.
            was_at = {
                pid: self._environment.get_body_location(pid)
                for pid in (agent_id, *adm.co_participants)
            }
            try:
                with observe_stage(Stage.ACTION, agent_id=agent_id):
                    exec_state = await adm.executor.start(
                        adm.action,
                        step,
                        agents=agents_dict,
                        environment=self._environment,
                        message_system=self._message_system,
                    )
            except Exception as exc:  # noqa: BLE001 — Rule 5: an executor exception must not kill the run
                # Don't invent an execution: nobody gets an entry; they re-plan next step. Any
                # forced teardown already happened, so this is a real bug: log at error.
                logger.error(
                    "executor_start_failed",
                    extra={
                        "step": step,
                        "agent_id": agent_id,
                        "action_type": adm.action.action_type.value,
                        "seized": adm.seized,
                        "error": str(exc),
                    },
                )
                continue

            # Perception ran earlier, so refresh the situation of anyone who moved, or later
            # cognition this step places him at his starting point.
            for pid, before in was_at.items():
                moved = agents_dict.get(pid)
                if moved is None or self._environment.get_body_location(pid) == before:
                    continue
                moved.refresh_situation(self._environment.spatial_for(agent_id=pid, step=step))

            # Records follow ``start()``, not ``claim_bodies``. A mismatch loses someone a step.
            enrolled = [pid for pid in exec_state.participant_ids if pid != agent_id]
            if set(enrolled) != set(adm.co_participants):
                logger.error(
                    "claim_bodies_diverged_from_start",
                    extra={
                        "step": step, "agent_id": agent_id,
                        "action_type": exec_state.action_type.value,
                        "claimed": adm.co_participants, "enrolled": enrolled,
                    },
                )

            self._executor_registry.add_active(exec_state)
            logger.debug(
                "action_started",
                extra={
                    "step": step,
                    "agent_id": agent_id,
                    "action_type": exec_state.action_type.value,
                    "remaining_steps": exec_state.remaining_steps,
                    "co_participants": enrolled,
                },
            )
            arbitration[agent_id] = self._start_arbitrated(
                plan, exec_state, location_id=self._environment.get_body_location(agent_id),
            )
            # Built after the initiator's start(): location and wording come from this execution.
            for pid in enrolled:
                pid_plan = plans_by_id.get(pid)
                if pid_plan is None:
                    # Unreachable: adjudication enlists only agents with a plan entry.
                    logger.warning(
                        "conscription_join_unresolved",
                        extra={"agent_id": pid, "initiator": agent_id, "step": step},
                    )
                    continue
                # dropped_own: his own decision was deferred at commit.
                logger.debug(
                    "initiative_conscripted",
                    extra={
                        "step": step, "agent_id": pid, "initiator": agent_id,
                        "dropped_own": pid_plan.action is not None,
                    },
                )
                arbitration[pid] = self._passive_join_arbitrated(pid_plan, exec_state)

        # Denials last, once enlistment is settled: one verdict per body per step, and an enlisted
        # body gets the join record.
        for agent_id, reason in denied.items():
            if agent_id in claimed_by:
                logger.info(
                    "denial_superseded_by_enrollment",
                    extra={"step": step, "agent_id": agent_id, "initiator": claimed_by[agent_id]},
                )
                continue
            plan = plans_by_id[agent_id]
            rejection = self._rejected_result(plan, reason=reason)
            # The decision's own text: a denied action never starts to generate one.
            failed_state = ActionExecutionState.create_failed(
                action_type=plan.action.action_type, initiator_id=agent_id,
                failure_result=rejection, started_step=step,
                purpose=plan.action.action_description or plan.action.action_type.value,
            )
            self._executor_registry.add_active(failed_state)
            arbitration[agent_id] = self._start_arbitrated(
                plan, failed_state, location_id=self._environment.get_body_location(agent_id),
            )

        return arbitration, seize_records

    def _duration_hint(self, steps: int) -> str:
        return describe_duration(steps, self._clock.config.seconds_per_step)

    def _start_arbitrated(
        self,
        plan: AgentStepPlan,
        exec_state: ActionExecutionState,
        *,
        location_id: str,
    ) -> ArbitratedAction:
        """The initiator's record for a newly admitted execution's first beat.

        Surfaces the executor's opening_outcome, else a generic marker that also names the
        actor. The real outcome lands at complete().
        """
        marker = exec_state.opening_outcome or (
            f"{self._directory.agent_name(plan.agent_id)}着手{exec_state.purpose}"
            f"（预计{self._duration_hint(exec_state.estimated_steps)}）"
        )
        stub = AgentAction(
            agent_id=plan.agent_id,
            step=plan.step,
            action_type=exec_state.action_type,
            action_description=exec_state.purpose,
        )
        # The three channels stay independent: observations come only from the executor's
        # opening_observations, never from the marker; factual_memory is empty because start
        # writes no memory.
        return ArbitratedAction(
            agent_id=plan.agent_id,
            action_result=ActionResult(
                action=stub,
                expected_outcome=exec_state.purpose,
                outcome=marker,
                observations=list(exec_state.opening_observations),
                # This beat did start; not a verdict on the action (the opening beat skips feedback).
                succeeded=True,
                factual_memory="",
            ),
            location_id=location_id,
            visible_agent_ids=list(plan.spatial.visible_agent_ids),
            message_ids=[m.id for m in plan.inbox],
            ongoing_execution_id=exec_state.execution_id,
            is_passive_join=False,
        )

    def _passive_join_arbitrated(
        self,
        plan: AgentStepPlan,
        exec_state: ActionExecutionState | None,
    ) -> ArbitratedAction:
        """The record for a body conscripted into someone else's action.

        ``plan.action`` may be None (decision failed, still pulled in); ``exec_state`` is
        authoritative.
        """
        initiator_name = self._directory.agent_name(exec_state.initiator_id) if exec_state else "对方"
        # Joiner's viewpoint: the initiator's purpose ("与李世民交谈") would be self-referential here.
        
        action_description = (
            participant_action_desc(self._directory, plan.agent_id, exec_state) if exec_state
            else (plan.action.action_description if plan.action else "")
        )
        purpose = exec_state.purpose if exec_state else (plan.action.action_description if plan.action else "")
        action_type = (
            exec_state.action_type if exec_state
            else (plan.action.action_type if plan.action else ActionType.REST)
        )
        exec_id = exec_state.execution_id if exec_state else None
        participant_name = self._directory.agent_name(plan.agent_id)
        # Wording follows the mode: "invited" for someone dragged off would launder compulsion
        # into consent.
        joined = "邀入" if self._mode_of_execution(exec_state) is Conscription.INVITE else "强制拉入"
        outcome = f"{participant_name}被{initiator_name}{joined}「{purpose}」"
        joined_at = self._environment.get_body_location(plan.agent_id)
        # No onlooker narration for joining: the starting execution already named both sides,
        # and carry's dedup wouldn't catch a second sentence for the same event. Onlooker text is
        # the executor's to author, not the arbiter's.
        stub = AgentAction(
            agent_id=plan.agent_id,
            step=plan.step,
            action_type=action_type,
            action_description=action_description,
            # Without a target the renderer can't draw any line.
            target=participant_target(plan.agent_id, exec_state) if exec_state else ActionTarget(),
        )
        return ArbitratedAction(
            agent_id=plan.agent_id,
            action_result=ActionResult(
                action=stub,
                expected_outcome="在被加入的行动中自然顺利进行",
                outcome=outcome,
                observations=[],
                succeeded=True,
                # Empty: ``join_ongoing_action`` writes no memory, and a filled field would only
                # pretend to reach it.
                factual_memory="",
            ),
            location_id=joined_at,
            visible_agent_ids=list(plan.spatial.visible_agent_ids),
            message_ids=[m.id for m in plan.inbox],
            ongoing_execution_id=exec_id,
            is_passive_join=True,
        )

    def _busy_reason(
        self, pid: str, admissions: dict[str, _Admission], claimed_by: dict[str, str],
    ) -> str:
        """What X is doing, as the rejected agent sees it (see _BUSY_WITH).

        Read from the books for this step's admissions, else the registry; an execution that
        finished on the step-start tick falls back to the vague phrase.
        """
        name = self._directory.agent_name(pid)
        owner = claimed_by.get(pid) or (pid if pid in admissions else None)
        if owner is not None:
            adm = admissions[owner]
            action_type = adm.action.action_type
            members = [owner, *adm.co_participants]
        else:
            exec_state = self._executor_registry.get_active_for_agent(pid)
            if exec_state is None:
                return f"{name}{_BUSY_UNSEEN}"
            action_type = exec_state.action_type
            members = list(exec_state.participant_ids)
        if action_type == ActionType.TALK:
            partners = [p for p in members if p != pid]
            if partners:
                who = "、".join(self._directory.agent_name(p) for p in partners)
                return f"{name}正与{who}交谈"
        return f"{name}{_BUSY_WITH.get(action_type, _BUSY_UNSEEN)}"

    def _rejected_result(
        self,
        plan: AgentStepPlan,
        *,
        reason: str,
    ) -> ActionResult:
        """The failure result for an intent the arbiter denies (enrollment contention).

        Carried into a create_failed execution, so it takes the one completion path like any
        other failure. Nothing happened in the world, so onlookers see nothing."""
        actor_name = self._directory.agent_name(plan.agent_id)
        desc = plan.action.action_description
        stub = AgentAction(
            agent_id=plan.agent_id,
            step=plan.step,
            action_type=plan.action.action_type,
            action_description=desc,
            target=plan.action.target,
        )
        # desc is the actor's own first-person wording: quote it and strip its end punctuation,
        # or it renders as "我本想我走到他面前…细节。，却因".
        quoted = strip_end_punct(desc)
        return ActionResult(
            action=stub,
            expected_outcome=plan.action.expected_outcome,
            outcome=f"{actor_name}本想做「{quoted}」，却因{reason}未能如愿。",
            observations=[],
            succeeded=False,
            failure_reason=reason,
            factual_memory=f"我本想做「{quoted}」，却因{reason}未能如愿。",
        )
