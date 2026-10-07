"""The action executor framework: ``ActionExecutionState``, ``ActionExecutor``, ``Conscription``,
and what an execution means per participant and on the record (``participant_*`` / ``*_semantics``).

Scene assembly lives in ``engine.scene``; outcome wording in ``engine.narration``.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, List

from core.interfaces.action import ActionResult, ActionTarget, ActionType, AgentAction, Observed, Ref
from core.interfaces.execution import TickResult
from engine.narration import (
    format_intent_clause, format_interrupt_reason_3p, format_interrupt_thought, scene_line,
)
from engine.scene import observe_location

if TYPE_CHECKING:
    from agent.agent import Agent
    from core.interfaces.directory import WorldDirectory
    from engine.environment import EnvironmentSystem
    from engine.message_system import MessageSystem


def participant_action_desc(
    directory: "WorldDirectory", agent_id: str, exec_state: "ActionExecutionState"
) -> str:
    """Viewpoint-correct description of exec_state for one participant.

    ``purpose`` is the initiator's framing (a TALK's "与李世民交谈" would be self-referential
    read by 李世民), so a conscripted participant gets a joiner-POV line instead.
    """
    if agent_id == exec_state.initiator_id:
        return exec_state.purpose
    initiator_name = directory.agent_name(exec_state.initiator_id)
    # No bracket around the purpose: callers quote this whole line in 「」, so one would nest.
    return f"参与{initiator_name}发起的行动，行动意图：{exec_state.purpose}"


def participant_intent(agent_id: str, exec_state: "ActionExecutionState") -> str:
    """Viewpoint-correct "what I'm after": only the initiator wrote ``expected_outcome``, so
    anyone else gets "" rather than an invented motive."""
    if agent_id != exec_state.initiator_id:
        return ""
    return format_intent_clause(exec_state.expected_outcome)


def participant_target(agent_id: str, exec_state: "ActionExecutionState") -> "ActionTarget":
    """Viewpoint-correct target for one participant.

    A conscripted participant drops himself from ``acts_on`` (or the renderer draws a line to
    himself); if nothing is left, from his side the act aims at the initiator. ``claims`` keeps
    only himself, matching his closing record (``_landing_target``).
    """
    if agent_id == exec_state.initiator_id:
        return exec_state.target
    acts_on = [ref for ref in exec_state.target.acts_on if ref.id != agent_id]
    if not acts_on:
        acts_on = [Ref.agent(exec_state.initiator_id)]
    return ActionTarget(
        acts_on=acts_on,
        claims=[ref for ref in exec_state.target.claims if ref.id == agent_id],
        reaches=[ref for ref in exec_state.target.reaches if ref.id != agent_id],
    )


def execution_annotations(
    directory: "WorldDirectory", exec_state: "ActionExecutionState"
) -> dict[str, Any]:
    """This execution's identity, attached to every LLM call it triggers, so the audit can
    own results that don't pair with that step's decision (multi-step completions,
    conscripts). Names, not ids: the audit's judge reads them."""
    return {
        "execution_id": exec_state.execution_id,
        "action_description": exec_state.purpose,
        "action_initiator": directory.agent_name(exec_state.initiator_id),
        "action_participants": [directory.agent_name(pid) for pid in exec_state.participant_ids],
    }


def target_semantics(target: "ActionTarget | None") -> dict[str, Any]:
    """The action's three relations (``ActionTarget``'s axes, unflattened) for the observer.

    Separate from ``action_semantics`` because tick beats have no result yet but still need
    the aim. Every id carries its ``kind`` because the renderer routes on it. Results
    (``affected_entity_ids``, ``overheard_by``) are not aims and don't belong here.
    """
    tgt = target if target is not None else ActionTarget()

    def refs(group: "list[Ref]") -> list[dict[str, str]]:
        return [{"kind": ref.kind, "id": ref.id} for ref in group]

    return {
        "target": {
            "acts_on": refs(tgt.acts_on),
            "claims": refs(tgt.claims),
            "reaches": refs(tgt.reaches),
        }
    }


def action_semantics(action: "AgentAction", result: "ActionResult") -> dict[str, Any]:
    """A render-neutral descriptor of what was done, to what — never how to draw it.

    - ``deed``: the observable verb (see ``Deed``).
    - ``target``: the aim (``target_semantics``); it survives failure, unlike
      ``affected_entity_ids``.
    - ``affected_entity_ids``: entities this deed changed or created. Which are new the
      observer derives from the entity table.

    The deed must name the act, not the consequence: "an object changed" is true of taking and
    of smashing alike. PHYSICAL's adjudicator declares it; every other type is its own deed.
    """
    aim = target_semantics(action.target)
    action_type = str(getattr(action.action_type, "value", action.action_type))
    # PHYSICAL's deed is empty when nothing was adjudicated. An empty deed is not the mark of
    # a non-event: a not_executed TALK still reports "talk", so read ``not_executed``.
    deed = result.deed if action_type == ActionType.PHYSICAL.value else action_type

    return {
        **aim,   # {"target": {acts_on, claims, reaches}}
        "deed": deed,
        # A rejected spawn has an empty ``entity_id`` and touched nothing.
        "affected_entity_ids": [
            *(c.entity_id for c in (result.entity_state_changes or [])),
            *(s.entity_id for s in (result.entity_spawns or []) if s.entity_id),
        ],
    }


@dataclass
class ActionExecutionState:
    """
    Cross-step state container for an ongoing multi-step action.
    Owned by ActionExecutorRegistry; survives across ticks until complete/interrupted.
    """

    execution_id: str
    action_type: ActionType
    initiator_id: str
    participant_ids: List[str]
    started_step: int
    estimated_steps: int
    remaining_steps: int
    purpose: str
    # Carried for the execution's whole life so results and ticks can say whom it was aimed
    # at. COVERT binds nothing here; its target lives only in action_description.
    target: ActionTarget = field(default_factory=ActionTarget)
    expected_outcome: str = ""
    # God-view opening beat at the start step.
    opening_outcome: str = ""
    # Bystander view of the opening, one entry per place (MOVE is seen at origin and
    # waypoint). Independent of opening_outcome, never derived from it; empty = nothing
    # observable, so a privileged opening can't leak.
    opening_observations: list[Observed] = field(default_factory=list)
    # Bodies that hear without being enrolled (TALK's listeners). Arbitration, conscription
    # and teardown must read participant_ids alone, so a listener never spends its turn.
    listener_ids: List[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        action_type: ActionType,
        initiator_id: str,
        participant_ids: List[str],
        purpose: str,
        started_step: int,
        estimated_steps: int,
        opening_outcome: str,
        target: ActionTarget,   # required: an executor must not be able to forget it
        expected_outcome: str = "",
        opening_observations: list[Observed] | None = None,
        listener_ids: List[str] | None = None,
    ) -> "ActionExecutionState":
        # The start step is the first step, so N steps = 1 start + N-1 ticks. remaining == 0
        # ⟺ ready to complete, which unifies immediate and ongoing actions. Born-zero callers
        # pass opening_outcome="": the same-step sweep replaces it with the completion.
        return cls(
            execution_id=(
                f"{action_type.value}_{initiator_id}_{started_step}_{uuid.uuid4().hex[:6]}"
            ),
            action_type=action_type,
            initiator_id=initiator_id,
            participant_ids=list(participant_ids),
            started_step=started_step,
            estimated_steps=estimated_steps,
            remaining_steps=max(0, estimated_steps - 1),
            purpose=purpose,
            target=target,
            expected_outcome=expected_outcome,
            opening_outcome=opening_outcome,
            opening_observations=list(opening_observations or []),
            listener_ids=list(listener_ids or []),
        )

    @classmethod
    def create_failed(
        cls,
        *,
        action_type: ActionType,
        initiator_id: str,
        failure_result: "ActionResult",
        started_step: int,
        purpose: str = "",
    ) -> "ActionExecutionState":
        """A feasibility/rejection failure as an actor-only, same-step-completing execution
        carrying its result in ``extra["completed_result"]``, so it flows through the one
        completion path."""
        # A "couldn't execute" non-event, not a defeat; the renderer mutes it.
        failure_result.not_executed = True
        state = cls.create(
            action_type=action_type,
            initiator_id=initiator_id,
            participant_ids=[initiator_id],
            purpose=purpose,
            started_step=started_step,
            estimated_steps=1,
            opening_outcome="",
            target=failure_result.action.target,
            expected_outcome=failure_result.expected_outcome,
        )
        state.extra["completed_result"] = failure_result
        return state


class Conscription(str, Enum):
    """How an action takes up someone else's body when he is already committed.

    - ``INVITE``: he must be free; if busy, the action is rejected (a world fact, not a failure).
    - ``COMPEL``: his earlier action is torn down (he remembers it) and he joins.

    Neither may take a body that has already spent its turn this step: only an earlier
    commitment can be overridden.
    """

    INVITE = "invite"
    COMPEL = "compel"


class ActionExecutor(ABC):
    """
    Type-specific action execution handler.
    Lifecycle: start → tick (×N-1 steps) → complete | interrupt.

    ``start()`` always returns an ``ActionExecutionState`` (failure → ``create_failed``) and
    ``complete()`` is the sole source of results; there is no separate immediate path.
    """

    # True if complete() mutates world infrastructure (MOVE's arrival). Such executors are
    # completed serially after the concurrent read-only batch, so judges see start-of-step state.
    mutates_world_during_complete: bool = False

    # How this action takes up other bodies (see Conscription); None = only its own.
    # Arbitration orders conscripting actions first (ExecutionArbiter._initiative_order).
    conscription: "Conscription | None" = None

    @abstractmethod
    async def start(
        self,
        action: AgentAction,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> ActionExecutionState:
        """
        Called during arbitration when an agent initiates a new action.

        Only feasibility checks, placeholder side effects (MOVE → IN_TRANSIT) and the opening
        narrative; never the result. A feasibility failure returns ``create_failed(...)``.
        """

    @abstractmethod
    async def tick(
        self,
        state: ActionExecutionState,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> List[TickResult]:
        """
        Called once per step while action is IN_PROGRESS, before new decisions.
        Returns per-agent intermediate results written as low-importance memories.
        """

    @abstractmethod
    async def complete(
        self,
        state: ActionExecutionState,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> List[ActionResult]:
        """
        Called when remaining_steps reaches 0; returns one final ActionResult per participant.

        If ``_stored_result(state)`` is non-None (a ``create_failed`` failure, or a result fixed
        at start() like SEND_MESSAGE), return it verbatim instead of re-adjudicating.
        """

    @staticmethod
    def _stored_result(state: ActionExecutionState) -> "ActionResult | None":
        """The pre-built result stored at start(), or None (see ``complete()``)."""
        return state.extra.get("completed_result")

    def claim_bodies(
        self,
        action: AgentAction,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
    ) -> list[str]:
        """Which other bodies this act will take: a read-only admission check, since arbitration
        settles the whole beat before ``start()`` mutates anything. ``start()`` re-checks with the
        same criteria and wins; arbitration logs a mismatch. Default ``[]``.
        """
        return []

    async def interrupt(
        self,
        state: ActionExecutionState,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem | None" = None,
        interrupted_agent_id: str | None = None,
        thought: str = "",
        cause: str = "",
    ) -> List[ActionResult]:
        """Handle forced interruption. Default: partial-completion results.

        ``interrupted_agent_id``: the participant breaking off; matters only for
        multi-participant actions (TALK).

        ``thought``: that agent's first-person reason from ``Agent.evaluate_interrupt``.

        ``cause``: a third-person phrase for when no participant chose this (death, director
        relocation, conscription). It must reach the first-person channel too: without it the
        model invents a decision ("我中途撂下了…") or narrates an act from the purpose text.
        ``interrupted_agent_id`` then names only whose body was taken.

        Interrupts produce no ``observations``: onlookers already perceive whatever cut it off.
        Accepted cost: some interrupts go unseen; fix that one channel if it ever matters.

        The coordinator's triggering signals stop at ``evaluate_interrupt``'s prompt: never
        splice them into outcomes or memories (a raw cognition signal would leak into
        third-person channels, and external threat goals are deliberately not memories; see
        agent/perception_layer.py).
        """
        thought_part = format_interrupt_thought(thought)
        results: List[ActionResult] = []
        for agent_id in state.participant_ids:
            agent = agents.get(agent_id)
            if agent is None:
                continue
            stub = AgentAction(
                agent_id=agent_id,
                step=step,
                action_type=state.action_type,
                action_description=state.purpose,
                target=state.target,
            )
            name = agent.personality.soul.name
            location = observe_location(environment, agent_id) if environment is not None else "此处"
            cut = f"因{cause}而中断" if cause else "被打断"
            gist = scene_line(location, f"{name}的「{state.purpose}」{cut}。")
            results.append(ActionResult(
                action=stub,
                expected_outcome=state.purpose,
                outcome=gist + format_interrupt_reason_3p(name, thought),
                gist=gist,
                succeeded=False,
                factual_memory=f"正在执行「{state.purpose}」时{cut}{thought_part}",
            ))
        return results
