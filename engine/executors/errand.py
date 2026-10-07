"""Errand action executor: assigns a task to an NPC tool-body to carry out elsewhere.

No LLM on this path, because there is nothing to adjudicate: an Npc has no will (it is
"assigned", never asked), the four axes of ``ErrandOrder`` are within any body's means, and
every failure is an objective check (busy or restrained → ``start`` rejects; unreachable →
``NpcRunner`` reports it; item not in the assigner's hands → rejected). Templates also can't
leak the order into the observation, where an LLM might.

Assigning takes one step: the point of sending someone is not waiting yourself.

The trip itself is driven by ``engine.npc_runner``, not here: a residue that outlives its
action belongs to world advancement. Don't stretch the execution to drive it: on the assigner
it would lock him for N beats, on the body it would push a non-``Agent`` id through
arbitration, and an N-step countdown can't model a state machine that can stall or turn back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from core.interfaces.action import (
    ActionResult, ActionType, AgentAction, EntityStateChange, ErrandOrder,
)
from core.interfaces.directory import WorldDirectory
from core.logging import get_logger
from core.interfaces.execution import TickResult
from engine.executors.base import ActionExecutionState, ActionExecutor
from engine.narration import observed_here, scene_line
from engine.scene import observe_location

if TYPE_CHECKING:
    from agent.agent import Agent
    from engine.environment import EnvironmentSystem
    from engine.message_system import MessageSystem

logger = get_logger(__name__)

#: Carries the ErrandOrder from start to complete in ``state.extra``.
_ORDER_KEY = "errand_order"


def _render_errand(
    order: ErrandOrder, *, environment: "EnvironmentSystem", directory: WorldDirectory,
) -> list[str]:
    """The only order → text conversion: reads ``ErrandOrder``'s four axes without a dispatch
    table, resolving every id here to a name (never a bare id).

    Each clause names the place rather than "那处": clauses are spliced into different
    sentences, where a back-reference can miss. The place-only cell is produced here too.
    """
    lines: list[str] = []
    destination = environment.narrative_location_name(order.destination_id)
    to_name = directory.agent_name(order.recipient_id) if order.recipient_id else ""
    if to_name and not order.item_id and not order.message:
        # Don't drop: the runner treats this as "see whether he's there", not a plain look.
        lines.append(f"去看看{to_name}在不在{destination}")
    if order.item_id:
        item = environment.get_entity(order.item_id)
        item_name = (item.name if item is not None else "") or "某物"
        lines.append(
            f"去到{destination}把{item_name}交到{to_name}手上" if to_name
            else f"去把{item_name}放在{destination}"
        )
    if order.message:
        lines.append(
            f"到{destination}对{to_name}带一句话：「{order.message}」" if to_name
            else f"去{destination}当众说出：「{order.message}」"
        )
    # All four axes empty = go take a look. That is a task, not "no instructions".
    return lines or [f"去{destination}走一趟"]


class ErrandExecutor(ActionExecutor):
    """Assigns a task to an NPC tool-body. Deterministic: it is told, it goes."""

    # The errand lands in ``start``; complete only reports it and declares the item handover.
    mutates_world_during_complete = False
    # Never put the Npc in claims: arbitration can't find it in agents and would reject the action.
    conscription = None

    def __init__(self, directory: WorldDirectory) -> None:
        self._directory = directory

    async def start(
        self,
        action: AgentAction,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> ActionExecutionState:
        order = getattr(action.intent, "errand", None)
        npc = environment.get_npc(order.npc_id) if isinstance(order, ErrandOrder) else None
        # Objective feasibility only ("can it be spared"), never "is it willing".
        reason = ""
        if not isinstance(order, ErrandOrder) or npc is None:
            reason = "此处没有可派的人手"
        elif npc.errand is not None:
            reason = f"{npc.name or '那人'}正为旁人跑一趟，一时抽不开身"
        elif npc.condition is not None:
            reason = f"{npc.name or '那人'}此刻动弹不得"
        elif order.item_id:
            # Don't rely on decision-side binding alone: the handover rewrites the holder
            # unconditionally, so an item held across town would jump into the bearer's hands.
            item = environment.get_entity(order.item_id)
            if item is None or item.owner_id != action.agent_id:
                reason = f"{(item.name if item is not None else '') or '那样东西'}已不在他手上"
        if reason:
            return self._unavailable(action, step, reason, environment=environment)

        # Check and occupy in one statement here, not at landing: otherwise two senders in one
        # beat both see the body free, and the loser's success memory is already written.
        # Starts in ``_enact`` run sequentially. The only remaining False is an assigner not on
        # solid ground.
        if not environment.assign_errand(order, requester_id=action.agent_id):
            return self._unavailable(
                action, step, "指派失败", environment=environment,
            )

        state = ActionExecutionState.create(
            action_type=ActionType.ERRAND,
            initiator_id=action.agent_id,
            participant_ids=[action.agent_id],
            purpose=action.action_description or "派人办事",
            started_step=step,
            # Giving an order takes one beat however far the bearer goes (see
            # test_executor_duration_authority). Don't trust the actor's estimate: one sentence
            # would pin him in place.
            estimated_steps=1,
            opening_outcome="",
            target=action.target,
            expected_outcome=action.expected_outcome,
        )
        state.extra[_ORDER_KEY] = order
        return state

    async def tick(
        self,
        state: ActionExecutionState,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> List[TickResult]:
        # Duration is fixed at 1 (born-zero); this is never reached.
        return []

    async def complete(
        self,
        state: ActionExecutionState,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> List[ActionResult]:
        stored = self._stored_result(state)
        if stored is not None:
            return [stored]

        location = observe_location(environment, state.initiator_id)
        order = state.extra.get(_ORDER_KEY)
        npc = environment.get_npc(order.npc_id) if isinstance(order, ErrandOrder) else None
        if not isinstance(order, ErrandOrder) or npc is None:
            # The bearer vanished this beat: null step, don't invent "he set off" (Rule 1).
            logger.warning(
                "errand_bearer_vanished",
                extra={"agent_id": state.initiator_id, "step": step},
            )
            return [self._null_step(state, step=step, location=location)]

        actor_name = self._directory.agent_name(state.initiator_id)
        bearer = npc.name or "那人"
        errand_clause = "，".join(_render_errand(
            order, environment=environment, directory=self._directory,
        ))
        # The item must reach the bearer now, or every carrying errand arrives empty-handed.
        # The gate in ``start`` guarantees he still holds it. No ``perception``: the handover is
        # folded into the single observation below.
        handover = [EntityStateChange(entity_id=order.item_id, owner_id=order.npc_id)] if (
            order.item_id
        ) else []

        # Three channels written separately: the observation must not contain what was ordered.
        outcome = scene_line(location, f"{actor_name}派{bearer}{errand_clause}。")
        item = environment.get_entity(order.item_id) if order.item_id else None
        handover_clause = f"把{(item.name if item is not None else '') or '某物'}交到他手上，" if (
            order.item_id
        ) else ""
        observation = scene_line(
            location,
            f"{actor_name}叫住{bearer}交代了几句，{handover_clause}{bearer}随即动身。",
        )
        return [ActionResult(
            action=self._stub(state, step),
            expected_outcome=state.expected_outcome or state.purpose,
            outcome=outcome,
            succeeded=True,
            observations=observed_here(environment, state.initiator_id, observation),
            # A lurker nearby hears what was ordered.
            happening=outcome,
            factual_memory=f"我派{bearer}{errand_clause}。",
            # A receipt, not an instruction: ``start`` already sent this errand out.
            errand_orders=[order],
            entity_state_changes=handover,
        )]

    def _stub(self, state: ActionExecutionState, step: int) -> AgentAction:
        return AgentAction(
            agent_id=state.initiator_id,
            step=step,
            action_type=ActionType.ERRAND,
            action_description=state.purpose,
            target=state.target,
            estimated_steps=1,
        )

    def _unavailable(
        self, action: AgentAction, step: int, reason: str, *,
        environment: "EnvironmentSystem",
    ) -> ActionExecutionState:
        """Can't send it: unavailable, not unwilling (an Npc has no will to reject with). A real
        world event, remembered normally; ``reason`` is a structured fact."""
        location = observe_location(environment, action.agent_id)
        actor_name = self._directory.agent_name(action.agent_id)
        outcome = scene_line(location, f"{actor_name}本想派人跑一趟，却因{reason}未能成行。")
        return ActionExecutionState.create_failed(
            action_type=ActionType.ERRAND,
            initiator_id=action.agent_id,
            failure_result=ActionResult(
                action=action,
                expected_outcome=action.expected_outcome,
                outcome=outcome,
                observations=observed_here(environment, action.agent_id, outcome),
                succeeded=False,
                failure_reason=reason,
                factual_memory=f"想派人跑一趟未果：{reason}",
            ),
            started_step=step,
            purpose=action.action_description or "派人办事",
        )

    def _null_step(
        self, state: ActionExecutionState, *, step: int, location: str,
    ) -> ActionResult:
        """The honest product when the bearer vanished: asserts neither that it was sent nor
        that it wasn't."""
        return ActionResult(
            action=self._stub(state, step),
            expected_outcome=state.expected_outcome or state.purpose,
            outcome=scene_line(location, (
                f"{self._directory.agent_name(state.initiator_id)}想派人办一件事，"
                f"一时未能确知派没派出去。"
            )),
            succeeded=False,
            adjudication_failed=True,
        )
