"""SEND_MESSAGE executor: words sent one way, to named recipients anywhere or to everyone here."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from core.action_duration import action_duration_steps
from core.interfaces.action import ActionResult, ActionType, AgentAction
from core.interfaces.directory import WorldDirectory
from core.interfaces.execution import TickResult
from core.interfaces.message import ANONYMOUS_SENDER_NAME
from engine.executors.base import ActionExecutionState, ActionExecutor
from engine.narration import scene_line
from engine.scene import observe_location

if TYPE_CHECKING:
    from agent.agent import Agent
    from engine.environment import EnvironmentSystem
    from engine.message_system import MessageSystem


class SendMessageExecutor(ActionExecutor):
    """Dispatches in ``start()`` and fixes the result there for ``complete()`` to return: once
    sent, the message is the message system's to carry, so the act is momentary."""

    momentary = True

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
        result = await self._handle_send_message(
            action, step, agents=agents, environment=environment, message_system=message_system,
        )
        state = ActionExecutionState.create(
            action_type=ActionType.SEND_MESSAGE,
            initiator_id=action.agent_id,
            participant_ids=[action.agent_id],
            purpose=action.action_description or "send message",
            started_step=step,
            estimated_steps=action_duration_steps(action.action_type, action.estimated_steps),
            opening_outcome="",
            target=action.target,
            expected_outcome=action.expected_outcome,
        )
        state.extra["completed_result"] = result
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
        # Momentary: settled in the step it was issued in, never ticked.
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
        return [stored] if stored is not None else []

    async def _handle_send_message(
        self,
        action: AgentAction,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> ActionResult:
        recipient_ids = action.target.acted_on_agents
        directed = bool(recipient_ids)
        # Directed messages cross locations (scope None); an undirected one reaches everyone here.
        location = environment.get_body_location(action.agent_id)
        location_scope = None if directed else location

        # Must match ``MessageSystem._resolve_receivers``.
        receivers = [
            aid for aid in (recipient_ids if directed else environment.agents_at(location))
            if aid != action.agent_id
        ]
        if not receivers:
            return self._unheard_result(action, step, environment=environment, directed=directed)

        if message_system is not None:
            sender_name = self._directory.agent_name(action.agent_id)
            await message_system.dispatch_from_action(
                action,
                current_step=step,
                sender_name=sender_name,
                location_scope=location_scope,
            )

        # The description is intent; only content is spoken to the recipients.
        message_text = action.content
        place = observe_location(environment, action.agent_id)
        doer_name = self._directory.agent_name(action.agent_id)
        signature = ANONYMOUS_SENDER_NAME if action.anonymous else ""
        if directed:
            recipient_text = "、".join(self._directory.agent_name(aid) for aid in recipient_ids)
            factual = f"我成功在{place}{signature}向{recipient_text}发送消息：「{message_text}」"
            outcome = scene_line(place, f"{doer_name}成功{signature}向{recipient_text}发送消息：「{message_text}」")
        else:
            # Name everyone: the memory must later answer "who heard this".
            heard_text = "、".join(self._directory.agent_name(aid) for aid in receivers)
            if action.anonymous:
                factual = f"我在{place}匿名向{heard_text}发送通知：「{message_text}」"
                outcome = scene_line(place, f"{doer_name}成功匿名向{heard_text}发送通知：「{message_text}」")
            else:
                factual = f"我在{place}当着{heard_text}的面宣告：「{message_text}」"
                outcome = scene_line(place, f"{doer_name}成功向在场的{heard_text}宣告：「{message_text}」")
        stub = AgentAction(
            agent_id=action.agent_id,
            step=step,
            action_type=ActionType.SEND_MESSAGE,
            action_description=action.action_description or "send message",
            content=action.content,
            anonymous=action.anonymous,
            estimated_steps=1,
            # The record is rebuilt from this stub; without the target a letter renders as a notice.
            target=action.target,
        )
        return ActionResult(
            action=stub,
            expected_outcome=action.expected_outcome,
            outcome=outcome,
            # No onlooker observation: the content would be delivered twice.
            observations=[],
            succeeded=True,
            factual_memory=factual,
        )

    def _unheard_result(
        self,
        action: AgentAction,
        step: int,
        *,
        environment: "EnvironmentSystem",
        directed: bool,
    ) -> ActionResult:
        """A message nobody could receive: a failure, or the sender would wait forever for a
        reply. Not dispatched."""
        message_text = action.content
        place = observe_location(environment, action.agent_id)
        doer_name = self._directory.agent_name(action.agent_id)
        signature = ANONYMOUS_SENDER_NAME if action.anonymous else ""
        if directed:
            reason = "没有能送到的人"
            factual = f"我在{place}{signature}发送消息：「{message_text}」，但没有一个人收得到"
            outcome = scene_line(place, f"{doer_name}{signature}发送消息「{message_text}」，但没有一个人收得到。")
        else:
            reason = "此处无人"
            verb = "匿名发送通知" if action.anonymous else "宣告"
            unheard = "没有人收到" if action.anonymous else "没有人听见"
            factual = f"我在{place}{verb}：「{message_text}」，但此处无人，{unheard}"
            outcome = scene_line(place, f"{doer_name}向四下{verb}「{message_text}」，但此处无人，{unheard}。")
        stub = AgentAction(
            agent_id=action.agent_id,
            step=step,
            action_type=ActionType.SEND_MESSAGE,
            action_description=action.action_description or "send message",
            content=action.content,
            anonymous=action.anonymous,
            estimated_steps=1,
            target=action.target,
        )
        return ActionResult(
            action=stub,
            expected_outcome=action.expected_outcome,
            outcome=outcome,
            observations=[],
            succeeded=False,
            failure_reason=reason,
            factual_memory=factual,
        )

