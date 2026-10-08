"""Simple action executor for REST and SEND_MESSAGE."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from core.interfaces.action import ActionResult, ActionType, AgentAction
from core.interfaces.directory import WorldDirectory
from core.duration import describe_duration
from core.prompts import strip_end_punct
from core.interfaces.execution import TickResult
from engine.executors.base import ActionExecutionState, ActionExecutor
from engine.narration import (
    format_intent_clause, format_interrupt_reason_3p, format_interrupt_thought, observed_here,
    scene_line,
)
from engine.scene import observe_location

if TYPE_CHECKING:
    from agent.agent import Agent
    from engine.environment import EnvironmentSystem
    from engine.message_system import MessageSystem


# Vitality restored per step of REST; read it against death_handler's passive decay (0.001/step).
_REST_RECOVERY_PER_STEP: float = 0.025


class SimpleExecutor(ActionExecutor):
    """Handles single-agent, content-free actions. SEND_MESSAGE dispatches in start() and
    fixes its result into extra["completed_result"] for complete() to return."""

    def __init__(self, directory: WorldDirectory, seconds_per_step: int = 3600) -> None:
        self._directory = directory
        self._seconds_per_step = seconds_per_step

    async def start(
        self,
        action: AgentAction,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> ActionExecutionState:
        if action.action_type == ActionType.SEND_MESSAGE:
            result = await self._handle_send_message(
                action, step, agents=agents, environment=environment, message_system=message_system,
            )
            state = ActionExecutionState.create(
                action_type=ActionType.SEND_MESSAGE,
                initiator_id=action.agent_id,
                participant_ids=[action.agent_id],
                purpose=action.action_description or "send message",
                started_step=step,
                estimated_steps=1,
                opening_outcome="",
                target=action.target,
                expected_outcome=action.expected_outcome,
            )
            state.extra["completed_result"] = result
            return state

        duration = action.estimated_steps
        purpose = action.action_description or action.action_type.value
        duration_label = describe_duration(duration, self._seconds_per_step)
        actor_name = self._directory.agent_name(action.agent_id)
        location = observe_location(environment, action.agent_id)
        if action.action_type == ActionType.REST:
            opening = scene_line(location, f"{actor_name}休息，打算歇{duration_label}。")
        else:
            opening = scene_line(location, f"{actor_name}开始着手做「{strip_end_punct(purpose)}」。")
        return ActionExecutionState.create(
            action_type=action.action_type,
            initiator_id=action.agent_id,
            participant_ids=[action.agent_id],
            purpose=purpose,
            started_step=step,
            estimated_steps=duration,
            opening_outcome=opening,
            # Public action: observation == opening.
            opening_observations=observed_here(environment, action.agent_id, opening),
            target=action.target,
            expected_outcome=action.expected_outcome,
        )

    async def tick(
        self,
        state: ActionExecutionState,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> List[TickResult]:
        actor_name = self._directory.agent_name(state.initiator_id)
        location = observe_location(environment, state.initiator_id)
        elapsed = state.estimated_steps - state.remaining_steps
        elapsed_label = describe_duration(elapsed, self._seconds_per_step)
        remaining_label = describe_duration(state.remaining_steps, self._seconds_per_step)
        if state.action_type == ActionType.REST:
            if elapsed <= 0:
                planned = describe_duration(state.estimated_steps, self._seconds_per_step)
                body = f"{actor_name}正在休息，刚歇下，打算歇{planned}。"
            elif state.remaining_steps > 0:
                body = f"{actor_name}正在休息，已歇{elapsed_label}，打算再歇{remaining_label}。"
            else:
                body = f"{actor_name}正在休息，已歇{elapsed_label}。"
        else:
            tail = f"，预计还需{remaining_label}" if state.remaining_steps > 0 else ""
            body = f"{actor_name}「{state.purpose}」仍在进行，已持续{elapsed_label}{tail}。"
        narrative = scene_line(location, body)
        return [TickResult(
            agent_id=state.initiator_id, outcome=narrative,
            observations=observed_here(environment, state.initiator_id, narrative),
        )]

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
            return [stored]  # SEND_MESSAGE dispatch result (fixed at start)
        factual = self._completed_factual(
            action_type=state.action_type,
            purpose=state.purpose,
            duration=state.estimated_steps,
            environment=environment,
            initiator_id=state.initiator_id,
        )
        outcome = self._completed_outcome(
            action_type=state.action_type,
            purpose=state.purpose,
            duration=state.estimated_steps,
            environment=environment,
            initiator_id=state.initiator_id,
        )
        stub = AgentAction(
            agent_id=state.initiator_id,
            step=step,
            action_type=state.action_type,
            action_description=state.purpose,
            target=state.target,
            estimated_steps=state.estimated_steps,
        )
        rest_recovery = -_REST_RECOVERY_PER_STEP * state.estimated_steps if state.action_type == ActionType.REST else 0.0
        return [ActionResult(
            action=stub,
            expected_outcome=state.expected_outcome or state.purpose,
            outcome=outcome,
            observations=observed_here(environment, state.initiator_id, outcome),
            succeeded=True,
            factual_memory=factual,
            vitality_damage=rest_recovery,
        )]

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

        # Memory uses his own description of the act, not the delivered words.
        intent_text = action.action_description or "（无明确内容）"
        # Quoted in the 3p line: unquoted, his first-person words read as the narrator's.
        quoted_intent = strip_end_punct(intent_text)
        place = observe_location(environment, action.agent_id)
        actor_name = self._directory.agent_name(action.agent_id)
        if directed:
            recipient_text = "、".join(self._directory.agent_name(aid) for aid in recipient_ids)
            factual = f"我成功在{place}传讯给{recipient_text}：{intent_text}"
            outcome = scene_line(place, f"{actor_name}成功向{recipient_text}传讯：「{quoted_intent}」。")
        else:
            # Name everyone: the memory must later answer "who heard this".
            heard_text = "、".join(self._directory.agent_name(aid) for aid in receivers)
            factual = f"我在{place}当着{heard_text}的面宣告：{intent_text}"
            outcome = scene_line(place, f"{actor_name}成功向在场的{heard_text}宣告：「{quoted_intent}」。")
        stub = AgentAction(
            agent_id=action.agent_id,
            step=step,
            action_type=ActionType.SEND_MESSAGE,
            action_description=action.action_description or "send message",
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
        intent_text = action.action_description or "（无明确内容）"
        quoted_intent = strip_end_punct(intent_text)
        place = observe_location(environment, action.agent_id)
        actor_name = self._directory.agent_name(action.agent_id)
        if directed:
            reason = "没有能送到的人"
            factual = f"我在{place}传讯：{intent_text}，但没有一个人收得到"
            outcome = scene_line(place, f"{actor_name}发出传讯「{quoted_intent}」，但没有一个人收得到。")
        else:
            reason = "此处无人"
            factual = f"我在{place}宣告：{intent_text}，但此处无人，没有人听见"
            outcome = scene_line(place, f"{actor_name}向四下宣告「{quoted_intent}」，但此处无人，没有人听见。")
        stub = AgentAction(
            agent_id=action.agent_id,
            step=step,
            action_type=ActionType.SEND_MESSAGE,
            action_description=action.action_description or "send message",
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
        elapsed = state.estimated_steps - state.remaining_steps
        elapsed_label = describe_duration(elapsed, self._seconds_per_step)
        # Only third-person cause may name what interrupted it (see base.interrupt).
        cut = f"因{cause}而中断" if cause else "被打断"
        rest_recovery = -_REST_RECOVERY_PER_STEP * elapsed if state.action_type == ActionType.REST and elapsed > 0 else 0.0
        thought_part = format_interrupt_thought(thought)
        results: List[ActionResult] = []
        for agent_id in state.participant_ids:
            if agent_id not in agents:
                continue
            stub = AgentAction(
                agent_id=agent_id,
                step=step,
                action_type=state.action_type,
                action_description=state.purpose,
                target=state.target,
            )
            actor_name = self._directory.agent_name(agent_id)
            location = observe_location(environment, agent_id) if environment is not None else "此处"
            reason_3p = format_interrupt_reason_3p(actor_name, thought)
            if state.action_type == ActionType.REST:
                # Intent only in 1p (see _completed_outcome).
                intent = format_intent_clause(self._rest_activity(state.purpose) or "")
                factual = intent + (
                    f"休息了{elapsed_label}便{cut}，勉强恢复了些体力{thought_part}"
                    if elapsed > 0
                    else f"刚要休息就{cut}{thought_part}"
                )
                gist = scene_line(location, (
                    f"{actor_name}休息了{elapsed_label}，{cut}。"
                    if elapsed > 0
                    else f"{actor_name}刚要休息就{cut}。"
                ))
            else:
                factual = (
                    f"「{state.purpose}」进行了{elapsed_label}便{cut}{thought_part}"
                    if elapsed > 0
                    else f"刚着手「{state.purpose}」就{cut}{thought_part}"
                )
                gist = scene_line(location, (
                    f"{actor_name}的「{state.purpose}」进行了{elapsed_label}便{cut}。"
                    if elapsed > 0
                    else f"{actor_name}刚着手「{state.purpose}」就{cut}。"
                ))
            results.append(ActionResult(
                action=stub,
                expected_outcome=state.purpose,
                outcome=gist + reason_3p,   # 3p full record (who, where, what, interrupted)
                gist=gist,
                succeeded=False,
                factual_memory=factual,
                vitality_damage=rest_recovery,
            ))
        return results

    def _completed_outcome(
        self,
        *,
        action_type: ActionType,
        purpose: str,
        duration: int,
        environment: "EnvironmentSystem",
        initiator_id: str,
    ) -> str:
        """3p outcome for a completed action; also the onlooker observation.

        Only the observable half of the result: recovery and his reason go only into 1p memory.
        Don't append ``action_description``: it's a first-person sentence and after his name
        reads as if he said it aloud.
        """
        actor_name = self._directory.agent_name(initiator_id)
        location = observe_location(environment, initiator_id)
        if action_type == ActionType.REST:
            label = describe_duration(duration, self._seconds_per_step)
            return scene_line(location, f"{actor_name}歇息，前后歇了{label}。")
        return scene_line(location, f"{actor_name}做完了：「{strip_end_punct(purpose)}」。")

    def _completed_factual(
        self,
        *,
        action_type: ActionType,
        purpose: str,
        duration: int,
        environment: "EnvironmentSystem",
        initiator_id: str,
    ) -> str:
        """factual_memory for a completed multi-step action (natural duration, no step ids)."""
        if action_type == ActionType.REST:
            location = observe_location(environment, initiator_id)
            duration_label = describe_duration(duration, self._seconds_per_step)
            intent = format_intent_clause(self._rest_activity(purpose) or "")
            return f"{intent}在{location}休息了{duration_label}，精力和状态都有所恢复。"
        return f"完成了：{purpose}。"

    @staticmethod
    def _rest_activity(purpose: str | None) -> str | None:
        """REST's specific reason ("闭目养神平复心绪") for 1p memory, or None for the
        ``ActionType.REST.value`` placeholder, which would leak as "我打算「rest」"."""
        text = (purpose or "").strip()
        if not text or text == ActionType.REST.value:
            return None
        # ``format_intent_clause`` strips trailing punctuation itself.
        return text
