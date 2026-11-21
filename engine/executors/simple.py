"""Simple action executor for REST."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from core.interfaces.action import ActionResult, ActionType, AgentAction
from core.action_duration import action_duration_steps
from core.interfaces.directory import WorldDirectory
from core.duration import describe_duration
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
    """REST: a single-agent act no one adjudicates, spanning as many steps as the doer rests."""

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
        duration = action_duration_steps(action.action_type, action.estimated_steps)
        purpose = action.action_description or action.action_type.value
        duration_label = describe_duration(duration, self._seconds_per_step)
        doer_name = self._directory.agent_name(action.agent_id)
        location = observe_location(environment, action.agent_id)
        opening = scene_line(location, f"{doer_name}休息，打算歇{duration_label}。")
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
        doer_name = self._directory.agent_name(state.initiator_id)
        location = observe_location(environment, state.initiator_id)
        elapsed = state.estimated_steps - state.remaining_steps
        elapsed_label = describe_duration(elapsed, self._seconds_per_step)
        remaining_label = describe_duration(state.remaining_steps, self._seconds_per_step)
        if elapsed <= 0:
            planned = describe_duration(state.estimated_steps, self._seconds_per_step)
            body = f"{doer_name}正在休息，刚歇下，打算歇{planned}。"
        elif state.remaining_steps > 0:
            body = f"{doer_name}正在休息，已歇{elapsed_label}，打算再歇{remaining_label}。"
        else:
            body = f"{doer_name}正在休息，已歇{elapsed_label}。"
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
        location = observe_location(environment, state.initiator_id)
        duration_label = describe_duration(state.estimated_steps, self._seconds_per_step)
        intent = format_intent_clause(self._rest_activity(state.purpose) or "")
        factual = f"{intent}在{location}休息了{duration_label}，精力和状态都有所恢复。"
        # Also the onlooker observation, so only the observable half: recovery and his reason
        # go only into 1p memory. Don't append the purpose: it's a first-person sentence and
        # after his name reads as if he said it aloud.
        outcome = scene_line(
            location, f"{self._directory.agent_name(state.initiator_id)}休息了{duration_label}。",
        )
        stub = AgentAction(
            agent_id=state.initiator_id,
            step=step,
            action_type=state.action_type,
            action_description=state.purpose,
            target=state.target,
            estimated_steps=state.estimated_steps,
        )
        rest_recovery = -_REST_RECOVERY_PER_STEP * state.estimated_steps
        return [ActionResult(
            action=stub,
            expected_outcome=state.expected_outcome or state.purpose,
            outcome=outcome,
            observations=observed_here(environment, state.initiator_id, outcome),
            succeeded=True,
            factual_memory=factual,
            vitality_damage=rest_recovery,
        )]

    async def interrupt(
        self,
        state: ActionExecutionState,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        interrupted_agent_id: str | None = None,
        thought: str = "",
        cause: str = "",
    ) -> List[ActionResult]:
        elapsed = state.estimated_steps - state.remaining_steps
        elapsed_label = describe_duration(elapsed, self._seconds_per_step)
        # Only third-person cause may name what interrupted it (see base.interrupt).
        cut = f"因{cause}而中断" if cause else "被打断"
        rest_recovery = -_REST_RECOVERY_PER_STEP * elapsed if elapsed > 0 else 0.0
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
            doer_name = self._directory.agent_name(agent_id)
            location = observe_location(environment, agent_id)
            reason_3p = format_interrupt_reason_3p(doer_name, thought)
            # Intent only in 1p (see complete()).
            intent = format_intent_clause(self._rest_activity(state.purpose) or "")
            factual = intent + (
                f"休息了{elapsed_label}便{cut}，勉强恢复了些体力{thought_part}"
                if elapsed > 0
                else f"刚要休息就{cut}{thought_part}"
            )
            gist = scene_line(location, (
                f"{doer_name}休息了{elapsed_label}，{cut}。"
                if elapsed > 0
                else f"{doer_name}刚要休息就{cut}。"
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

    @staticmethod
    def _rest_activity(purpose: str | None) -> str | None:
        """REST's specific reason ("闭目养神平复心绪") for 1p memory, or None for the
        ``ActionType.REST.value`` placeholder, which would leak as "我打算「rest」"."""
        text = (purpose or "").strip()
        if not text or text == ActionType.REST.value:
            return None
        # ``format_intent_clause`` strips trailing punctuation itself.
        return text
