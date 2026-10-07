"""Registry for action executors and active execution states."""

from __future__ import annotations

from typing import List

from core.interfaces.action import ActionType
from engine.executors.base import ActionExecutionState, ActionExecutor


class ActionExecutorRegistry:
    """
    Manages executor implementations and active multi-step execution states.
    Owned by NarrativeRuntime; one instance per world run.
    """

    def __init__(self) -> None:
        self._executors: dict[ActionType, ActionExecutor] = {}
        self._active: dict[str, ActionExecutionState] = {}  # execution_id → state

    def register(self, action_type: ActionType, executor: ActionExecutor) -> None:
        self._executors[action_type] = executor

    def get_executor(self, action_type: ActionType) -> ActionExecutor | None:
        return self._executors.get(action_type)

    def add_active(self, state: ActionExecutionState) -> None:
        self._active[state.execution_id] = state

    def remove_active(self, execution_id: str) -> None:
        self._active.pop(execution_id, None)

    def get_active_for_agent(self, agent_id: str) -> ActionExecutionState | None:
        for state in self._active.values():
            if agent_id in state.participant_ids:
                return state
        return None

    def is_agent_active(self, agent_id: str) -> bool:
        return self.get_active_for_agent(agent_id) is not None

    def all_active(self) -> List[ActionExecutionState]:
        return list(self._active.values())
