"""Action executor framework."""

from __future__ import annotations

from core.interfaces.action import ActionType
from core.interfaces.directory import WorldDirectory
from core.interfaces.llm import LLMRouter
from engine.executors.base import ActionExecutionState, ActionExecutor
from engine.executors.covert import CovertExecutor
from engine.executors.errand import ErrandExecutor
from engine.executors.movement import MovementExecutor
from engine.executors.physical import PhysicalExecutor
from engine.executors.registry import ActionExecutorRegistry
from engine.executors.simple import SimpleExecutor
from engine.executors.social import SocialExecutor
from engine.executors.work import WorkExecutor


def build_default_registry(
    llm_router: LLMRouter,
    directory: WorldDirectory,
    *,
    seconds_per_step: int = 3600,
    world_start_second_of_day: int = 0,
) -> ActionExecutorRegistry:
    """Create and configure the default executor registry for a world run."""
    registry = ActionExecutorRegistry()
    simple = SimpleExecutor(directory, seconds_per_step=seconds_per_step)
    social = SocialExecutor(
        llm_router, directory, seconds_per_step=seconds_per_step, world_start_second_of_day=world_start_second_of_day)
    movement = MovementExecutor(directory, seconds_per_step=seconds_per_step)
    work = WorkExecutor(llm_router, directory, seconds_per_step=seconds_per_step)
    physical = PhysicalExecutor(llm_router, directory, seconds_per_step=seconds_per_step)
    covert = CovertExecutor(
        llm_router, directory, seconds_per_step=seconds_per_step,
        world_start_second_of_day=world_start_second_of_day)
    errand = ErrandExecutor(directory)

    registry.register(ActionType.REST, simple)
    registry.register(ActionType.SEND_MESSAGE, simple)
    registry.register(ActionType.TALK, social)
    registry.register(ActionType.MOVE, movement)
    registry.register(ActionType.WORK, work)
    registry.register(ActionType.PHYSICAL, physical)
    registry.register(ActionType.COVERT, covert)
    registry.register(ActionType.ERRAND, errand)

    # The arbiter assumes every ActionType has an executor; a missing one is a config error (Rule 3).
    missing = [t for t in ActionType if registry.get_executor(t) is None]
    if missing:
        raise ValueError(
            f"build_default_registry: no executor for action types: "
            f"{', '.join(t.value for t in missing)}"
        )

    return registry


__all__ = [
    "ActionExecutionState",
    "ActionExecutor",
    "ActionExecutorRegistry",
    "CovertExecutor",
    "ErrandExecutor",
    "MovementExecutor",
    "PhysicalExecutor",
    "SimpleExecutor",
    "SocialExecutor",
    "WorkExecutor",
    "build_default_registry",
]
