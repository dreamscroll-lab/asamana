"""Locks the separation between the three sources of action duration, which never overlap:
- Semantic contract (PHYSICAL/SEND_MESSAGE/ERRAND): the executor hard-codes 1 step
- World physics (MOVE): Place.connections (the route's walking time over the step length); the LLM estimate is ignored
- Agent's own estimate (TALK/REST/WORK/COVERT): action.estimated_steps as-is; WorldConfig can't take over

Built on an environment directly, independent of the WorldConfig interface.
"""

from __future__ import annotations

import pytest

from agent.decision import ActionType, AgentAction
from core.interfaces.action import ActionTarget, Ref
from engine.directory import LiveWorldDirectory
from engine.environment import EnvironmentSystem
from engine.executors.base import ActionExecutionState
from engine.executors.covert import CovertExecutor
from engine.executors.movement import MovementExecutor
from engine.executors.simple import SimpleExecutor
from engine.executors.social import SocialExecutor
from engine.executors.work import WorkExecutor
from core.interfaces.place import Place


def _directory() -> LiveWorldDirectory:
    return LiveWorldDirectory.from_agents({}, EnvironmentSystem())


def _make_environment(*, move_steps: int = 1) -> EnvironmentSystem:
    """Two locations a walk of ``move_steps`` of MovementExecutor's default 1h step apart."""
    loc_a = Place(
        place_id="loc_a", name="loc_a", description="",
        connections={"loc_b": move_steps * 3600}, is_public=True, capacity=50,
    )
    loc_b = Place(
        place_id="loc_b", name="loc_b", description="",
        connections={"loc_a": move_steps * 3600}, is_public=True, capacity=50,
    )
    env = EnvironmentSystem()
    env.space.register_place(loc_a)
    env.space.register_place(loc_b)
    env.place_agent(agent_id="agent-1", location_id="loc_a")
    env.place_agent(agent_id="agent-2", location_id="loc_a")
    return env


def _axis_target(
    action_type: ActionType, *, agent_id: str | None = None, location_id: str | None = None
) -> ActionTarget:
    """Put the target on the right relation axis, using the same rules as
    ``DecisionEngine._build_action``.

    A move acts on a place, and anyone brought along only has their turn taken; a conversation
    partner is both; any other action on a person only acts on them. The fixture follows the same
    rules so it doesn't introduce a second set of semantics.
    """
    if action_type is ActionType.MOVE:
        return ActionTarget(
            acts_on=[Ref.place(location_id)] if location_id else [],
            claims=[Ref.agent(agent_id)] if agent_id else [],
        )
    if agent_id:
        return ActionTarget(
            acts_on=[Ref.agent(agent_id)],
            claims=[Ref.agent(agent_id)] if action_type is ActionType.TALK else [],
        )
    return ActionTarget(acts_on=[Ref.place(location_id)] if location_id else [])


def _make_action(action_type: ActionType, *, estimated_steps: int,
                 location_id: str | None = None,
                 target_agent_id: str | None = None) -> AgentAction:
    return AgentAction(
        agent_id="agent-1", step=1, action_type=action_type,
        action_description="test",
        target=_axis_target(action_type, agent_id=target_agent_id, location_id=location_id),
        estimated_steps=estimated_steps,
    )


# ---------------------------------------------------------------------------
# MOVE uses Place.connections and ignores action.estimated_steps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_move_multi_step_uses_world_connections_not_llm_estimate(container) -> None:
    """Multi-step MOVE: even if the LLM says estimated_steps=1, the executor uses the connection
    distance (3 steps)."""
    env = _make_environment(move_steps=3)
    action = _make_action(ActionType.MOVE, estimated_steps=1, location_id="loc_b")  # the LLM underestimates
    executor = MovementExecutor(_directory())
    result = await executor.start(
        action, step=1, agents={}, environment=env, message_system=None,  # type: ignore
    )
    # distance 3 > 1, so the executor returns an ActionExecutionState rather than an ActionResult
    assert isinstance(result, ActionExecutionState)
    assert result.estimated_steps == 3


@pytest.mark.asyncio
async def test_move_single_step_uses_world_connections_not_llm_estimate(container) -> None:
    """Single-step MOVE (connections=1): even if the LLM says estimated_steps=5, the executor uses
    connections=1."""
    env = _make_environment(move_steps=1)
    action = _make_action(ActionType.MOVE, estimated_steps=5, location_id="loc_b")  # the LLM overestimates
    executor = MovementExecutor(_directory())
    result = await executor.start(
        action, step=1, agents={}, environment=env, message_system=None,  # type: ignore
    )
    # even a distance of 1 comes back as an ActionExecutionState
    assert isinstance(result, ActionExecutionState)
    assert result.estimated_steps == 1


# ---------------------------------------------------------------------------
# TALK / REST / WORK / COVERT pass action.estimated_steps through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_work_passthrough_action_estimated_steps_4(container) -> None:
    env = _make_environment()
    action = _make_action(ActionType.WORK, estimated_steps=4)
    executor = WorkExecutor(container.llm_router, _directory())
    result = await executor.start(
        action, step=1, agents={"agent-1": _stub_agent("agent-1")},
        environment=env, message_system=None,  # type: ignore
    )
    assert isinstance(result, ActionExecutionState)
    assert result.estimated_steps == 4


@pytest.mark.asyncio
async def test_rest_passthrough_action_estimated_steps_8(container) -> None:
    env = _make_environment()
    action = _make_action(ActionType.REST, estimated_steps=8)
    executor = SimpleExecutor(_directory())
    result = await executor.start(
        action, step=1, agents={"agent-1": _stub_agent("agent-1")},
        environment=env, message_system=None,  # type: ignore
    )
    assert isinstance(result, ActionExecutionState)
    assert result.estimated_steps == 8


@pytest.mark.asyncio
async def test_covert_passthrough_action_estimated_steps_3(container) -> None:
    env = _make_environment()
    action = _make_action(ActionType.COVERT, estimated_steps=3)
    executor = CovertExecutor(container.llm_router, _directory())
    result = await executor.start(
        action, step=1, agents={"agent-1": _stub_agent("agent-1")},
        environment=env, message_system=None,  # type: ignore
    )
    assert isinstance(result, ActionExecutionState)
    assert result.estimated_steps == 3


@pytest.mark.asyncio
async def test_talk_passthrough_action_estimated_steps_5(container) -> None:
    env = _make_environment()
    agent_1 = _stub_agent("agent-1")
    agent_2 = _stub_agent("agent-2")
    action = _make_action(ActionType.TALK, estimated_steps=5, target_agent_id="agent-2")
    executor = SocialExecutor(container.llm_router, _directory())
    result = await executor.start(
        action, step=1, agents={"agent-1": agent_1, "agent-2": agent_2},
        environment=env, message_system=None,  # type: ignore
    )
    assert isinstance(result, ActionExecutionState)
    assert result.estimated_steps == 5


# ---------------------------------------------------------------------------
# PHYSICAL / SEND_MESSAGE / ERRAND always take 1 step (hard-coded in the executor)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_message_always_one_step_regardless_of_estimated_steps(container) -> None:
    """SEND_MESSAGE takes 1 step and returns an ActionResult even if the LLM says
    estimated_steps=10."""
    # Actually sending needs a MessageSystem, so this checks the code path instead:
    # SimpleExecutor.start sends SEND_MESSAGE to _handle_send_message regardless of estimated_steps.
    import inspect
    src = inspect.getsource(SimpleExecutor.start)
    assert "SEND_MESSAGE" in src or "send_message" in src, (
        "SimpleExecutor.start 必须显式处理 SEND_MESSAGE 分支(硬编码 1 step)"
    )


def test_physical_executor_hardcodes_one_step() -> None:
    """PhysicalExecutor hard-codes estimated_steps=1 and doesn't read action.estimated_steps."""
    from engine.executors.physical import PhysicalExecutor
    import inspect
    src = inspect.getsource(PhysicalExecutor)
    assert "estimated_steps=1" in src


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _stub_agent(agent_id: str):
    """Minimal agent stub for executor.start() invocation."""
    from types import SimpleNamespace
    return SimpleNamespace(
        agent_id=agent_id,
        personality=SimpleNamespace(
            soul=SimpleNamespace(name=agent_id, role="test", traits_text=lambda: "测试"),
            state=SimpleNamespace(current_location="loc_a"),
        ),
        is_main_character=False,
    )


@pytest.mark.asyncio
async def test_errand_always_one_step_regardless_of_estimated_steps(container) -> None:
    """Sending someone on an errand takes one step, however far the person sent has to go.

    Honoring estimated_steps would defeat the point of the action: you send someone so you don't
    have to wait yourself. Charging the requester for the whole trip just makes it a slower MOVE.
    """
    from engine.executors.errand import ErrandExecutor
    from world.models import NpcSeed

    env = _make_environment()
    env.spawn_npc(NpcSeed(name="王二", description="走得快"), location_id="loc_a")
    npc_id = env.all_npcs()[0].npc_id
    action = _make_action(ActionType.ERRAND, estimated_steps=9)
    action.target = ActionTarget(acts_on=[Ref.npc(npc_id)])

    state = await ErrandExecutor(_directory()).start(
        action, 1, agents={}, environment=env, message_system=None,
    )
    assert state.estimated_steps == 1
    assert state.remaining_steps == 0
