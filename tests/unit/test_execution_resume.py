"""A multi-step act under way when a world's session is rebuilt (a restart, a run stopped at the
actor's turn, a recast) goes on from where the snapshot left it.

The registry of executions lives in process memory; the step snapshot is what survives. Unsaved,
every act in progress would vanish on restore with no result: the feed would keep a 进行中 beat
that never closes, a traveler would snap back to his origin, and nobody would remember what they
had been doing.
"""

from __future__ import annotations

import pytest

from agent.personality import ActionStatus, AgentActivityStatus
from core.container import Container
from core.interfaces.action import (
    KIND_ITEM, KIND_LOCATION, ActionTarget, ActionType, AgentAction, Observed, Ref,
)
from engine.application import NarrativeApplication
from engine.environment import IN_TRANSIT
from engine.executors.base import ActionExecutionState
from tests.unit.confirmed_world import confirmed_world
from world.initializer import WorldInitializer


@pytest.fixture()
def container(mock_build_container: Container) -> Container:
    return mock_build_container


def _a_full_execution() -> ActionExecutionState:
    return ActionExecutionState(
        execution_id="move_agent-a_3_abc123",
        action_type=ActionType.MOVE,
        initiator_id="agent-a",
        participant_ids=["agent-a", "agent-b"],
        started_step=3,
        estimated_steps=4,
        remaining_steps=2,
        purpose="带他去后花园",
        target=ActionTarget(
            acts_on=[Ref(KIND_LOCATION, "garden")],
            claims=[Ref.agent("agent-b")],
            reaches=[Ref.entity("seal", KIND_ITEM)],
        ),
        expected_outcome="避开耳目",
        opening_outcome="他带着她出了大殿。",
        opening_observations=[
            Observed(location_id="hall", text="他们出了大殿。"),
            Observed(location_id="corridor", text="有人走过。", strength=0.4),
        ],
        listener_ids=["agent-c"],
        talk_topic="",
        extra={"origin": "hall", "destination": "garden", "path": ["hall", "corridor", "garden"],
               "arrivals": [0, 2, 4], "travel_seconds": 14400},
    )


def test_an_execution_comes_back_from_its_snapshot_payload_whole() -> None:
    execution = _a_full_execution()
    assert ActionExecutionState.from_payload(execution.to_payload()) == execution


def test_an_execution_that_would_not_survive_the_snapshot_is_refused_when_saved() -> None:
    """The snapshot writer stringifies what JSON can't hold, which no restore could read back."""
    execution = _a_full_execution()
    execution.extra["handle"] = object()
    with pytest.raises(TypeError):
        execution.to_payload()


async def _begin_work(application: NarrativeApplication, world_id: str, *, steps: int) -> tuple[str, str]:
    """Set one character on a WORK of ``steps`` steps between steps, as the start step would."""
    session = application.get_session(world_id)
    runtime = session.runtime
    agent_id = next(iter(session.world.agents))
    agent = session.world.agents[agent_id]
    executor = runtime._executor_registry.get_executor(ActionType.WORK)
    action = AgentAction(
        agent_id=agent_id, step=0, action_type=ActionType.WORK,
        action_description="在书房起草奏章", target=ActionTarget(), estimated_steps=steps,
    )
    execution = await executor.start(
        action, 0, agents=session.world.agents, environment=runtime._environment,
        message_system=runtime._message_system,
    )
    runtime._executor_registry.add_active(execution)
    agent.personality.begin_action(
        step=0, description=action.action_description,
        activity_status=AgentActivityStatus.WORKING, target=None, estimated_steps=steps,
    )
    return agent_id, execution.execution_id


@pytest.mark.asyncio
async def test_a_restored_world_carries_on_with_the_act_and_ends_it(container, test_config) -> None:
    application = NarrativeApplication(container, test_config)
    world = await confirmed_world(application, "玄武门之变", template="changan_iso")
    agent_id, execution_id = await _begin_work(application, world.world_id, steps=4)
    await application.run_world(world.world_id, steps=1)

    saved = await container.snapshot.load(world.world_id, 1)
    assert [e["execution_id"] for e in saved.engine.active_executions] == [execution_id]

    restored = NarrativeApplication(container, test_config)
    await restored.restore_session(world.world_id)
    session = restored.get_session(world.world_id)
    (execution,) = session.runtime._executor_registry.all_active()
    assert (execution.execution_id, execution.remaining_steps) == (execution_id, 2)
    state = session.world.agents[agent_id].personality.state
    assert (state.action_status, state.current_action) == (ActionStatus.IN_PROGRESS, "在书房起草奏章")

    await restored.run_world(world.world_id, steps=2)

    ended = await container.snapshot.load(world.world_id, 3)
    assert [
        e["phase"] for e in ended.events
        if e.get("kind") == "action" and e.get("execution_id") == execution_id
    ] == ["ongoing_complete"]
    assert not session.runtime._executor_registry.all_active()
    assert session.world.agents[agent_id].personality.state.action_status != ActionStatus.IN_PROGRESS


@pytest.mark.asyncio
async def test_an_act_its_snapshot_never_saved_leaves_its_character_free(container, test_config) -> None:
    """The character's state says busy but the snapshot holds no act: nothing would ever tick or
    end it, so the character is idle rather than busy forever."""
    application = NarrativeApplication(container, test_config)
    world = await confirmed_world(application, "玄武门之变", template="changan_iso")
    agent_id, _ = await _begin_work(application, world.world_id, steps=4)
    await application.run_world(world.world_id, steps=1)
    saved = await container.snapshot.load(world.world_id, 1)
    saved.engine.active_executions = []
    await container.snapshot.save(world.world_id, 1, saved)

    restored = await WorldInitializer(container).restore(world.world_id)

    assert restored.active_executions == []
    state = restored.agents[agent_id].personality.state
    assert (state.action_status, state.current_action) == (ActionStatus.IDLE, None)


@pytest.mark.asyncio
async def test_an_act_that_cannot_be_saved_costs_only_itself(container, test_config) -> None:
    """Rule 6: the step and the rest of its snapshot still commit."""
    application = NarrativeApplication(container, test_config)
    world = await confirmed_world(application, "玄武门之变", template="changan_iso")
    _, execution_id = await _begin_work(application, world.world_id, steps=4)
    (execution,) = application.get_session(world.world_id).runtime._executor_registry.all_active()
    execution.extra["handle"] = object()

    await application.run_world(world.world_id, steps=1)

    saved = await container.snapshot.load(world.world_id, 1)
    assert saved is not None and saved.engine.active_executions == []


@pytest.mark.asyncio
async def test_an_act_someone_in_it_has_no_record_of_is_dropped_whole(container, test_config) -> None:
    application = NarrativeApplication(container, test_config)
    world = await confirmed_world(application, "玄武门之变", template="changan_iso")
    agent_id, _ = await _begin_work(application, world.world_id, steps=4)
    await application.run_world(world.world_id, steps=1)
    saved = await container.snapshot.load(world.world_id, 1)
    saved.engine.active_executions[0]["participant_ids"].append(
        next(aid for aid in world.agents if aid != agent_id)
    )
    await container.snapshot.save(world.world_id, 1, saved)

    restored = await WorldInitializer(container).restore(world.world_id)

    assert restored.active_executions == []
    assert restored.agents[agent_id].personality.state.action_status == ActionStatus.IDLE


@pytest.mark.asyncio
async def test_a_traveler_restored_mid_trip_stays_on_the_road(container, test_config) -> None:
    """His own location changes only on arrival, so standing him there would put him back at his
    origin; nor is he sent back as a body in transit with no trip would be."""
    application = NarrativeApplication(container, test_config)
    world = await confirmed_world(application, "玄武门之变", template="changan_iso")
    agent_id, _ = await _begin_work(application, world.world_id, steps=4)
    await application.run_world(world.world_id, steps=1)
    # Recast the saved act as a trip caught between two waypoints.
    saved = await container.snapshot.load(world.world_id, 1)
    origin = saved.state.agents[agent_id]["location_id"]
    destination = next(
        place_id for place_id in world.world_config.get_places() if place_id != origin
    )
    payload = saved.engine.active_executions[0]
    payload["action_type"] = ActionType.MOVE.value
    payload["target"] = {"acts_on": [{"kind": KIND_LOCATION, "id": destination}], "claims": [], "reaches": []}
    payload["extra"] = {"origin": origin, "destination": destination, "path": [origin, destination],
                        "arrivals": [0, 4], "travel_seconds": 14400}
    saved.state.agents[agent_id]["location_id"] = IN_TRANSIT
    saved.engine.environment["transit_origins"] = {agent_id: origin}
    await container.snapshot.save(world.world_id, 1, saved)

    restored = await WorldInitializer(container).restore(world.world_id)

    assert restored.environment.get_body_location(agent_id) == IN_TRANSIT
    assert restored.environment.transit_origin(agent_id) == origin
    (resumed,) = restored.active_executions
    assert (resumed.action_type, resumed.extra["destination"]) == (ActionType.MOVE, destination)
