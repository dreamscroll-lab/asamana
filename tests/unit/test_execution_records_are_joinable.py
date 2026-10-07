"""Every record an execution produces must carry the execution it came from.

A joint action emits one record per participant (each writes their own memory/emotion/relations);
the observer folds them back into one deed by joining on ``execution_id``.

``completion_record`` stamps ``execution_id`` itself and takes the execution in its signature, so no
caller can forget. Without the stamp, executions that complete on a later step and interrupts can't
be joined, and the feed prints the deed twice ("X 被加入 Y 发起的【…】"). These tests check the stamp
at the source, because running the sim can't provoke the scenario reliably.
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from core.interfaces.action import ActionType, Ref
from engine.executors.registry import ActionExecutorRegistry
from tests.unit.test_action_executors import (
    _directory,
    _make_action,
    _make_agent,
    _make_environment,
)
from engine.environment import EnvironmentSystem
from engine.execution_processor import ExecutionProcessor
from engine.executors.social import SocialExecutor


class _ScriptedLLM:
    """Dialogue + per-side memory summary, enough to complete a TALK."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, scene, messages, **kwargs):  # noqa: ANN001, ANN401
        self.calls += 1
        content = (
            '{"dialogue": [{"speaker": "A", "line": "来了"}, {"speaker": "B", "line": "嗯"}]}'
            if self.calls == 1
            else '{"fact": "我们谈过了", "succeeded": true, "direction": "neutral"}'
        )
        return type("R", (), {"content": content, "model": "test"})()

    async def complete_with_retry(self, scene, messages, **kwargs):  # noqa: ANN001, ANN401
        return await self.complete(scene, messages, **kwargs)


@pytest.mark.asyncio
async def test_a_talk_finishing_on_a_later_step_leaves_joinable_records(container) -> None:
    """The exact shape the observer saw as two loose cards: a 2-person TALK that runs several
    steps and finalises later. Both participants' records must name the same execution."""
    env = _make_environment()
    agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
    agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
    agents = {"agent-a": agent_a, "agent-b": agent_b}
    directory = _directory(agents)

    executor = SocialExecutor(_ScriptedLLM(), directory)  # type: ignore[arg-type]
    registry = ActionExecutorRegistry()
    registry.register(ActionType.TALK, executor)
    processor = ExecutionProcessor(
        executor_registry=registry,
        environment=env,
        message_system=None,  # type: ignore[arg-type]  # TALK is same-location: never dispatched
        directory=directory,
    )

    action = _make_action(
        agent_id="agent-a", action_type=ActionType.TALK, target_agent_id="agent-b",
        estimated_steps=3,  # multi-step: it will NOT complete on the step it began
    )
    state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
    registry.add_active(state)
    state.remaining_steps = 0  # …and now it is ready to finalise, on a later step

    records = await processor.tick_ongoing_executions(agents, step=3)

    completions = [r for r in records if r.get("phase") == "ongoing_complete"]
    assert len(completions) == 2, "a 2-person TALK yields one record per participant"
    assert {r["agent_id"] for r in completions} == {"agent-a", "agent-b"}
    # THE INVARIANT: both name the same execution, so a client can fold them into one deed.
    execution_ids = {r.get("execution_id") for r in completions}
    assert execution_ids == {state.execution_id}, "records must be joinable on execution_id"
    assert None not in execution_ids
    # And the fold can tell whose deed it is — the initiator's record speaks for it.
    assert all(r.get("initiator_id") == "agent-a" for r in completions)


@pytest.mark.asyncio
async def test_completion_record_stamps_the_execution_for_every_caller() -> None:
    """The stamp lives in completion_record itself, not in each caller: build one straight from the
    processor and look."""
    from core.interfaces.action import ActionResult, ActionTarget, AgentAction
    from engine.directory import LiveWorldDirectory
    from engine.executors.base import ActionExecutionState

    env = EnvironmentSystem()
    processor = ExecutionProcessor(
        executor_registry=ActionExecutorRegistry(),
        environment=env,
        message_system=None,  # type: ignore[arg-type]
        directory=LiveWorldDirectory.from_agents({}, env),
    )
    target = ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")])
    state = ActionExecutionState.create(
        action_type=ActionType.TALK, initiator_id="agent-a",
        participant_ids=["agent-a", "agent-b"], purpose="谈", started_step=1,
        estimated_steps=1, opening_outcome="", target=target,
    )
    result = ActionResult(
        action=AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.TALK,
            action_description="谈", target=target,
        ),
        expected_outcome="",
        outcome="谈完了。",
    )

    record = processor.completion_record(state, result, {})

    assert record["execution_id"] == state.execution_id


@pytest.mark.asyncio
async def test_a_move_relands_the_anchor_before_the_feedback_reads_it() -> None:
    """An agent who arrives must experience it where he arrived: the situation anchor is re-pointed
    at the destination before feedback runs.

    Perception runs before tick (runtime ``_perceive_all_agents``), so otherwise the whole feedback
    chain reads the starting place and says "I am at A" next to "went from B to A".
    """
    from core.interfaces.action import ActionResult, ActionTarget
    from engine.executors.registry import ActionExecutorRegistry

    env = _make_environment()

    class _Spy:
        """Records two things: the place seen when re-anchoring, and whether that happened before finalize."""

        agent_id = "agent-a"
        is_active = True

        def __init__(self) -> None:
            self.anchored_at: str | None = None
            self.anchored_before_finalize = False

        def refresh_situation(self, spatial) -> None:  # noqa: ANN001
            self.anchored_at = spatial.location_view.name if spatial.location_view else None

        async def finalize_ongoing_action(self, *, result, step) -> None:  # noqa: ANN001
            self.anchored_before_finalize = self.anchored_at is not None

    processor = ExecutionProcessor(
        executor_registry=ActionExecutorRegistry(),
        environment=env,
        message_system=None,  # type: ignore[arg-type]
        directory=_directory({}),
    )
    action = _make_action(agent_id="agent-a", action_type=ActionType.MOVE)
    action.target = ActionTarget(acts_on=[Ref.place("garden")])
    result = ActionResult(action=action, expected_outcome="到花园", outcome="他到了花园。", succeeded=True)

    # The world has already moved him (move_body in movement.complete runs just before this).
    env.move_body(body_id="agent-a", location_id="garden")
    spy = _Spy()
    await processor.land_result_feedback(result, {"agent-a": spy}, 7)  # type: ignore[arg-type]

    assert spy.anchored_at == env.narrative_location_name("garden")
    assert spy.anchored_before_finalize, "补锚点必须在 finalize 之前，否则反馈读到的仍是旧地方"

    # Complement: an action that doesn't move anyone isn't re-anchored, so feedback doesn't build a
    # spatial view for nothing.
    still = _Spy()
    talk = _make_action(agent_id="agent-a", action_type=ActionType.TALK, target_agent_id="agent-b")
    idle = ActionResult(action=talk, expected_outcome="谈一谈", outcome="他们谈过了。", succeeded=True)
    await processor.land_result_feedback(idle, {"agent-a": still}, 7)  # type: ignore[arg-type]
    assert still.anchored_at is None


@pytest.mark.asyncio
async def test_a_leg_of_the_road_moves_the_anchor_with_the_body() -> None:
    """Travel moves the body every step, so the anchor has to follow. Perception runs before tick,
    so without re-anchoring it stays at the previous waypoint.

    The test is whether the location actually changed, not the action type, so any executor that
    moves someone during tick is covered without each one remembering to do it.
    """
    from engine.executors.base import ActionExecutionState
    from engine.executors.registry import ActionExecutorRegistry

    env = _make_environment()

    class _Walker:
        """An executor that only moves one step forward: it changes the world in tick and emits no narration."""

        async def tick(self, state, step, *, agents, environment, message_system):  # noqa: ANN001
            environment.move_body(body_id=state.initiator_id, location_id="garden")
            return []

    class _Spy:
        agent_id = "agent-a"
        is_active = True
        is_main_character = False

        def __init__(self) -> None:
            self.anchored_at: str | None = None
            self.personality = SimpleNamespace(state=SimpleNamespace(
                action_remaining_steps=3, action_status=None, current_action=None,
            ), update_action_status=lambda **kw: None)

        def refresh_situation(self, spatial) -> None:  # noqa: ANN001
            self.anchored_at = spatial.location_view.name if spatial.location_view else None

    registry = ActionExecutorRegistry()
    registry.register(ActionType.MOVE, _Walker())  # type: ignore[arg-type]
    registry.add_active(ActionExecutionState(
        execution_id="e1", action_type=ActionType.MOVE, initiator_id="agent-a",
        participant_ids=["agent-a"], started_step=1, estimated_steps=4, remaining_steps=3,
        purpose="赶路",
    ))
    processor = ExecutionProcessor(
        executor_registry=registry, environment=env,
        message_system=None,  # type: ignore[arg-type]
        directory=_directory({}),
    )

    spy = _Spy()
    await processor.tick_ongoing_executions({"agent-a": spy}, 2)  # type: ignore[arg-type]
    assert spy.anchored_at == env.narrative_location_name("garden")

    # Complement: no move this step means no re-anchor, so ongoing actions don't build a spatial view
    # for nothing.
    still = _Spy()
    await processor.tick_ongoing_executions({"agent-a": still}, 3)  # type: ignore[arg-type]
    assert still.anchored_at is None
