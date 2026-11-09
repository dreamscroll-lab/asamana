"""Momentary acts don't occupy the body (``ActionExecutor.momentary``).

The one test is whether, once issued, the act still needs the doer's own body. A character holds
at most one body-occupying act per step alongside its momentary acts: issuing one leaves the doer
free to be invited or compelled the same step, and being taken neither revokes nor defers it.
"""

from __future__ import annotations

import pytest

from agent.personality import ActionStatus, AgentActivityStatus
from core.interfaces.action import ActionTarget, ActionType, AgentAction, Ref
from engine.executors import build_default_registry
from engine.executors.base import ActionExecutionState, ActionExecutor, Conscription
from engine.executors.registry import ActionExecutorRegistry
from tests.unit.test_runtime_smoke import _plan_for, _two_rooms_with


def _send(agent_id: str, to: str, words: str) -> AgentAction:
    return AgentAction(
        agent_id=agent_id, step=1, action_type=ActionType.SEND_MESSAGE,
        action_description=f"给人传话：{words}", content=words,
        target=ActionTarget(acts_on=[Ref.agent(to)]), estimated_steps=1,
    )


def test_every_action_type_declares_whether_it_is_momentary(container) -> None:
    """Today SEND_MESSAGE and ERRAND hand the act over to the world; every other type keeps the
    doer's body busy."""
    from engine.directory import LiveWorldDirectory
    from engine.environment import EnvironmentSystem

    registry = build_default_registry(
        container.llm_router, LiveWorldDirectory.from_agents({}, EnvironmentSystem()),
    )
    momentary = {t for t in ActionType if registry.is_momentary(t)}
    assert momentary == {ActionType.SEND_MESSAGE, ActionType.ERRAND}


def test_a_momentary_executor_cannot_take_bodies() -> None:
    class _Grabbing(ActionExecutor):
        momentary = True
        conscription = Conscription.COMPEL

        async def start(self, *a, **k):  # noqa: ANN002, ANN003
            raise NotImplementedError

        async def tick(self, *a, **k):  # noqa: ANN002, ANN003
            raise NotImplementedError

        async def complete(self, *a, **k):  # noqa: ANN002, ANN003
            raise NotImplementedError

    with pytest.raises(ValueError):
        ActionExecutorRegistry().register(ActionType.SEND_MESSAGE, _Grabbing())


def test_a_momentary_execution_never_answers_for_the_body(container) -> None:
    from engine.directory import LiveWorldDirectory
    from engine.environment import EnvironmentSystem

    registry = build_default_registry(
        container.llm_router, LiveWorldDirectory.from_agents({}, EnvironmentSystem()),
    )
    send = ActionExecutionState.create(
        action_type=ActionType.SEND_MESSAGE, initiator_id="a", participant_ids=["a"],
        purpose="传话", started_step=1, estimated_steps=1, opening_outcome="", target=ActionTarget(),
    )
    registry.add_active(send)
    assert registry.get_active_for_agent("a") is None
    assert registry.get(send.execution_id) is send


@pytest.mark.asyncio
async def test_a_sender_is_still_invited_and_the_message_stands(container) -> None:
    """B sends a letter and A invites B into a talk in the same step: B joins the talk, and his
    letter still goes. His decision isn't set aside, so nothing lands in his foiled buffer."""
    env, agents, rt = _two_rooms_with(
        container, "world-momentary-invite", {"agent-a": "李世民", "agent-b": "长孙无忌", "agent-c": "房玄龄"},
    )
    talk = AgentAction(
        agent_id="agent-a", step=1, action_type=ActionType.TALK, action_description="商议对策",
        target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]),
        estimated_steps=3,
    )
    send = _send("agent-b", "agent-c", "今夜子时，玄武门见")
    planned = [
        ("main", agents["agent-b"], _plan_for(agents["agent-b"], send, env)),
        ("main", agents["agent-a"], _plan_for(agents["agent-a"], talk, env)),
    ]

    arb, _, momentary = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001

    assert arb["agent-b"].is_passive_join is True
    assert [m.ongoing_execution_id is not None for m in momentary["agent-b"]] == [True]
    pending = await rt._message_system.peek_pending()  # noqa: SLF001
    assert [m.content for m in pending] == ["今夜子时，玄武门见"]

    records = await rt._commit_execution(planned, arb, momentary, agents)  # noqa: SLF001
    b = agents["agent-b"]
    assert b.memory_system.recent_foiled_attempts(1) == []
    # His body is in the talk; the letter didn't take it.
    assert b.personality.state.action_status == ActionStatus.IN_PROGRESS
    assert b.personality.state.activity_status == AgentActivityStatus.TALKING
    # One record per execution he stands in: the letter, then the talk.
    b_types = [r["action_type"] for r in records if r["agent_id"] == "agent-b"]
    assert b_types == ["send_message", "talk"]

    pending_execs = rt._same_step_pending()  # noqa: SLF001
    assert [e.action_type for e in pending_execs] == [ActionType.SEND_MESSAGE]
    results = await rt._processor.finalize_executions_batch(pending_execs, agents, 1)  # noqa: SLF001
    assert all(r.succeeded for rs in results.values() for r in rs)
    # The letter's feedback is recorded but leaves the body in the talk.
    assert b.personality.state.action_status == ActionStatus.IN_PROGRESS
    assert b.personality.state.current_action == arb["agent-b"].action_result.action.action_description
    assert b.personality.state.last_action == send.action_description


@pytest.mark.asyncio
async def test_a_sender_drawn_into_a_long_talk_leaves_both_on_his_decision_trace(container) -> None:
    """B sends a letter and is drawn into A's three-step talk the same step. His verdict is his
    letter's; the talk his body joined is stamped beside it. Without it nothing on B's side of this
    step tells of the talk until it settles steps later, and his breaking it off then reads as
    breaking off a talk that never began."""
    from core.interfaces.trace import Stage
    from providers.trace.in_memory import InMemoryTraceSink
    from tests.unit.test_runtime_smoke import _arb_runtime, _trace_call

    world_id = "world-momentary-joined-trace"
    env, agents, _ = _two_rooms_with(
        container, world_id, {"agent-a": "李世民", "agent-b": "长孙无忌", "agent-c": "房玄龄"},
    )
    sink = InMemoryTraceSink()
    for aid in ("agent-a", "agent-b"):
        sink.record_llm_call(_trace_call(world_id, agent_id=aid, stage=Stage.DECISION))
    sink.record_llm_call(_trace_call(world_id, agent_id="agent-b", stage=Stage.MOTIVATION))
    rt = _arb_runtime(container, world_id, agents, env, trace_sink=sink)
    talk = AgentAction(
        agent_id="agent-a", step=1, action_type=ActionType.TALK, action_description="商议对策",
        target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]),
        estimated_steps=3,
    )
    planned = [
        ("main", agents["agent-b"], _plan_for(agents["agent-b"], _send("agent-b", "agent-c", "速来"), env)),
        ("main", agents["agent-a"], _plan_for(agents["agent-a"], talk, env)),
    ]

    arb, _, momentary = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    await rt._commit_execution(planned, arb, momentary, agents)  # noqa: SLF001

    decisions = {c.agent_id: c.extra for c in sink.llm_calls if c.stage == Stage.DECISION.value}
    assert decisions["agent-b"] == {"verdict": "executed", "joined_execution": {
        "execution_id": arb["agent-b"].ongoing_execution_id, "initiator": "李世民",
        "purpose": "商议对策", "spans_steps": 3}}
    # The initiator's body joined nothing: his talk is his own.
    assert "joined_execution" not in decisions["agent-a"]
    # His letter landed its motivation: the goals behind it are live, so it isn't set aside as a
    # conscripted body's is.
    motivation = next(c for c in sink.llm_calls if c.stage == Stage.MOTIVATION.value)
    assert motivation.adopted is not False and motivation.reject_reason == ""


@pytest.mark.asyncio
async def test_a_sender_carried_off_still_sent_it_and_both_land_in_order(container) -> None:
    """A carries B off (COMPEL) the step B sends a letter: the letter goes, and in the sweep the
    letter settles before the carry, so the body-occupying act is what B last did."""
    env, agents, rt = _two_rooms_with(
        container, "world-momentary-carry", {"agent-a": "尉迟敬德", "agent-b": "李建成", "agent-c": "魏征"},
    )
    carry = AgentAction(
        agent_id="agent-a", step=1, action_type=ActionType.MOVE, action_description="拽着他去园中",
        target=ActionTarget(acts_on=[Ref.place("garden")], claims=[Ref.agent("agent-b")]),
        estimated_steps=1,
    )
    send = _send("agent-b", "agent-c", "速来救我")
    planned = [
        ("main", agents["agent-a"], _plan_for(agents["agent-a"], carry, env)),
        ("main", agents["agent-b"], _plan_for(agents["agent-b"], send, env)),
    ]

    arb, _, momentary = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    assert arb["agent-b"].is_passive_join is True and arb["agent-b"].invited_by == ""
    assert "agent-b" in momentary

    records = await rt._commit_execution(planned, arb, momentary, agents)  # noqa: SLF001
    b = agents["agent-b"]
    assert b.memory_system.recent_foiled_attempts(1) == []

    pending = rt._same_step_pending()  # noqa: SLF001
    assert [e.action_type for e in pending] == [ActionType.SEND_MESSAGE, ActionType.MOVE]
    results = await rt._processor.finalize_executions_batch(pending, agents, 1)  # noqa: SLF001
    for exec_state in pending:
        rt._processor.merge_completion_into_records(  # noqa: SLF001
            exec_state, results.get(exec_state.execution_id, []), records, agents,
        )
    # He was carried to the garden, but the letter was sent, and is placed, where he stood.
    assert env.get_body_location("agent-b") == "garden"
    letter = next(r for r in records if r["action_type"] == "send_message")
    assert letter["phase"] == "settled" and letter["location_id"] == "palace"
    assert [m.content for m in await rt._message_system.peek_pending()] == ["速来救我"]  # noqa: SLF001
    # The carry landed after the letter: it is his last act, and it freed his body.
    assert b.personality.state.last_action != send.action_description
    assert b.personality.state.action_status != ActionStatus.IN_PROGRESS


@pytest.mark.asyncio
async def test_a_momentary_act_alone_leaves_the_body_free(container) -> None:
    """With nobody taking him, a sender's body is never marked as busy with the letter."""
    env, agents, rt = _two_rooms_with(
        container, "world-momentary-alone", {"agent-b": "长孙无忌", "agent-c": "房玄龄"},
    )
    send = _send("agent-b", "agent-c", "明日早朝再议")
    planned = [("main", agents["agent-b"], _plan_for(agents["agent-b"], send, env))]

    arb, _, momentary = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    assert "agent-b" not in arb and "agent-b" in momentary
    records = await rt._commit_execution(planned, arb, momentary, agents)  # noqa: SLF001
    assert [r["agent_id"] for r in records] == ["agent-b"]
    b = agents["agent-b"]
    assert b.personality.state.action_status != ActionStatus.IN_PROGRESS
    await rt._processor.finalize_executions_batch(rt._same_step_pending(), agents, 1)  # noqa: SLF001
    assert b.personality.state.action_status != ActionStatus.IN_PROGRESS
    assert b.personality.state.last_action == send.action_description


@pytest.mark.asyncio
async def test_a_momentary_execution_left_active_is_voided_never_ticked(caplog) -> None:
    """A momentary act settles in the step it was issued in. One still active at the tick is a
    broken invariant: ticking it would advance its doer's act in progress, so it is removed and
    voided as a null step, and the break is logged."""
    from engine.directory import LiveWorldDirectory
    from engine.environment import EnvironmentSystem
    from engine.execution_processor import ExecutionProcessor
    from engine.executors.send_message import SendMessageExecutor

    class _Doer:
        is_active = True

        def __init__(self) -> None:
            self.advanced = False
            self.voided: list[str] = []

        def advance_action(self, **_kw) -> None:  # noqa: ANN003
            self.advanced = True

        def void_momentary_act(self, *, step: int, description: str) -> None:
            self.voided.append(description)

    env = EnvironmentSystem()
    directory = LiveWorldDirectory.from_agents({}, env)
    registry = ActionExecutorRegistry()
    registry.register(ActionType.SEND_MESSAGE, SendMessageExecutor(directory))
    registry.add_active(ActionExecutionState(
        execution_id="e1", action_type=ActionType.SEND_MESSAGE, initiator_id="agent-b",
        participant_ids=["agent-b"], started_step=1, estimated_steps=3, remaining_steps=2,
        purpose="给人传话",
    ))
    processor = ExecutionProcessor(
        executor_registry=registry, environment=env,
        message_system=None,  # type: ignore[arg-type]
        directory=directory,
    )
    doer = _Doer()

    records = await processor.tick_ongoing_executions({"agent-b": doer}, 2)  # type: ignore[arg-type]

    assert records == []
    assert registry.get("e1") is None
    assert doer.advanced is False
    assert doer.voided == ["给人传话"]
    assert "momentary_execution_outlived_step" in caplog.text
