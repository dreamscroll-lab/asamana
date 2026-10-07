"""An interrupted agent must remember WHY he broke off.

``Agent.evaluate_interrupt``'s first-person ``thought`` is the only carrier of the reason (the
triggering signals stop at the decision, see ActionExecutor.interrupt), so every interruptible
executor must land it in the agent's own memory: in ``factual_memory`` directly, or via the LLM
that writes it. MovementExecutor has no such LLM, so it appends ``thought`` itself; otherwise the
memory keeps the effect ("刚从X动身前往Y，就停下") with its cause amputated.

``outcome`` is the god-view full record, so it states the reason too, quoted and attributed. An
interrupt emits no bystander channel (see ``ActionExecutor.interrupt``); that is pinned here too.
"""

from __future__ import annotations

import pytest

from core.interfaces.action import ActionTarget, ActionType, AgentAction, Ref
from engine.executors.base import ActionExecutionState
from engine.executors.movement import MovementExecutor
from engine.executors.simple import SimpleExecutor
from tests.unit.test_action_executors import _directory, _make_environment

_THOUGHT = "我改主意了，此刻回头才是活路"


@pytest.mark.asyncio
async def test_movement_interrupt_remembers_the_thought() -> None:
    """A journey abandoned mid-way: the memory must say why he turned back."""
    env = _make_environment()
    executor = MovementExecutor(_directory())
    action = AgentAction(
        agent_id="agent-a", step=1, action_type=ActionType.MOVE,
        action_description="前往集市", target=ActionTarget(acts_on=[Ref.place("market")]),
        estimated_steps=3,
    )
    state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
    state.remaining_steps = 2  # part-way there

    (result,) = await executor.interrupt(
        state, 2, agents={"agent-a": None}, environment=env, thought=_THOUGHT,
    )

    assert _THOUGHT in result.factual_memory, "the mover must remember why he stopped"
    # The god-view record states the cause (attributed) — every other outcome states why an
    # action ended, and an interrupt must not be the one that reports an effect with no cause.
    assert _THOUGHT in result.outcome
    assert result.observations == []


@pytest.mark.asyncio
async def test_rest_interrupt_remembers_the_thought() -> None:
    """The same contract on a solo non-move action, so the guard isn't movement-specific."""
    env = _make_environment()
    executor = SimpleExecutor(_directory())
    state = ActionExecutionState.create(
        action_type=ActionType.REST, initiator_id="agent-a", participant_ids=["agent-a"],
        purpose="小憩片刻", started_step=1, estimated_steps=3, opening_outcome="",
        target=ActionTarget(),
    )
    state.remaining_steps = 1

    (result,) = await executor.interrupt(
        state, 3, agents={"agent-a": None}, environment=env, thought=_THOUGHT,
    )

    assert _THOUGHT in result.factual_memory
    assert _THOUGHT in result.outcome        # god-view full record
    assert result.observations == []


@pytest.mark.asyncio
async def test_an_exposed_covert_leaks_the_deed_but_never_the_thought(container) -> None:
    """COVERT is the one interrupt path whose outcome doubles as the bystanders' observation.

    So the reason must hang off ``outcome`` alone: appending it to the shared string would publish
    the actor's private thought to everyone who saw him. The two channels are built separately.
    """
    from tests.unit.test_action_executors import _make_agent
    from engine.executors.covert import CovertExecutor

    env = _make_environment()
    agent = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
    agents = {"agent-a": agent}
    executor = CovertExecutor(container.llm_router, _directory(agents))
    action = AgentAction(
        agent_id="agent-a", step=1, action_type=ActionType.COVERT,
        action_description="潜入库房", target=ActionTarget(), estimated_steps=3,
    )
    state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
    state.remaining_steps = 1

    (result,) = await executor.interrupt(
        state, 3, agents=agents, environment=env, thought=_THOUGHT,
    )

    assert _THOUGHT in result.outcome
    # Not even exposure is delivered: a COVERT exposed on the interrupted step goes unperceived
    # (accepted cost; see the interrupt contract).
    assert result.observations == []


_CAUSE = "被甲强行拉走"


@pytest.mark.asyncio
async def test_a_seized_talker_is_told_what_was_done_to_him() -> None:
    """``cause`` takes the same path as ``thought``: the reason for the interruption must reach each
    participant's first-person memory.

    A cause means no participant chose to stop, so neither memory may say they walked away. Without
    a cause, the model fills the gap with a place name taken from purpose.
    """
    from engine.executors.social import SocialExecutor
    from tests.unit.test_condition_reaches_every_prompt import _Capture, _agent

    env = _make_environment()
    llm = _Capture('{"fact":"f"}')
    executor = SocialExecutor(llm, _directory(), seconds_per_step=3600)
    agents = {"agent-a": _agent("agent-a", "甲"), "agent-b": _agent("agent-b", "乙")}
    state = ActionExecutionState.create(
        action_type=ActionType.TALK, initiator_id="agent-a",
        participant_ids=["agent-a", "agent-b"], purpose="命他退往东宫偏殿静思",
        started_step=1, estimated_steps=3, opening_outcome="", target=ActionTarget(),
    )
    state.extra["target_id"] = "agent-b"
    state.remaining_steps = 1

    results = await executor.interrupt(
        state, 3, agents=agents, environment=env,
        interrupted_agent_id="agent-b", thought="", cause=_CAUSE,
    )

    assert len(results) == 2
    prompts = "\n".join(m.content for call in llm.messages for m in call)
    # Each was asked once, and both times got the cause.
    assert prompts.count(_CAUSE) == 2
    # The one who was dragged away didn't choose to stop; he made no decision.
    assert "撂下" not in prompts
    # First-person memory must be anchored to a place (same as work's interrupt memory): without
    # an anchor, the model picks a place name out of purpose.
    assert prompts.count("我此刻在") == 2
