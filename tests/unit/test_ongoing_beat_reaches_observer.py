"""A multi-step action must reach the observer as THREE beats: begin, still in progress, result.

The runtime must pass through the middle beat ExecutionProcessor builds every step, or a
multi-step act reads as "he set out to do X" … hours of silence … "X turned out thus".

The tick record must carry ``agent_name`` (else the read model falls back to the agent_id, an id
in the narrative layer), ``execution_id`` (ties it to its act) and ``action_description``.
"""

from __future__ import annotations

import pytest

from core.interfaces.action import ActionTarget, ActionType, AgentAction, Ref
from engine.execution_processor import ExecutionProcessor
from engine.executors.registry import ActionExecutorRegistry
from engine.executors.work import WorkExecutor
from interaction.models import _action_from_record
from tests.unit.test_action_executors import _directory, _make_agent, _make_environment
from tests.unit.bus_tap import tap


async def _tick_a_multi_step_work(container) -> dict:
    """One WORK, three steps long, advanced by one step — i.e. the middle beat."""
    env = _make_environment()
    agent = _make_agent(container, world_id="w", agent_id="agent-a", name="李渊")
    agents = {"agent-a": agent}
    directory = _directory(agents)
    executor = WorkExecutor(container.llm_router, directory)
    registry = ActionExecutorRegistry()
    registry.register(ActionType.WORK, executor)
    processor = ExecutionProcessor(
        executor_registry=registry, environment=env,
        message_system=None,  # type: ignore[arg-type]  # WORK never dispatches
        directory=directory,
    )
    state = await executor.start(
        AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.WORK,
            action_description="重新审视案头那份手诏", target=ActionTarget(), estimated_steps=3,
        ),
        1, agents=agents, environment=env, message_system=None,
    )
    registry.add_active(state)

    records = await processor.tick_ongoing_executions(agents, step=2)

    (tick,) = [r for r in records if r.get("phase") == "ongoing_tick"]
    return tick


@pytest.mark.asyncio
async def test_the_progress_beat_is_a_full_observer_record(container) -> None:
    tick = await _tick_a_multi_step_work(container)

    # It says WHO — by name. An agent_id here would be an id in the narrative layer: the read
    # model falls back to it, and it would be printed.
    assert tick["agent_name"] == "李渊"
    assert tick["agent_name"] != tick["agent_id"]
    # It says WHICH ACT — so a client can tie it to the begin/complete beats around it, and can
    # fold a joint action's per-participant ticks into one.
    assert tick["execution_id"]
    assert tick["initiator_id"] == "agent-a"
    assert tick["action_description"]
    # And it says something NEW: that it is still going, and how far along.
    assert tick["outcome"]


@pytest.mark.asyncio
async def test_the_progress_beat_says_who_it_is_aimed_at(container) -> None:
    """The middle beat must carry the aim, not only the begin and complete beats.

    Both consumers fail SILENTLY on an empty list: the renderer re-aims the body every tick, so a
    figure mid-conversation turns to face nobody; and the pair drops out of the layout's "dealing
    with each other" set, drifting apart until the result snaps them back together.
    """
    env = _make_environment()
    a = _make_agent(container, world_id="w", agent_id="agent-a", name="李世民")
    b = _make_agent(container, world_id="w", agent_id="agent-b", name="魏征")
    agents = {"agent-a": a, "agent-b": b}
    directory = _directory(agents)
    executor = WorkExecutor(container.llm_router, directory)   # any multi-step executor
    registry = ActionExecutorRegistry()
    registry.register(ActionType.WORK, executor)
    processor = ExecutionProcessor(
        executor_registry=registry, environment=env,
        message_system=None,  # type: ignore[arg-type]
        directory=directory,
    )
    state = await executor.start(
        AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.WORK,
            action_description="与魏征议事", target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
            estimated_steps=3,
        ),
        1, agents=agents, environment=env, message_system=None,
    )
    registry.add_active(state)

    (tick,) = [r for r in await processor.tick_ongoing_executions(agents, step=2)
               if r.get("phase") == "ongoing_tick"]

    assert tick["target"]["acts_on"] == [{"kind": "agent", "id": "agent-b"}]
    # And it survives into the read model the renderer actually consumes.
    assert [r.id for r in _action_from_record(tick).target.acts_on] == ["agent-b"]


@pytest.mark.asyncio
async def test_the_progress_beat_survives_the_read_model_without_leaking_an_id(container) -> None:
    summary = _action_from_record(await _tick_a_multi_step_work(container))

    assert summary.phase == "ongoing_tick"
    assert summary.agent_name == "李渊"
    assert "agent-" not in summary.agent_name  # the fallback must not put the id here
    assert summary.execution_id
    assert summary.action_description


@pytest.mark.asyncio
async def test_the_progress_beat_actually_reaches_the_step_stream(container) -> None:
    """The runtime builds the middle beat and must not filter it out of the action stream.

    Also pins the flag that tells the observer an act is still running — without it, the begin
    record's OPENING ("X着手做…") gets printed as though it were a result, and the same sentence
    comes out twice on one card.
    """
    published = tap(container.event_bus)
    from agent.personality import ActionStatus
    from core.interfaces.llm import LLMScene
    from tests.unit.test_runtime_smoke import _build_agent, _build_runtime

    world_id = "world-ongoing-beat"
    runtime, environment, _ = _build_runtime(container, world_id=world_id)
    environment.place_agent(agent_id="agent-1", location_id="palace")
    agent = _build_agent(container, world_id=world_id, agent_id="agent-1", name="李渊",
                         is_main_character=True)
    # A WORK that SPANS steps — the whole point. (The shared helper pins estimated_steps=1,
    # which is born-zero: it starts and finishes on one step and never ticks at all.)
    container.llm_router.get(LLMScene.AGENT_DECISION_MAIN).fixed_response = (
        '{"selected_index": 3, "action_description": "重新审视案头那份手诏", "estimated_steps": 3}'
    )

    await runtime.run_step([agent])                       # step 1: he sets out
    begin = published()[0]["actions"][0]
    assert begin["phase"] == "begin", "a multi-step opening must stay an opening, not 'settled'"
    assert agent.personality.state.action_status == ActionStatus.IN_PROGRESS

    await runtime.run_step([agent])                       # step 2: still at it
    ticks = [r for r in published()[0]["actions"]
             if r.get("phase") == "ongoing_tick"]

    assert ticks, "the middle beat must reach the observer's stream, not just the CLI log"
    assert ticks[0]["execution_id"] == begin["execution_id"]  # …tied to the act it belongs to
    # Never an id. This fixture's directory is empty, so the name degrades to "某人" (someone) per the
    # directory's contract; the exact-name case is pinned above.
    assert "agent-" not in ticks[0]["agent_name"]
