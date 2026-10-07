"""InjectionDispatcher: the dispatch path shared by both authors.

These tests pin the answers that would drift if the code branched by author: who counts as affected,
where the injection lands, and whose name is on an injected message. The LLM event editor and the
human director must get the same answer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.interfaces.urgency import Urgency
from engine.broadcast import BroadcastChannel
from engine.directory import LiveWorldDirectory
from engine.environment import EnvironmentSystem
from engine.execution_processor import ExecutionProcessor
from engine.executors.registry import ActionExecutorRegistry
from engine.injection import Author, BroadcastSpec, InjectionDispatcher, MessageSpec
from engine.message_system import MessageSystem
from engine.world_mutation import EntityMutation, SpawnMutation, WorldMutationChannel
from core.interfaces.severity import Severity
from providers.message.in_memory import InMemoryMessageProvider
from core.interfaces.place import Place


def _agent(agent_id: str, name: str, location: str, *, active: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=agent_id,
        is_active=active,
        personality=SimpleNamespace(
            soul=SimpleNamespace(name=name),
            state=SimpleNamespace(current_location=location),
        ),
    )


def _setup():
    environment = EnvironmentSystem()
    for place_id, name in (("palace", "太极宫"), ("market", "西市")):
        environment.space.register_place(Place(place_id=place_id, name=name))
    agents = {
        "a1": _agent("a1", "李世民", "palace"),
        "a2": _agent("a2", "李建成", "market"),
        "a3": _agent("a3", "李元吉", "palace", active=False),
    }
    for aid, agent in agents.items():
        environment.place_agent(agent_id=aid, location_id=agent.personality.state.current_location)
    message_system = MessageSystem(InMemoryMessageProvider(), world_id="w1")
    directory = LiveWorldDirectory.from_agents(agents, environment)
    broadcast_channel = BroadcastChannel()
    dispatcher = InjectionDispatcher(
        broadcast_channel=broadcast_channel,
        message_system=message_system,
        mutation_channel=WorldMutationChannel(
            environment=environment,
            seconds_per_step=3600,
            processor=ExecutionProcessor(
                executor_registry=ActionExecutorRegistry(),
                environment=environment,
                message_system=message_system,
                directory=directory,
            ),
        ),
        directory=directory,
    )
    return dispatcher, agents, environment, message_system, broadcast_channel


@pytest.mark.asyncio
@pytest.mark.parametrize("author", list(Author))
async def test_a_broadcast_touches_the_living_who_hear_it_whoever_wrote_it(author) -> None:
    """A broadcast-only injection still affects people: the living who are present count, the dead
    and those elsewhere don't."""
    dispatcher, agents, *_ = _setup()

    committed = await dispatcher.dispatch(
        author=author, step=3, agents=agents, narrative_desc="宫中起火",
        broadcast=BroadcastSpec(content="宫中火起。", severity=Severity.HIGH, location_scope="palace"),
        message=None,
    )

    assert committed is not None
    assert committed.target_ids == ("a1",)
    assert committed.event.affected_names == ["李世民"]
    assert committed.event.location_label == "太极宫"
    assert committed.event.authored_by is author


@pytest.mark.asyncio
@pytest.mark.parametrize("author", list(Author))
async def test_a_spawn_without_a_broadcast_still_says_where_it_landed(author) -> None:
    dispatcher, agents, environment, *_ = _setup()

    committed = await dispatcher.dispatch(
        author=author, step=3, agents=agents, narrative_desc="西市多了一口箱子",
        broadcast=None, message=None,
        mutations=[SpawnMutation(
            observation="街心忽然多了一口木箱。", location_id="market",
            name="木箱", entity_type="item",
        )],
    )

    assert committed is not None
    assert committed.event.dispatched_to == ["mutation"]
    assert committed.event.location_label == "西市"
    assert committed.target_ids == ("a2",)
    assert [e.name for e in environment.get_items_at("market")] == ["木箱"]


@pytest.mark.asyncio
@pytest.mark.parametrize("author", list(Author))
async def test_an_injected_message_is_unsigned_and_only_metadata_names_the_author(author) -> None:
    """The recipient reads a message of unknown origin; authorship lives only in code-layer metadata."""
    dispatcher, agents, _, message_system, _ = _setup()

    await dispatcher.dispatch(
        author=author, step=3, agents=agents, narrative_desc="有人递来密信",
        broadcast=None,
        message=MessageSpec(content="今夜别出门。", recipients=["a2"], urgency=Urgency.HIGH),
    )

    [message] = await message_system.peek_pending()
    assert (message.sender_id, message.sender_name) == ("narrator", "不知来源")
    assert message.sender_is_agent is False
    assert message.metadata == {"source": author.value, "narrative": True}


@pytest.mark.asyncio
async def test_nothing_dispatched_means_no_event() -> None:
    dispatcher, agents, *_ = _setup()

    committed = await dispatcher.dispatch(
        author=Author.SYSTEM, step=3, agents=agents, narrative_desc="空",
        broadcast=None, message=None,
        mutations=[EntityMutation(observation="…", entity_id="missing", destroyed=True)],
    )

    assert committed is None
