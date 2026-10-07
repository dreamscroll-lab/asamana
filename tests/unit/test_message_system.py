"""Unit tests for the engine message system (a pure transport pipe).

Contract (see the engine/message_system.py module docstring):
- Scope: carries only cross-location / cross-time traffic; same-location goes through
  carry_observation, not MessageSystem
- Lossless transport: inbox.content is byte-identical to the sent content
- Data-driven: no PropagationType; dispatch by recipients x location_scope x deliver_step
- No side effects: no relation updates, no memory writes, no interrupts, no LLM calls
- Actions and messages decoupled: TALK/COVERT never enter MessageSystem
"""

from __future__ import annotations

import pytest

from agent.decision import ActionType, AgentAction
from core.interfaces.action import ActionTarget, Ref
from core.interfaces.message import Message
from engine.environment import EnvironmentSystem
from engine.message_system import MessageSystem
from core.interfaces.urgency import Urgency
from worlds.tiled import TiledWorldConfig


def _make_message_system(container: object, *, world_id: str = "world-1") -> MessageSystem:
    return MessageSystem(container.message_provider, world_id=world_id)  # type: ignore[attr-defined]


def _direct(
    *,
    msg_id: str,
    world_id: str = "world-1",
    sender_id: str,
    recipient_id: str,
    content: str,
    deliver_step: int,
    urgency: Urgency = Urgency.NORMAL,
) -> Message:
    return Message(
        id=msg_id,
        world_id=world_id,
        sender_id=sender_id,
        content=content,
        recipients=[recipient_id],
        location_scope=None,
        deliver_step=deliver_step,
        created_step=deliver_step,
        urgency=urgency,
    )


def _broadcast(
    *,
    msg_id: str,
    world_id: str = "world-1",
    sender_id: str,
    content: str,
    deliver_step: int,
    location_scope: str | None = None,
) -> Message:
    return Message(
        id=msg_id,
        world_id=world_id,
        sender_id=sender_id,
        content=content,
        recipients=None,
        location_scope=location_scope,
        deliver_step=deliver_step,
        created_step=deliver_step,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Time dimension (deliver_step)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_message_is_collected_only_at_or_after_deliver_step(container: object) -> None:
    system = _make_message_system(container)

    await system.publish(
        _direct(
            msg_id="m-future",
            sender_id="agent-1",
            recipient_id="agent-2",
            content="Hello.",
            deliver_step=3,
        )
    )

    assert await system.collect(step=2) == []
    messages = await system.collect(step=3)
    assert len(messages) == 1
    assert messages[0].id == "m-future"


@pytest.mark.asyncio
async def test_delayed_message_via_deliver_step_in_future(container: object) -> None:
    """Delay is carried by the deliver_step field; no separate propagation type is needed."""
    system = _make_message_system(container)

    msg = _direct(
        msg_id="m-delayed",
        sender_id="agent-1",
        recipient_id="agent-2",
        content="后天的事。",
        deliver_step=7,
    )
    msg.created_step = 5  # created at 5, delivered at 7 -> 2-step delay
    await system.publish(msg)

    assert await system.collect(step=6) == []
    messages = await system.collect(step=7)
    assert len(messages) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch: the four-way recipients x location_scope matrix
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_directed_delivery_recipients_list_no_location(container: object) -> None:
    """recipients=list, location=None -> delivered to the listed agents."""
    system = _make_message_system(container)

    await system.publish(
        _direct(
            msg_id="m-direct",
            sender_id="agent-2",
            recipient_id="agent-1",
            content="Meet me at dawn.",
            deliver_step=1,
        )
    )

    delivery = await system.deliver_for_agents(step=1, agent_ids=["agent-1", "agent-2"])

    assert "m-direct" in [m.id for m in delivery.inbox_for("agent-1")]
    assert "m-direct" not in [m.id for m in delivery.inbox_for("agent-2")]


@pytest.mark.asyncio
async def test_broadcast_no_recipients_no_location_reaches_all_except_sender(
    container: object,
) -> None:
    """recipients=None, location=None -> world-wide broadcast (sender excluded)."""
    system = _make_message_system(container)

    await system.publish(
        _broadcast(
            msg_id="m-bc-all",
            sender_id="agent-1",
            content="Emergency proclamation.",
            deliver_step=1,
        )
    )

    delivery = await system.deliver_for_agents(step=1, agent_ids=["agent-1", "agent-2", "agent-3"])

    assert "m-bc-all" not in [m.id for m in delivery.inbox_for("agent-1")]
    assert "m-bc-all" in [m.id for m in delivery.inbox_for("agent-2")]
    assert "m-bc-all" in [m.id for m in delivery.inbox_for("agent-3")]


@pytest.mark.asyncio
async def test_broadcast_with_location_scope_only_reaches_agents_in_that_location(
    container: object,
) -> None:
    """recipients=None, location=L -> delivered to every agent in L."""
    system = _make_message_system(container)
    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))  # type: ignore[attr-defined]
    environment.place_agent(agent_id="agent-1", location_id="taiji_palace")
    environment.place_agent(agent_id="agent-2", location_id="taiji_palace")
    environment.place_agent(agent_id="agent-3", location_id="market")

    await system.publish(
        _broadcast(
            msg_id="m-bc-loc",
            sender_id="agent-1",
            content="城门即将关闭！",
            deliver_step=1,
            location_scope="taiji_palace",
        )
    )

    delivery = await system.deliver_for_agents(
        step=1,
        agent_ids=["agent-1", "agent-2", "agent-3"],
        environment=environment,
    )

    assert "m-bc-loc" in [m.id for m in delivery.inbox_for("agent-2")]
    assert "m-bc-loc" not in [m.id for m in delivery.inbox_for("agent-1")]  # the sender doesn't receive it
    assert "m-bc-loc" not in [m.id for m in delivery.inbox_for("agent-3")]  # not in the location


@pytest.mark.asyncio
async def test_recipients_list_intersect_with_location_scope(container: object) -> None:
    """recipients=list, location=L -> intersection (listed agents who are in L)."""
    system = _make_message_system(container)
    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))  # type: ignore[attr-defined]
    environment.place_agent(agent_id="agent-1", location_id="taiji_palace")
    environment.place_agent(agent_id="agent-2", location_id="taiji_palace")
    environment.place_agent(agent_id="agent-3", location_id="market")

    msg = Message(
        id="m-intersect",
        world_id="world-1",
        sender_id="agent-1",
        content="只在宫里的二三号听。",
        recipients=["agent-2", "agent-3"],
        location_scope="taiji_palace",
        deliver_step=1,
        created_step=1,
    )
    await system.publish(msg)

    delivery = await system.deliver_for_agents(
        step=1,
        agent_ids=["agent-1", "agent-2", "agent-3"],
        environment=environment,
    )

    # agent-2 is listed and in palace -> receives
    assert "m-intersect" in [m.id for m in delivery.inbox_for("agent-2")]
    # agent-3 is listed but in market -> does not receive
    assert "m-intersect" not in [m.id for m in delivery.inbox_for("agent-3")]


@pytest.mark.asyncio
async def test_unknown_recipient_goes_to_undelivered(container: object) -> None:
    system = _make_message_system(container)

    await system.publish(
        _direct(
            msg_id="m-lost",
            sender_id="agent-1",
            recipient_id="ghost-agent",
            content="This will be lost.",
            deliver_step=1,
        )
    )

    delivery = await system.deliver_for_agents(step=1, agent_ids=["agent-1", "agent-2"])

    assert "m-lost" in [m.id for m in delivery.undelivered_messages]


# ─────────────────────────────────────────────────────────────────────────────
# Lossless transport
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inbox_content_is_byte_identical_to_sent_content(container: object) -> None:
    """The pipe does no rendering. Message.content in the inbox is identical to the published
    content."""
    system = _make_message_system(container)

    sent_content = "这是非常具体且不应被改写的原话:细节 A、细节 B、细节 C。"
    msg = _direct(
        msg_id="m-loss-less",
        sender_id="sender",
        recipient_id="receiver",
        content=sent_content,
        deliver_step=1,
    )
    await system.publish(msg)

    delivery = await system.deliver_for_agents(step=1, agent_ids=["sender", "receiver"])
    received = delivery.inbox_for("receiver")[0]

    assert received.content == sent_content
    assert received is msg  # the same object: never copied or rewritten


# ─────────────────────────────────────────────────────────────────────────────
# The recipient decides: urgency is a meta signal
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_urgency_field_is_preserved_through_delivery(container: object) -> None:
    system = _make_message_system(container)

    await system.publish(
        _direct(
            msg_id="m-urgent",
            sender_id="agent-1",
            recipient_id="agent-2",
            content="紧急!",
            deliver_step=1,
            urgency=Urgency.HIGH,
        )
    )

    delivery = await system.deliver_for_agents(step=1, agent_ids=["agent-1", "agent-2"])
    received = delivery.inbox_for("agent-2")[0]
    assert received.urgency == Urgency.HIGH


# ─────────────────────────────────────────────────────────────────────────────
# dispatch_from_action (cross-location actions only)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_from_action_with_targets_creates_directed_message(
    container: object,
) -> None:
    system = _make_message_system(container)

    queued = await system.dispatch_from_action(
        AgentAction(
            agent_id="agent-1",
            step=6,
            action_type=ActionType.SEND_MESSAGE,
            action_description="Warn the prince about danger.",
            target=ActionTarget(acts_on=[Ref.agent("agent-2")]),
            inner_monologue="urgent warning",
        ),
        current_step=6,
        sender_name="Li Shimin",
        location_scope="taiji_palace",
    )

    assert queued.recipients == ["agent-2"]
    assert queued.location_scope == "taiji_palace"
    assert queued.deliver_step == 6
    assert queued.intent == "urgent warning"
    assert queued.urgency == Urgency.NORMAL


@pytest.mark.asyncio
async def test_dispatch_delivers_message_content_not_action_description(
    container: object,
) -> None:
    """What is delivered is the words addressed to the recipient (content=message_content), not the
    sender's narration of their own action (action_description). They are two different narrative
    voices."""
    system = _make_message_system(container)

    queued = await system.dispatch_from_action(
        AgentAction(
            agent_id="agent-1",
            step=6,
            action_type=ActionType.SEND_MESSAGE,
            action_description="我向父皇传讯，揭发太子谋反",        # the sender's own action narration
            content="父皇！太子欲谋逆，此乃铁证，请明察！",          # the words sent to the recipient
            target=ActionTarget(acts_on=[Ref.agent("agent-2")]),
        ),
        current_step=6,
        sender_name="李世民",
    )

    assert queued.content == "父皇！太子欲谋逆，此乃铁证，请明察！"
    assert "我向父皇传讯" not in queued.content


@pytest.mark.asyncio
async def test_dispatch_from_action_without_targets_creates_broadcast(container: object) -> None:
    system = _make_message_system(container)

    queued = await system.dispatch_from_action(
        AgentAction(
            agent_id="agent-1",
            step=5,
            action_type=ActionType.SEND_MESSAGE,
            action_description="昭告天下",
            target=ActionTarget(),  # no agent_ids → broadcast
        ),
        current_step=5,
    )

    assert queued.recipients is None  # None = broadcast
    assert queued.location_scope is None


@pytest.mark.asyncio
async def test_dispatch_from_action_propagates_action_urgency(
    container: object,
) -> None:
    system = _make_message_system(container)

    queued = await system.dispatch_from_action(
        AgentAction(
            agent_id="agent-1",
            step=10,
            action_type=ActionType.SEND_MESSAGE,
            action_description="快来救我！",
            target=ActionTarget(acts_on=[Ref.agent("agent-2")]),
            inner_monologue="情况危急",
            urgency=Urgency.HIGH,
        ),
        current_step=10,
        sender_name="小明",
    )

    assert queued.urgency == Urgency.HIGH


# ─────────────────────────────────────────────────────────────────────────────
# MessageDelivery.as_dict / empty state
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_as_dict_shape(container: object) -> None:
    system = _make_message_system(container)

    await system.publish(
        _direct(
            msg_id="m-snap",
            sender_id="a",
            recipient_id="b",
            content="snapshot test",
            deliver_step=5,
        )
    )

    delivery = await system.deliver_for_agents(step=5, agent_ids=["a", "b"])
    d = delivery.as_dict()

    assert d["step"] == 5
    assert "delivered" in d and len(d["delivered"]) == 1
    assert d["delivered"][0]["recipients"] == ["b"]
    assert "received" not in d  # payload has no received/perceived_content (lossless transport)
    assert "inboxes" in d
    assert "undelivered" in d


@pytest.mark.asyncio
async def test_empty_step_returns_empty_delivery(container: object) -> None:
    system = _make_message_system(container)

    delivery = await system.deliver_for_agents(step=99, agent_ids=["a", "b"])

    assert delivery.inbox_for("a") == []
    assert delivery.inbox_for("b") == []
    assert delivery.delivered_messages == []
    assert delivery.undelivered_messages == []
