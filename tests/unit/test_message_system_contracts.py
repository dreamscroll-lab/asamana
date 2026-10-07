"""Negative assertions ("what MessageSystem must not do") and cross-field consistency, locking the
design invariants in the engine/message_system.py module docstring. Happy-path tests live in
test_message_system.py.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.decision import ActionType, AgentAction
from core.interfaces.action import ActionTarget, Ref
from core.interfaces.message import Message
from core.interfaces.urgency import Urgency
from engine.message_system import MessageSystem


def _make_message_system(container: object, *, world_id: str = "w1") -> MessageSystem:
    return MessageSystem(container.message_provider, world_id=world_id)


def _msg(
    *, mid: str = "m1", sender: str = "agent-1",
    recipients: list[str] | None = None,
    location_scope: str | None = None,
    deliver_step: int = 1,
    content: str = "hello",
    urgency: Urgency = Urgency.NORMAL,
    actor_ids: tuple[str, ...] = (),
) -> Message:
    return Message(
        id=mid, world_id="w1",
        sender_id=sender, content=content,
        recipients=recipients, location_scope=location_scope,
        deliver_step=deliver_step, created_step=deliver_step,
        urgency=urgency, actor_ids=actor_ids,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lossless transport: byte-level end to end + same object reference
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_p2_message_object_identity_preserved_through_pipe(container) -> None:
    """Lossless transport, strengthened: the inbox Message is the very object published, ruling out any rebuild."""
    system = _make_message_system(container)
    original = _msg(mid="m-id", sender="agent-1", recipients=["agent-2"],
                    deliver_step=1, content="精确字节内容\n含中文与换行")
    await system.publish(original)
    delivery = await system.deliver_for_agents(step=1, agent_ids=["agent-1", "agent-2"])
    inbox = delivery.inbox_for("agent-2")
    assert len(inbox) == 1
    # key invariant: the same object reference, not a rebuilt Message(... content=...)
    assert inbox[0] is original
    # byte-level backstop
    assert inbox[0].content == "精确字节内容\n含中文与换行"
    assert inbox[0].id == "m-id"
    assert inbox[0].urgency == Urgency.NORMAL


# ─────────────────────────────────────────────────────────────────────────────
# No side effects + no LLM dependency
# ─────────────────────────────────────────────────────────────────────────────


def test_p4_message_system_module_does_not_import_llm_router() -> None:
    """engine/message_system.py must not import LLMRouter / LLMScene / extract_json.
    MessageSystem is a lossless pipe; any such import means it has started altering messages in
    transit.
    """
    src = Path("engine/message_system.py").read_text(encoding="utf-8")
    forbidden = ["LLMRouter", "LLMScene", "extract_json", "_llm_", "complete_with_retry"]
    found = [token for token in forbidden if token in src]
    assert not found, (
        f"engine/message_system.py 重新出现禁用 LLM 依赖: {found}。"
        f"无损传输与无副作用契约要求 MessageSystem 不调 LLM。"
    )


def test_p4_message_system_constructor_does_not_accept_llm_router() -> None:
    """MessageSystem.__init__ must not take an llm_router parameter."""
    sig = inspect.signature(MessageSystem.__init__)
    params = set(sig.parameters.keys())
    assert "llm_router" not in params, (
        "MessageSystem.__init__ 出现 llm_router 参数。"
        "无损传输与无副作用要求 MessageSystem 不依赖 LLM。"
    )


@pytest.mark.asyncio
async def test_p4_deliver_for_agents_does_not_call_relation_or_memory(container) -> None:
    """No side effects: deliver_for_agents calls no relation_system / memory_system method."""
    system = _make_message_system(container)
    await system.publish(_msg(mid="m1", sender="a1", recipients=["a2"], deliver_step=1))

    # Sentinel mocks: any call from MessageSystem fails the test.
    relation_mock = MagicMock()
    memory_mock = MagicMock()

    await system.deliver_for_agents(step=1, agent_ids=["a1", "a2"])

    assert relation_mock.method_calls == []
    assert memory_mock.method_calls == []


# ─────────────────────────────────────────────────────────────────────────────
# Self-exclusion: the sender is never a receiver
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_p3_sender_never_in_receivers_across_all_four_combos(container) -> None:
    """Across all four field combinations: the sender never receives its own message."""
    from engine.environment import EnvironmentSystem
    from core.interfaces.place import Place
    env = EnvironmentSystem()
    env.space.register_place(Place(
        place_id="loc_a", name="loc_a", description="",
        connections={}, is_public=True, capacity=50,
    ))
    env.place_agent(agent_id="a1", location_id="loc_a")
    env.place_agent(agent_id="a2", location_id="loc_a")

    system = _make_message_system(container)
    agent_ids = ["a1", "a2"]

    # case 1: directed, including the sender itself
    await system.publish(_msg(mid="c1", sender="a1", recipients=["a1", "a2"], deliver_step=1))
    # case 2: world-wide broadcast (recipients=None, location_scope=None)
    await system.publish(_msg(mid="c2", sender="a1", recipients=None, location_scope=None, deliver_step=1))
    # case 3: location-scoped broadcast
    await system.publish(_msg(mid="c3", sender="a1", recipients=None, location_scope="loc_a", deliver_step=1))
    # case 4: intersection (list and location)
    await system.publish(_msg(mid="c4", sender="a1", recipients=["a1", "a2"], location_scope="loc_a", deliver_step=1))

    delivery = await system.deliver_for_agents(step=1, agent_ids=agent_ids, environment=env)
    sender_inbox = delivery.inbox_for("a1")
    assert sender_inbox == [], (
        f"自我排除违规:sender a1 收到了自己的消息(共 {len(sender_inbox)} 条:"
        f"{[m.id for m in sender_inbox]})"
    )
    assert {m.id for m in delivery.inbox_for("a2")} == {"c1", "c2", "c3", "c4"}


@pytest.mark.asyncio
async def test_every_member_of_the_acting_body_is_left_out_of_all_four_combos(container) -> None:
    """Self-exclusion: members of the acting party never receive what they did, in all four cases,
    even when the speaker and the author are different people."""
    from engine.environment import EnvironmentSystem
    from core.interfaces.place import Place
    env = EnvironmentSystem()
    env.space.register_place(Place(
        place_id="loc_a", name="loc_a", description="",
        connections={}, is_public=True, capacity=50,
    ))
    for aid in ("a1", "a2", "a3"):
        env.place_agent(agent_id=aid, location_id="loc_a")

    system = _make_message_system(container)
    agent_ids = ["a1", "a2", "a3"]
    # The sender is the one speaking; a2 dictated the words. Both belong to the acting party.
    body = ("mouth", "a2")
    for mid, recipients, scope in (
        ("c1", ["a1", "a2", "a3"], None),
        ("c2", None, None),
        ("c3", None, "loc_a"),
        ("c4", ["a1", "a2", "a3"], "loc_a"),
    ):
        await system.publish(_msg(
            mid=mid, sender="mouth", recipients=recipients,
            location_scope=scope, deliver_step=1, actor_ids=body,
        ))

    delivery = await system.deliver_for_agents(step=1, agent_ids=agent_ids, environment=env)
    assert delivery.inbox_for("a2") == []
    assert {m.id for m in delivery.inbox_for("a1")} == {"c1", "c2", "c3", "c4"}
    assert {m.id for m in delivery.inbox_for("a3")} == {"c1", "c2", "c3", "c4"}


def test_the_delivery_rules_survive_a_snapshot_round_trip() -> None:
    """A restored message is delivered by the same rule. Dropping a field would silently change the
    rule."""
    import dataclasses
    import json

    from core.serialization import dump_json
    from providers.snapshot.file import _message_from_dict

    original = Message(
        id="m1", world_id="w1", sender_id="mouth", content="城门今夜不开",
        recipients=None, location_scope="loc_a", deliver_step=3, created_step=2,
        sender_name="王二", sender_is_agent=False, actor_ids=("mouth", "a2"),
    )
    restored = _message_from_dict(json.loads(dump_json(dataclasses.asdict(original))))
    assert restored.actor_ids == ("mouth", "a2")
    assert restored.sender_is_agent is False


# ─────────────────────────────────────────────────────────────────────────────
# Actions and messages decoupled: dispatch_from_action metadata
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_p5_dispatch_from_action_stamps_source_metadata(container) -> None:
    """dispatch_from_action stamps source=action + action_type in Message.metadata.

    This is the caller/system contract that lets a receiver tell "this came from an action
    dispatch" apart from "the narrator published it directly".
    """
    system = _make_message_system(container)
    action = AgentAction(
        agent_id="a1", step=1,
        action_type=ActionType.SEND_MESSAGE,
        action_description="紧急召集",
        target=ActionTarget(acts_on=[Ref.agent("a2")]),
        urgency=Urgency.HIGH,
    )
    message = await system.dispatch_from_action(action, current_step=1, sender_name="李建成")
    assert message.metadata.get("source") == "action"
    assert message.metadata.get("action_type") == ActionType.SEND_MESSAGE.value
    # urgency passes through; the recipient decides
    assert message.urgency == Urgency.HIGH


# ─────────────────────────────────────────────────────────────────────────────
# Scope: same-location executors stay out of MessageSystem (via source inspection)
# ─────────────────────────────────────────────────────────────────────────────


def test_p1_same_location_executors_do_not_call_message_system_methods() -> None:
    """Same-location executors (TALK / COVERT / WORK / MOVE) must not call message_system write
    methods: their outcomes propagate via ActionResult -> _carry_step_observations -> ambient_events.

    SimpleExecutor is exempt: its SEND_MESSAGE is cross-location. Only method calls are checked, since
    the interface signature needs the ``MessageSystem`` type hint.
    """
    same_location_executor_files = [
        "engine/executors/social.py",   # TALK
        "engine/executors/covert.py",   # COVERT
        "engine/executors/work.py",     # WORK
        "engine/executors/movement.py", # MOVE
    ]
    forbidden_calls = [
        "message_system.publish(",
        "message_system.dispatch_from_action(",
    ]
    violations: list[tuple[str, str]] = []
    for filepath in same_location_executor_files:
        src = Path(filepath).read_text(encoding="utf-8")
        for token in forbidden_calls:
            if token in src:
                violations.append((filepath, token))
    assert not violations, (
        "同地点不走消息系统违规:以下同地点 executor 调用了 MessageSystem 写入方法,"
        "违反「同地点信息走 carry」契约:\n"
        + "\n".join(f"  {f}: 含 {t}" for f, t in violations)
    )
