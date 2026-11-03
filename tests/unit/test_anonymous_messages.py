"""Anonymous messages retain provenance without giving recipients an identified sender."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent.actor_operations import build_actor_operation
from agent.decision import DecisionEngine, candidate_lists
from core.interfaces.action import ActionTarget, ActionType, AgentAction, Ref
from core.interfaces.actor import ActorAction, ActorAnswerRejected, ActorIntent, ActorOperation, ActorOperationParameters
from core.interfaces.llm import LLMScene
from core.interfaces.message import Message
from core.interfaces.perception import PerceivedIdentity
from core.interfaces.snapshot import EngineState, WorldSnapshot
from core.interfaces.urgency import Urgency
from engine.executors.simple import SimpleExecutor
from engine.message_system import MessageSystem
from engine.step_record import message_events
from interaction.actor_view import step_view_for
from providers.snapshot.file import FileSnapshotProvider
from tests.unit.test_action_executors import _directory, _make_environment
from tests.unit.test_actor_action import HERE, _index_of, _packet
from tests.unit.test_actor_view import ME, _body, _letter, _step
from interaction.models import LocationScope, WorldScope
from tests.unit.test_agent_known_names import _make_agent
from tests.unit.test_decision import _make_packet, _make_personality
from tests.unit.test_perception_layer import _make_layer, _make_spatial
from tests.unit.test_world_pressure_prompt import _ev, _msg, _stub_agent


def _message(*, anonymous: bool = True, **overrides: Any) -> Message:
    data: dict[str, Any] = dict(
        id="letter", world_id="world-1", sender_id="sender", sender_name="匿名" if anonymous else "甲",
        anonymous=anonymous, content="请在午后来找我", recipients=["agent-1"], location_scope=None,
        created_step=1, deliver_step=1, urgency=Urgency.HIGH,
    )
    return Message(**(data | overrides))


@pytest.mark.asyncio
@pytest.mark.parametrize("anonymous", [False, True])
async def test_send_preserves_provenance_routing_and_signature(container: Any, anonymous: bool) -> None:
    system = MessageSystem(container.message_provider, world_id="world-1")
    action = AgentAction(
        ActionType.SEND_MESSAGE, agent_id="sender", content="正文中的署名也要原样保留：甲",
        anonymous=anonymous, target=ActionTarget(acts_on=[Ref.agent("sender"), Ref.agent("reader")]),
    )
    sent = await system.dispatch_from_action(action, current_step=1, sender_name="甲")
    assert sent.sender_id == "sender"
    assert sent.sender_name == ("匿名" if anonymous else "甲")
    assert sent.anonymous is anonymous
    delivery = await system.deliver_for_agents(step=1, agent_ids=["sender", "reader"])
    assert delivery.inbox_for("sender") == []
    assert delivery.inbox_for("reader") == [sent]
    assert sent.content == action.content
    [record] = message_events(delivery, where={"reader": "hall"})
    assert record["sender_id"] == "sender" and record["anonymous"] is anonymous


@pytest.mark.asyncio
@pytest.mark.parametrize("anonymous", [False, True])
async def test_pending_letter_roundtrips_through_file_snapshot(tmp_path: Path, anonymous: bool) -> None:
    provider = FileSnapshotProvider(str(tmp_path))
    original = _message(anonymous=anonymous)
    snapshot = WorldSnapshot(world_id="world-1", step=1, timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), engine=EngineState(pending_messages=[original]))
    await provider.save("world-1", 1, snapshot)
    restored = await provider.load("world-1", 1)
    assert restored is not None
    assert restored.engine.pending_messages == [original]


@pytest.mark.asyncio
async def test_redacted_delivery_does_not_erase_persisted_sender(container: Any, tmp_path: Path) -> None:
    from interaction.replayer import Replayer
    from tests.unit.step_view import events

    message = _message()
    system = MessageSystem(container.message_provider, world_id=message.world_id)
    await system.publish(message)
    delivery = await system.deliver_for_agents(step=1, agent_ids=["agent-1"])
    snapshot = WorldSnapshot(
        world_id=message.world_id, step=1, timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        events=message_events(delivery, where={"agent-1": "hall"}),
    )
    provider = FileSnapshotProvider(str(tmp_path))
    await provider.save(message.world_id, 1, snapshot)
    replay = Replayer(message.world_id, provider)
    complete = await replay.get_step(1)
    [(_, hidden)] = events(step_view_for(complete, "agent-1"), "message")
    assert hidden.sender_id is None
    assert (hidden.message_id, hidden.receiver_ids, hidden.content) == (message.id, ["agent-1"], message.content)
    [(_, original)] = events(complete, "message")
    [(_, reloaded)] = events(await replay.get_step(1), "message")
    assert original.sender_id == reloaded.sender_id == message.sender_id
    assert snapshot.events[0]["sender_id"] == message.sender_id


@pytest.mark.asyncio
@pytest.mark.parametrize("directed", [False, True])
async def test_executor_keeps_anonymous_choice_in_action_and_own_memory(container: Any, directed: bool) -> None:
    system = MessageSystem(container.message_provider, world_id="world-1")
    action = AgentAction(
        ActionType.SEND_MESSAGE, agent_id="agent-a", anonymous=True, content="请留步",
        target=ActionTarget(acts_on=[Ref.agent("agent-b")]) if directed else ActionTarget(),
    )
    env = _make_environment()
    executor = SimpleExecutor(_directory())
    state = await executor.start(action, 1, agents={}, environment=env, message_system=system)
    [result] = await executor.complete(state, 1, agents={}, environment=env, message_system=system)
    assert result.succeeded and result.action.anonymous
    assert "匿名" in result.factual_memory
    if not directed:
        assert "当着" not in result.factual_memory and "宣告" not in result.outcome
    [sent] = await system.peek_pending()
    assert sent.anonymous and sent.sender_id == "agent-a"


@pytest.mark.parametrize("scope", ["direct", "place", "world"])
def test_actor_response_removes_anonymous_identity_but_internal_view_keeps_it(scope: str) -> None:
    letter = replace(_letter(1, [ME] if scope != "world" else None),
                     anonymous=True, sender_name="匿名", scope=scope, receiver_ids=[ME])
    original = _step(locations={"hall": LocationScope(location_id="hall", agents=[_body(ME)], events=[letter])})
    if scope == "world":
        original = _step(world=WorldScope(events=[letter]))
    actor = step_view_for(original, ME)
    [received] = actor.world.events if scope == "world" else actor.locations["hall"].events
    assert received.sender_id is None and received.sender_name == "匿名" and received.anonymous
    assert letter.sender_id == "agent-x"
    assert "agent-x" not in json.dumps(asdict(actor), ensure_ascii=False)


@pytest.mark.asyncio
async def test_anonymous_message_does_not_enter_identity_relations_or_contact_list(container: Any) -> None:
    agent = _make_agent(container)
    agent.remember_agent("sender", PerceivedIdentity(name="甲", gender="女"))
    msg = _message()
    spatial = replace(_make_spatial(), location_id="loc")
    await agent.perceive_step(spatial=spatial, inbox=[msg], broadcasts=[], step=1)
    assert agent._known_agents["sender"] == PerceivedIdentity(name="甲", gender="女")
    assert "sender" not in [r.target_agent_id for r in await agent._perceive_relevant_relations(spatial, [msg], current_step=1)]
    assert candidate_lists(_make_packet(inbox=[msg])).roster_ids == []
    assert await agent.relation_system.all_relations() == []


@pytest.mark.asyncio
async def test_anonymous_memory_keeps_signature_without_relation_or_sender_metadata(container: Any) -> None:
    layer, _ = _make_layer(container)
    [item] = await layer._collect_items(_make_spatial(), [_message()], [])
    assert item.content == "来自匿名的消息：请在午后来找我"
    assert item.related_agents == []
    assert item.metadata["sender_id"] is None


@pytest.mark.asyncio
async def test_mixed_letters_do_not_attach_signed_relation_to_anonymous_letter() -> None:
    ev = _ev()
    agent = _stub_agent(name="乙", role="", location="hall")
    agent.relation_system.load_existing = AsyncMock(return_value=None)
    anonymous = _msg("匿名正文", sender_id="sender", sender_name="匿名", anonymous=True)
    assert await ev._sender_relations(agent, [anonymous], exclude_ids=set()) == {}
    agent.relation_system.load_existing.assert_not_awaited()
    signed = _msg("实名正文", sender_id="sender", sender_name="甲")
    signals = ev._signals(
        inbox=[signed, anonymous], broadcasts=[], ambient=[],
        sender_relations={"sender": "仅限实名的关系"}, visible_ids=set(),
    )
    prompt = ev._build_agent_prompt(agent, signals=signals, situation_header="")
    assert prompt.count("仅限实名的关系") == 1
    anonymous_line = next(line for line in prompt.splitlines() if "匿名正文" in line)
    assert "关系" not in anonymous_line and "甲" not in anonymous_line and "sender" not in anonymous_line


@pytest.mark.parametrize("anonymous", [False, True])
def test_actor_binding_preserves_anonymity(container: Any, anonymous: bool) -> None:
    engine = DecisionEngine(container.llm_router)
    action = ActorAction(ActionType.SEND_MESSAGE, "我传信", person_ids=(HERE,), message_content="请留步", anonymous=anonymous)
    bound = engine.bind_actor_action(action, personality=_make_personality(), packet=_packet())
    assert bound.anonymous is anonymous


@pytest.mark.asyncio
@pytest.mark.parametrize("anonymous", [False, True])
async def test_explicit_actor_signature_cannot_be_overridden(container: Any, anonymous: bool) -> None:
    engine = DecisionEngine(container.llm_router)
    packet = _packet()
    intent = ActorIntent(
        "turn", operation=ActorOperation.BROADCAST, target=Ref.place(packet.spatial.location_id),
        parameters=ActorOperationParameters(message="请留步", anonymous=anonymous),
    )
    base = build_actor_operation(intent, packet)
    provider = container.llm_router.get(LLMScene.ACTOR_INTENT)
    provider.fixed_response = json.dumps({"decision": "act", "anonymous": not anonymous})
    with pytest.raises(ActorAnswerRejected, match="operation_constraint_conflict"):
        await engine.interpret_actor_decision("发出通知", personality=_make_personality(), packet=packet, intent=intent, base=base)
    provider.fixed_response = json.dumps({"decision": "act"})
    decision, _ = await engine.interpret_actor_decision("发出通知", personality=_make_personality(), packet=packet, intent=intent, base=base)
    assert decision.action.anonymous is anonymous


@pytest.mark.asyncio
@pytest.mark.parametrize("anonymous", [False, True])
async def test_text_intent_interprets_signature_before_binding(container: Any, anonymous: bool) -> None:
    engine = DecisionEngine(container.llm_router)
    provider = container.llm_router.get(LLMScene.ACTOR_INTENT)
    provider.fixed_response = json.dumps({
        "decision": "act", "selected_index": _index_of(ActionType.SEND_MESSAGE), "person_indices": [1],
        "anonymous": anonymous, "message_content": "请留步", "action_description": "我发出传信",
    })
    decision, _ = await engine.interpret_actor_decision(
        "匿名传信" if anonymous else "传信", personality=_make_personality(), packet=_packet(),
    )
    assert decision.action.anonymous is anonymous


@pytest.mark.asyncio
@pytest.mark.parametrize("anonymous", [False, True])
async def test_autonomous_decision_preserves_signature_choice(container: Any, anonymous: bool) -> None:
    engine = DecisionEngine(container.llm_router)
    provider = container.llm_router.get(LLMScene.AGENT_DECISION_MAIN)
    provider.fixed_response = json.dumps({
        "selected_index": _index_of(ActionType.SEND_MESSAGE), "inner_monologue": "我需要把话送到",
        "person_indices": [1], "anonymous": anonymous, "message_content": "请留步", "action_description": "我传信",
    })
    result = await engine.decide(personality=_make_personality(), packet=_make_packet(visible_agent_ids=["reader"]))
    assert result.action is not None and result.action.anonymous is anonymous


def test_actor_api_accepts_only_boolean_anonymity_on_applicable_operation() -> None:
    from interaction.api.actor import _decode_intent

    data = {"turn_id": "turn", "operation": "broadcast", "target": {"kind": "location", "id": "hall"},
            "parameters": {"message": "请留步", "anonymous": True}}
    assert _decode_intent(data).parameters.anonymous is True
    with pytest.raises(TypeError, match="expected true or false"):
        _decode_intent({**data, "parameters": {"message": "请留步", "anonymous": "true"}})
    with pytest.raises(ValueError, match="does not apply"):
        _decode_intent({**data, "operation": "talk", "target": {"kind": "agent", "id": "reader"},
                        "parameters": {"anonymous": True}})


@pytest.mark.parametrize("action_type", [ActionType.REST, ActionType.TALK])
def test_irrelevant_model_anonymity_is_ignored(container: Any, action_type: ActionType) -> None:
    from agent.decision import _actor_action_from_payload

    action = _actor_action_from_payload({
        "selected_index": _index_of(action_type), "action_description": "我行动", "anonymous": True,
    }, candidate_lists(_packet()))
    assert action is not None and not action.anonymous


def test_anonymous_message_is_absent_from_recipient_send_action_history() -> None:
    from interaction.models import ActionPrivateView, ActionSharedView, ActionSummary

    action = ActionSummary(
        seq=1, initiator_id="sender", participant_ids=["sender"], action_type="send_message",
        anonymous=True, public=None, shared=ActionSharedView(audience=["sender"], outcome="甲匿名传信给乙", gist="匿名传信"),
        private=[ActionPrivateView(agent_id="sender", action_description="我传信")],
    )
    view = _step(locations={"hall": LocationScope(location_id="hall", agents=[_body(ME)], events=[action])})
    assert step_view_for(view, ME).locations["hall"].events == []


def test_anonymous_prompt_and_actor_entry_share_the_public_signature() -> None:
    from agent.actor_perception import build_perceived_entry
    from core.prompts import render_perceived_signals

    msg = _message()
    lines = render_perceived_signals(spatial=_make_spatial(), inbox=[msg], broadcasts=[])
    assert any("匿名" in line and msg.content in line for line in lines)
    assert "sender" not in "\n".join(lines)
    perceived = build_perceived_entry(1, [msg], [])
    assert perceived is not None and perceived.messages[0].sender_name == "匿名"


def test_anonymous_status_is_an_action_property_in_step_view() -> None:
    from interaction.models import StepView
    from tests.unit.step_view import events, snapshot_of

    snapshot = snapshot_of(records=[{
        "agent_id": "sender", "initiator_id": "sender", "participant_ids": ["sender"],
        "execution_id": "send", "action_type": "send_message", "phase": "settled", "location_id": "hall",
        "anonymous": True, "action_description": "我传信", "observations": [],
    }])
    assert snapshot.events[0]["anonymous"] is True
    assert "anonymous" not in snapshot.events[0]["parts"][0]
    view = StepView.from_snapshot(snapshot)
    [(_, action)] = events(view, "action")
    assert action.anonymous is True
    assert "anonymous" not in asdict(action.private[0])
    [(_, own)] = events(step_view_for(view, "sender"), "action")
    assert own.anonymous is True
    assert events(step_view_for(view, "reader"), "action") == []


def test_visible_action_keeps_anonymous_status_when_private_parts_are_filtered() -> None:
    from interaction.models import ActionPrivateView, ActionSummary, ObservedView

    action = ActionSummary(
        seq=1, initiator_id="sender", participant_ids=["sender"], action_type="send_message",
        anonymous=True, public=ObservedView("有人发出了一条消息"),
        private=[ActionPrivateView(agent_id="sender", inner_monologue="我的打算")],
    )
    view = _step(locations={"hall": LocationScope(location_id="hall", agents=[_body(ME)], events=[action])})
    [observed] = step_view_for(view, ME).locations["hall"].events
    assert observed.anonymous is True
    assert observed.private == [] and observed.shared is None


def test_prompt_signature_immediately_follows_message_content(container: Any) -> None:
    engine = DecisionEngine(container.llm_router)
    intent_keys = list(engine._intent_output_fields())
    assert intent_keys.index("anonymous") == intent_keys.index("message_content") + 1
    assert intent_keys.index("person_indices") < intent_keys.index("action_description") < intent_keys.index("message_content")
    assert intent_keys.index("anonymous") < intent_keys.index("urgency")
    system, _, _ = engine._build_decision_prompt(_make_personality(), _make_packet(), [])
    lines = system.splitlines()
    content_index = next(i for i, line in enumerate(lines) if line.startswith('  "message_content":'))
    assert lines[content_index + 1].startswith('  "anonymous":')
    assert system.index('  "person_indices":') < system.index('  "action_description":') < system.index('  "message_content":')
    assert system.index('  "anonymous":') < system.index('  "urgency":')


def test_prompt_ties_anonymity_to_the_recipient_not_to_secrecy(container: Any) -> None:
    """A directed message reaches no onlooker (the executor emits no observation), so secrecy is
    never a reason to sign anonymously; only hiding the sender from whoever receives it is. Without
    saying so, the model signs secret letters to its closest allies anonymously."""
    from agent.decision import _ACTION_SPACE

    engine = DecisionEngine(container.llm_router)
    system, user, _ = engine._build_decision_prompt(_make_personality(), _make_packet(), _ACTION_SPACE)
    assert "只有收件人看得到，旁人既看不到也听不到" in user
    anonymous = next(line for line in system.splitlines() if line.startswith('  "anonymous":'))
    assert "跟这话有多机密、多要紧无关" in anonymous
    assert "不想让收到的人知道是我" in anonymous


def test_actor_intent_reads_secrecy_words_as_secrecy_not_anonymity(container: Any) -> None:
    """An actor who says "quietly tell him" wants a private letter, which every directed message
    already is; only asking to hide who sent it makes it anonymous."""
    engine = DecisionEngine(container.llm_router)
    field = engine._intent_output_fields()["anonymous"]
    assert "不让收件人知道是谁发的" in field
    assert "「悄悄」" in field and "说的是保密" in field
    assert "只有收件人看得到，旁人既看不到也听不到" in engine._intent_system_prompt()


@pytest.mark.parametrize("action_type", [ActionType.REST, ActionType.TALK])
def test_direct_actor_binding_ignores_anonymity_on_other_actions(container: Any, action_type: ActionType) -> None:
    action = ActorAction(action_type, "我行动", anonymous=True, person_ids=(HERE,) if action_type is ActionType.TALK else ())
    bound = DecisionEngine(container.llm_router).bind_actor_action(action, personality=_make_personality(), packet=_packet())
    assert not bound.anonymous
