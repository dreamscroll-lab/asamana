"""The actor's action form: bound by the same rules as a model's decision, drafted from a
sentence onto the character's own lists, and briefed from the same perception."""

from __future__ import annotations

import json

import pytest

from agent.actor_briefing import build_actor_briefing
from agent.actor_perception import build_perceived_entry
from agent.decision import _ACTION_SPACE, _SEND_MESSAGE_EXPECTED_OUTCOME, DecisionEngine, candidate_lists
from agent.memory_types import MemoryStream
from agent.relation import PerceivedRelation
from core.interfaces.action import ActionType, Ref
from core.interfaces.actor import (
    ActorAction,
    ActorAnswerRejected,
    ActorDecision,
    BriefMessage,
)
from core.interfaces.llm import LLMScene
from core.interfaces.message import Message
from core.interfaces.perception import (
    Broadcast, BroadcastType, LocationView, NpcIdentity, PerceivedIdentity, PerceivedNpc, PerceivedPresence, VisibleEntity,
)
from core.interfaces.urgency import Urgency
from core.prompts import relation_legend
from tests.unit.test_decision_m4_gaps import _make_packet, _make_personality, _mem

ME = "agent-shimin"
HERE = "agent-jiancheng"
AWAY = "agent-yuanji"


def _packet():
    packet = _make_packet(
        visible_agent_ids=[HERE],
        visible_agents={HERE: PerceivedIdentity(name="李建成", gender="男")},
        reachable_location_ids=["xuanwu_gate"],
        location_views={"xuanwu_gate": LocationView(name="玄武门")},
        relevant_relations=[PerceivedRelation(
            trust=0.2, affection=-0.3, target_agent_id=AWAY, target_agent_name="李元吉",
            target_agent_gender="男", labels=["兄弟:弟弟"], history_summary="素来不睦",
        )],
        factual_memories=[_mem("昨日在殿上争执", MemoryStream.FACTUAL, mid="f1")],
        experiential_memories=[_mem("我心里憋着一口气", MemoryStream.EXPERIENTIAL, mid="e1")],
    )
    packet.spatial.visible_agents[HERE] = PerceivedPresence(
        PerceivedIdentity(name="李建成", gender="男"), condition="面带倦色",
    )
    packet.spatial.visible_entities = [
        VisibleEntity(entity_id="sword", name="佩剑", entity_type="item", holder_id=ME),
        VisibleEntity(entity_id="memorial", name="奏章", entity_type="item", content="请削秦王兵权"),
    ]
    return packet


@pytest.fixture
def engine(container) -> DecisionEngine:
    return DecisionEngine(container.llm_router)


def _bind(engine: DecisionEngine, action: ActorAction):
    return engine.bind_actor_action(action, personality=_make_personality(), packet=_packet())


def test_a_form_binds_like_the_models_answer(engine) -> None:
    packet = _packet()
    by_model = engine._parse_llm_selection(  # noqa: SLF001
        json.dumps({
            "selected_index": next(i for i, c in enumerate(_ACTION_SPACE) if c.action_type is ActionType.TALK),
            "person_indices": [1],
            "action_description": "与大哥当面谈谈",
            "message_content": "大哥，我们好好谈谈",
        }),
        _ACTION_SPACE,
        packet,
    )

    by_actor = _bind(engine, ActorAction(
        action_type=ActionType.TALK, action_description="与大哥当面谈谈",
        person_ids=(HERE,), message_content="大哥，我们好好谈谈",
    ))

    assert by_actor.target == by_model.target
    assert by_actor.action_description == by_model.action_description
    assert by_actor.agent_id == ME


def test_a_target_the_character_does_not_perceive_is_refused(engine) -> None:
    with pytest.raises(ActorAnswerRejected) as refused:
        _bind(engine, ActorAction(
            action_type=ActionType.SEND_MESSAGE, action_description="给陌生人捎话",
            person_ids=("agent-stranger",), message_content="速来",
        ))
    assert refused.value.reason == "person_ids_not_perceived"


@pytest.mark.parametrize("action, reason", [
    (ActorAction(action_type=ActionType.TALK, action_description="找四弟说话", person_ids=(AWAY,)),
     "talk_target_not_present"),
    (ActorAction(action_type=ActionType.MOVE, action_description="出门走走"), "move_without_destination"),
    (ActorAction(action_type=ActionType.SEND_MESSAGE, action_description="说点什么"), "send_without_channel"),
    # No words to deliver: the description is never sent in their place.
    (ActorAction(action_type=ActionType.SEND_MESSAGE, action_description="给四弟捎话", person_ids=(AWAY,)),
     "message_required"),
    # Only what I hold can be handed over: the memorial lies here.
    (ActorAction(action_type=ActionType.PHYSICAL, action_description="把奏章递给大哥",
                 physical_entity_id="memorial", physical_recipient_id=HERE),
     "physical_recipient_not_held"),
])
def test_the_types_own_rules_apply(engine, action, reason) -> None:
    with pytest.raises(ActorAnswerRejected) as refused:
        _bind(engine, action)
    assert refused.value.reason == reason


def test_a_message_reaches_someone_absent(engine) -> None:
    bound = _bind(engine, ActorAction(
        action_type=ActionType.SEND_MESSAGE, action_description="给四弟捎话",
        person_ids=(AWAY,), message_content="今夜别出门", urgency=Urgency.HIGH,
    ))
    assert bound.target.acts_on == [Ref.agent(AWAY)]
    assert bound.content == "今夜别出门"
    assert bound.urgency is Urgency.HIGH


def test_a_hand_over_names_the_thing_and_the_person(engine) -> None:
    bound = _bind(engine, ActorAction(
        action_type=ActionType.PHYSICAL, action_description="把佩剑递给大哥",
        physical_entity_id="sword", physical_recipient_id=HERE,
    ))
    assert bound.target.acted_on_ids == ["sword"]
    assert bound.target.reaches == [Ref.agent(HERE)]


def _reply(container, **payload) -> None:
    container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = json.dumps(payload)


def _index_of(action_type: ActionType) -> int:
    return next(i for i, c in enumerate(_ACTION_SPACE) if c.action_type is action_type)


async def _read(engine, text: str, *, short_term_goals=None, long_term_goals=None):
    return await engine.interpret_actor_decision(
        text,
        personality=_make_personality(short_term_goals=short_term_goals, long_term_goals=long_term_goals),
        packet=_packet(),
    )


@pytest.mark.asyncio
async def test_words_read_as_an_action_name_targets_by_id(container, engine) -> None:
    _reply(
        container, reason="想和大哥当面说清楚", decision="act", selected_index=_index_of(ActionType.TALK),
        person_indices=[1, 9], message_content="大哥，借一步说话",
        action_description="请大哥借一步说话", estimated_steps=2,
    )

    reading = await _read(engine, "跟大哥说借一步说话", short_term_goals=["探明大哥的打算"])

    assert reading == (ActorDecision(action=ActorAction(
        action_type=ActionType.TALK, action_description="请大哥借一步说话",
        person_ids=(HERE,), message_content="大哥，借一步说话", estimated_steps=2,
    )), "想和大哥当面说清楚")
    assert _bind(engine, reading[0].action).target.acts_on == [Ref.agent(HERE)]


@pytest.mark.asyncio
@pytest.mark.parametrize("action_type,related,unrelated", [
    (ActionType.PHYSICAL, {"physical_person_index": 1}, {"person_indices": [2], "destination_index": 1}),
    (ActionType.TALK, {"person_indices": [1]}, {"physical_entity_index": 1}),
    (ActionType.SEND_MESSAGE, {"person_indices": [2], "message_content": "请留步"}, {"physical_person_index": 1}),
    (ActionType.MOVE, {"destination_index": 1}, {"person_indices": [1]}),
    (ActionType.WORK, {}, {"physical_person_index": 1, "message_content": "不应传出的话"}),
    (ActionType.REST, {}, {"destination_index": 1}),
    (ActionType.COVERT, {}, {"person_indices": [1], "physical_entity_index": 1}),
    (ActionType.ERRAND, {"errand_npc_index": 1, "errand_destination_index": 1}, {"person_indices": [1]}),
])
async def test_free_text_and_autonomous_decisions_bind_only_relevant_fields(container, engine, action_type: ActionType, related: dict, unrelated: dict) -> None:
    packet = _packet()
    packet.spatial.visible_npcs["runner"] = PerceivedNpc(NpcIdentity("信使", "男"))
    payload = {
        "selected_index": _index_of(action_type), "action_description": "我按计划行动", "estimated_steps": 3,
        "inner_monologue": "我想清楚了", "expected_outcome": "完成眼前的事", "urgency": "high",
        "unknown_field": {"noise": True}, **related, **unrelated,
    }
    selection = engine._parse_llm_selection(payload, _ACTION_SPACE, packet)
    assert selection is not None
    assert selection.inner_monologue == "我想清楚了" and selection.expected_outcome == "完成眼前的事"
    assert selection.estimated_steps == 3 and selection.urgency is Urgency.HIGH
    _reply(container, decision="act", **payload)
    reading = await engine.interpret_actor_decision("具体行动", personality=_make_personality(), packet=packet)
    assert reading is not None
    decision = reading[0]
    bound = engine.bind_actor_action(decision.action, personality=_make_personality(), packet=packet)
    assert bound.action_type is action_type and bound.target == selection.target
    assert bound.intent.errand == selection.errand
    # A reading is a whole decision: what the step is for, and how pressing words are, reach
    # the action as they do from the model's own decision (a message's expectation is pinned).
    assert decision.action.expected_outcome == "完成眼前的事"
    assert bound.expected_outcome == (
        _SEND_MESSAGE_EXPECTED_OUTCOME if action_type is ActionType.SEND_MESSAGE else "完成眼前的事"
    )
    assert decision.action.estimated_steps == 3
    expected_urgency = Urgency.HIGH if action_type is ActionType.SEND_MESSAGE else Urgency.NORMAL
    assert decision.action.urgency is expected_urgency and bound.urgency is expected_urgency
    if action_type not in (ActionType.TALK, ActionType.SEND_MESSAGE):
        assert selection.message_content == ""
        assert decision.action.person_ids == () and decision.action.message_content == ""


@pytest.mark.asyncio
async def test_an_intent_to_do_nothing_this_step_reads_as_idling(container, engine) -> None:
    _reply(container, reason="说的是先等等", decision="idle")
    assert await _read(engine, "先等等看") == (ActorDecision(idle=True), "说的是先等等")


@pytest.mark.asyncio
async def test_words_that_name_nothing_doable_read_as_nothing(container, engine) -> None:
    _reply(container, reason="只是感慨", decision="unclear")
    assert await _read(engine, "唉") == (None, "只是感慨")
    _reply(container, reason="?", decision="wait")
    assert await _read(engine, "随便") is None

    async def _down(*_args, **_kwargs):
        raise RuntimeError("provider down")

    container.llm_router.get(LLMScene.ACTOR_INTENT).complete = _down  # type: ignore[method-assign]
    assert await _read(engine, "去玄武门") is None


def _intent_prompt(engine, text: str, **kwargs) -> tuple[str, str]:
    personality = _make_personality(short_term_goals=["探明大哥的打算"])
    return engine._intent_system_prompt(), engine._intent_user_prompt(  # noqa: SLF001
        text, personality, _packet(), recent=kwargs.get("recent", ()), recalled=kwargs.get("recalled", ()),
        fixed=kwargs.get("fixed"),
    )


def test_the_reading_prompt_shows_the_lists_and_what_the_character_knows(engine) -> None:
    system, user = _intent_prompt(
        engine, "把奏章拿给大哥看", recent=["昨日：我去了东宫"],
        recalled=[_mem("大哥曾说奏章之事要当面谈", MemoryStream.FACTUAL, mid="r1")],
    )

    assert "#1 李建成（男，在场，面带倦色）" in user
    assert "#2 李元吉（男，不在场）" in user
    assert "#1 玄武门（脚程" in user and "xuanwu_gate" not in user
    assert "昨日：我去了东宫" in user and "大哥曾说奏章之事要当面谈" in user
    assert "李元吉" in user.split("【名单】")[0]   # relations come before the lists
    assert user.rstrip().endswith("只输出 JSON。")
    assert "【意图】\n把奏章拿给大哥看" in user
    # The parser reads a character's intent: who drives the character is no part of it.
    assert "演员" not in system + user and "指令" not in system + user and "用户" not in system + user
    # The prefix stays the same for every character and sentence.
    assert "李" not in system and "奏章" not in system
    assert "【已定下的】" not in user
    assert system.index("【角色】") < system.index("【任务】") < system.index("【输出】")
    # The relation legend is a constant: in the cached prefix, not beside the relations.
    assert relation_legend() in system and relation_legend() not in user
    assert system.index("【说明】") < system.index(relation_legend()) < system.index("【输出】")


def test_the_reading_prompt_carries_what_just_reached_the_character(engine) -> None:
    from core.interfaces.perception import AmbientEvent

    packet = _packet()
    packet.inbox = [Message(
        id="m1", world_id="w", sender_id=HERE, sender_name="李建成", content="今夜子时，玄武门见",
        recipients=[ME], location_scope=None, created_step=1, deliver_step=2, urgency=Urgency.HIGH,
    )]
    packet.broadcasts = [Broadcast(
        content="东宫走水", source="system", broadcast_type=BroadcastType.WORLD_EVENT, deliver_step=2,
    )]
    packet.spatial.ambient_events = [AmbientEvent(content="门外有人在争吵")]

    user = engine._intent_user_prompt(  # noqa: SLF001
        "回他的信", _make_personality(), packet, recent=(), recalled=(), fixed=None,
    )

    known = user[user.index("【我知道的】"):user.index("【名单】")]
    assert "李建成：今夜子时，玄武门见" in known
    assert "东宫走水" in known and "门外有人在争吵" in known


def test_the_reading_prompt_leaves_out_what_only_weighs_what_to_do(engine) -> None:
    """The ends come only from the intent: what the character would weigh to choose them
    (persona, mood, needs, goals) stays out, so the reading has nothing to add an end from."""
    personality = _make_personality(short_term_goals=["探明大哥的打算"], long_term_goals=["登基为帝"])
    user = engine._intent_user_prompt(  # noqa: SLF001
        "去玄武门", personality, _packet(), recent=(), recalled=(), fixed=None,
    )

    assert personality.soul.identity_text() in user
    assert "探明大哥的打算" not in user and "登基为帝" not in user
    assert "心境" not in user and "性格" not in user and "背景" not in user


def test_the_reading_fields_come_in_dependency_order(engine) -> None:
    system, _user = _intent_prompt(engine, "去玄武门")
    schema = system[system.index("【输出】"):]
    order = ["reason", "decision", "selected_index", "person_indices", "errand_recipient_index",
             "action_description", "talk_topic", "message_content", "errand_message",
             "expected_outcome", "estimated_steps", "urgency"]
    positions = [schema.index(f'"{key}"') for key in order]
    assert positions == sorted(positions)
    assert '"operation"' not in schema and '"operation_consistent"' not in schema
    assert '"item_name"' in schema and '"item_id"' not in schema


def test_one_system_prompt_serves_free_words_and_a_fixed_operation(engine) -> None:
    base = ActorAction(action_type=ActionType.MOVE, action_description="我前往玄武门", destination_id="xuanwu_gate")
    free_system, _ = _intent_prompt(engine, "去玄武门")
    fixed_system, fixed_user = _intent_prompt(engine, "悄悄地去", fixed=(base, {"destination_index": 1, "selected_index": 0}))

    assert free_system == fixed_system
    fixed = fixed_user[fixed_user.index("【已定下的】"):fixed_user.index("【意图】")]
    assert "我前往玄武门" in fixed and '"destination_index": 1' in fixed


def test_the_briefing_lists_are_the_binding_lists() -> None:
    packet = _packet()
    lists = candidate_lists(packet)

    briefing = build_actor_briefing(packet)

    assert "character" not in vars(briefing)
    assert ME not in lists.roster_ids
    assert [p.agent_id for p in briefing.people] == lists.roster_ids == [HERE, AWAY]
    assert [p.present for p in briefing.people] == [True, False]
    assert [p.location_id for p in briefing.places] == lists.reachable_ids
    assert [(t.entity_id, t.name) for t in briefing.entities] == [("sword", "佩剑"), ("memorial", "奏章")]


def test_the_perceived_entry_is_the_letters_and_broadcasts_that_reached_the_character() -> None:
    letter = Message(
        id="m1", world_id="w", sender_id=HERE, sender_name="李建成", content="今夜子时，玄武门见",
        recipients=[ME], location_scope=None, created_step=1, deliver_step=2, urgency=Urgency.HIGH,
    )
    fire = Broadcast(
        content="东宫走水", source="system", broadcast_type=BroadcastType.WORLD_EVENT, deliver_step=2,
    )

    perception = build_perceived_entry(2, [letter], [fire])

    assert build_perceived_entry(2, [], []) is None   # nothing arrived, nothing to record
    assert perception is not None and perception.acted_upon == ()
    # What landed on him is told with what arrived, in the same step's entry.
    together = build_perceived_entry(2, [letter], [], ["李建成一把将我推开"])
    assert together is not None and together.acted_upon == ("李建成一把将我推开",)
    assert len(together.messages) == 1
    assert perception.messages == (BriefMessage(sender_name="李建成", content="今夜子时，玄武门见", urgency="high"),)
    assert perception.broadcasts == ("东宫走水",)
