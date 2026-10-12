"""The actor's action form: bound by the same rules as a model's decision, drafted from a
sentence onto the character's own lists, and briefed from the same perception."""

from __future__ import annotations

import json

import pytest

from agent.actor_briefing import build_actor_briefing
from agent.decision import _ACTION_SPACE, DecisionEngine, candidate_lists
from agent.memory_types import MemoryStream
from agent.relation import PerceivedRelation
from core.interfaces.action import ActionType, Ref
from core.interfaces.actor import (
    ActorAction,
    ActorAnswerRejected,
    ActorDecision,
    ActorInterruptAnswer,
    ActorWait,
    BriefMemoryKind,
)
from core.interfaces.llm import LLMScene
from core.interfaces.perception import LocationView, PerceivedIdentity, PerceivedPresence, VisibleEntity
from core.interfaces.urgency import Urgency
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
    assert by_actor.agent_id == ME and by_actor.llm_model == "actor"


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

    assert reading == ActorDecision(action=ActorAction(
        action_type=ActionType.TALK, action_description="请大哥借一步说话",
        person_ids=(HERE,), message_content="大哥，借一步说话", estimated_steps=2,
    ))
    # No goal was stated, so the goals stay as they are; and the action binds as read.
    assert reading.short_term_goals is None and reading.long_term_goals is None
    assert _bind(engine, reading.action).target.acts_on == [Ref.agent(HERE)]


@pytest.mark.asyncio
async def test_stated_goals_change_only_what_was_stated(container, engine) -> None:
    _reply(
        container, reason="放下探听，改为稳住大哥，先等等", decision="wait", wait_steps=3,
        new_short_term_goals=["稳住大哥"], dropped_short_term_goal_indices=[1, 7],
    )

    reading = await _read(
        engine, "不探了，先稳住大哥，等三个时辰再说",
        short_term_goals=["探明大哥的打算", "备好马匹"], long_term_goals=["登基为帝"],
    )

    assert reading.wait == ActorWait(steps=3)
    assert reading.short_term_goals == ("备好马匹", "稳住大哥")
    assert reading.long_term_goals is None


@pytest.mark.asyncio
async def test_words_that_name_nothing_doable_read_as_nothing(container, engine) -> None:
    _reply(container, reason="只是感慨", decision="unclear")
    assert await _read(engine, "唉") is None

    async def _down(*_args, **_kwargs):
        raise RuntimeError("provider down")

    container.llm_router.get(LLMScene.ACTOR_INTENT).complete = _down  # type: ignore[method-assign]
    assert await _read(engine, "去玄武门") is None


@pytest.mark.asyncio
async def test_break_off_words_keep_what_comes_next_verbatim(container, engine) -> None:
    async def _ask(text: str):
        return await engine.interpret_actor_interrupt(text, personality=_make_personality(), doing="誊抄奏章")

    _reply(container, reason="要停下赶去玄武门", interrupt=True, thought="此事耽搁不得", next="去玄武门")
    assert await _ask("先放下，去玄武门") == ActorInterruptAnswer(
        interrupt=True, thought="此事耽搁不得", then="去玄武门",
    )

    _reply(container, reason="接着抄", interrupt=False, thought="", next="去玄武门")
    assert (await _ask("不理会，接着抄")).then == ""

    # The reading is a plain instruction about the character; who gave it is not the model's concern.
    system, user = (m.content for m in container.llm_router.get(LLMScene.ACTOR_INTENT).call_history[-1])
    assert "演员" not in system + user
    assert user.startswith("角色手头正在做：誊抄奏章") and "【指令】\n不理会，接着抄" in user


def test_the_reading_prompt_shows_the_lists_and_goals_the_words_bind_against(engine) -> None:
    system, user = engine._build_interpret_prompt(  # noqa: SLF001
        "把奏章拿给大哥看", _make_personality(short_term_goals=["探明大哥的打算"]), _packet(),
    )

    assert "#1 李建成（男，在场，面带倦色）" in user
    assert "#2 李元吉（男，不在场）" in user
    assert "#1 玄武门（脚程" in user and "xuanwu_gate" not in user
    assert user.rstrip().endswith("只输出 JSON。")
    assert "【指令】\n把奏章拿给大哥看" in user
    assert "演员" not in system + user
    assert "  #1 探明大哥的打算" in user
    # The prefix stays the same for every character and sentence.
    assert "李" not in system and "奏章" not in system


def test_the_briefing_lists_are_the_binding_lists() -> None:
    packet = _packet()
    lists = candidate_lists(packet)

    briefing = build_actor_briefing(_make_personality(), packet, seconds_per_step=3600)

    assert [p.agent_id for p in briefing.people] == lists.roster_ids == [HERE, AWAY]
    assert [(p.present, p.condition) for p in briefing.people] == [(True, "面带倦色"), (False, "")]
    assert [p.location_id for p in briefing.places] == lists.reachable_ids
    assert [(t.entity_id, t.held_by_me) for t in briefing.entities] == [("sword", True), ("memorial", False)]
    assert briefing.entities[1].content == "请削秦王兵权"
    assert [(r.agent_id, r.labels, r.summary) for r in briefing.relations] == [(AWAY, ("兄弟:弟弟",), "素来不睦")]
    assert [(m.memory_id, m.kind) for m in briefing.memories] == [
        ("f1", BriefMemoryKind.FACTUAL), ("e1", BriefMemoryKind.EXPERIENTIAL),
    ]
    assert briefing.location_name == "秦王府" and briefing.condition == ""
