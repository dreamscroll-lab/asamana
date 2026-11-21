"""What a conversation did to the things in its speakers' hands.

A dialogue may write a thing changing hands; only a ruling read off the transcript and checked
against the world lands it, and every record of the conversation is written after that ruling and
told what really changed. Otherwise a "给您" that changed nothing is remembered as done, and the
next step finds the thing still in the giver's pocket.
"""

from __future__ import annotations

import json

import pytest

from agent.decision import ActionType, AgentAction
from core.interfaces.action import ActionTarget, Ref
from core.interfaces.entity import EntityPresence, WorldEntity, WorldEntityType
from engine.directory import LiveWorldDirectory
from engine.environment import EnvironmentSystem
from engine.execution_processor import ExecutionProcessor
from engine.executors.base import ActionExecutionState
from engine.executors.registry import ActionExecutorRegistry
from engine.executors.social import _DIALOGUE_SYSTEM, SocialExecutor
from engine.executors.talk_entities import (
    _SYSTEM as _ENTITY_SYSTEM,
    NO_ENTITY_CHANGE,
    judge_talk_entity_changes,
)
from core.interfaces.place import Place

from tests.unit.test_action_executors import _make_agent

A, B, C = "agent-a", "agent-b", "agent-c"


def _env(*held: WorldEntity) -> EnvironmentSystem:
    env = EnvironmentSystem()
    env.space.register_place(Place(place_id="hall", name="大堂", description="", is_public=True, capacity=50))
    for aid in (A, B, C):
        env.place_agent(agent_id=aid, location_id="hall")
    for entity in held:
        env.register_entity(entity)
    return env


def _thing(
    entity_id: str, name: str, holder: str, kind: WorldEntityType = WorldEntityType.ITEM, *, content: str = "",
) -> WorldEntity:
    return WorldEntity(
        entity_id=entity_id, name=name, entity_type=kind,
        presence=EntityPresence.HELD, presence_ref=holder, state="sealed", content=content,
    )


def _lying(entity_id: str, name: str, kind: WorldEntityType = WorldEntityType.ITEM) -> WorldEntity:
    return WorldEntity(
        entity_id=entity_id, name=name, entity_type=kind,
        presence=EntityPresence.AT_LOCATION, presence_ref="hall",
    )


class _Router:
    """Answers each kind of call by its system prompt and records the order they came in."""

    def __init__(self, entities: str, *, other: str | None = None) -> None:
        self.entities = entities
        self.other = other
        self.calls: list[tuple[str, str]] = []

    async def complete(self, scene, messages, **kwargs):  # noqa: ANN001, ANN003
        system, user = messages[0].content, messages[-1].content
        if system == _ENTITY_SYSTEM:
            kind, content = "entities", self.entities
        elif system == _DIALOGUE_SYSTEM:
            kind, content = "dialogue", json.dumps({"reason": "r", "dialogue": [
                {"speaker": 1, "line": "东西拿来。"}, {"speaker": 2, "line": "给您。"},
            ], "observation": ""}, ensure_ascii=False)
        elif self.other is not None:
            kind, content = "other", self.other
        else:
            kind, content = "record", '{"fact": "谈了一场。", "success": true, "relation": "neutral", "why": ""}'
        self.calls.append((kind, user))

        class _R:
            pass

        response = _R()
        response.content = content
        return response


def _agents(container) -> dict:
    return {
        A: _make_agent(container, world_id="w", agent_id=A, name="林峰"),
        B: _make_agent(container, world_id="w", agent_id=B, name="吴明"),
        C: _make_agent(container, world_id="w", agent_id=C, name="郑洁"),
    }


async def _talk(container, router: _Router, env: EnvironmentSystem):
    agents = _agents(container)
    executor = SocialExecutor(router, LiveWorldDirectory.from_agents(agents, env))
    action = AgentAction(
        agent_id=A, step=1, action_type=ActionType.TALK, action_description="要他交出东西",
        target=ActionTarget(acts_on=[Ref.agent(B)], claims=[Ref.agent(B)], reaches=[Ref.agent(C)]),
        estimated_steps=1,
    )
    state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
    return await executor.complete(state, 1, agents=agents, environment=env, message_system=None)


def _ruling(*changes: dict) -> str:
    return json.dumps({"changes": list(changes)}, ensure_ascii=False)


def _entry(item: int, holder: int | None = None, *, state: str = "", destroyed: bool = False, **made: str) -> dict:
    entry = {"reason": "r", "entity": item, "state": state, "destroyed": destroyed,
             "name": "", "description": "", "content": "", **made}
    if holder is not None:
        entry["holder"] = holder
    return entry


@pytest.mark.asyncio
async def test_a_handover_the_dialogue_shows_lands_once_and_every_record_is_told(container) -> None:
    router = _Router(_ruling(_entry(3, 1)))
    results = await _talk(container, router, _env(_thing("u", "加密U盘", B)))

    initiator, addressee = results
    [change] = initiator.entity_state_changes
    assert change.entity_id == "u" and change.owner_id == A
    assert "吴明把加密U盘交到了林峰手上" in change.perception
    assert addressee.entity_state_changes == []   # landed once, not per side

    kinds = [kind for kind, _ in router.calls]
    # The ruling comes before every record, and each record (both sides, the listener) is told.
    assert kinds.index("entities") < min(i for i, k in enumerate(kinds) if k == "record")
    records = [user for kind, user in router.calls if kind == "record"]
    assert len(records) == 3
    assert all("吴明把加密U盘交到了林峰手上" in user for user in records)


@pytest.mark.asyncio
async def test_the_ruling_sees_what_the_dialogue_s_scene_puts_within_reach(container) -> None:
    router = _Router(_ruling())
    await _talk(container, router, _env(_thing("u", "加密U盘", B), _thing("n", "纸条", C), _lying("k", "钥匙")))

    [user] = [user for kind, user in router.calls if kind == "entities"]
    assert "#1 林峰" in user and "#2 吴明" in user
    assert "#3 加密U盘" in user and "由吴明持有" in user
    assert "#4 钥匙" in user                    # lying here
    # A bystander's thing isn't the conversation's to hand over.
    assert "纸条" not in user
    # Ids never cross into the prompt.
    assert "agent-" not in user and "#3 u" not in user


@pytest.mark.asyncio
async def test_words_alone_change_nothing_and_the_records_hear_so(container) -> None:
    router = _Router(_ruling())
    results = await _talk(container, router, _env(_thing("u", "加密U盘", B)))

    assert all(r.entity_state_changes == [] and r.entity_spawns == [] for r in results)
    records = [user for kind, user in router.calls if kind == "record"]
    assert records and all(NO_ENTITY_CHANGE in user for user in records)


async def _judge(env: EnvironmentSystem, answer: str, container):  # noqa: ANN001
    agents = _agents(container)
    return await judge_talk_entity_changes(
        _Router(answer), LiveWorldDirectory.from_agents(agents, env), env,
        speaker_ids=(A, B), transcript="林峰：东西拿来。\n吴明：给您。", location="大堂", step=1,
    )


@pytest.mark.asyncio
async def test_a_thing_not_in_either_speaker_s_hands_is_not_handed_over(container) -> None:
    """A person's number in the item slot, or a number past the list, binds nothing."""
    env = _env(_thing("u", "加密U盘", B))
    changes = await _judge(env, _ruling(_entry(1, 1), _entry(4, 1)), container)
    assert changes.state_changes == () and changes.record_text() == NO_ENTITY_CHANGE


@pytest.mark.asyncio
async def test_a_holder_naming_nobody_drops_the_entry(container) -> None:
    """An item's number in the holder slot isn't repaired into some other holder."""
    env = _env(_thing("u", "加密U盘", B))
    changes = await _judge(env, _ruling(_entry(3, 3, state="已打开")), container)
    assert changes.state_changes == ()


@pytest.mark.asyncio
async def test_an_untakeable_thing_is_not_handed_over(container) -> None:
    env = _env(_thing("m", "石碑", B, kind=WorldEntityType.LANDMARK))
    changes = await _judge(env, _ruling(_entry(3, 1)), container)
    assert changes.state_changes == ()


@pytest.mark.asyncio
async def test_an_entry_that_changes_nothing_lands_nothing(container) -> None:
    env = _env(_thing("u", "加密U盘", B))
    changes = await _judge(env, _ruling(_entry(3, 2)), container)
    assert changes.state_changes == () and changes.lines == ()


@pytest.mark.asyncio
async def test_a_thing_destroyed_is_not_also_handed_over(container) -> None:
    env = _env(_thing("u", "加密U盘", B))
    changes = await _judge(env, _ruling(_entry(3, 1, state="碎了", destroyed=True)), container)
    [change] = changes.state_changes
    assert change.destroyed and change.owner_id is None and change.location_id is None
    assert changes.lines == ("吴明手上的加密U盘被毁掉了。",)


@pytest.mark.asyncio
async def test_a_thing_opened_and_handed_over_is_one_change_read_in_order(container) -> None:
    env = _env(_thing("u", "加密U盘", B))
    changes = await _judge(env, _ruling(_entry(3, 1, state="已打开")), container)
    [change] = changes.state_changes
    assert (change.new_state, change.owner_id) == ("已打开", A)
    assert changes.lines == ("吴明手上的加密U盘，状态变为「已打开」。", "吴明把加密U盘交到了林峰手上。")


@pytest.mark.asyncio
async def test_a_thing_set_down_lies_here(container) -> None:
    env = _env(_thing("u", "加密U盘", B))
    changes = await _judge(env, _ruling(_entry(3, 0)), container)
    [change] = changes.state_changes
    assert change.location_id == "hall" and change.owner_id is None
    assert changes.lines == ("吴明把加密U盘放在了此处。",)


@pytest.mark.asyncio
async def test_a_missing_holder_keeps_the_thing_where_it_is(container) -> None:
    env = _env(_thing("u", "加密U盘", B))
    changes = await _judge(env, _ruling(_entry(3, state="已打开")), container)
    [change] = changes.state_changes
    assert change.new_state == "已打开" and change.owner_id is None and change.location_id is None
    assert change.perception == ""


@pytest.mark.asyncio
async def test_one_thing_settles_once(container) -> None:
    env = _env(_thing("u", "加密U盘", B))
    changes = await _judge(env, _ruling(_entry(3, 1), _entry(3, 0)), container)
    [change] = changes.state_changes
    assert change.owner_id == A


@pytest.mark.asyncio
async def test_the_ruling_sees_what_a_thing_says(container) -> None:
    """A change to the words replaces them whole, so the ruling has to see what is there."""
    router = _Router(_ruling())
    await _talk(container, router, _env(_thing("n", "纸条", B, content="三点见")))
    [user] = [user for kind, user in router.calls if kind == "entities"]
    assert "三点见" in user


@pytest.mark.asyncio
async def test_new_words_land_whole_and_the_records_are_not_told_them(container) -> None:
    env = _env(_thing("n", "纸条", B, content="三点见"))
    changes = await _judge(env, _ruling(_entry(3, content="三点见，带上钥匙")), container)
    [change] = changes.state_changes
    assert change.new_content == "三点见，带上钥匙" and change.owner_id is None
    assert changes.lines == ("吴明手上的纸条写上了新的内容。",)


@pytest.mark.asyncio
async def test_a_thing_can_be_renamed_and_change_its_look(container) -> None:
    env = _env(_thing("w", "木块", B))
    changes = await _judge(env, _ruling(_entry(3, name="木雕", description="雕成了一只鸟")), container)
    [change] = changes.state_changes
    assert (change.new_name, change.new_description) == ("木雕", "雕成了一只鸟")
    assert changes.lines == ("吴明手上的木块成了木雕。", "吴明手上的木块，样子变为「雕成了一只鸟」。")


@pytest.mark.asyncio
async def test_values_copied_back_unchanged_are_no_change(container) -> None:
    env = _env(_thing("n", "纸条", B, content="三点见"))
    changes = await _judge(env, _ruling(_entry(3, 2, name="纸条", state="sealed", content="三点见")), container)
    assert changes.state_changes == ()


@pytest.mark.asyncio
async def test_a_ruined_thing_takes_no_new_name_or_words(container) -> None:
    env = _env(_thing("n", "纸条", B, content="三点见"))
    changes = await _judge(env, _ruling(_entry(3, destroyed=True, name="碎纸", content="看不清了")), container)
    [change] = changes.state_changes
    assert change.destroyed and change.new_name == "" and change.new_content == ""
    assert changes.lines == ("吴明手上的纸条被毁掉了。",)


@pytest.mark.asyncio
async def test_a_thing_on_the_ground_can_be_picked_up(container) -> None:
    env = _env(_lying("k", "钥匙"))
    changes = await _judge(env, _ruling(_entry(3, 1)), container)
    [change] = changes.state_changes
    assert change.owner_id == A and change.location_id is None
    assert changes.lines == ("林峰拿起了钥匙。",)
    assert "林峰拿起了钥匙" in change.perception


def test_a_past_deed_claimed_in_the_dialogue_is_not_a_change_now() -> None:
    """"I already took it" says nothing about this talk; the thing stays where it is given."""
    assert "只认这场对话里当场做的" in _ENTITY_SYSTEM


@pytest.mark.asyncio
async def test_a_thing_on_the_ground_can_change_where_it_lies(container) -> None:
    env = _env(_lying("w", "涂鸦墙", WorldEntityType.LANDMARK))
    changes = await _judge(env, _ruling(_entry(3, 0, state="添了新字")), container)
    [change] = changes.state_changes
    assert change.new_state == "添了新字" and change.owner_id is None and change.location_id is None
    assert changes.lines == ("摆在此处的涂鸦墙，状态变为「添了新字」。",)


@pytest.mark.asyncio
async def test_a_thing_on_the_ground_left_as_it_was_is_no_change(container) -> None:
    env = _env(_lying("k", "钥匙"))
    changes = await _judge(env, _ruling(_entry(3, 0)), container)
    assert changes.state_changes == ()


@pytest.mark.asyncio
async def test_nothing_within_reach_rules_nothing(container) -> None:
    """No thing in either speaker's hands or on the ground: no call, and the records are told
    nothing changed."""
    router = _Router(_ruling(_entry(0, 1, name="字条")))
    results = await _talk(container, router, _env(_thing("n", "纸条", C)))   # a bystander's only

    assert "entities" not in [kind for kind, _ in router.calls]
    assert all(r.entity_state_changes == [] and r.entity_spawns == [] for r in results)
    records = [user for kind, user in router.calls if kind == "record"]
    assert records and all(NO_ENTITY_CHANGE in user for user in records)


@pytest.mark.asyncio
async def test_a_thing_made_on_the_spot_is_held_privately_or_left_in_sight(container) -> None:
    env = _env(_lying("k", "钥匙"))   # something within reach, or nothing is ruled
    changes = await _judge(env, _ruling(
        _entry(0, 2, name="字条", description="一张对折的纸", content="三点见"),
        _entry(0, 0, name="草图"),
        _entry(0, 3, name="名单"),   # names no speaker: dropped
        _entry(0, 1),                # no name: dropped
    ), container)
    held, left = changes.spawns
    assert held.holder_id == B and not held.is_public and held.perception == ""
    assert held.content == "三点见"
    assert left.holder_id is None and left.is_public and "草图被留在了此处" in left.perception

@pytest.mark.asyncio
async def test_a_failed_ruling_changes_nothing(container) -> None:
    env = _env(_thing("u", "加密U盘", B))
    changes = await _judge(env, "not json", container)
    assert changes.state_changes == () and changes.spawns == ()
    assert changes.record_text() == NO_ENTITY_CHANGE


@pytest.mark.asyncio
async def test_a_live_talk_is_ruled_like_any_other(container) -> None:
    """What the transcript shows done lands, whoever spoke it."""
    from core.interfaces.actor import ActorDialogueTurn
    from tests.unit.test_talk_live import HERO, OTHER, _converse, _ScriptedActor, _setup, _talk as _live_talk

    _executor, env, agents = _setup(container, _ScriptedActor([ActorDialogueTurn(line="（递过去）给您。", end=True)]))
    env.register_entity(_thing("u", "加密U盘", HERO))
    spoken = json.dumps({
        "thought": "收下。", "line": "（接过）好。", "end": True,
        "fact": "我们谈了一场。", "success": True, "relation": "neutral",
    }, ensure_ascii=False)
    router = _Router(_ruling(_entry(3, 2)), other=spoken)
    executor = SocialExecutor(router, LiveWorldDirectory.from_agents(agents, env))
    _state, results = await _converse(executor, env, agents, _live_talk(HERO, OTHER))

    assert [kind for kind, _ in router.calls].count("entities") == 1
    [change] = [c for r in results for c in r.entity_state_changes]
    assert change.entity_id == "u" and change.owner_id == OTHER


def test_a_work_keys_its_own_things_and_what_lies_where_it_is(container) -> None:
    """Its judge numbers both, so a same-step grab at either waits for it."""
    env = _env(_thing("u", "加密U盘", A), _thing("n", "纸条", B), _lying("k", "钥匙"))
    processor = ExecutionProcessor(
        executor_registry=ActionExecutorRegistry(), environment=env, message_system=None,
        directory=LiveWorldDirectory.from_agents({}, env),
    )
    work = ActionExecutionState.create(
        action_type=ActionType.WORK, initiator_id=A, participant_ids=[A], purpose="写",
        started_step=1, estimated_steps=1, opening_outcome="", target=None,
    )
    assert processor._execution_item_keys(work) == {"u", "k"}  # noqa: SLF001


def test_a_talk_keys_the_things_in_its_speakers_hands(container) -> None:
    """So a talk and a same-step grab at one of those things don't judge the same starting world."""
    env = _env(_thing("u", "加密U盘", B), _thing("n", "纸条", C), _lying("k", "钥匙"))
    processor = ExecutionProcessor(
        executor_registry=ActionExecutorRegistry(), environment=env, message_system=None,
        directory=LiveWorldDirectory.from_agents({}, env),
    )
    talk = ActionExecutionState.create(
        action_type=ActionType.TALK, initiator_id=A, participant_ids=[A, B], purpose="谈",
        started_step=1, estimated_steps=1, opening_outcome="",
        target=ActionTarget(acts_on=[Ref.agent(B)], claims=[Ref.agent(B)]),
    )
    # Neither a bystander's thing nor the ground: keying the ground would queue every
    # conversation at a place behind one another.
    assert processor._execution_item_keys(talk) == {"u"}  # noqa: SLF001
