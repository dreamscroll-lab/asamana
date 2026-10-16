"""Actor API: casting at confirmation, the turn as the actor sees it, the actor's words read at
the turn, answering, the thread, the pace, the director's routes closed while an actor performs, and the exit."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from agent.decision import _ACTION_SPACE
from core.interfaces.action import ActionType
from core.interfaces.actor import ActorTurn, TurnKind
from core.interfaces.llm import LLMScene
from engine.application import NarrativeApplication
from world import WorldCatalog

from interaction.api import create_app
from interaction.api.actor import _decode_amendments


async def _unconfirmed_world(container, test_config, catalog: WorldCatalog) -> tuple[str, str]:
    """A world still under review, and a character to cast."""
    world = await NarrativeApplication(container, test_config, catalog=catalog).build_world(
        "玄武门之变", template="changan_iso",
    )
    return world.world_id, next(iter(world.agents))


def _client(container, test_config, catalog: WorldCatalog) -> TestClient:
    return TestClient(create_app(container, test_config, catalog=catalog))


@pytest.mark.parametrize("field,value", [
    ("status", "completed"), ("origin", "residue"), ("due_step", 10),
    ("created_step", 0), ("progress_summary", "完成"), ("last_evaluated_step", 1),
])
def test_goal_amendments_reject_read_only_fields(field: str, value: Any) -> None:
    with pytest.raises(TypeError):
        _decode_amendments({"short_term_goals": [{"id": "stg-0-1", "text": "目标", field: value}]})


def _until(probe: Callable[[], Any], *, timeout: float = 20.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (value := probe()) is not None:
            return value
        time.sleep(0.02)
    raise AssertionError("timed out")


def _open_turn(client: TestClient, world_id: str) -> dict[str, Any]:
    return _until(lambda: client.get(f"/api/worlds/{world_id}/actor/turn").json()["turn"])


def _intent(container, **payload) -> None:
    container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = json.dumps(payload, ensure_ascii=False)


def _talk_index() -> int:
    return next(i for i, c in enumerate(_ACTION_SPACE) if c.action_type is ActionType.TALK)


def _work_index() -> int:
    return next(i for i, c in enumerate(_ACTION_SPACE) if c.action_type is ActionType.WORK)


@pytest.mark.asyncio
async def test_casting_happens_at_confirmation(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)

    with _client(mock_build_container, test_config, catalog) as client:
        unknown = client.post(f"/api/worlds/{world_id}/confirm", json={"character_id": "nobody"})
        confirmed = client.post(f"/api/worlds/{world_id}/confirm", json={"character_id": character_id})
        recast = client.post(f"/api/worlds/{world_id}/confirm", json={"character_id": character_id})
        again = client.post(f"/api/worlds/{world_id}/confirm")

    assert unknown.status_code == 400
    assert confirmed.status_code == 200
    assert confirmed.json()["confirmed"] is True
    assert confirmed.json()["actor_character_id"] == character_id
    assert confirmed.json()["actor_exited"] is False
    # The cast is locked with the setup; confirming again without one changes nothing.
    assert recast.status_code == 409
    assert again.status_code == 200 and again.json()["actor_character_id"] == character_id


@pytest.mark.asyncio
async def test_public_cast_is_identical_before_during_and_after_performing(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)

    with _client(mock_build_container, test_config, catalog) as client:
        before = client.get(f"/api/worlds/{world_id}/agents").json()
        assert client.post(f"/api/worlds/{world_id}/confirm", json={"character_id": character_id}).status_code == 200
        performing = client.get(f"/api/worlds/{world_id}/agents").json()
        assert client.post(f"/api/worlds/{world_id}/actor/exit").status_code == 200
        released = client.get(f"/api/worlds/{world_id}/agents").json()
        assert client.get(f"/api/worlds/{world_id}/actor/candidates").status_code == 404

    assert before == performing == released
    assert character_id in {c["agent_id"] for c in before}
    assert all(set(candidate) == {
        "agent_id", "name", "gender", "age", "role", "background", "appearance", "color",
    } for candidate in before)


@pytest.mark.asyncio
async def test_after_the_directors_review_nobody_can_be_cast(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)

    with _client(mock_build_container, test_config, catalog) as client:
        reviewed = client.post(f"/api/worlds/{world_id}/review")
        cast = client.post(f"/api/worlds/{world_id}/confirm", json={"character_id": character_id})
        as_director = client.post(f"/api/worlds/{world_id}/confirm")

    assert reviewed.status_code == 200 and reviewed.json()["director_reviewed"] is True
    assert cast.status_code == 409
    assert as_director.status_code == 200 and as_director.json()["actor_character_id"] is None


@pytest.mark.asyncio
async def test_the_directors_review_is_closed_while_an_actor_performs(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)

    with _client(mock_build_container, test_config, catalog) as client:
        client.post(f"/api/worlds/{world_id}/confirm", json={"character_id": character_id})
        review = client.post(f"/api/worlds/{world_id}/review")
        meta = client.get(f"/api/worlds/{world_id}").json()

    assert review.status_code == 409
    # Who the main characters are is not the actor's to know.
    assert meta["main_agent_names"] == []


@pytest.mark.asyncio
async def test_emotions_are_the_vocabulary_an_amendment_picks_from(mock_build_container, test_config) -> None:
    from agent.personality import EmotionType

    catalog = WorldCatalog(None)
    world_id, _character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)

    with _client(mock_build_container, test_config, catalog) as client:
        emotions = client.get(f"/api/worlds/{world_id}/actor/emotions").json()

    assert [e["value"] for e in emotions] == [e.value for e in EmotionType]
    assert {e["value"]: e["label"] for e in emotions}["anger"] == EmotionType.ANGER.label


@pytest.mark.asyncio
async def test_the_actor_reads_confirms_and_answers_a_turn(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)

    with _client(mock_build_container, test_config, catalog) as client:
        client.post(f"/api/worlds/{world_id}/confirm", json={"character_id": character_id})
        assert client.post(f"/api/worlds/{world_id}/run", json={"steps": 1}).status_code == 200
        turn = _open_turn(client, world_id)
        assert (turn["agent_id"], turn["kind"]) == (character_id, "decide")
        assert "people" in turn["briefing"]

        # Words naming nobody the character can talk to: refused with the reason, nothing answered.
        _intent(mock_build_container, reason="找人说话", decision="act", selected_index=_talk_index(),
                person_indices=[99], action_description="找人说话")
        refused = client.post(f"/api/worlds/{world_id}/actor/say", json={"text": "找个人说话"}).json()
        _intent(mock_build_container, reason="只是叹气", decision="unclear")
        unreadable = client.post(f"/api/worlds/{world_id}/actor/say", json={"text": "唉"}).json()
        _intent(mock_build_container, reason="誊抄奏章", decision="act", selected_index=_work_index(),
                action_description="亲自誊抄奏章", estimated_steps=1)
        reading = client.post(f"/api/worlds/{world_id}/actor/say", json={"text": "抄奏章"}).json()

        assert refused == {"accepted": False, "reason": "talk_without_target"}
        assert unreadable == {"accepted": False, "reason": "unreadable"}
        assert reading == {"accepted": True, "reason": None}
        assert client.get(f"/api/worlds/{world_id}/actor/turn").json()["turn"] is not None

        # The reading itself is in the thread, where the actor reads it.
        said = client.get(f"/api/worlds/{world_id}/actor/thread").json()["entries"][-1]
        assert said["kind"] == "said" and said["turn_id"] == turn["turn_id"]
        assert said["reading"]["action"]["action_type"] == "work"
        stale = client.post(f"/api/worlds/{world_id}/actor/answer", json={
            "turn_id": "a-closed-turn", "answer": said["reading"],
        })
        answered = client.post(f"/api/worlds/{world_id}/actor/answer", json={
            "turn_id": said["turn_id"], "answer": said["reading"],
        })
        assert stale.status_code == 409
        assert answered.json() == {"accepted": True, "reason": None}

        meta = _until(lambda: (
            m if (m := client.get(f"/api/worlds/{world_id}").json())["run_state"] == "completed" else None
        ))
        thread = client.get(f"/api/worlds/{world_id}/actor/thread").json()["entries"]
    assert meta["current_step"] == 1
    assert [e["kind"] for e in thread] == ["asked", "said", "said", "said", "answered", "began", "ended"]


@pytest.mark.asyncio
async def test_words_and_break_offs_outside_a_turn_are_refused(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)

    with _client(mock_build_container, test_config, catalog) as client:
        client.post(f"/api/worlds/{world_id}/confirm", json={"character_id": character_id})
        said = client.post(f"/api/worlds/{world_id}/actor/say", json={"text": "抄奏章"})
        # Idle, so nothing to break off.
        break_off = client.post(f"/api/worlds/{world_id}/actor/break-off").json()

    assert said.status_code == 409
    assert break_off == {"accepted": False, "reason": "nothing_to_break_off"}


@pytest.mark.asyncio
async def test_the_pace_is_kept_with_the_casting(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)

    with _client(mock_build_container, test_config, catalog) as client:
        before = client.get(f"/api/worlds/{world_id}").json()["actor_pace"]
        no_actor = client.put(f"/api/worlds/{world_id}/actor/pace", json={"pace": "when_reached"})
        client.post(f"/api/worlds/{world_id}/confirm", json={"character_id": character_id})
        unknown = client.put(f"/api/worlds/{world_id}/actor/pace", json={"pace": "lazy"})
        set_ = client.put(f"/api/worlds/{world_id}/actor/pace", json={"pace": "when_reached"})
        after = client.get(f"/api/worlds/{world_id}").json()["actor_pace"]

    assert before == "every_step"
    assert no_actor.status_code == 409
    assert unknown.status_code == 400
    assert set_.json() == {"pace": "when_reached"}
    assert after == "when_reached"


@pytest.mark.asyncio
async def test_a_refused_or_malformed_answer_leaves_the_turn_open(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)

    with _client(mock_build_container, test_config, catalog) as client:
        client.post(f"/api/worlds/{world_id}/confirm", json={"character_id": character_id})
        client.post(f"/api/worlds/{world_id}/run", json={"steps": 1})
        turn = _open_turn(client, world_id)

        def answer(body: Any):
            return client.post(f"/api/worlds/{world_id}/actor/answer", json={
                "turn_id": turn["turn_id"], "answer": body,
            })

        unknown_key = answer({"idle": True, "mood": "calm"})
        no_type = answer({"action": {"action_type": None, "action_description": "走走"}})
        not_a_flag = answer({"idle": "yes"})
        no_turn = client.post(f"/api/worlds/{world_id}/actor/answer", json={"answer": {"idle": True}})
        not_an_accept = answer({"accept": "yes"})
        not_a_line = answer({"line": 3, "end": False})
        # A line that doesn't end the conversation has to say something.
        nothing_said = answer({"line": " ", "end": False})
        # Well formed, but a decision turn takes no interrupt, invitation or line.
        wrong_kinds = [answer({"interrupt": True}), answer({"accept": True}), answer({"line": "大哥", "end": False})]

        malformed = (unknown_key, no_type, not_a_flag, no_turn, not_an_accept, not_a_line, nothing_said)
        assert [r.status_code for r in malformed] == [400] * 7
        assert [r.status_code for r in wrong_kinds] == [409, 409, 409]
        assert client.get(f"/api/worlds/{world_id}/actor/turn").json()["turn"]["turn_id"] == turn["turn_id"]

        assert answer({"idle": True}).json()["accepted"] is True
        client.post(f"/api/worlds/{world_id}/stop")


@pytest.mark.asyncio
async def test_the_actor_reads_and_amends_the_mind_between_steps(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)

    with _client(mock_build_container, test_config, catalog) as client:
        client.post(f"/api/worlds/{world_id}/confirm", json={"character_id": character_id})
        before = client.get(f"/api/worlds/{world_id}/actor/mind").json()
        saved = client.post(f"/api/worlds/{world_id}/actor/amend", json={
            "emotion": {"emotion": "anger", "intensity": 0.7, "valence": -0.6},
            "short_term_goals": [{"id": None, "text": "今夜先见长孙无忌"}],
            "insights": ["大哥终究容不下我"],
        })
        refused = client.post(f"/api/worlds/{world_id}/actor/amend", json={
            "relations": [{"agent_id": "nobody", "trust": 0.5}],
        })
        malformed = client.post(f"/api/worlds/{world_id}/actor/amend", json={"mood": "calm"})
        after = client.get(f"/api/worlds/{world_id}/actor/mind").json()

    assert set(before) >= {"emotion", "long_term_goals", "short_term_goals", "relations"}
    assert before["pending"] is None
    assert saved.json() == {"accepted": True, "reason": None, "landed": False}
    assert refused.json() == {"accepted": False, "reason": "relation_not_listed", "landed": False}
    assert malformed.status_code == 400
    assert after["pending"]["short_term_goals"] == [{"id": None, "text": "今夜先见长孙无忌"}]
    assert after["pending"]["insights"] == ["大哥终究容不下我"]
    assert after["emotion"] == before["emotion"]


@pytest.mark.asyncio
async def test_director_routes_close_while_an_actor_performs(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)

    with _client(mock_build_container, test_config, catalog) as client:
        client.post(f"/api/worlds/{world_id}/confirm", json={"character_id": character_id})
        directive = client.post(f"/api/worlds/{world_id}/direct", json={"text": "敲响全城的钟"})
        directives = client.get(f"/api/worlds/{world_id}/directives")

        exited = client.post(f"/api/worlds/{world_id}/actor/exit")
        exit_again = client.post(f"/api/worlds/{world_id}/actor/exit")
        directives_after = client.get(f"/api/worlds/{world_id}/directives")

    assert directive.status_code == 403 and directives.status_code == 403
    assert exited.status_code == 200
    assert (exited.json()["actor_character_id"], exited.json()["actor_exited"]) == (character_id, True)
    assert exit_again.status_code == 409
    assert directives_after.status_code == 200


@pytest.mark.asyncio
async def test_turns_and_entries_reach_the_websocket(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, _character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)
    turn = ActorTurn(agent_id="a1", step=3, kind=TurnKind.INTERRUPT, doing="誊抄奏章")

    with _client(mock_build_container, test_config, catalog) as client:
        with client.websocket_connect(f"/ws/{world_id}") as ws:
            assert ws.receive_json()["type"] == "status"
            assert ws.receive_json()["type"] == "latest_step"
            mock_build_container.event_bus.publish({"type": "actor_turn", "world_id": world_id, "turn": turn})
            mock_build_container.event_bus.publish({"type": "actor_turn", "world_id": world_id, "turn": None})
            mock_build_container.event_bus.publish(
                {"type": "actor_entry", "world_id": world_id, "entry": {"kind": "ended", "fact": "我抄完了"}},
            )
            opened, closed, entry = ws.receive_json(), ws.receive_json(), ws.receive_json()

    assert entry == {"type": "actor_entry", "data": {"kind": "ended", "fact": "我抄完了"}}
    assert opened["type"] == "actor_turn"
    assert (opened["data"]["step"], opened["data"]["kind"], opened["data"]["doing"]) == (3, "interrupt", "誊抄奏章")
    assert closed == {"type": "actor_turn", "data": None}


@pytest.mark.asyncio
async def test_observation_routes_answer_as_the_character_until_the_actor_exits(
    mock_build_container, test_config,
) -> None:
    catalog = WorldCatalog(None)
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)
    snapshot = await mock_build_container.snapshot.load(world_id, 0)
    assert snapshot is not None
    private = {
        "core_traits": ["谨慎"], "core_values": ["家人"], "self_image": "守护者",
        "life_goal": "保全家人", "secret": "藏有密信",
    }
    for profile in snapshot.build["character_profiles"].values():
        profile.update(private)
    await mock_build_container.snapshot.save(world_id, 0, snapshot)

    with _client(mock_build_container, test_config, catalog) as client:
        profiles = {
            agent_id: client.get(f"/api/worlds/{world_id}/agents/{agent_id}/profile").json()
            for agent_id in snapshot.build["character_profiles"]
        }
        client.post(f"/api/worlds/{world_id}/confirm", json={"character_id": character_id})
        performing = {
            agent_id: client.get(f"/api/worlds/{world_id}/agents/{agent_id}/profile").json()
            for agent_id in profiles
        }
        graph = client.get(f"/api/worlds/{world_id}/graph").json()
        step = client.get(f"/api/worlds/{world_id}/steps/0").json()
        client.post(f"/api/worlds/{world_id}/actor/exit")
        released = {
            agent_id: client.get(f"/api/worlds/{world_id}/agents/{agent_id}/profile").json()
            for agent_id in profiles
        }

    assert performing[character_id] == profiles[character_id]
    others = [p for agent_id, p in performing.items() if agent_id != character_id]
    assert others
    for profile in others:
        expected = profiles[profile["agent_id"]] | {
            "core_traits": [], "core_values": [], "self_image": "", "life_goal": "",
            "secret": "", "is_main_character": None,
        }
        assert profile == expected
    assert all({key: profile[key] for key in private} == private for profile in profiles.values())
    assert all(e["from_id"] == character_id for e in graph["edges"])
    assert step["observer"] == {"injections": []}
    assert released == profiles


