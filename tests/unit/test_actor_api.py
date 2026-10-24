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
from interaction.application import NarrativeApplication
from world import WorldCatalog
from world.setup import WorldSetup

from interaction.api import create_app
from interaction.api.actor import _decode_amendments
from interaction.api.serialization import to_wire_json
from interaction.models import agent_profile


async def _unconfirmed_world(container, test_config, catalog: WorldCatalog) -> tuple[str, str]:
    """A world still under review, and a character to cast."""
    application = NarrativeApplication(container, test_config, catalog=catalog)
    world_id = await application.build_world("玄武门之变", template="changan_iso")
    return world_id, (await WorldSetup.load(container.snapshot, world_id)).characters[0].agent_id


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
    with client.websocket_connect(f"/ws/{world_id}") as ws:
        while True:
            frame = ws.receive_json()
            if frame["type"] == "actor_turn" and frame["data"] is not None:
                return frame["data"]


def _intent(container, **payload) -> None:
    container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = json.dumps(payload, ensure_ascii=False)


def _talk_index() -> int:
    return next(i for i, c in enumerate(_ACTION_SPACE) if c.action_type is ActionType.TALK)


def _work_index() -> int:
    return next(i for i, c in enumerate(_ACTION_SPACE) if c.action_type is ActionType.WORK)


@pytest.mark.asyncio
@pytest.mark.parametrize("private_route", ["setup", "profile"])
async def test_private_reads_close_actor_casting(mock_build_container, test_config, private_route) -> None:
    catalog = WorldCatalog()
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)
    path = "setup" if private_route == "setup" else f"agents/{character_id}/profile"
    with _client(mock_build_container, test_config, catalog) as client:
        assert client.get(f"/api/worlds/{world_id}/{path}").status_code == 200
        assert catalog.director_reviewed(world_id)
        response = client.post(f"/api/worlds/{world_id}/confirm", json={"character_id": character_id})
        assert response.status_code == 409
        assert client.post(f"/api/worlds/{world_id}/confirm").status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("private_route", ["setup", "profile", "review"])
@pytest.mark.parametrize("failure", ["refresh", "write"])
async def test_private_reads_fail_closed_when_review_cannot_be_recorded(
    mock_build_container, test_config, monkeypatch, private_route, failure,
) -> None:
    catalog = WorldCatalog()
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)
    if failure == "refresh":
        monkeypatch.setattr(catalog, "_refreshed_for_write", lambda *_: False)
    else:
        def fail_write() -> None:
            raise OSError("Catalog storage unavailable")

        monkeypatch.setattr(catalog, "_flush", fail_write)
    path = f"agents/{character_id}/profile" if private_route == "profile" else private_route
    with _client(mock_build_container, test_config, catalog) as client:
        response = (
            client.post(f"/api/worlds/{world_id}/{path}") if private_route == "review"
            else client.get(f"/api/worlds/{world_id}/{path}")
        )
    assert response.status_code == 503
    assert response.json() == {"detail": "The director review could not be recorded."}


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
async def test_public_npcs_are_identical_before_during_and_after_performing(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)
    setup = await mock_build_container.snapshot.load_world_setup(world_id)
    setup["npc_seeds"] = [{
        "npc_id": "npc-messenger", "name": "信使", "gender": "男", "age": 34,
        "description": "善于跑腿", "location": setup["characters"][0]["initial_location"],
    }]
    await mock_build_container.snapshot.save_world_setup(world_id, setup)

    with _client(mock_build_container, test_config, catalog) as client:
        before = client.get(f"/api/worlds/{world_id}/npcs").json()
        assert client.post(f"/api/worlds/{world_id}/confirm", json={"character_id": character_id}).status_code == 200
        performing = client.get(f"/api/worlds/{world_id}/npcs").json()
        assert client.post(f"/api/worlds/{world_id}/actor/exit").status_code == 200
        released = client.get(f"/api/worlds/{world_id}/npcs").json()

    assert before == performing == released
    assert [npc["npc_id"] for npc in before] == ["npc-messenger"]


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
        refused = client.post(f"/api/worlds/{world_id}/actor/intent", json={"turn_id": turn["turn_id"], "text": "找个人说话"}).json()
        _intent(mock_build_container, reason="只是叹气", decision="unclear")
        unreadable = client.post(f"/api/worlds/{world_id}/actor/intent", json={"turn_id": turn["turn_id"], "text": "唉"}).json()
        _intent(mock_build_container, reason="誊抄奏章", decision="act", selected_index=_work_index(),
                action_description="亲自誊抄奏章", estimated_steps=1)
        decision_summary = client.post(f"/api/worlds/{world_id}/actor/intent", json={"turn_id": turn["turn_id"], "text": "抄奏章"}).json()

        assert refused == {"status": "rejected", "reason": "talk_without_target", "decision_summary": None}
        assert unreadable == {"status": "rejected", "reason": "unreadable", "decision_summary": None}
        assert decision_summary["status"] == "needs_confirmation" and decision_summary["reason"] is None
        assert _open_turn(client, world_id)["turn_id"] == turn["turn_id"]

        # The decision summary itself is in the thread, where the actor reads it.
        said = client.get(f"/api/worlds/{world_id}/actor/thread").json()["entries"][-1]
        assert said["kind"] == "said" and said["turn_id"] == turn["turn_id"]
        assert said["decision_summary"]["operation"] == "work"
        assert "action" not in said["decision_summary"]
        stale = client.post(f"/api/worlds/{world_id}/actor/answer", json={
            "turn_id": "a-closed-turn", "answer": {"decision_summary_id": said["decision_summary"]["decision_summary_id"]},
        })
        answered = client.post(f"/api/worlds/{world_id}/actor/answer", json={
            "turn_id": said["turn_id"], "answer": {"decision_summary_id": said["decision_summary"]["decision_summary_id"]},
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
        said = client.post(f"/api/worlds/{world_id}/actor/intent", json={"turn_id": "closed-turn", "text": "抄奏章"})
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
        invalid_thought = answer({"interrupt": True, "thought": 3})
        long_thought = answer({"interrupt": True, "thought": "想" * 501})
        # Well formed, but a decision turn takes no interrupt, invitation or line.
        wrong_kinds = [answer({"interrupt": True}), answer({"accept": True}), answer({"line": "大哥", "end": False})]

        malformed = (unknown_key, no_type, not_a_flag, no_turn, not_an_accept, not_a_line, nothing_said, invalid_thought, long_thought)
        assert [r.status_code for r in malformed] == [400] * len(malformed)
        assert [r.status_code for r in wrong_kinds] == [409, 409, 409]
        assert _open_turn(client, world_id)["turn_id"] == turn["turn_id"]

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
async def test_actor_turn_is_delivered_and_restored_only_through_websocket(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)
    base = f"/api/worlds/{world_id}"
    with _client(mock_build_container, test_config, catalog) as client:
        assert client.get(f"{base}/actor/turn").status_code == 404
        assert "/api/worlds/{world_id}/actor/turn" not in client.get("/openapi.json").json()["paths"]
        assert client.post(f"{base}/confirm", json={"character_id": character_id}).status_code == 200
        with client.websocket_connect(f"/ws/{world_id}") as ws:
            assert ws.receive_json()["type"] == "status"
            assert ws.receive_json()["type"] == "latest_step"
            assert client.post(f"{base}/run", json={"steps": 1}).status_code == 200
            while True:
                frame = ws.receive_json()
                if frame["type"] == "actor_turn" and frame["data"] is not None:
                    turn = frame["data"]
                    break
            assert turn["kind"] == "decide" and turn["agent_id"] == character_id
            assert _open_turn(client, world_id) == turn
            answered = client.post(f"{base}/actor/answer", json={
                "turn_id": turn["turn_id"], "answer": {"idle": True},
            })
            assert answered.json()["accepted"] is True
            while True:
                frame = ws.receive_json()
                if frame["type"] == "actor_turn":
                    assert frame["data"] is None
                    break
        _until(lambda: meta if (meta := client.get(base).json())["run_state"] == "completed" else None)


@pytest.mark.asyncio
async def test_turns_and_entries_reach_the_websocket(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, _character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)
    await NarrativeApplication(mock_build_container, test_config, catalog=catalog).confirm_world(world_id)
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
    setup = await mock_build_container.snapshot.load_world_setup(world_id)
    assert setup is not None
    private = {
        "core_traits": ["谨慎"], "core_values": ["家人"], "self_image": "守护者",
        "life_goal": "保全家人", "secret": "藏有密信",
    }
    for definition in setup["characters"]:
        definition["soul_data"].update(private)
    await mock_build_container.snapshot.save_world_setup(world_id, setup)
    profiles = {
        d.agent_id: to_wire_json(agent_profile(d)) for d in WorldSetup.from_dict(setup).characters
    }

    with _client(mock_build_container, test_config, catalog) as client:
        assert client.post(
            f"/api/worlds/{world_id}/confirm", json={"character_id": character_id},
        ).status_code == 200
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




@pytest.mark.asyncio
@pytest.mark.parametrize("description", ["", " \n "])
async def test_broadcast_skips_interpretation_and_uses_only_session_model_gate(
    mock_build_container, test_config, monkeypatch, description: str,
) -> None:
    from unittest.mock import AsyncMock, Mock

    from interaction.api.app import ApiServices

    catalog = WorldCatalog(None)
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)
    base = f"/api/worlds/{world_id}"
    with _client(mock_build_container, test_config, catalog) as client:
        client.post(f"{base}/confirm", json={"character_id": character_id})
        client.post(f"{base}/run", json={"steps": 1})
        turn = _open_turn(client, world_id)
        model_gate = Mock()
        parser = AsyncMock(side_effect=AssertionError("must not parse"))
        monkeypatch.setattr(ApiServices, "require_model_keys", model_gate)
        monkeypatch.setattr(mock_build_container.llm_router.get(LLMScene.ACTOR_INTENT), "complete", parser)
        rejected = client.post(f"{base}/actor/intent", json={
            "turn_id": turn["turn_id"], "operation": "broadcast",
            "target": {"kind": "location", "id": turn["briefing"]["location"]["location_id"]},
            "parameters": {"message": "请大家留意这里"}, "description": "只悄悄告诉一个人",
        })
        assert rejected.status_code == 400
        assert "broadcast intent takes no description" in rejected.json()["detail"]
        model_gate.assert_not_called()
        parser.assert_not_awaited()
        submitted = client.post(f"{base}/actor/intent", json={
            "turn_id": turn["turn_id"], "operation": "broadcast",
            "target": {"kind": "location", "id": turn["briefing"]["location"]["location_id"]},
            "parameters": {"message": "请大家留意这里"}, "description": description,
        })
        assert submitted.status_code == 200
        assert submitted.json()["status"] == "submitted"
        # Session access checks keys independently of intent interpretation.
        model_gate.assert_called_once_with()
        parser.assert_not_awaited()
        _until(lambda: m if (m := client.get(base).json())["run_state"] == "completed" else None)
        thread = client.get(f"{base}/actor/thread").json()["entries"]
        said = next(entry for entry in thread if entry["kind"] == "said")
        assert said["intent"]["description"] == description
        assert said["decision_summary"]["parameters"]["message"] == "请大家留意这里"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["work", "rest", "covert"])
async def test_self_operations_use_turn_identity_and_follow_description_requirements(
    mock_build_container, test_config, monkeypatch, operation: str,
) -> None:
    from unittest.mock import AsyncMock

    catalog = WorldCatalog(None)
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)
    provider = mock_build_container.llm_router.get(LLMScene.ACTOR_INTENT)
    provider.fixed_response = json.dumps({
        "decision": "act",
        "action_description": "我按描述完成自己的行动", "estimated_steps": 3,
    })
    parser = AsyncMock(wraps=provider.complete)
    monkeypatch.setattr(provider, "complete", parser)
    base = f"/api/worlds/{world_id}"
    with _client(mock_build_container, test_config, catalog) as client:
        client.post(f"{base}/confirm", json={"character_id": character_id})
        client.post(f"{base}/run", json={"steps": 1})
        turn = _open_turn(client, world_id)
        assert turn["agent_id"] == character_id
        assert "character" not in turn["briefing"]
        assert character_id not in {person["agent_id"] for person in turn["briefing"]["people"]}
        capabilities = client.get(f"{base}/actor/operations", params={
            "turn_id": turn["turn_id"], "target_kind": "agent", "target_id": character_id,
        })
        assert capabilities.status_code == 200
        assert [capability["operation"] for capability in capabilities.json()["operations"]] == ["work", "rest", "covert"]
        assert all(capability["available"] for capability in capabilities.json()["operations"])
        requirements = {c["operation"]: c["required_parameters"] for c in capabilities.json()["operations"]}
        assert requirements == {"work": ["description"], "rest": [], "covert": ["description"]}
        target = {"kind": "agent", "id": character_id}
        payload = {"turn_id": turn["turn_id"], "operation": operation, "target": target, "parameters": {}}
        if operation != "rest":
            rejected = client.post(f"{base}/actor/intent", json=payload)
            assert rejected.json()["reason"] == "description_required"
            parser.assert_not_awaited()
            payload["description"] = "整理账册" if operation == "work" else "观察此处正在进行的谈话"
        submitted = client.post(f"{base}/actor/intent", json=payload)
        assert submitted.status_code == 200
        if operation == "rest":
            assert submitted.json()["status"] == "submitted"
            parser.assert_not_awaited()
        else:
            assert submitted.json()["status"] == "needs_confirmation"
            parser.assert_awaited_once()
            assert client.post(f"{base}/actor/answer", json={
                "turn_id": turn["turn_id"], "answer": {
                    "decision_summary_id": submitted.json()["decision_summary"]["decision_summary_id"],
                },
            }).json()["accepted"] is True
        _until(lambda: m if (m := client.get(base).json())["run_state"] == "completed" else None)
        thread = client.get(f"{base}/actor/thread").json()["entries"]
        said = next(entry for entry in thread if entry["kind"] == "said" and entry["decision_summary"] is not None)
        step_seconds = client.get(base).json()["seconds_per_step"]
        assert said["decision_summary"]["estimated_duration_seconds"] == step_seconds * (1 if operation == "rest" else 3)
        answered = next(entry for entry in thread if entry["kind"] == "answered")
        assert said["intent"]["operation"] == said["decision_summary"]["operation"] == operation
        assert said["intent"]["target"] == said["decision_summary"]["target"] == target
        assert answered["answer"] == {"decision_summary_id": said["decision_summary"]["decision_summary_id"]}
        assert "action_description" not in json.dumps(thread)
    snapshot = await mock_build_container.snapshot.load(world_id, 1)
    assert snapshot.actor_thread == thread
