"""Actor API: casting at confirmation, the turn as the actor sees it, reading their words,
answering, the director's routes closed while an actor performs, and the exit."""

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


async def _unconfirmed_world(container, test_config, catalog: WorldCatalog) -> tuple[str, str]:
    """A world still under review, and a character to cast."""
    world = await NarrativeApplication(container, test_config, catalog=catalog).build_world(
        "玄武门之变", template="changan_iso",
    )
    return world.world_id, next(iter(world.agents))


def _client(container, test_config, catalog: WorldCatalog) -> TestClient:
    return TestClient(create_app(container, test_config, catalog=catalog))


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
async def test_candidates_show_only_what_anyone_could_see(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)

    with _client(mock_build_container, test_config, catalog) as client:
        candidates = client.get(f"/api/worlds/{world_id}/actor/candidates").json()

    assert character_id in {c["agent_id"] for c in candidates}
    assert set(candidates[0]) == {"agent_id", "name", "gender", "age", "role", "color", "is_main_character"}


@pytest.mark.asyncio
async def test_the_actor_reads_confirms_and_answers_a_turn(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)

    with _client(mock_build_container, test_config, catalog) as client:
        client.post(f"/api/worlds/{world_id}/confirm", json={"character_id": character_id})
        assert client.post(f"/api/worlds/{world_id}/run", json={"steps": 1}).status_code == 200
        turn = _open_turn(client, world_id)
        assert (turn["agent_id"], turn["kind"]) == (character_id, "decide")
        assert turn["briefing"]["location_name"]

        # Words naming nobody the character can talk to: refused with the reason, nothing answered.
        _intent(mock_build_container, reason="找人说话", decision="act", selected_index=_talk_index(),
                person_indices=[99], action_description="找人说话")
        refused = client.post(f"/api/worlds/{world_id}/actor/interpret", json={"text": "找个人说话"}).json()
        _intent(mock_build_container, reason="只是叹气", decision="unclear")
        unreadable = client.post(f"/api/worlds/{world_id}/actor/interpret", json={"text": "唉"}).json()
        _intent(mock_build_container, reason="先等着", decision="wait", wait_steps=2)
        reading = client.post(f"/api/worlds/{world_id}/actor/interpret", json={"text": "等两个时辰"}).json()

        assert (refused["accepted"], refused["reason"]) == (False, "talk_without_target")
        assert (unreadable["accepted"], unreadable["reason"]) == (False, "unreadable")
        assert reading["accepted"] is True
        assert reading["reading"]["wait"] == {"steps": 2}
        assert client.get(f"/api/worlds/{world_id}/actor/turn").json()["turn"] is not None

        assert reading["turn_id"] == turn["turn_id"]
        stale = client.post(f"/api/worlds/{world_id}/actor/answer", json={
            "turn_id": "a-closed-turn", "answer": reading["reading"],
        })
        answered = client.post(f"/api/worlds/{world_id}/actor/answer", json={
            "turn_id": reading["turn_id"], "answer": reading["reading"],
        })
        assert stale.status_code == 409
        assert answered.json() == {"accepted": True, "reason": None}

        meta = _until(lambda: (
            m if (m := client.get(f"/api/worlds/{world_id}").json())["run_state"] == "completed" else None
        ))
    assert meta["current_step"] == 1


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

        not_listed = answer({"wait": {}, "amendments": {
            "memories": [{"memory_id": "not-listed", "text": "改写"}], "relations": [],
        }})
        unknown_key = answer({"wait": {}, "mood": "calm"})
        out_of_range = answer({"wait": {}, "amendments": {
            "memories": [], "relations": [{"agent_id": character_id, "trust": 3}],
        }})
        no_type = answer({"action": {"action_type": None, "action_description": "走走"}})

        assert not_listed.json() == {"accepted": False, "reason": "memory_not_listed"}
        assert [r.status_code for r in (unknown_key, out_of_range, no_type)] == [400, 400, 400]
        assert client.get(f"/api/worlds/{world_id}/actor/turn").json()["turn"]["turn_id"] == turn["turn_id"]

        assert answer({"wait": {}}).json()["accepted"] is True
        client.post(f"/api/worlds/{world_id}/stop")


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
async def test_turns_reach_the_websocket(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, _character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)
    turn = ActorTurn(agent_id="a1", step=3, kind=TurnKind.INTERRUPT, doing="誊抄奏章")

    with _client(mock_build_container, test_config, catalog) as client:
        with client.websocket_connect(f"/ws/{world_id}") as ws:
            assert ws.receive_json()["type"] == "status"
            assert ws.receive_json()["type"] == "latest_step"
            mock_build_container.event_bus.publish({"type": "actor_turn", "world_id": world_id, "turn": turn})
            mock_build_container.event_bus.publish({"type": "actor_turn", "world_id": world_id, "turn": None})
            opened, closed = ws.receive_json(), ws.receive_json()

    assert opened["type"] == "actor_turn"
    assert (opened["data"]["step"], opened["data"]["kind"], opened["data"]["doing"]) == (3, "interrupt", "誊抄奏章")
    assert closed == {"type": "actor_turn", "data": None}


@pytest.mark.asyncio
async def test_observation_routes_answer_as_the_character_until_the_actor_exits(
    mock_build_container, test_config,
) -> None:
    catalog = WorldCatalog(None)
    world_id, character_id = await _unconfirmed_world(mock_build_container, test_config, catalog)

    with _client(mock_build_container, test_config, catalog) as client:
        client.post(f"/api/worlds/{world_id}/confirm", json={"character_id": character_id})
        performing = client.get(f"/api/worlds/{world_id}/agents").json()
        graph = client.get(f"/api/worlds/{world_id}/graph").json()
        step = client.get(f"/api/worlds/{world_id}/steps/0").json()
        client.post(f"/api/worlds/{world_id}/actor/exit")
        released = client.get(f"/api/worlds/{world_id}/agents").json()

    others = [p for p in performing if p["agent_id"] != character_id]
    assert others and all(p["is_main_character"] is None and p["secret"] == "" for p in others)
    assert all(e["from_id"] == character_id for e in graph["edges"])
    assert step["observer"] == {"injections": []}
    assert all(p["is_main_character"] is not None for p in released)
