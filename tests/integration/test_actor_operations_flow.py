"""Explicit actor intent reaches execution and persisted thread without intent parsing."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
import pytest

from config.models import Config
from core.container import Container
from core.interfaces.llm import LLMScene
from engine.application import NarrativeApplication
from interaction.api import create_app
from world import WorldCatalog


def _until(probe: Callable[[], Any]) -> Any:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        result = probe()
        if result is not None:
            return result
        time.sleep(0.02)
    raise AssertionError("Actor operation did not finish")


def _assert_public(value: object) -> None:
    if isinstance(value, dict):
        assert not {
            "action_description", "inner_monologue", "expected_outcome",
            "estimated_steps", "physical_entity_id", "person_ids",
        }.intersection(value)
        for child in value.values():
            _assert_public(child)
    elif isinstance(value, list):
        for child in value:
            _assert_public(child)


@pytest.mark.asyncio
async def test_explicit_move_executes_without_parsing_and_persists_public_thread(
    mock_build_container: Container,
    test_config: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = mock_build_container
    catalog = WorldCatalog(None)
    world = await NarrativeApplication(container, test_config, catalog=catalog).build_world(
        "玄武门之变", template="changan_iso",
    )
    character_id = next(iter(world.agents))
    parser = AsyncMock(side_effect=AssertionError("Explicit operations must not parse intent"))
    monkeypatch.setattr(container.llm_router.get(LLMScene.ACTOR_INTENT), "complete", parser)
    base = f"/api/worlds/{world.world_id}"

    with TestClient(create_app(container, test_config, catalog=catalog)) as client:
        assert client.post(f"{base}/confirm", json={"character_id": character_id}).status_code == 200
        assert client.post(f"{base}/run", json={"steps": 1}).status_code == 200
        with client.websocket_connect(f"/ws/{world.world_id}") as ws:
            while True:
                frame = ws.receive_json()
                if frame["type"] == "actor_turn" and frame["data"] is not None:
                    turn = frame["data"]
                    break
        destination = turn["briefing"]["places"][0]["location_id"]
        target = {"kind": "location", "id": destination}
        capabilities = client.get(f"{base}/actor/operations", params={
            "turn_id": turn["turn_id"], "target_kind": "location", "target_id": destination,
        })
        assert capabilities.status_code == 200
        assert any(
            item["operation"] == "move" and item["available"]
            for item in capabilities.json()["operations"]
        )

        response = client.post(f"{base}/actor/intent", json={
            "turn_id": turn["turn_id"], "operation": "move", "target": target,
            "parameters": {},
        })
        assert response.status_code == 200
        assert response.json()["status"] == "submitted"
        _until(lambda: (
            meta if (meta := client.get(base).json())["run_state"] == "completed" else None
        ))
        thread = client.get(f"{base}/actor/thread").json()["entries"]
        began = next(entry for entry in thread if entry["kind"] == "began")
        said = next(entry for entry in thread if entry["kind"] == "said")
        seconds_per_step = client.get(base).json()["seconds_per_step"]
        assert said["action_summary"]["estimated_duration_seconds"] == began["total_steps"] * seconds_per_step
        _assert_public(thread)
        stale = client.post(f"{base}/actor/intent", json={
            "turn_id": turn["turn_id"], "operation": "move", "target": target,
            "parameters": {},
        })
        assert stale.status_code == 409

    parser.assert_not_awaited()
    snapshot = await container.snapshot.load(world.world_id, 1)
    assert snapshot is not None
    assert snapshot.actor_thread == thread
    _assert_public(snapshot.actor_thread)
