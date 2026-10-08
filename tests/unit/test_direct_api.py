"""Director API: submitting interventions and single-stepping.

These ``interaction/`` paths change the world. The main product semantics verified: "can't be
carried out" is a normal answer, not an error.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from datetime import datetime, timezone

from core.interfaces.llm import LLMScene
from core.interfaces.snapshot import WorldSnapshot
from engine.application import NarrativeApplication
from engine.run_control import RunState
from world import WorldCatalog

from interaction.api import create_app


def _client(container, test_config, catalog: WorldCatalog) -> TestClient:
    return TestClient(create_app(container, test_config, catalog=catalog))


async def _built_world(container, test_config, catalog: WorldCatalog) -> str:
    """A confirmed world with a live session, the state the director acts on."""
    app = NarrativeApplication(container, test_config, catalog=catalog)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    catalog.set_confirmed(world.world_id)
    return world.world_id


def _directive_json(**overrides) -> str:
    payload = {
        "reason": "导演要放个消息",
        "feasible": True,
        "refusal": "",
        "broadcast": {"content": "钟声大作。", "severity": "medium",
                      "location_scope": None, "phenomenon": "none"},
        "message": None,
        "mutations": [],
        "narrative_desc": "钟声大作",
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


async def _seed_intervention(container, world_id: str, *, step: int, authored_by: str = "director"):
    """Put one already-landed injection on the record, in the shape the runtime writes."""
    await container.snapshot.save(world_id, step, WorldSnapshot(
        world_id=world_id,
        step=step,
        timestamp=datetime.now(timezone.utc),
        world_time={"iso": f"step={step:04d} time=19:00", "label": "武德九年，六月初一"},
        events_this_step=[{
            "id": "ev-1",
            "step": step,
            "narrative_desc": "钟声大作",
            "authored_by": authored_by,
            "directive_text": "敲响全城的钟",
            "receipt": {"delivered_to": [{"agent_id": "a1", "name": "李世民"}],
                        "pressure": [], "decided": [], "interrupted": []},
        }],
    ))


@pytest.mark.asyncio
async def test_directive_is_accepted_and_queued(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id = await _built_world(mock_build_container, test_config, catalog)
    mock_build_container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json()

    with _client(mock_build_container, test_config, catalog) as client:
        res = client.post(f"/api/worlds/{world_id}/direct", json={"text": "敲响全城的钟"})

    assert res.status_code == 200
    body = res.json()
    assert body["accepted"] is True
    assert body["preview"] == "钟声大作"
    assert body["queued"] == 1


@pytest.mark.asyncio
async def test_unfeasible_directive_is_a_200_with_a_reason_not_an_error(
    mock_build_container, test_config,
) -> None:
    """"Can't be carried out" is a normal answer, not a client error: the directive just needs
    stating more clearly. The reason is returned verbatim."""
    catalog = WorldCatalog(None)
    world_id = await _built_world(mock_build_container, test_config, catalog)
    mock_build_container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json(
        feasible=False, refusal="没说清是谁,请指名道姓。", broadcast=None,
    )

    with _client(mock_build_container, test_config, catalog) as client:
        res = client.post(f"/api/worlds/{world_id}/direct", json={"text": "让他去那边"})

    assert res.status_code == 200
    assert res.json()["accepted"] is False
    assert res.json()["reason"] == "没说清是谁,请指名道姓。"


@pytest.mark.asyncio
async def test_empty_and_overlong_directives_are_rejected_before_the_llm(
    mock_build_container, test_config,
) -> None:
    catalog = WorldCatalog(None)
    world_id = await _built_world(mock_build_container, test_config, catalog)

    with _client(mock_build_container, test_config, catalog) as client:
        assert client.post(f"/api/worlds/{world_id}/direct", json={"text": "  "}).status_code == 400
        # A few thousand characters of fiction isn't a directive; putting it into the parse prompt
        # only invites improvisation.
        long = client.post(f"/api/worlds/{world_id}/direct", json={"text": "字" * 501})
        assert long.status_code == 400


@pytest.mark.asyncio
async def test_directive_on_a_missing_world_is_404(mock_build_container, test_config) -> None:
    with _client(mock_build_container, test_config, WorldCatalog(None)) as client:
        res = client.post("/api/worlds/nope/direct", json={"text": "放把火"})
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_step_route_starts_the_advance_and_reports_run_state(
    mock_build_container, test_config,
) -> None:
    """The route wires single-stepping to the application and reports the run state.

    Don't assert "step 1 is on disk" here: advancing is a fire-and-forget background task and
    sleeping for it makes the test flaky. ``test_run_control.py`` checks the advance at the
    application layer, where the task can be awaited.
    """
    catalog = WorldCatalog(None)
    world_id = await _built_world(mock_build_container, test_config, catalog)

    with _client(mock_build_container, test_config, catalog) as client:
        res = client.post(f"/api/worlds/{world_id}/step")

    assert res.status_code == 200
    assert res.json()["world_id"] == world_id
    assert res.json()["run_state"] is not None


@pytest.mark.asyncio
async def test_step_on_a_missing_world_is_404(mock_build_container, test_config) -> None:
    with _client(mock_build_container, test_config, WorldCatalog(None)) as client:
        assert client.post("/api/worlds/nope/step").status_code == 404


@pytest.mark.asyncio
async def test_an_accepted_directive_advances_a_world_that_would_not_move(
    mock_build_container, test_config, monkeypatch,
) -> None:
    """Accepting a directive means applying it: a queued directive nobody picks up leaves the user
    waiting at a map that never moves."""
    catalog = WorldCatalog(None)
    app = NarrativeApplication(mock_build_container, test_config, catalog=catalog)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    mock_build_container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json()
    stepped: list[str] = []
    monkeypatch.setattr(app, "step_world", stepped.append)

    result = await app.submit_directive(world.world_id, "敲响全城的钟")

    assert result.accepted is True
    assert stepped == [world.world_id]


@pytest.mark.asyncio
async def test_a_refused_directive_does_not_move_the_world(
    mock_build_container, test_config, monkeypatch,
) -> None:
    """Nothing was injected, so there's no reason to touch the world."""
    catalog = WorldCatalog(None)
    app = NarrativeApplication(mock_build_container, test_config, catalog=catalog)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    mock_build_container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json(
        feasible=False, refusal="没说清是谁。", broadcast=None,
    )
    stepped: list[str] = []
    monkeypatch.setattr(app, "step_world", stepped.append)

    await app.submit_directive(world.world_id, "让他去那边")

    assert stepped == []


@pytest.mark.asyncio
async def test_directing_a_running_world_does_not_cut_its_run_short(
    mock_build_container, test_config, monkeypatch,
) -> None:
    """``step_world`` means "take one step, then stop". Calling it on a world that is six steps into
    a ten-step run would end that run at step seven, when nothing needed doing: the next step was
    already on its way."""
    catalog = WorldCatalog(None)
    app = NarrativeApplication(mock_build_container, test_config, catalog=catalog)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    mock_build_container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json()
    app.get_session(world.world_id).status = RunState.RUNNING
    stepped: list[str] = []
    monkeypatch.setattr(app, "step_world", stepped.append)

    assert (await app.submit_directive(world.world_id, "敲响全城的钟")).accepted is True
    assert stepped == []


@pytest.mark.asyncio
async def test_directing_a_paused_world_advances_it_by_exactly_one_step(
    mock_build_container, test_config, monkeypatch,
) -> None:
    """A ten-step run paused at step six advances to step seven after an injection, then stays
    paused.

    Pausing means "look before advancing" (pause, inject, step once, look). ``step_once`` lets one
    beat through and parks again; the remaining steps survive (checked in ``test_run_control.py``).
    """
    catalog = WorldCatalog(None)
    app = NarrativeApplication(mock_build_container, test_config, catalog=catalog)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    mock_build_container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json()
    app.get_session(world.world_id).status = RunState.PAUSED
    stepped: list[str] = []
    monkeypatch.setattr(app, "step_world", stepped.append)

    await app.submit_directive(world.world_id, "敲响全城的钟")

    assert stepped == [world.world_id]


@pytest.mark.asyncio
async def test_the_record_of_past_directives_is_readable(
    mock_build_container, test_config,
) -> None:
    """"Which directives have I given this world" must be answerable, and the narrative feed can't
    answer it: it only holds steps you watched live. This route reads everything from snapshots.

    This test covers route → Replayer → serialization; ``test_director_runtime_integration.py``
    checks through a real step loop that the director's words land in the snapshot.
    """
    catalog = WorldCatalog(None)
    world_id = await _built_world(mock_build_container, test_config, catalog)
    await _seed_intervention(mock_build_container, world_id, step=3)

    with _client(mock_build_container, test_config, catalog) as client:
        res = client.get(f"/api/worlds/{world_id}/directives")

    assert res.status_code == 200
    body = res.json()
    assert [it["directive_text"] for it in body] == ["敲响全城的钟"]
    assert body[0]["narrative"] == "钟声大作"
    # No receipt: "who it moved" is unreadable outside the step where it landed, so it stays on
    # the feed card next to that step's actions.
    assert "receipt" not in body[0]
    # Times use the world's own calendar. "Step N" is a scheduling coordinate and never goes into a
    # human-readable field.
    assert body[0]["time_label"] == "武德九年，六月初一"


@pytest.mark.asyncio
async def test_the_llm_editors_own_twists_are_not_my_interventions(
    mock_build_container, test_config,
) -> None:
    """This record answers "what did I do", not "what was injected": events the LLM editor wrote
    share the snapshot stream (told apart by ``authored_by``) and would bury the human's entries.
    """
    catalog = WorldCatalog(None)
    world_id = await _built_world(mock_build_container, test_config, catalog)
    await _seed_intervention(mock_build_container, world_id, step=3, authored_by="system")

    with _client(mock_build_container, test_config, catalog) as client:
        res = client.get(f"/api/worlds/{world_id}/directives")

    assert res.json() == []


@pytest.mark.asyncio
async def test_a_refused_directive_leaves_nothing_in_the_record(
    mock_build_container, test_config,
) -> None:
    """A rejected directive made nothing happen, so it must not appear in the world's record:
    replay would show something that never happened. The rejection reason is shown in the moment.
    """
    catalog = WorldCatalog(None)
    world_id = await _built_world(mock_build_container, test_config, catalog)
    mock_build_container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json(
        feasible=False, refusal="没说清是谁。", broadcast=None,
    )

    with _client(mock_build_container, test_config, catalog) as client:
        client.post(f"/api/worlds/{world_id}/direct", json={"text": "让他去那边"})
        res = client.get(f"/api/worlds/{world_id}/directives")

    assert res.json() == []


@pytest.mark.asyncio
async def test_a_world_under_review_cannot_be_stepped_or_directed(
    mock_build_container, test_config,
) -> None:
    """Step and directive advance the world just as run does, so they share run's confirm gate."""
    catalog = WorldCatalog(None)
    app = NarrativeApplication(mock_build_container, test_config, catalog=catalog)
    world = await app.build_world("宫廷权谋", template="changan_iso")

    with _client(mock_build_container, test_config, catalog) as client:
        step = client.post(f"/api/worlds/{world.world_id}/step")
        direct = client.post(f"/api/worlds/{world.world_id}/direct", json={"text": "敲响全城的钟"})

    assert (step.status_code, direct.status_code) == (409, 409)
    assert await mock_build_container.snapshot.list_steps(world.world_id) == [0]


@pytest.mark.asyncio
async def test_a_reset_erases_the_directive_history_it_rolled_back(
    mock_build_container, test_config,
) -> None:
    """A reset deletes every step after 0, and the next run reuses their numbers: a directive the
    reset erased must not linger in the history, nor hide a new one at the same step."""
    catalog = WorldCatalog(None)
    world_id = await _built_world(mock_build_container, test_config, catalog)
    await _seed_intervention(mock_build_container, world_id, step=3)

    with _client(mock_build_container, test_config, catalog) as client:
        assert len(client.get(f"/api/worlds/{world_id}/directives").json()) == 1
        assert client.post(f"/api/worlds/{world_id}/reset").status_code == 200
        await _seed_intervention(mock_build_container, world_id, step=1)
        await _seed_intervention(mock_build_container, world_id, step=3)
        history = client.get(f"/api/worlds/{world_id}/directives").json()

    assert [r["step"] for r in history] == [1, 3]
