"""API layer tests: observe endpoints (graph/map/profile) + lifecycle + world delete."""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from core.interfaces.snapshot import WorldSnapshot
from engine.clock import WorldTime, WorldTimeConfig
from world import WorldCatalog
from world.catalog import MAX_WORLD_NAME_LEN
from worlds.tiled import EXAMPLE_PREFIX, list_templates

from interaction.api import create_app


def _client(container, test_config, *, catalog: WorldCatalog | None = None) -> TestClient:
    # In-memory catalog so tests never touch the on-disk registry.
    return TestClient(create_app(container, test_config, catalog=catalog or WorldCatalog(None)))


def _snapshot(world_id: str, step: int, relations: dict, *, profiles: dict | None = None) -> WorldSnapshot:
    return WorldSnapshot(
        world_id=world_id,
        step=step,
        timestamp=datetime.now(),
        world_time=WorldTime.from_step(step, WorldTimeConfig()).clock_payload(),
        agent_states={
            "agent-1": {"agent_id": "agent-1", "agent_name": "甲", "is_main_character": True, "current_location": "宫"},
            "agent-2": {"agent_id": "agent-2", "agent_name": "乙", "is_main_character": False, "current_location": "宫"},
        },
        agent_relations=relations,
        metadata={"character_profiles": profiles} if profiles else {},
    )


def _relation(from_id: str, to_id: str, *, affection: float, label: str) -> dict:
    return {
        "world_id": "world-api",
        "from_id": from_id,
        "to_id": to_id,
        "trust_objective": 0.6,
        "affection_objective": affection,
        "labels": [label] if label else [],
        "updated_step": 1,
        "history_summary": "",
        "interaction_count": 2,
    }


@pytest.mark.asyncio
async def test_graph_endpoint_reads_relations_for_requested_step(container, test_config) -> None:
    world_id = "world-api"
    await container.snapshot.save(world_id, 0, _snapshot(
        world_id, 0,
        {"agent-1->agent-2": _relation("agent-1", "agent-2", affection=0.5, label="盟友")},
        profiles={"agent-1": {"role": "皇子"}, "agent-2": {"role": "谋士"}},
    ))
    await container.snapshot.save(world_id, 1, _snapshot(
        world_id, 1,
        {"agent-1->agent-2": _relation("agent-1", "agent-2", affection=-0.6, label="政敌")},
    ))

    client = _client(container, test_config)
    step0 = client.get(f"/api/worlds/{world_id}/graph", params={"step": 0}).json()
    assert {n["id"] for n in step0["nodes"]} == {"agent-1", "agent-2"}
    assert step0["edges"][0]["labels"] == ["盟友"]
    latest = client.get(f"/api/worlds/{world_id}/graph").json()
    assert latest["edges"][0]["labels"] == ["政敌"]


@pytest.mark.asyncio
async def test_graph_endpoint_drops_neutral_perception_edges(container, test_config) -> None:
    # A baseline record (trust 0.5 / affection 0.0 / no labels / no interactions) is
    # persisted whenever two agents merely perceive each other; it is not a narrative
    # bond and must not clutter the graph. A genuine edge alongside it survives.
    world_id = "world-neutral"
    neutral = {
        "world_id": world_id, "from_id": "agent-1", "to_id": "agent-2",
        "trust_objective": 0.5, "affection_objective": 0.0,
        "labels": [], "updated_step": 0, "history_summary": "", "interaction_count": 0,
    }
    genuine = _relation("agent-2", "agent-1", affection=0.4, label="旧识")
    await container.snapshot.save(world_id, 0, _snapshot(
        world_id, 0,
        {"agent-1->agent-2": neutral, "agent-2->agent-1": genuine},
    ))

    client = _client(container, test_config)
    edges = client.get(f"/api/worlds/{world_id}/graph").json()["edges"]
    assert len(edges) == 1
    assert edges[0]["from_id"] == "agent-2" and edges[0]["labels"] == ["旧识"]


@pytest.mark.asyncio
async def test_map_endpoint_serves_frozen_per_world_copy(container, test_config) -> None:
    # A world's map is frozen at build; the endpoint must serve that copy verbatim,
    # never re-reading the (possibly since-edited) template. Prove it by freezing a
    # sentinel that no template contains.
    world_id = "world-map"
    frozen = {"tiledversion": "1.10", "layers": [], "sentinel": "FROZEN-AT-BUILD"}
    await container.snapshot.save_world_map(world_id, frozen)

    client = _client(container, test_config)
    got = client.get(f"/api/worlds/{world_id}/map").json()
    assert got["sentinel"] == "FROZEN-AT-BUILD"


@pytest.mark.asyncio
async def test_map_endpoint_falls_back_to_template_for_legacy_worlds(container, test_config) -> None:
    # A world with no per-world map copy falls back to the live template named
    # in its config, so its map still renders.
    world_id = "world-legacy"
    await container.snapshot.save_world_config(world_id, {"runtime_context": {"template": "changan_iso"}})

    client = _client(container, test_config)
    resp = client.get(f"/api/worlds/{world_id}/map")
    assert resp.status_code == 200
    assert "layers" in resp.json()  # a real tmj document


@pytest.mark.asyncio
async def test_delete_world_removes_it(container, test_config) -> None:
    world_id = "world-api"
    await container.snapshot.save(world_id, 0, _snapshot(world_id, 0, {}))
    client = _client(container, test_config)
    assert client.get(f"/api/worlds/{world_id}").status_code == 200
    assert client.delete(f"/api/worlds/{world_id}").json()["deleted"] is True
    assert client.get(f"/api/worlds/{world_id}").status_code == 404
    assert client.delete(f"/api/worlds/{world_id}").status_code == 404


@pytest.mark.asyncio
async def test_websocket_sends_initial_status_and_snapshot(container, test_config) -> None:
    # `from __future__ import annotations` + a function-local `WebSocket` import
    # makes FastAPI mis-read `websocket` as a query param and reject every
    # handshake (close 1008). The connection must accept and stream.
    world_id = "world-api"
    await container.snapshot.save(world_id, 0, _snapshot(world_id, 0, {}))
    client = _client(container, test_config)
    with client.websocket_connect(f"/ws/{world_id}") as ws:
        first = ws.receive_json()
        assert first["type"] == "status"
        second = ws.receive_json()
        assert second["type"] == "snapshot"
        assert second["data"]["step"] == 0


@pytest.mark.asyncio
async def test_a_step_published_while_connecting_is_not_lost(
    container, test_config, monkeypatch
) -> None:
    """There must be no window between "read the latest snapshot" and "subscribe" on connect —
    live mode doesn't backfill, so a missed step is missing for good.

    The initial-snapshot awaits are real file I/O. The publish is slipped inside that I/O (after
    a frame arrives the server has most likely subscribed already), so a later subscription
    necessarily drops it.

    On failure this test hangs rather than errors: only waiting proves the event wasn't lost, and
    a sentinel event can't bound the wait because under the race it also lands before the
    subscription.
    """
    from interaction.world_manager import WorldManager

    world_id = "world-api"
    await container.snapshot.save(world_id, 0, _snapshot(world_id, 0, {}))

    original = WorldManager.list_steps

    async def publish_mid_flight(self, wid: str):
        # The engine advances a step just while the initial frame is still being sent.
        container.event_bus.publish(
            {"type": "step", "world_id": world_id, "step": 1, "world_time": {}, "agents": {}}
        )
        return await original(self, wid)

    monkeypatch.setattr(WorldManager, "list_steps", publish_mid_flight)

    client = _client(container, test_config)
    with client.websocket_connect(f"/ws/{world_id}") as ws:
        assert ws.receive_json()["type"] == "status"
        assert ws.receive_json()["type"] == "snapshot"
        pushed = ws.receive_json()
        assert pushed["type"] == "step" and pushed["data"]["step"] == 1


@pytest.mark.asyncio
async def test_the_step_the_initial_snapshot_carried_is_not_pushed_twice(
    container, test_config
) -> None:
    """Subscribing first costs a little overlap: the pump needn't re-send the step the initial
    snapshot already delivered."""
    world_id = "world-api"
    await container.snapshot.save(world_id, 0, _snapshot(world_id, 0, {}))
    client = _client(container, test_config)
    with client.websocket_connect(f"/ws/{world_id}") as ws:
        assert ws.receive_json()["type"] == "status"
        assert ws.receive_json()["type"] == "snapshot"      # step 0
        container.event_bus.publish(
            {"type": "step", "world_id": world_id, "step": 0, "world_time": {}, "agents": {}}
        )
        container.event_bus.publish(
            {"type": "step", "world_id": world_id, "step": 1, "world_time": {}, "agents": {}}
        )
        nxt = ws.receive_json()
        assert nxt["type"] == "step" and nxt["data"]["step"] == 1   # 0 is skipped


@pytest.mark.asyncio
async def test_agent_profile_returns_static_identity(container, test_config) -> None:
    world_id = "world-api"
    await container.snapshot.save(world_id, 0, _snapshot(
        world_id, 0, {},
        profiles={"agent-1": {
            "name": "甲", "role": "皇子", "age": 24, "gender": "男",
            "background": "自幼习武", "core_traits": ["果决", "隐忍"],
            "core_values": ["社稷", "成王败寇"], "self_image": "我乃天策上将",
            "life_goal": "夺取帝国最高权力", "is_main_character": True,
        }},
    ))
    client = _client(container, test_config)
    prof = client.get(f"/api/worlds/{world_id}/agents/agent-1/profile").json()
    assert prof["name"] == "甲" and prof["role"] == "皇子"
    assert prof["core_traits"] == ["果决", "隐忍"] and prof["is_main_character"] is True
    assert prof["core_values"] == ["社稷", "成王败寇"]
    assert prof["self_image"] == "我乃天策上将" and prof["life_goal"] == "夺取帝国最高权力"
    assert prof["age"] == 24 and prof["gender"] == "男"
    assert client.get(f"/api/worlds/{world_id}/agents/nobody/profile").status_code == 404


@pytest.mark.asyncio
async def test_agent_roster_carries_demographics(container, test_config) -> None:
    """The cast roster carries each agent's gender and age.

    These two fields alone pick a figure's body on the map (frontend/src/phaser/skins.ts),
    and nothing else reads them. Losing them fails silently: every agent falls through to the
    default adult male body and nothing errors.
    """
    world_id = "world-api"
    await container.snapshot.save(world_id, 0, _snapshot(
        world_id, 0, {},
        profiles={
            "agent-1": {"name": "甲", "role": "皇子", "age": 24, "gender": "男"},
            "agent-2": {"name": "乙", "role": "女官", "age": 61, "gender": "女"},
        },
    ))
    roster = _client(container, test_config).get(f"/api/worlds/{world_id}/agents").json()
    assert [(p["age"], p["gender"]) for p in roster] == [(24, "男"), (61, "女")]


@pytest.mark.asyncio
async def test_agent_profile_reads_character_profiles_whole(container, test_config) -> None:
    world_id = "world-api"
    snap = _snapshot(
        world_id, 0, {},
        profiles={"agent-1": {
            "name": "甲", "role": "皇子", "core_traits": ["果决"],
            "core_values": ["社稷", "成王败寇"], "life_goal": "夺取帝国最高权力",
            "self_image": "我乃天策上将",
        }},
    )
    await container.snapshot.save(world_id, 0, snap)
    prof = _client(container, test_config).get(f"/api/worlds/{world_id}/agents/agent-1/profile").json()
    assert prof["core_traits"] == ["果决"]
    assert prof["core_values"] == ["社稷", "成王败寇"]
    assert prof["life_goal"] == "夺取帝国最高权力" and prof["self_image"] == "我乃天策上将"


@pytest.mark.asyncio
async def test_confirm_locks_world(container, test_config) -> None:
    world_id = "world-api"
    await container.snapshot.save(world_id, 0, _snapshot(world_id, 0, {}))
    catalog = WorldCatalog(None)
    catalog.register(world_id, theme="宫廷", world_name="长安")
    client = _client(container, test_config, catalog=catalog)

    assert client.get(f"/api/worlds/{world_id}").json()["confirmed"] is False
    assert client.post(f"/api/worlds/{world_id}/confirm").json()["confirmed"] is True
    assert client.get(f"/api/worlds/{world_id}").json()["confirmed"] is True
    # Idempotent, and 404 for unknown worlds.
    assert client.post(f"/api/worlds/{world_id}/confirm").json()["confirmed"] is True
    assert client.post("/api/worlds/nope/confirm").status_code == 404


@pytest.mark.asyncio
async def test_rename_world_changes_only_the_label(container, test_config) -> None:
    """Renaming changes the display name in the catalog — the world itself (the build-time frozen
    analysis in the snapshot) is untouched, as are the entry's other fields: a confirmed world is
    still confirmed after a rename."""
    world_id = "world-api"
    await container.snapshot.save(world_id, 0, _snapshot(world_id, 0, {}))
    catalog = WorldCatalog(None)
    catalog.register(world_id, theme="宫廷", world_name="长安")
    client = _client(container, test_config, catalog=catalog)
    client.post(f"/api/worlds/{world_id}/confirm")

    renamed = client.patch(f"/api/worlds/{world_id}", json={"world_name": "  武德九年  "})
    assert renamed.status_code == 200
    # Whitespace trimmed, and the answer is the world's fresh metadata.
    assert renamed.json()["world_name"] == "武德九年"
    assert renamed.json()["confirmed"] is True and renamed.json()["theme"] == "宫廷"
    # Both read paths agree — the list is what the sidebar renders.
    assert client.get(f"/api/worlds/{world_id}").json()["world_name"] == "武德九年"
    listed = client.get("/api/worlds").json()
    assert [w["world_name"] for w in listed if w["world_id"] == world_id] == ["武德九年"]


@pytest.mark.asyncio
async def test_rename_world_rejects_empty_and_unknown(container, test_config) -> None:
    world_id = "world-api"
    await container.snapshot.save(world_id, 0, _snapshot(world_id, 0, {}))
    catalog = WorldCatalog(None)
    catalog.register(world_id, theme="宫廷", world_name="长安")
    client = _client(container, test_config, catalog=catalog)

    # A name is the user's own text: reject rather than invent or truncate one.
    assert client.patch(f"/api/worlds/{world_id}", json={"world_name": "   "}).status_code == 400
    assert client.patch(f"/api/worlds/{world_id}", json={}).status_code == 400
    assert client.patch(
        f"/api/worlds/{world_id}", json={"world_name": "长" * (MAX_WORLD_NAME_LEN + 1)}
    ).status_code == 400
    assert client.patch("/api/worlds/nope", json={"world_name": "x"}).status_code == 404
    # None of the rejections touched the stored name.
    assert client.get(f"/api/worlds/{world_id}").json()["world_name"] == "长安"


@pytest.mark.asyncio
async def test_run_blocked_until_confirmed(container, test_config) -> None:
    world_id = "world-api"
    await container.snapshot.save(world_id, 0, _snapshot(world_id, 0, {}))
    catalog = WorldCatalog(None)
    catalog.register(world_id, theme="宫廷", world_name="长安")
    client = _client(container, test_config, catalog=catalog)

    blocked = client.post(f"/api/worlds/{world_id}/run", json={"steps": 1})
    assert blocked.status_code == 409
    assert "confirm" in blocked.json()["detail"].lower()


@pytest.mark.asyncio
async def test_run_without_steps_advances_exactly_one_step(container, test_config, monkeypatch) -> None:
    """No duration given = run one step, never "run forever".

    Every step makes several LLM calls per character, so an unbounded default would let one
    missing field push the bill arbitrarily high without the caller seeing anything. 0 / negative
    / non-integer are all rejected and never silently become some other number.
    """
    from engine.application import NarrativeApplication

    world_id = "world-api"
    await container.snapshot.save(world_id, 0, _snapshot(world_id, 0, {}))
    catalog = WorldCatalog(None)
    catalog.register(world_id, theme="宫廷", world_name="长安")
    catalog.set_confirmed(world_id, True)

    requested: list[int] = []

    # A live session already: no restore, which needs a persisted world config.
    monkeypatch.setattr(NarrativeApplication, "has_session", lambda self, wid: True)
    monkeypatch.setattr(
        NarrativeApplication, "start_run",
        lambda self, wid, *, steps: requested.append(steps),
    )
    client = _client(container, test_config, catalog=catalog)
    run = lambda body: client.post(f"/api/worlds/{world_id}/run", json=body)

    assert run({}).status_code == 200
    assert run({"steps": None}).status_code == 200
    assert requested == [1, 1]

    # 2.7 / true / "3" aren't step counts: truncated to 2, quietly turned into 1, looks like a
    # number — running a rewritten request is far worse than rejecting it.
    for bad in (0, -5, "abc", 2.7, True, "3"):
        assert run({"steps": bad}).status_code == 400, bad
    assert requested == [1, 1]


@pytest.mark.asyncio
async def test_list_worlds_includes_run_state(container, test_config) -> None:
    world_id = "world-api"
    await container.snapshot.save(world_id, 0, _snapshot(world_id, 0, {}))
    catalog = WorldCatalog(None)
    catalog.register(world_id, theme="宫廷", world_name="长安")
    worlds = _client(container, test_config, catalog=catalog).get("/api/worlds").json()
    assert any(w["world_id"] == world_id and "run_state" in w for w in worlds)


@pytest.mark.asyncio
async def test_list_worlds_comes_oldest_first(container, test_config) -> None:
    """The endpoint returns entries in ascending creation time (the worlds' own order); id
    lexical order is irrelevant, and those without a time go last. "Newest first" is a display
    preference the client reverses itself."""
    catalog = WorldCatalog(None)
    catalog.register("a-newest", created_at=datetime(2026, 9, 3))
    catalog.register("b-undated")
    catalog.register("c-oldest", created_at=datetime(2026, 9, 1))
    catalog.register("d-middle", created_at=datetime(2026, 9, 2))

    worlds = _client(container, test_config, catalog=catalog).get("/api/worlds").json()

    assert [w["world_id"] for w in worlds] == ["c-oldest", "d-middle", "a-newest", "b-undated"]


@pytest.mark.asyncio
async def test_map_endpoint_serves_world_template_tmj(container, test_config) -> None:
    """The map API serves the same .tmj the world was built on, so the renderer
    and the engine share one map artifact (no bundled frontend copy)."""
    world_id = "world-map"
    await container.snapshot.save_world_config(world_id, {
        "entities": {"taiji_palace": {
            "entity_id": "taiji_palace", "name": "太极宫", "entity_type": "location",
            "connections": {"dongong": 1}, "capacity": 30,
        }},
        "runtime_context": {"template": "changan_iso"},
    })
    client = _client(container, test_config)
    tmj = client.get(f"/api/worlds/{world_id}/map").json()
    assert tmj["width"] == 48 and "layers" in tmj  # a real template document

    # Object layers may sit inside Tiled GROUPS, so the walk has to recurse —
    # exactly as worlds/tiled.py does. A flat scan finds nothing on a grouped map.
    def locations(layers: list) -> list:
        found = []
        for layer in layers:
            found += locations(layer.get("layers", []))
            found += [o for o in layer.get("objects", []) if o.get("type") == "location"]
        return found

    assert len(locations(tmj["layers"])) == 25  # Tang Chang'an, engine-schema'd
    # unknown world → 404
    assert _client(container, test_config).get("/api/worlds/nope/map").status_code == 404


@pytest.mark.asyncio
async def test_map_asset_endpoint_serves_tileset_art_and_refuses_to_escape(container, test_config) -> None:
    """The backend owns the map AND the art it names, so a template is a drop-in:
    ONE self-contained `<name>/` directory (map + art). The renderer asks by the relative path
    written in the .tmj and never has to know which template a world is on.

    The path arrives from the URL, so the traversal and suffix guards are the
    security boundary of this route, not a nicety."""
    world_id = "world-asset"
    await container.snapshot.save_world_config(world_id, {
        "entities": {},
        "runtime_context": {"template": "changan_iso"},
    })
    client = _client(container, test_config)

    # The index says where each image the map NAMES is reachable — the client
    # reads URLs from here instead of composing them, so the art can move.
    index = client.get(f"/api/worlds/{world_id}/map/assets").json()
    tmj = client.get(f"/api/worlds/{world_id}/map").json()
    named = {ts["image"] for ts in tmj["tilesets"]}
    assert set(index) == named and len(named) == 15

    ok = client.get(index["tilesets/base/ground/ground.png"])
    assert ok.status_code == 200
    assert ok.content[:8] == b"\x89PNG\r\n\x1a\n"  # real bytes, not an error page
    for url in index.values():  # every advertised URL actually resolves
        assert client.get(url).status_code == 200

    # Holding the bytes already must cost 0 bytes, not a re-download. Map art is
    # megabytes and re-requested on every visit; without this the client pays for
    # all of it every time (Starlette emits an ETag but never answers a
    # conditional request).
    url = index["tilesets/base/ground/ground.png"]
    etag = ok.headers["etag"]
    again = client.get(url, headers={"If-None-Match": etag})
    assert again.status_code == 304
    assert not again.content
    # A client holding a DIFFERENT version still gets the real bytes.
    assert client.get(url, headers={"If-None-Match": '"stale"'}).status_code == 200

    # …and every way out of the template directory is closed.
    for escape in (
        "../changan_iso/map.tmj",                 # a sibling template's map data
        "../../tiled.py",                     # source, two levels up
        "tilesets/../../../config/config.yaml",  # traversal buried mid-path
    ):
        assert client.get(f"/api/worlds/{world_id}/map/assets/{escape}").status_code == 404
    assert client.get(f"/api/worlds/{world_id}/map/assets/nope.png").status_code == 404


@pytest.mark.asyncio
async def test_frozen_assets_win_over_the_live_template(container, test_config) -> None:
    """A world's map is frozen at build, so the art it names must be frozen with it.

    Otherwise the freeze is only half done: the world keeps its own geometry but
    reads it through whatever the template's tilesets look like TODAY, and a tile
    that moved within a sheet renders as garbage. Prove the precedence by freezing
    a sentinel image that no template contains."""
    world_id = "world-frozen-art"
    sentinel = b"\x89PNG\r\n\x1a\n" + b"FROZEN-AT-BUILD"
    await container.snapshot.save_world_config(world_id, {
        "entities": {}, "runtime_context": {"template": "changan_iso"},
    })
    await container.snapshot.save_world_asset(
        world_id, "tilesets/base/ground/ground.png", sentinel
    )

    client = _client(container, test_config)
    got = client.get(f"/api/worlds/{world_id}/map/assets/tilesets/base/ground/ground.png")
    assert got.status_code == 200
    assert got.content == sentinel, "served the live template instead of the frozen copy"

    # Frozen art is written once and addressed by world id, so it can never change
    # under this URL — the client should stop asking entirely rather than pay a
    # revalidation round trip per tileset on every visit.
    assert "immutable" in got.headers["cache-control"]
    # Live template art is still edited by hand, so it only earns revalidation.
    live = client.get(f"/api/worlds/{world_id}/map/assets/tilesets/base/roads/roads.png")
    assert live.headers["cache-control"] == "no-cache"

    # An asset this world never froze still falls back to the template, so a world
    # without frozen art keeps rendering.
    fallback = client.get(f"/api/worlds/{world_id}/map/assets/tilesets/base/roads/roads.png")
    assert fallback.status_code == 200 and fallback.content[:8] == b"\x89PNG\r\n\x1a\n"


# ---- map template catalogue + per-world template selection -------------------


def test_templates_endpoint_lists_every_installed_map(container, test_config) -> None:
    """A map is available because its directory is on disk — that IS the registration.

    The endpoint is what makes a dropped-in map reachable without a frontend or
    config change, so it must describe each template well enough to pick from
    (name, era, how big a world it is) and must not know any map by name.
    """
    entries = _client(container, test_config).get("/api/templates").json()

    assert entries, "no map templates served"
    by_name = {entry["template"]: entry for entry in entries}
    assert set(by_name) == set(list_templates())
    for entry in entries:
        assert entry["world_name"], f"{entry['template']}: nothing to show in a picker"
        assert entry["era_name"]
        assert entry["description"]
        assert entry["location_count"] > 0


def test_reference_maps_never_reach_the_picker(tmp_path, monkeypatch) -> None:
    """A reference map is held to the whole contract but is not a place a story
    happens, so it must never reach the list a user picks a world from.

    Planted rather than read off disk: a test that reads disk quietly stops proving
    anything once the last installed sample is removed."""
    from worlds import tiled

    monkeypatch.setattr(tiled, "TEMPLATES_DIR", tmp_path)
    for name in ("shelved_city", f"{EXAMPLE_PREFIX}example_reference"):
        (tmp_path / name).mkdir()
        (tmp_path / name / tiled.MAP_FILENAME).write_text("{}", encoding="utf-8")

    assert tiled.list_templates() == ["shelved_city"]
    assert tiled.list_templates(include_examples=True) == [
        f"{EXAMPLE_PREFIX}example_reference",
        "shelved_city",
    ]


def test_create_world_rejects_an_unknown_template(container, test_config) -> None:
    """Caught at the request, not minutes later inside a background build.

    The build runs as a job the client polls, so a template that does not exist
    would otherwise surface as a failed job long after the request succeeded —
    and a traversal-shaped name must never reach the filesystem at all.
    """
    client = _client(container, test_config)
    for bogus in ("does-not-exist", "../config", "changan_iso/../changan_iso"):
        response = client.post("/api/worlds", json={"theme": "任意主题", "template": bogus})
        assert response.status_code == 400, bogus
        assert "template" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_build_world_offers_the_right_maps_to_choose_between(container, test_config) -> None:
    """Naming a map settles it; naming none puts every map up for the choice.

    Two worlds in one deployment can sit on different maps, so the map is picked per
    build rather than per config. With nothing named, the whole catalogue goes to the
    builder, which picks from the theme.
    """
    from engine.application import NarrativeApplication

    application = NarrativeApplication(container, test_config, catalog=WorldCatalog(None))
    seen: list[list] = []

    async def capture(*, theme, world_configs=None, **kwargs):
        seen.append(list(world_configs or []))
        raise RuntimeError("stop once the candidates are assembled")

    application._builder.build = capture  # type: ignore[method-assign]
    installed = list_templates()

    for template in ("metro", None):
        with pytest.raises(RuntimeError):
            await application.build_world(
                "任意主题", template=template, available_templates=installed
            )

    named, unnamed = seen

    assert len(named) == 1, "a named map leaves nothing to choose between"
    assert named[0].to_runtime_context()["template"] == "metro"

    assert [c.to_runtime_context()["template"] for c in unnamed] == installed, (
        "with no map named, every installed map must be a candidate"
    )


@pytest.mark.asyncio
async def test_build_world_with_no_map_available_fails(container, test_config) -> None:
    """Building a world with no map to choose must fail, not fall back to some default map.

    An empty candidate list has two origins — no map installed, or the caller didn't list the
    installed ones — and both are config errors. Falling back to a default map would build the
    world on a base nobody chose, and the base underlies every later step.
    """
    from engine.application import NarrativeApplication

    application = NarrativeApplication(container, test_config, catalog=WorldCatalog(None))
    with pytest.raises(ValueError, match="No world templates"):
        await application.build_world("任意主题", available_templates=[])


# ---- editable interface content ----------------------------------------------


def test_content_endpoint_serves_the_authored_presets(container, test_config) -> None:
    """Sample themes are content, authored in a file rather than compiled in.

    Each one has to be usable as-is: a title for the button and the full theme it
    puts in the box. A half-filled entry would render an unlabelled button or
    drop an empty prompt into the form, so it is dropped instead of shown.
    """
    presets = _client(container, test_config).get("/api/content").json()["presets"]

    assert presets, "no sample themes served"
    for preset in presets:
        assert preset["title"].strip(), preset
        assert preset["theme"].strip(), preset


def test_content_is_reread_per_request(container, test_config, tmp_path, monkeypatch) -> None:
    """An edit shows up on the next page load — no rebuild, no restart.

    That is the entire reason this text lives in a mounted file instead of in the
    frontend bundle, so caching it would spend the only thing it bought.
    """
    from config.content import CONTENT_ENV_VAR

    path = tmp_path / "content.yaml"
    monkeypatch.setenv(CONTENT_ENV_VAR, str(path))
    client = _client(container, test_config)

    path.write_text("presets:\n  - title: 甲\n    theme: 第一版主题\n", encoding="utf-8")
    assert client.get("/api/content").json()["presets"] == [
        {"title": "甲", "theme": "第一版主题"}
    ]

    path.write_text("presets:\n  - title: 乙\n    theme: 改过的主题\n", encoding="utf-8")
    assert client.get("/api/content").json()["presets"] == [
        {"title": "乙", "theme": "改过的主题"}
    ]


@pytest.mark.parametrize(
    "body",
    [
        "",                                    # empty file
        "presets: []",                         # authored as nothing
        "presets:\n  - title: 只有标题\n",       # no theme → unusable
        "presets:\n  - theme: 只有主题\n",       # no title → unlabelled
        "presets: 这不是一个列表",                # wrong shape
        "presets:\n  - [unclosed",             # not even YAML
    ],
)
def test_broken_content_degrades_instead_of_breaking_the_home_screen(
    container, test_config, tmp_path, monkeypatch, body: str
) -> None:
    """A typo in a list of sample prompts must not take the home screen down.

    It is logged as an error for someone to fix, but the endpoint still answers:
    losing the suggestions costs a convenience, losing the page costs everything.
    """
    from config.content import CONTENT_ENV_VAR

    path = tmp_path / "content.yaml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv(CONTENT_ENV_VAR, str(path))

    response = _client(container, test_config).get("/api/content")
    assert response.status_code == 200
    assert response.json()["presets"] == []


def test_missing_content_file_is_simply_no_content(
    container, test_config, tmp_path, monkeypatch
) -> None:
    """The file is optional — a deployment that ships none just shows fewer things."""
    from config.content import CONTENT_ENV_VAR

    monkeypatch.setenv(CONTENT_ENV_VAR, str(tmp_path / "nope.yaml"))
    assert _client(container, test_config).get("/api/content").json()["presets"] == []


def test_server_shutdown_lets_running_worlds_finish_their_step(
    container, test_config, monkeypatch,
) -> None:
    """The app's shutdown hands control to NarrativeApplication.shutdown, which stops runs at a
    step boundary instead of leaving them to be cancelled mid-step."""
    from engine.application import NarrativeApplication

    called: list[bool] = []

    async def _shutdown(self) -> None:
        called.append(True)

    monkeypatch.setattr(NarrativeApplication, "shutdown", _shutdown)
    with _client(container, test_config):
        assert called == []
    assert called == [True]


@pytest.mark.asyncio
async def test_advancing_a_world_reads_its_metadata_once() -> None:
    """The run / step / directive gate reads the world's metadata once, not again for the session."""
    from types import SimpleNamespace

    from interaction.api.app import ApiServices

    reads: list[str] = []

    async def get_world(world_id: str):
        reads.append(world_id)
        return SimpleNamespace(confirmed=True)

    services = ApiServices(
        container=SimpleNamespace(model_keys=True),
        config=None,
        application=SimpleNamespace(has_session=lambda wid: True),
        manager=SimpleNamespace(get_world=get_world),
    )

    await services.ensure_advanceable("w")

    assert reads == ["w"]
