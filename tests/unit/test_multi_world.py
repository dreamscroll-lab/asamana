"""Multi-world concurrency: isolation of events, config copies, and stores."""

from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

from core.container import Container
from core.event_bus import NarrativeEventBus
from core.serialization import dump_json
from engine.application import NarrativeApplication
from providers.message.in_memory import InMemoryMessageProvider
from providers.snapshot.file import FileSnapshotProvider
from providers.snapshot.in_memory import InMemorySnapshotProvider
from world.initializer import WorldInitializer
from engine.environment import UNPLACED
from core.interfaces.place import Place
from world.stored_config import StoredWorldConfig, serialize_world_config
from worlds.tiled import TiledWorldConfig
from tests.unit.bus_tap import tap


@pytest.fixture()
def container(mock_build_container: Container) -> Container:
    """Multi-world tests build full worlds: alias the shared mock-build container."""
    return mock_build_container


def test_a_subscriber_that_stopped_reading_cannot_grow_the_bus_without_bound() -> None:
    """Subscriber queues are bounded and drop the oldest entry when full.

    Observation is a transient view. Dropping a frame is better than exhausting process memory, and
    snapshots hold the durable record.
    """
    from core.event_bus import _SUBSCRIBER_QUEUE_SIZE as CAP

    bus = NarrativeEventBus()
    queue = bus.subscribe("w")
    for step in range(CAP + 50):
        bus.publish({"type": "step", "world_id": "w", "step": step})

    assert queue.qsize() == CAP
    # The newest entries are kept: the oldest are evicted rather than the newest refused.
    assert queue.get_nowait()["step"] == 50


def test_one_worlds_event_storm_cannot_evict_another_worlds_frames() -> None:
    """A subscriber receives only its own world's events, or a burst from another world would evict
    its frames.

    The queue is capped and drops the head when full, so in a queue shared by every world the
    evicted item may be this subscriber's own undelivered event. Live view doesn't backfill, so that
    step is lost for the session (the map jumps from N to N+2).
    """
    from core.event_bus import _SUBSCRIBER_QUEUE_SIZE as CAP

    bus = NarrativeEventBus()
    watching_b = bus.subscribe("world-b")

    bus.publish({"type": "step", "world_id": "world-b", "step": 1})
    for step in range(CAP * 2):                      # burst from world-a
        bus.publish({"type": "step", "world_id": "world-a", "step": step})

    assert watching_b.qsize() == 1
    got = watching_b.get_nowait()
    assert (got["world_id"], got["step"]) == ("world-b", 1)


def test_a_subscriber_can_still_take_every_world() -> None:
    """Without a world, a subscriber receives everything. This is for process-level observers and
    tests."""
    bus = NarrativeEventBus()
    everything = bus.subscribe()
    bus.publish({"type": "step", "world_id": "world-a", "step": 1})
    bus.publish({"type": "step", "world_id": "world-b", "step": 1})
    assert everything.qsize() == 2


def test_unsubscribe_drops_only_that_queue() -> None:
    bus = NarrativeEventBus()
    first = bus.subscribe("w")
    second = bus.subscribe("w")
    bus.unsubscribe(first)
    bus.publish({"type": "step", "world_id": "w", "step": 1})
    assert first.qsize() == 0
    assert second.qsize() == 1


# ---------------------------------------------------------------------------
# WorldConfig copies + persisted asset
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_copies_template_and_leaves_it_pristine(container, test_config) -> None:
    """initialize() gives each world a private entity copy; build-time placement
    (which mutates location agent_ids by reference) never reaches the template."""
    application = NarrativeApplication(container, test_config)
    template = TiledWorldConfig(template="changan_iso")
    entity_id = next(iter(template.get_places()))

    world = await application.build_world("玄武门之变", template="changan_iso")

    assert (
        world.world_config.get_places()[entity_id]
        is not template.get_places()[entity_id]
    )
    # Occupancy can't be stored in this base (``WorldEntity`` has no such field), so step-0
    # occupants can't be baked in here. The guard for that lives in test_environment.


def test_stored_world_config_round_trips_template_data(container) -> None:
    template = TiledWorldConfig(template="changan_iso")
    data = json.loads(dump_json(serialize_world_config(template)))  # full JSON round trip
    stored = StoredWorldConfig(data)

    assert stored.get_places() == template.get_places()
    assert stored.get_location_aliases() == template.get_location_aliases()
    assert stored.get_world_description() == template.get_world_description()
    assert stored.to_runtime_context() == template.to_runtime_context()

    aliases = template.get_location_aliases()
    probes = list(template.get_places())[:2] + list(aliases)[:2] + ["不存在的地方"]
    for probe in probes:
        assert stored.resolve_location_id(probe) == template.resolve_location_id(probe)


@pytest.mark.asyncio
async def test_restore_uses_persisted_config_not_current_template(container, test_config) -> None:
    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")

    # Edit the template. ``Place`` is frozen (space is fixed at world build), so the only way to
    # edit it is to replace the entry.
    tainted = TiledWorldConfig(template="changan_iso")
    place_id = next(iter(tainted.get_places()))
    original_name = tainted.get_places()[place_id].name
    tainted._places_cache[place_id] = dataclasses.replace(
        tainted.get_places()[place_id], name="模板被改后的名字",
    )

    restored = await WorldInitializer(container).restore(world.world_id)

    assert restored.world_config.get_places()[place_id].name == original_name


@pytest.mark.asyncio
async def test_restore_does_not_resurrect_baked_in_occupancy(container, test_config) -> None:
    """After restore, an agent stands in exactly one place: the one recorded in his save.

    An agent who moved and ends up in both his old and new places on restore breaks both visibility
    and capacity checks. The environment is the only source of truth for occupancy and isn't
    persisted with the world config assets, so this checks the environment's final state.
    """
    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")
    await application.run_world(world.world_id, steps=1)

    # Simulate the agent having ended the run in a different location.
    agent_id = next(iter(world.agents))
    stored = await container.agent_store.load_agent_state(world.world_id, agent_id)
    old_location = stored.current_location
    other_location = next(
        eid
        for eid, e in world.world_config.get_places().items()
        if isinstance(e, Place) and eid != old_location
    )
    stored.current_location = other_location
    await container.agent_store.save_agent_state(world.world_id, agent_id, stored)

    restored = await WorldInitializer(container).restore(world.world_id)
    env = restored.environment
    assert env.get_body_location(agent_id) == other_location
    assert agent_id not in env.bodies_at(old_location)
    assert agent_id in env.bodies_at(other_location)


@pytest.mark.asyncio
async def test_restore_without_persisted_config_raises(container) -> None:
    initializer = WorldInitializer(container)
    with pytest.raises(ValueError, match="No persisted world config"):
        await initializer.restore("never-built-world")


@pytest.mark.asyncio
async def test_snapshot_providers_world_config_asset(tmp_path, container) -> None:
    data = serialize_world_config(TiledWorldConfig(template="changan_iso"))

    in_memory = InMemorySnapshotProvider()
    assert await in_memory.load_world_config("w") is None
    await in_memory.save_world_config("w", data)
    assert await in_memory.load_world_config("w") == data

    file_provider = FileSnapshotProvider(path=str(tmp_path))
    await file_provider.save_world_config("w", data)
    target = tmp_path / "w" / "world_config.json"
    assert target.exists()
    assert not (tmp_path / "w" / "world_config.json.tmp").exists()
    loaded = await file_provider.load_world_config("w")
    assert StoredWorldConfig(loaded).get_places() == TiledWorldConfig(template="changan_iso").get_places()
    # The config asset is a world-lifetime asset: step cleanup must not delete it.
    await file_provider.delete_steps_after("w", 0)
    assert await file_provider.load_world_config("w") == loaded


# ---------------------------------------------------------------------------
# run_world reentry guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_world_rejects_concurrent_runs_of_same_world(container, test_config) -> None:
    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")

    results = await asyncio.gather(
        application.run_world(world.world_id, steps=1),
        application.run_world(world.world_id, steps=1),
        return_exceptions=True,
    )
    errors = [r for r in results if isinstance(r, BaseException)]
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert "already running" in str(errors[0])

    # The latch resets after the run: the world can run again.
    await application.run_world(world.world_id, steps=1)


# ---------------------------------------------------------------------------
# Message provider partition
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_message_provider_isolates_worlds() -> None:
    from core.interfaces.message import Message

    def make(world_id: str, msg_id: str, deliver_step: int = 1) -> Message:
        return Message(
            id=msg_id,
            world_id=world_id,
            sender_id="sender",
            content="hello",
            recipients=["receiver"],
            location_scope=None,
            deliver_step=deliver_step,
            created_step=0,
        )

    provider = InMemoryMessageProvider()
    await provider.enqueue(make("world-a", "a-1"))
    await provider.enqueue(make("world-a", "a-2", deliver_step=9))
    await provider.enqueue(make("world-b", "b-1"))

    assert [m.id for m in await provider.dequeue_ready("world-a", 1)] == ["a-1"]
    assert [m.id for m in await provider.peek_pending("world-a")] == ["a-2"]
    assert [m.id for m in await provider.peek_pending("world-b")] == ["b-1"]

    await provider.clear("world-a")
    assert await provider.peek_pending("world-a") == []
    assert [m.id for m in await provider.peek_pending("world-b")] == ["b-1"]

    await provider.clear()
    assert await provider.peek_pending("world-b") == []


# ---------------------------------------------------------------------------
# Two worlds run concurrently without crosstalk
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_worlds_run_concurrently_without_crosstalk(container, test_config) -> None:
    application = NarrativeApplication(container, test_config)
    world_a = await application.build_world("玄武门之变", template="changan_iso")
    world_b = await application.build_world("贞观之治", template="changan_iso")
    assert world_a.world_id != world_b.world_id

    # Private config copies: same template entity, distinct objects per world.
    entity_id = next(iter(TiledWorldConfig(template="changan_iso").get_places()))
    assert (
        world_a.world_config.get_places()[entity_id]
        is not world_b.world_config.get_places()[entity_id]
    )

    published_a = tap(container.event_bus, world_a.world_id)
    published_b = tap(container.event_bus, world_b.world_id)
    results = await asyncio.gather(
        application.run_world(world_a.world_id, steps=2),
        application.run_world(world_b.world_id, steps=2),
        return_exceptions=True,
    )
    assert not [r for r in results if isinstance(r, BaseException)]

    # Each world's observer gets exactly its own steps, none of the other's.
    events_a = published_a()
    events_b = published_b()
    assert [e["step"] for e in events_a if e.get("type") == "step"] == [1, 2]
    assert [e["step"] for e in events_b if e.get("type") == "step"] == [1, 2]

    # Stores stay world-scoped: each world sees exactly its own snapshots and agents.
    assert await container.snapshot.list_steps(world_a.world_id) == [0, 1, 2]
    assert await container.snapshot.list_steps(world_b.world_id) == [0, 1, 2]
    assert len(await container.agent_store.list_agent_ids(world_a.world_id)) == len(world_a.agents)
    assert len(await container.agent_store.list_agent_ids(world_b.world_id)) == len(world_b.agents)


@pytest.mark.asyncio
async def test_restore_does_not_place_dead_agent(container, test_config) -> None:
    """Dead agents are not placed. Death handling already removed the body from the environment; an
    unconditional place on restore would put it back in the room, visible to everyone there for
    good, since death handling only processes this step's new deaths and would never clear it."""
    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")
    await application.run_world(world.world_id, steps=1)

    agent_id = next(iter(world.agents))
    stored = await container.agent_store.load_agent_state(world.world_id, agent_id)
    stored.vitality = 0.0
    await container.agent_store.save_agent_state(world.world_id, agent_id, stored)

    restored = await WorldInitializer(container).restore(world.world_id)
    assert restored.agents[agent_id].is_active is False
    env = restored.environment
    assert env.get_body_location(agent_id) == UNPLACED
    assert all(
        agent_id not in env.bodies_at(lid)
        for lid in restored.world_config.get_places()
    )


@pytest.mark.asyncio
async def test_restore_brings_the_mindless_bodies_back(container, test_config) -> None:
    """NPCs must still exist in a world that runs right after build. Only the environment snapshot
    restores them; there is no other path.

    Restore rebuilds entity seeds but not Npcs (rebuilding would collide with the restored ids).
    Skip this and the world has no NPCs from step one, so ERRAND silently can never reach one.
    """
    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")
    seeded = {npc.npc_id: npc.name for npc in world.environment.all_npcs()}
    assert seeded, "这个世界本来就没有工具人,测不到东西"

    restored = await WorldInitializer(
        container, max_npcs=test_config.engine.max_npcs
    ).restore(world.world_id)
    env = restored.environment

    assert {npc.npc_id for npc in env.all_npcs()} == set(seeded)
    for npc_id in seeded:
        where = env.get_body_location(npc_id)
        assert where, f"{seeded[npc_id]} 还原后不在任何地方"
        assert npc_id in env.bodies_at(where)
        assert env.is_npc(npc_id)


@pytest.mark.asyncio
async def test_restoring_at_step_zero_leaves_the_entities_exactly_as_seeded(
    container, test_config
) -> None:
    """Replaying the environment at step 0 must be idempotent for entities: the seeds were just
    created, and the snapshot describes the same set.

    Seed ids are the sha1 of the name, so replay takes the "overwrite if found" branch. If ids ever
    become random, this test sees entities double, because the snapshot copy gets registered again
    as if created at runtime.
    """
    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")
    built = {
        e.entity_id: (e.state, e.presence, e.presence_ref)
        for e in world.environment.all_live_entities()
    }
    assert built

    restored = await WorldInitializer(
        container, max_npcs=test_config.engine.max_npcs
    ).restore(world.world_id)
    got = {
        e.entity_id: (e.state, e.presence, e.presence_ref)
        for e in restored.environment.all_live_entities()
    }
    assert got == built


@pytest.mark.asyncio
async def test_a_snapshot_written_by_an_older_shape_still_loads(tmp_path) -> None:
    """Unknown keys in a save are dropped on read. Removing a dataclass field must not make every
    existing snapshot unreadable.

    Splatting ``WorldSnapshot(**data)`` directly raises TypeError once a field is removed, breaking
    that world's restore and replay with a cryptic "unexpected keyword argument". agent_store
    filters with _AGENT_STATE_FIELDS for the same reason.
    """
    import json as _json

    provider = FileSnapshotProvider(path=str(tmp_path))
    path = tmp_path / "w" / "step_000003.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps({
        "world_id": "w", "step": 3, "timestamp": "2026-01-01T00:00:00",
        "agent_states": {"a1": {"agent_name": "甲"}},
        # a field no longer on the dataclass but still present in existing saves
        "a_field_we_since_removed": {"whatever": 1},
    }, ensure_ascii=False), encoding="utf-8")

    snap = await provider.load("w", 3)

    assert snap is not None and snap.step == 3
    assert snap.agent_states["a1"]["agent_name"] == "甲"
