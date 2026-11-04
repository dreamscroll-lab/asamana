"""Confirmed world identity caches and fresh draft reads."""

from unittest.mock import AsyncMock

import pytest

from interaction.world_manager import WorldManager
from providers.snapshot.in_memory import InMemorySnapshotProvider
from tests.unit.world_setups import setup_dict
from world import WorldCatalog


@pytest.mark.asyncio
async def test_confirmed_identities_and_profiles_are_cached_per_world() -> None:
    snapshots = InMemorySnapshotProvider()
    original = setup_dict({"a": {"name": "甲"}})
    original["npc_seeds"] = [
        {"npc_id": "n", "name": "门房", "gender": "男", "age": 40,
         "description": "熟悉宅院", "location": "palace"},
    ]
    await snapshots.save_world_setup("left", original)
    await snapshots.save_world_setup("right", setup_dict({"a": {"name": "乙"}}))
    catalog = WorldCatalog(None)
    for world_id in ("left", "right"):
        catalog.register(world_id)
        catalog.set_confirmed(world_id)
    manager = WorldManager(snapshot_provider=snapshots, catalog=catalog)
    read = AsyncMock(wraps=snapshots.load_world_setup)
    snapshots.load_world_setup = read

    identities = await manager.npc_identities("left")
    assert identities is not None and identities["n"].name == "门房"
    assert await manager.npc_identities("left") is identities
    assert read.await_count == 1
    assert await manager.npc_identities("right") == {}
    assert await manager.npc_identities("left") is identities
    left_profiles = await manager.character_profiles("left")
    right_profiles = await manager.character_profiles("right")
    assert left_profiles["a"].name == "甲"
    assert right_profiles["a"].name == "乙"
    reads = read.await_count
    assert await manager.character_profiles("left") is left_profiles
    assert await manager.character_profiles("right") is right_profiles
    assert await manager.npc_identities("left") is identities
    assert await manager.npc_identities("right") == {}
    assert read.await_count == reads


@pytest.mark.asyncio
async def test_missing_setup_does_not_cache_an_empty_npc_identity_table() -> None:
    snapshots = InMemorySnapshotProvider()
    manager = WorldManager(snapshot_provider=snapshots)
    assert await manager.npc_identities("pending") is None
    await snapshots.save_world_setup("pending", setup_dict({}))
    assert await manager.npc_identities("pending") == {}


@pytest.mark.asyncio
async def test_draft_npc_identities_are_read_fresh_and_only_frozen_at_confirmation() -> None:
    snapshots = InMemorySnapshotProvider()
    catalog = WorldCatalog(None)
    catalog.register("draft")
    manager = WorldManager(snapshot_provider=snapshots, catalog=catalog)
    setup = setup_dict({})
    setup["npc_seeds"] = [{
        "npc_id": "n", "name": "门房", "gender": "男", "age": 40,
        "description": "熟悉宅院", "location": "palace",
    }]
    await snapshots.save_world_setup("draft", setup)
    first = await manager.npc_identities("draft")
    assert first is not None and first["n"].name == "门房"

    setup["npc_seeds"][0]["name"] = "管事"
    await snapshots.save_world_setup("draft", setup)
    revised = await manager.npc_identities("draft")
    assert revised is not None and revised["n"].name == "管事"
    assert revised is not first

    catalog.set_confirmed("draft")
    frozen = await manager.npc_identities("draft")
    assert frozen is not None and frozen["n"].name == "管事"
    assert frozen is not revised
    assert await manager.npc_identities("draft") is frozen


@pytest.mark.asyncio
async def test_draft_character_profiles_are_read_fresh_and_only_frozen_at_confirmation() -> None:
    snapshots = InMemorySnapshotProvider()
    catalog = WorldCatalog(None)
    catalog.register("draft")
    manager = WorldManager(snapshot_provider=snapshots, catalog=catalog)
    await snapshots.save_world_setup("draft", setup_dict({"a": {"name": "甲", "secret": "旧密信"}}))
    first = await manager.character_profiles("draft")
    assert first["a"].secret == "旧密信"

    await snapshots.save_world_setup("draft", setup_dict({"a": {"name": "甲", "secret": "新密信"}}))
    revised = await manager.character_profiles("draft")
    assert revised["a"].secret == "新密信"
    assert revised is not first

    catalog.set_confirmed("draft")
    frozen = await manager.character_profiles("draft")
    assert frozen["a"].secret == "新密信"
    assert frozen is not revised
    assert await manager.character_profiles("draft") is frozen


@pytest.mark.asyncio
async def test_character_profile_read_during_confirmation_does_not_cache_a_draft() -> None:
    snapshots = InMemorySnapshotProvider()
    catalog = WorldCatalog(None)
    catalog.register("draft")
    manager = WorldManager(snapshot_provider=snapshots, catalog=catalog)
    await snapshots.save_world_setup("draft", setup_dict({"a": {"name": "甲", "role": "初稿"}}))
    load = snapshots.load_world_setup

    async def confirm_during_load(world_id: str) -> dict:
        draft = await load(world_id)
        catalog.set_confirmed(world_id)
        return draft

    snapshots.load_world_setup = confirm_during_load
    assert (await manager.character_profiles("draft"))["a"].role == "初稿"
    snapshots.load_world_setup = load
    await snapshots.save_world_setup("draft", setup_dict({"a": {"name": "甲", "role": "定稿"}}))
    assert (await manager.character_profiles("draft"))["a"].role == "定稿"


@pytest.mark.asyncio
async def test_character_and_place_identities_are_kept_only_once_confirmed() -> None:
    snapshots = InMemorySnapshotProvider()
    await snapshots.save_world_setup("w", setup_dict({"a": {"name": "甲", "role": "皇子"}}))
    await snapshots.save_world_config("w", {"places": {
        "palace": {"place_id": "palace", "name": "太极宫", "description": "正殿"},
    }})
    catalog = WorldCatalog(None)
    catalog.register("w")
    manager = WorldManager(snapshot_provider=snapshots, catalog=catalog)

    draft = await manager.character_identities("w")
    assert draft["a"].name == "甲" and draft["a"].role == "皇子"
    assert await manager.character_identities("w") is not draft
    places = await manager.place_identities("w")
    assert places is not None and places["palace"].name == "太极宫"
    assert await manager.place_identities("w") is not places

    catalog.set_confirmed("w")
    frozen = await manager.character_identities("w")
    assert await manager.character_identities("w") is frozen
    frozen_places = await manager.place_identities("w")
    assert await manager.place_identities("w") is frozen_places
    assert await manager.character_identities("none") == {}
    assert await manager.place_identities("none") is None
