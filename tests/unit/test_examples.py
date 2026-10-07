"""Example worlds: first-start seeding, and the bundled worlds still load in the current format."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from config import load_config
from config.models import Config, ProviderConfig
from interaction.examples import EXAMPLES_DIR, SEEDED_MARKER, seed_examples
from interaction.world_manager import WorldManager
from providers.agent_store.file import FileAgentStore
from providers.snapshot.file import FileSnapshotProvider
from providers.vector_store.file import FileVectorStore
from world import WorldCatalog


def _file_config(root: Path) -> Config:
    config = load_config(Path("config/config.test.yaml"))
    config.snapshot = ProviderConfig(provider="file", params={"path": str(root / "snapshots")})
    config.agent_store = ProviderConfig(provider="file", params={"path": str(root / "agents")})
    config.vector_store = ProviderConfig(provider="file", params={"path": str(root / "vectors")})
    config.observability.enabled = True
    config.observability.provider = "jsonl"
    config.observability.params = {"base_dir": str(root / "traces")}
    return config


def _bundle(root: Path, world_ids: list[str]) -> Path:
    """A minimal examples/ tree with one file per store for each world."""
    bundle = root / "examples"
    entries = {}
    for world_id in world_ids:
        (bundle / "snapshots" / world_id).mkdir(parents=True)
        (bundle / "snapshots" / world_id / "step_000000.json").write_text("{}", encoding="utf-8")
        (bundle / "agents" / world_id).mkdir(parents=True)
        (bundle / "agents" / world_id / "agent-a.json").write_text("{}", encoding="utf-8")
        (bundle / "traces" / world_id).mkdir(parents=True)
        (bundle / "traces" / world_id / "build.jsonl").write_text("", encoding="utf-8")
        (bundle / "vectors").mkdir(exist_ok=True)
        (bundle / "vectors" / f"{world_id}_agent-a_memory.0000.jsonl").write_text("", encoding="utf-8")
        entries[world_id] = {
            "theme": f"theme {world_id}",
            "world_name": f"name {world_id}",
            "created_at": "2026-10-07 05:36:32.341217",
            "confirmed": True,
        }
    (bundle / "vectors" / "other-world_agent-b_memory.0000.jsonl").write_text("", encoding="utf-8")
    (bundle / "worlds.json").write_text(json.dumps(entries), encoding="utf-8")
    return bundle


def test_a_fresh_install_gets_every_store_and_a_confirmed_catalog_entry(tmp_path: Path) -> None:
    data = tmp_path / "data"
    bundle = _bundle(tmp_path, ["w1", "w2"])
    catalog_path = str(data / "worlds.json")

    seeded = seed_examples(_file_config(data), catalog_path, examples_dir=bundle)

    assert seeded == ["w1", "w2"]
    for world_id in seeded:
        assert (data / "snapshots" / world_id / "step_000000.json").exists()
        assert (data / "agents" / world_id / "agent-a.json").exists()
        assert (data / "traces" / world_id / "build.jsonl").exists()
        assert (data / "vectors" / f"{world_id}_agent-a_memory.0000.jsonl").exists()
        entry = WorldCatalog(catalog_path).get(world_id)
        assert entry is not None
        assert entry["world_name"] == f"name {world_id}"
        assert entry["confirmed"] is True
    assert not (data / "vectors" / "other-world_agent-b_memory.0000.jsonl").exists()
    assert (data / SEEDED_MARKER).exists()


def test_an_install_that_predates_the_examples_is_only_marked(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    catalog_path = data / "worlds.json"
    catalog_path.write_text("{}", encoding="utf-8")

    seeded = seed_examples(
        _file_config(data), str(catalog_path), examples_dir=_bundle(tmp_path, ["w1"])
    )

    assert seeded == []
    assert not (data / "snapshots").exists()
    assert json.loads(catalog_path.read_text(encoding="utf-8")) == {}
    assert (data / SEEDED_MARKER).exists()


@pytest.mark.parametrize("catalog_after", ["emptied", "deleted"])
def test_seeding_happens_once_per_data_directory(tmp_path: Path, catalog_after: str) -> None:
    """Deleting every world, or deleting worlds.json by hand, must not bring the examples back."""
    data = tmp_path / "data"
    config = _file_config(data)
    bundle = _bundle(tmp_path, ["w1"])
    catalog_path = data / "worlds.json"
    assert seed_examples(config, str(catalog_path), examples_dir=bundle) == ["w1"]
    shutil.rmtree(data / "snapshots" / "w1")
    if catalog_after == "emptied":
        WorldCatalog(str(catalog_path)).remove("w1")
    else:
        catalog_path.unlink()

    assert seed_examples(config, str(catalog_path), examples_dir=bundle) == []
    assert not (data / "snapshots" / "w1").exists()


def test_a_wiped_data_directory_is_a_fresh_install(tmp_path: Path) -> None:
    data = tmp_path / "data"
    config = _file_config(data)
    bundle = _bundle(tmp_path, ["w1"])
    catalog_path = str(data / "worlds.json")
    assert seed_examples(config, catalog_path, examples_dir=bundle) == ["w1"]
    shutil.rmtree(data)

    assert seed_examples(config, catalog_path, examples_dir=bundle) == ["w1"]


def test_a_world_whose_data_already_exists_is_left_alone(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "snapshots" / "w1").mkdir(parents=True)
    catalog_path = str(data / "worlds.json")

    seeded = seed_examples(_file_config(data), catalog_path, examples_dir=_bundle(tmp_path, ["w1", "w2"]))

    assert seeded == ["w2"]
    assert list((data / "snapshots" / "w1").iterdir()) == []
    assert WorldCatalog(catalog_path).get("w1") is None


def test_stores_that_are_not_file_backed_get_nothing(tmp_path: Path) -> None:
    data = tmp_path / "data"
    catalog_path = data / "worlds.json"
    config = load_config(Path("config/config.test.yaml"))
    assert config.snapshot.provider != "file"

    seeded = seed_examples(config, str(catalog_path), examples_dir=_bundle(tmp_path, ["w1"]))

    assert seeded == []
    assert not catalog_path.exists()
    assert not (data / SEEDED_MARKER).exists()


def test_traces_are_skipped_when_observability_is_off(tmp_path: Path) -> None:
    data = tmp_path / "data"
    config = _file_config(data)
    config.observability.enabled = False

    seeded = seed_examples(config, str(data / "worlds.json"), examples_dir=_bundle(tmp_path, ["w1"]))

    assert seeded == ["w1"]
    assert (data / "snapshots" / "w1").exists()
    assert not (data / "traces").exists()


@pytest.mark.asyncio
async def test_the_bundled_worlds_load_with_the_current_code(tmp_path: Path) -> None:
    """Fails when a snapshot, agent or vector format change leaves the shipped examples behind:
    regenerate them from a fresh run rather than editing the files by hand."""
    entries = json.loads((EXAMPLES_DIR / "worlds.json").read_text(encoding="utf-8"))
    assert entries
    data = tmp_path / "data"
    config = _file_config(data)
    catalog_path = str(data / "worlds.json")

    assert sorted(seed_examples(config, catalog_path)) == sorted(entries)

    snapshots = FileSnapshotProvider(str(data / "snapshots"))
    agents = FileAgentStore(str(data / "agents"))
    manager = WorldManager(snapshot_provider=snapshots, catalog=WorldCatalog(catalog_path))
    vectors = FileVectorStore(str(data / "vectors"))
    for world_id in entries:
        steps = await snapshots.list_steps(world_id)
        assert steps
        for step in steps:
            assert await snapshots.load(world_id, step) is not None, (world_id, step)
        meta = await manager.get_world(world_id)
        assert meta is not None
        assert meta.confirmed
        assert meta.current_step == max(steps)
        agent_ids = await agents.list_agent_ids(world_id)
        assert agent_ids
        for agent_id in agent_ids:
            assert await agents.load_agent_state(world_id, agent_id) is not None, (world_id, agent_id)
        vector_files = sorted((data / "vectors").glob(f"{world_id}_*.jsonl"))
        assert vector_files, world_id
        for vector_file in vector_files:
            header = json.loads(vector_file.read_text(encoding="utf-8").splitlines()[0])
            assert await vectors.list_all(header["_meta"]["collection"]), vector_file.name
