"""Contract tests: WorldCatalog instances sharing one file (as separate processes do) don't overwrite each other's writes."""

from __future__ import annotations

import json
from pathlib import Path

from world.catalog import WorldCatalog


def test_register_merges_entries_from_another_instance(tmp_path: Path) -> None:
    """The catalog is read once at construction and overwritten whole on write. When two long-lived
    instances (e.g. two processes building concurrently) register in turn, the later writer must
    re-read and merge from disk first, or the earlier one's world disappears from the index."""
    path = str(tmp_path / "worlds.json")
    cat_a = WorldCatalog(path)
    cat_b = WorldCatalog(path)  # both instances are constructed before either writes (both in-memory snapshots empty)

    cat_a.register("world-a", theme="玄武门", world_name="长安甲")
    cat_b.register("world-b", theme="安史乱", world_name="长安乙")

    on_disk = json.loads((tmp_path / "worlds.json").read_text(encoding="utf-8"))
    assert set(on_disk) == {"world-a", "world-b"}
    # The later writer also sees the merged result itself.
    assert set(cat_b.entries()) == {"world-a", "world-b"}


def test_register_no_tmp_leftover(tmp_path: Path) -> None:
    cat = WorldCatalog(str(tmp_path / "worlds.json"))
    cat.register("w1")
    assert list(tmp_path.glob("*.tmp")) == []


def test_confirmed_defaults_false_and_persists(tmp_path: Path) -> None:
    path = str(tmp_path / "worlds.json")
    cat = WorldCatalog(path)
    cat.register("w1", theme="宫变")
    assert cat.get("w1")["confirmed"] is False

    cat.set_confirmed("w1", True)
    assert cat.get("w1")["confirmed"] is True
    # Survives a fresh instance (reads back from disk).
    assert WorldCatalog(path).get("w1")["confirmed"] is True
    # No-op for unknown worlds.
    cat.set_confirmed("nope", True)
    assert cat.get("nope") is None


def test_set_name_renames_without_disturbing_the_entry(tmp_path: Path) -> None:
    """Renaming touches only world_name; every other display hint (confirmed in particular) must be
    kept. Flipping a confirmed world back to unconfirmed on rename would close the narrative-run gate again."""
    path = str(tmp_path / "worlds.json")
    cat = WorldCatalog(path)
    cat.register("w1", theme="宫变", world_name="长安")
    cat.set_confirmed("w1", True)

    cat.set_name("w1", "武德九年")
    entry = cat.get("w1")
    assert entry["world_name"] == "武德九年"
    assert entry["theme"] == "宫变" and entry["confirmed"] is True
    # Survives a fresh instance (reads back from disk).
    assert WorldCatalog(path).get("w1")["world_name"] == "武德九年"
    # No-op for unknown worlds.
    cat.set_name("nope", "x")
    assert cat.get("nope") is None


def test_set_name_merges_entries_from_another_instance(tmp_path: Path) -> None:
    """Same lost-update discipline as register/set_confirmed: re-read from disk before renaming, or a
    world another process built in the meantime is erased by this whole-file overwrite."""
    path = str(tmp_path / "worlds.json")
    cat_a = WorldCatalog(path)
    cat_a.register("world-a", world_name="甲")
    cat_b = WorldCatalog(path)  # constructed before world-b exists

    cat_a.register("world-b", world_name="乙")  # a world created by another "process"
    cat_b.set_name("world-a", "甲改")

    on_disk = json.loads((tmp_path / "worlds.json").read_text(encoding="utf-8"))
    assert set(on_disk) == {"world-a", "world-b"}
    assert on_disk["world-a"]["world_name"] == "甲改"


def test_an_unreadable_catalog_is_not_overwritten_with_a_stale_copy(tmp_path: Path) -> None:
    """If the re-read fails, don't write: missing one entry beats erasing other processes' entries.

    `_flush` overwrites the whole file. Writing after a failed read does exactly what re-reading
    before a write is meant to prevent: the world data is all still on disk, but list / dashboard
    never see it.
    """
    path = tmp_path / "worlds.json"
    seeded = WorldCatalog(str(path))
    seeded.register("world-a", world_name="长安甲")
    seeded.register("world-b", world_name="长安乙")

    cat = WorldCatalog(str(path))
    path.write_text("{ 这不是合法 JSON", encoding="utf-8")   # broken by an external tool / a slip in a manual edit

    cat.register("world-c", world_name="长安丙")             # must not overwrite the whole file from a stale copy

    assert path.read_text(encoding="utf-8").startswith("{ 这不是")


def test_a_read_failure_at_construction_does_not_wipe_the_index(tmp_path: Path) -> None:
    """Worst case: the read at construction fails and the in-memory copy is empty. If the first
    register wrote anyway, the whole index would be overwritten down to a single world."""
    path = tmp_path / "worlds.json"
    WorldCatalog(str(path)).register("world-a", world_name="长安甲")
    good = path.read_text(encoding="utf-8")

    path.write_text("<<corrupt>>", encoding="utf-8")
    cat = WorldCatalog(str(path))                 # read fails at construction → _entries is empty
    assert cat.entries() == {}

    cat.register("world-b", world_name="长安乙")   # an empty copy must never be written to disk
    assert path.read_text(encoding="utf-8") == "<<corrupt>>"

    path.write_text(good, encoding="utf-8")       # recovers once the file is fixed
    cat.register("world-b", world_name="长安乙")
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {"world-a", "world-b"}


def test_entries_sees_a_world_registered_by_another_instance(tmp_path: Path) -> None:
    """Enumeration must see worlds registered by another process (a CLI build).

    The long-running backend's in-memory copy is frozen at its startup; if enumeration doesn't
    re-read from disk, that world never gets listed, so it can't be confirmed or run.
    """
    path = str(tmp_path / "worlds.json")
    server = WorldCatalog(path)          # long-running process, read once at startup
    server.register("w-existing", world_name="甲")
    cli = WorldCatalog(path)             # another process
    cli.register("w-from-cli", world_name="乙")

    assert set(server.entries()) == {"w-existing", "w-from-cli"}
    # The enumeration read already synced memory, so per-world lookups don't need their own disk reads.
    assert server.get("w-from-cli") is not None
