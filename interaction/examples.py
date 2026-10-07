"""Example worlds shipped in ``examples/``, copied into a fresh install on first start.

``examples/`` mirrors the on-disk layout of the file-backed stores (``snapshots/<id>/``,
``agents/<id>/``, ``vectors/<id>_*.jsonl``, ``traces/<id>/``) plus ``worlds.json``, the catalog
entries. Seeding copies each world into the paths the config names and registers it through
``WorldCatalog``, so a new user can browse and replay finished worlds before configuring any keys.

It runs once per data directory, recorded by a marker file beside the catalog. Don't key it on the
catalog file alone: deleting ``worlds.json`` by hand would then bring back examples the user had
deleted. An install whose catalog already exists predates the examples and only gets the marker.
It must run before the container is built, because the file vector store reads its whole directory
at construction.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from config.models import Config
from core.logging import get_logger
from world import WorldCatalog

logger = get_logger(__name__)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

# The bundle is in the file providers' format, so it can only be copied into those providers.
_FILE_PROVIDER = "file"
_FILE_TRACE_PROVIDER = "jsonl"

SEEDED_MARKER = ".examples_seeded"


def seed_examples(
    config: Config, catalog_path: str, *, examples_dir: Path = EXAMPLES_DIR
) -> list[str]:
    """Copy the bundled worlds into a fresh install and return the ids registered."""

    catalog_file = examples_dir / "worlds.json"
    marker = Path(catalog_path).with_name(SEEDED_MARKER)
    if marker.exists() or not catalog_file.exists():
        return []
    stores = _store_paths(config)
    if stores is None:
        logger.info("examples_skipped", extra={"reason": "stores are not file-backed"})
        return []
    if Path(catalog_path).exists():
        _mark(marker)
        return []

    entries: dict[str, dict[str, Any]] = json.loads(catalog_file.read_text(encoding="utf-8"))
    catalog = WorldCatalog(catalog_path)
    seeded: list[str] = []
    for world_id, entry in entries.items():
        snapshot_dir = stores["snapshots"] / world_id
        # Ids are never reused, so an existing directory is this world's own data: leave it be.
        if snapshot_dir.exists():
            continue
        shutil.copytree(examples_dir / "snapshots" / world_id, snapshot_dir)
        shutil.copytree(examples_dir / "agents" / world_id, stores["agents"] / world_id)
        stores["vectors"].mkdir(parents=True, exist_ok=True)
        for vector_file in (examples_dir / "vectors").glob(f"{world_id}_*.jsonl"):
            shutil.copy2(vector_file, stores["vectors"] / vector_file.name)
        if "traces" in stores and (examples_dir / "traces" / world_id).exists():
            shutil.copytree(examples_dir / "traces" / world_id, stores["traces"] / world_id)
        catalog.register(
            world_id,
            theme=entry.get("theme", ""),
            world_name=entry.get("world_name", ""),
            created_at=datetime.fromisoformat(entry["created_at"]) if entry.get("created_at") else None,
        )
        catalog.set_confirmed(world_id, bool(entry.get("confirmed", False)))
        seeded.append(world_id)
    _mark(marker)
    logger.info("examples_seeded", extra={"world_ids": seeded})
    return seeded


def _mark(marker: Path) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def _store_paths(config: Config) -> dict[str, Path] | None:
    """Where each store keeps its files, or None if a required store isn't file-backed.

    Traces are optional: with observability off or on another provider the worlds still replay.
    """

    required = {
        "snapshots": config.snapshot,
        "agents": config.agent_store,
        "vectors": config.vector_store,
    }
    paths: dict[str, Path] = {}
    for name, store in required.items():
        if store.provider != _FILE_PROVIDER or "path" not in store.params:
            return None
        paths[name] = Path(store.params["path"])
    observability = config.observability
    if (
        observability.enabled
        and observability.provider == _FILE_TRACE_PROVIDER
        and "base_dir" in observability.params
    ):
        paths["traces"] = Path(observability.params["base_dir"])
    return paths
