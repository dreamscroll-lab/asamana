"""World registry catalog (worlds.json).

A thin index of *which worlds exist*, mapping ``world_id`` to display hints
(theme / world_name / created_at). It is the authoritative source
for **enumeration** (listing worlds); per-world live metadata — current step,
status, main agents — is derived from snapshots elsewhere, not from here.

Lives in ``world/`` so both the engine (``NarrativeApplication.build_world``,
which writes an entry as a build-time side effect) and the interaction layer
(``WorldManager``, which reads to list worlds) can depend on it downward without
the engine reaching up into interaction.

A ``None`` path keeps the catalog in-memory only, so tests (and any build wired
with no catalog) never touch the on-disk ``./data/worlds.json``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.logging import get_logger
from core.serialization import atomic_write_text, dump_json

logger = get_logger(__name__)

# The on-disk catalog location. The single owner of this path — the API factory
# and the CLI both back their shared catalog with it (a ``None`` path keeps a
# catalog in-memory only, which is what tests use).
DEFAULT_CATALOG_PATH = "./data/worlds.json"

# Upper bound on a world's display name. The catalog owns the field, so the bound
# lives with it; the API rejects anything longer rather than truncating (a name is
# the user's own text — silently shortening it is worse than saying no).
MAX_WORLD_NAME_LEN = 40


class WorldCatalog:
    """Read/write index of known worlds, optionally persisted to a JSON file."""

    def __init__(self, path: str | None = None) -> None:
        self._path: Path | None = Path(path) if path else None
        self._entries: dict[str, dict[str, Any]] = {}
        self._refresh_from_disk()

    def _refresh_from_disk(self) -> bool:
        """Load the on-disk catalog into memory. Returns whether the in-memory copy now matches disk.

        On a read failure, leave ``_entries`` untouched and return False so the writer skips this write.
        ``_flush`` overwrites the whole file, so writing from a copy that isn't in sync with disk does
        exactly what re-reading before a write is meant to prevent: it erases worlds another process
        just registered. The worst case is at construction: if that read fails, ``_entries`` is still
        empty, and the first register overwrites the whole index with a single world.
        """
        if self._path is None or not self._path.exists():
            return True     # nothing on disk to sync with; the in-memory copy is the whole truth
        try:
            loaded = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("world_catalog_load_failed", extra={"path": str(self._path)})
            return False
        self._entries = {wid: dict(entry) for wid, entry in loaded.items()}
        return True

    def _refreshed_for_write(self, world_id: str, op: str) -> bool:
        """Re-read before writing; if the read fails, don't write. Missing one entry beats erasing other processes' entries."""
        if self._refresh_from_disk():
            return True
        logger.error(
            "world_catalog_write_skipped",
            extra={"world_id": world_id, "op": op, "path": str(self._path)},
        )
        return False

    def register(
        self,
        world_id: str,
        *,
        theme: str = "",
        world_name: str = "",
        created_at: datetime | None = None,
    ) -> None:
        """Record (or overwrite) a world's display hints and flush to disk."""

        # Re-read before writing: the catalog is read once at construction and overwritten whole on
        # write, so a world another process built while this one was alive would be erased from the
        # index by this _flush (its data stays on disk, but list/dashboard never see it). Re-reading
        # shrinks the lost-update window from a whole build to milliseconds.
        if not self._refreshed_for_write(world_id, "register"):
            return
        self._entries[world_id] = {
            "theme": theme,
            "world_name": world_name or world_id,
            "created_at": created_at,
            # A freshly built world starts unconfirmed: the user reviews the cast
            # and relationships, then confirms to lock initialization and unlock
            # the narrative run. Set via set_confirmed, read back in build_world_meta.
            "confirmed": False,
        }
        self._flush()

    def remove(self, world_id: str) -> None:
        """Drop a world's entry from the index and flush to disk.

        Same re-read-before-write guard as ``register``. Missing entries are a no-op.
        """

        if not self._refreshed_for_write(world_id, "remove"):
            return
        self._entries.pop(world_id, None)
        self._flush()

    def set_confirmed(self, world_id: str, value: bool = True) -> None:
        """Mark a world's initialization as confirmed (locked) or not.

        Same re-read-before-write guard as ``register``. A no-op if the world is not cataloged.
        """

        if not self._refreshed_for_write(world_id, "set_confirmed"):
            return
        entry = self._entries.get(world_id)
        if entry is None:
            return
        entry["confirmed"] = bool(value)
        self._flush()

    def set_name(self, world_id: str, world_name: str) -> None:
        """Rename a world's display name.

        Only the catalog's display hint changes: ``analysis.world_name`` is a
        build-time artifact frozen in the step-0 snapshot and reaches no runtime
        prompt, so renaming here is a labeling act, not a change to the world.

        Same re-read-before-write guard as ``register``. A no-op if the world is not cataloged.
        """

        if not self._refreshed_for_write(world_id, "set_name"):
            return
        entry = self._entries.get(world_id)
        if entry is None:
            return
        entry["world_name"] = world_name
        self._flush()

    def entries(self) -> dict[str, dict[str, Any]]:
        """Return a copy of all catalog entries keyed by world_id, re-read from disk.

        Enumeration must re-read first: a world registered by another process (a CLI build) exists only
        on disk, while the long-running process's in-memory copy is frozen at its startup. Without the
        re-read, that world never shows up in the world list, so it can't be confirmed or run.

        The re-read lives here, not in ``get``: ``get`` is called per world, so reading there turns one
        listing into N disk reads, and this path sits under the world-list polling. One read per
        enumeration is cheap.
        """

        self._refresh_from_disk()
        return {wid: dict(entry) for wid, entry in self._entries.items()}

    def get(self, world_id: str) -> dict[str, Any] | None:
        """Return a copy of one world's entry, or None if not cataloged."""

        entry = self._entries.get(world_id)
        return dict(entry) if entry is not None else None

    def _flush(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self._path, dump_json(self._entries))
