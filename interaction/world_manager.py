"""Read-oriented world catalog and interaction entry points."""

from __future__ import annotations

from typing import Any

from core.interfaces.snapshot import SnapshotProvider, WorldSnapshot
from world import WorldCatalog

from interaction.models import WorldMeta, build_world_meta
from interaction.replayer import Replayer


class WorldManager:
    """Coordinate read-only interaction flows for known worlds."""

    def __init__(
        self,
        *,
        snapshot_provider: SnapshotProvider,
        catalog: WorldCatalog | None = None,
    ) -> None:
        self._snapshot_provider = snapshot_provider
        # Step-0 character_profiles cache, keyed by world_id. Shared by the API
        # routers so the static-identity read (name/role/colour) happens once. Step 0 is
        # frozen at build, so an entry never goes stale; a deleted world's entry is a few
        # KB of dead weight, cheaper than a delete hook (same trade as maps._ground_cache).
        self._profile_cache: dict[str, dict[str, Any]] = {}
        # Share the same WorldCatalog instance the build path writes to (so a world
        # registered during build_world is visible here without a reload). When none
        # is injected, an in-memory one (tests).
        self._catalog = catalog if catalog is not None else WorldCatalog()

    @property
    def snapshot_provider(self) -> SnapshotProvider:
        return self._snapshot_provider

    async def confirm_world(self, world_id: str) -> WorldMeta:
        """Lock a world's initialization after the user reviews it.

        Idempotent. Raises ``ValueError`` if the world is unknown so the API can
        map it to a 404.
        """

        meta = await self.get_world(world_id)
        if meta is None:
            raise ValueError(f"World not found: {world_id}")
        self._catalog.set_confirmed(world_id, True)
        return await self.get_world(world_id)

    async def rename_world(self, world_id: str, world_name: str) -> WorldMeta:
        """Change a world's display name (the catalog entry only).

        The step-0 ``analysis.world_name`` only fed the build prompts, so it is left alone.
        Raises ``ValueError`` (→ 404) if the world has no catalog entry. Checked against the
        catalog, not ``get_world``: ``set_name`` would silently no-op on a world without one.
        """

        if self._catalog.get(world_id) is None:
            raise ValueError(f"World not found: {world_id}")
        self._catalog.set_name(world_id, world_name)
        return await self.get_world(world_id)

    async def list_worlds(self) -> list[WorldMeta]:
        """List known worlds from the local catalog and persisted snapshots, oldest first.

        Ordered by creation time, which is a fact about the worlds rather than a display
        choice; clients can reverse it. Worlds with no creation time sort last, and ties break
        on world_id so the order is deterministic.
        """

        worlds = [await self.get_world(world_id) for world_id in self._catalog.entries()]
        return sorted(
            (world for world in worlds if world is not None),
            # Compare timestamps: comparing aware and naive datetimes raises TypeError.
            key=lambda w: (
                w.created_at is None, w.created_at.timestamp() if w.created_at else 0.0, w.world_id,
            ),
        )

    async def get_world(self, world_id: str) -> WorldMeta | None:
        """Return derived metadata for one known world."""

        # Read only two snapshots. The world list polls this every few seconds for every world,
        # and snapshot reads are synchronous JSON parsing, so reading the whole history would
        # block the event loop and stall running worlds. Identity (main characters, description,
        # step length) is fixed at build time in step 0, and progress only needs the last step.
        steps = await self._snapshot_provider.list_steps(world_id)
        wanted = sorted({0, max(steps)} & set(steps)) if steps else []
        snapshots = [
            snapshot
            for step in wanted
            if (snapshot := await self._snapshot_provider.load(world_id, step)) is not None
        ]
        entry = self._catalog.get(world_id)
        if not snapshots and entry is None:
            return None
        return build_world_meta(world_id=world_id, snapshots=snapshots, catalog_entry=entry)

    async def latest_snapshot(self, world_id: str) -> WorldSnapshot | None:
        """Return the latest persisted snapshot for a world."""

        return await self._snapshot_provider.load_latest(world_id)

    async def character_profiles(self, world_id: str) -> dict[str, Any]:
        """Static per-agent identity written at step 0, keyed by agent_id.

        Cached per world (identity is frozen at build). Returns an empty mapping
        for worlds with no step-0 snapshot.
        """

        if world_id not in self._profile_cache:
            initial = await self._snapshot_provider.load(world_id, 0)
            self._profile_cache[world_id] = (
                dict(initial.metadata.get("character_profiles", {})) if initial else {}
            )
        return self._profile_cache[world_id]

    async def list_steps(self, world_id: str) -> list[int]:
        """Return persisted steps for a known world."""

        return await self._snapshot_provider.list_steps(world_id)

    def create_replayer(self, world_id: str) -> Replayer:
        """Build a snapshot-backed replayer for one world."""

        return Replayer(world_id, self._snapshot_provider)

