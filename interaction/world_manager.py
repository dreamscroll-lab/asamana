"""Read-oriented world catalog and interaction entry points."""

from __future__ import annotations

from typing import Any

from core.interfaces.snapshot import SnapshotProvider, WorldSnapshot
from world import AgentDefinition, WorldCatalog
from world.setup import WorldSetup

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
        # Character profiles read off the world's setup, keyed by world_id. Shared by the API
        # routers so the static-identity read (name/role/color) happens once. The setup changes
        # only by a revision before the world is confirmed, which calls ``forget_profiles``; a
        # deleted world's entry is a few KB of dead weight, cheaper than a delete hook (same
        # trade as maps._ground_cache).
        self._profile_cache: dict[str, dict[str, Any]] = {}
        # Share the same WorldCatalog instance the build path writes to (so a world
        # registered during build_world is visible here without a reload). When none
        # is injected, an in-memory one (tests).
        self._catalog = catalog if catalog is not None else WorldCatalog()

    @property
    def snapshot_provider(self) -> SnapshotProvider:
        return self._snapshot_provider

    async def rename_world(self, world_id: str, world_name: str) -> WorldMeta:
        """Change a world's display name (the catalog entry only).

        The setup's ``analysis.world_name`` only fed the build prompts, so it is left alone.
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

        # The world list polls this every few seconds for every world, and reads are synchronous
        # JSON parsing: read the setup and the latest snapshot only, never the history.
        entry = self._catalog.get(world_id)
        setup = await self.world_setup(world_id)
        latest = await self._snapshot_provider.load_latest(world_id)
        if entry is None and setup is None and latest is None:
            return None
        return build_world_meta(world_id=world_id, latest_snapshot=latest, setup=setup, catalog_entry=entry)

    async def world_setup(self, world_id: str) -> WorldSetup | None:
        return await WorldSetup.load(self._snapshot_provider, world_id)

    async def latest_snapshot(self, world_id: str) -> WorldSnapshot | None:
        """Return the latest persisted snapshot for a world."""

        return await self._snapshot_provider.load_latest(world_id)

    async def character_profiles(self, world_id: str) -> dict[str, Any]:
        """Static per-agent identity from the world's setup, keyed by agent_id.

        Returns an empty mapping for worlds with no setup.
        """

        if world_id not in self._profile_cache:
            setup = await self.world_setup(world_id)
            self._profile_cache[world_id] = (
                {d.agent_id: _profile(d) for d in setup.characters} if setup else {}
            )
        return self._profile_cache[world_id]

    def forget_profiles(self, world_id: str) -> None:
        """Drop the cached profiles after the world's setup was revised."""
        self._profile_cache.pop(world_id, None)

    async def list_steps(self, world_id: str) -> list[int]:
        """Return persisted steps for a known world."""

        return await self._snapshot_provider.list_steps(world_id)

    def create_replayer(self, world_id: str) -> Replayer:
        """Build a snapshot-backed replayer for one world."""

        return Replayer(world_id, self._snapshot_provider)



def _profile(definition: AgentDefinition) -> dict[str, Any]:
    soul = definition.soul
    return {
        "name": soul.name,
        "role": soul.role,
        "age": soul.age,
        "gender": soul.gender,
        "background": soul.background,
        "appearance": soul.appearance,
        "color": soul.color,
        "core_traits": list(soul.core_traits),
        "core_values": list(soul.core_values),
        "self_image": soul.self_image,
        "life_goal": soul.life_goal or "",
        "secret": soul.secret,
        "is_main_character": definition.is_main_character,
    }
