"""Read-oriented world catalog and interaction entry points."""

from __future__ import annotations

from core.interfaces.snapshot import SnapshotProvider, WorldSnapshot
from world import WorldCatalog
from world.setup import WorldSetup

from interaction.models import AgentProfile, NpcIdentity, WorldMeta, agent_profile, build_world_meta
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
        # Only confirmed setups are immutable; drafts are read anew on each request.
        self._profile_cache: dict[str, dict[str, AgentProfile]] = {}
        self._npc_identity_cache: dict[str, dict[str, NpcIdentity]] = {}
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

    async def character_profiles(self, world_id: str) -> dict[str, AgentProfile]:
        """Per-agent profiles, cached only after the world's setup is confirmed.

        Returns an empty mapping for worlds with no setup.
        """

        if world_id in self._profile_cache:
            return self._profile_cache[world_id]
        entry = self._catalog.get(world_id)
        confirmed = bool(entry and entry.get("confirmed"))
        setup = await self.world_setup(world_id)
        profiles = {d.agent_id: agent_profile(d) for d in setup.characters} if setup else {}
        if confirmed and setup is not None:
            self._profile_cache[world_id] = profiles
        return profiles

    async def npc_identities(self, world_id: str) -> dict[str, NpcIdentity] | None:
        """Public NPC identities; only confirmed worlds have immutable identity caches."""
        if world_id in self._npc_identity_cache:
            return self._npc_identity_cache[world_id]
        entry = self._catalog.get(world_id)
        confirmed = bool(entry and entry.get("confirmed"))
        setup = await self.world_setup(world_id)
        if setup is None:
            return None
        identities = {
            seed.npc_id: NpcIdentity(
                npc_id=seed.npc_id,
                name=seed.name,
                gender=seed.gender,
                age=seed.age,
                description=seed.description,
            )
            for seed in setup.npc_seeds
        }
        if confirmed:
            self._npc_identity_cache[world_id] = identities
        return identities

    async def list_steps(self, world_id: str) -> list[int]:
        """Return persisted steps for a known world."""

        return await self._snapshot_provider.list_steps(world_id)

    def create_replayer(self, world_id: str) -> Replayer:
        """Build a snapshot-backed replayer for one world."""

        return Replayer(world_id, self._snapshot_provider)
