"""Read-oriented world catalog and interaction entry points."""

from __future__ import annotations

from typing import Awaitable, Callable, TypeVar

from core.interfaces.snapshot import SnapshotProvider, WorldSnapshot
from world import WorldCatalog
from world.setup import WorldSetup
from world.stored_config import StoredWorldConfig

from interaction.models import (
    AgentProfile,
    CharacterIdentity,
    NpcIdentity,
    PlaceIdentity,
    WorldMeta,
    agent_profile,
    build_world_meta,
)
from interaction.replayer import Replayer

_Table = TypeVar("_Table")


class WorldManager:
    """Coordinate read-only interaction flows for known worlds."""

    def __init__(
        self,
        *,
        snapshot_provider: SnapshotProvider,
        catalog: WorldCatalog | None = None,
    ) -> None:
        self._snapshot_provider = snapshot_provider
        # Per-world tables kept by ``_confirmed_table``: only a confirmed world's are final.
        self._profile_cache: dict[str, dict[str, AgentProfile]] = {}
        self._character_identity_cache: dict[str, dict[str, CharacterIdentity]] = {}
        self._npc_identity_cache: dict[str, dict[str, NpcIdentity]] = {}
        self._place_identity_cache: dict[str, dict[str, PlaceIdentity]] = {}
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
        return build_world_meta(world_id=world_id, latest_snapshot=latest, setup=setup, catalog=self._catalog)

    async def world_setup(self, world_id: str) -> WorldSetup | None:
        """The world's setup (see ``WorldSetup.load``); None for a world that has none."""
        return await WorldSetup.load(self._snapshot_provider, world_id)

    async def latest_snapshot(self, world_id: str) -> WorldSnapshot | None:
        """Return the latest persisted snapshot for a world."""

        return await self._snapshot_provider.load_latest(world_id)

    async def _confirmed_table(
        self,
        cache: dict[str, _Table],
        world_id: str,
        read: Callable[[], Awaitable[_Table | None]],
    ) -> _Table | None:
        """A per-world table, kept once the world is confirmed: from then on it is final.

        A draft's table is read anew on each request, since its setup is still being revised.
        Confirmation is checked before the read, so one landing mid-read never keeps a draft.
        """
        if world_id in cache:
            return cache[world_id]
        confirmed = self._catalog.is_confirmed(world_id)
        table = await read()
        if confirmed and table is not None:
            cache[world_id] = table
        return table

    async def character_profiles(self, world_id: str) -> dict[str, AgentProfile]:
        """Per-character profiles, public identity and inner life; empty for a world with no setup."""

        async def read() -> dict[str, AgentProfile] | None:
            setup = await self.world_setup(world_id)
            return {d.agent_id: agent_profile(d) for d in setup.characters} if setup else None

        profiles = await self._confirmed_table(self._profile_cache, world_id, read)
        return profiles if profiles is not None else {}

    async def character_identities(self, world_id: str) -> dict[str, CharacterIdentity]:
        """The characters' public identities; empty for a world with no setup."""

        async def read() -> dict[str, CharacterIdentity] | None:
            setup = await self.world_setup(world_id)
            if setup is None:
                return None
            return {
                d.agent_id: CharacterIdentity(
                    agent_id=d.agent_id,
                    name=d.soul.name,
                    gender=d.soul.gender,
                    age=d.soul.age,
                    role=d.soul.role,
                    background=d.soul.background,
                    appearance=d.soul.appearance,
                    color=d.soul.color,
                )
                for d in setup.characters
            }

        identities = await self._confirmed_table(self._character_identity_cache, world_id, read)
        return identities if identities is not None else {}

    async def npc_identities(self, world_id: str) -> dict[str, NpcIdentity] | None:
        """The NPCs' public identities; None for a world with no setup."""

        async def read() -> dict[str, NpcIdentity] | None:
            setup = await self.world_setup(world_id)
            if setup is None:
                return None
            return {
                seed.npc_id: NpcIdentity(
                    npc_id=seed.npc_id,
                    name=seed.name,
                    gender=seed.gender,
                    age=seed.age,
                    description=seed.description,
                )
                for seed in setup.npc_seeds
            }

        return await self._confirmed_table(self._npc_identity_cache, world_id, read)

    async def place_identities(self, world_id: str) -> dict[str, PlaceIdentity] | None:
        """The places' public identities, from the map frozen at build; None for a world with none."""

        async def read() -> dict[str, PlaceIdentity] | None:
            data = await self._snapshot_provider.load_world_config(world_id)
            if not data:
                return None
            return {
                place_id: PlaceIdentity(location_id=place_id, name=place.name, description=place.description)
                for place_id, place in StoredWorldConfig(data).get_places().items()
            }

        return await self._confirmed_table(self._place_identity_cache, world_id, read)

    async def list_steps(self, world_id: str) -> list[int]:
        """Return persisted steps for a known world."""

        return await self._snapshot_provider.list_steps(world_id)

    def create_replayer(self, world_id: str) -> Replayer:
        """Build a snapshot-backed replayer for one world."""

        return Replayer(world_id, self._snapshot_provider)
