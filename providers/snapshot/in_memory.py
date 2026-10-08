"""In-memory snapshot provider."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.factory import ComponentKind, ProviderFactory
from core.interfaces.snapshot import SnapshotProvider, WorldSnapshot


@ProviderFactory.register("in_memory", kind=ComponentKind.SNAPSHOT)
class InMemorySnapshotProvider(SnapshotProvider):
    """Persist snapshots in memory."""

    def __init__(self) -> None:
        self._snapshots: Dict[Tuple[str, int], WorldSnapshot] = {}
        self._world_configs: Dict[str, dict[str, Any]] = {}
        self._world_maps: Dict[str, dict[str, Any]] = {}
        self._world_assets: Dict[Tuple[str, str], bytes] = {}

    async def save(self, world_id: str, step: int, snapshot: WorldSnapshot) -> None:
        self._snapshots[(world_id, step)] = snapshot

    async def load(self, world_id: str, step: int) -> Optional[WorldSnapshot]:
        return self._snapshots.get((world_id, step))

    async def list_steps(self, world_id: str) -> List[int]:
        return sorted(
            step
            for (stored_world_id, step) in self._snapshots
            if stored_world_id == world_id
        )

    async def load_latest(self, world_id: str) -> Optional[WorldSnapshot]:
        steps = await self.list_steps(world_id)
        if not steps:
            return None
        return await self.load(world_id, steps[-1])

    async def delete_steps_after(self, world_id: str, step: int) -> None:
        keys = [k for k in self._snapshots if k[0] == world_id and k[1] > step]
        for k in keys:
            del self._snapshots[k]

    async def delete_world(self, world_id: str) -> None:
        keys = [k for k in self._snapshots if k[0] == world_id]
        for k in keys:
            del self._snapshots[k]
        self._world_configs.pop(world_id, None)
        self._world_maps.pop(world_id, None)
        for key in [k for k in self._world_assets if k[0] == world_id]:
            del self._world_assets[key]

    async def save_world_config(self, world_id: str, data: dict[str, Any]) -> None:
        self._world_configs[world_id] = data

    async def load_world_config(self, world_id: str) -> Optional[dict[str, Any]]:
        return self._world_configs.get(world_id)

    async def save_world_map(self, world_id: str, data: dict[str, Any]) -> None:
        self._world_maps[world_id] = data

    async def load_world_map(self, world_id: str) -> Optional[dict[str, Any]]:
        return self._world_maps.get(world_id)

    async def save_world_asset(self, world_id: str, rel_path: str, data: bytes) -> None:
        self._world_assets[(world_id, rel_path)] = data

    async def load_world_asset(self, world_id: str, rel_path: str) -> Optional[bytes]:
        return self._world_assets.get((world_id, rel_path))
