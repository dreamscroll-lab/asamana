"""File-system snapshot provider."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from core.factory import ComponentKind, ProviderFactory
from core.interfaces.message import Message
from core.interfaces.perception import Broadcast, BroadcastType
from core.interfaces.snapshot import SnapshotProvider, WorldSnapshot
from core.logging import get_logger
from core.serialization import atomic_write_text, dump_json, read_json_file

logger = get_logger(__name__)

# Unknown keys are dropped on read, so removing a dataclass field doesn't make every existing
# snapshot unreadable (same as _AGENT_STATE_FIELDS in agent_store).
_SNAPSHOT_FIELDS = frozenset(f.name for f in dataclasses.fields(WorldSnapshot))

_STEP_RE = re.compile(r"^step_(\d+)\.json$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _validate_world_id(world_id: str) -> None:
    if not _SAFE_ID_RE.match(world_id):
        raise ValueError(
            f"Invalid world_id {world_id!r}: must contain only alphanumerics, hyphens, and underscores"
        )


@ProviderFactory.register("file", kind=ComponentKind.SNAPSHOT)
class FileSnapshotProvider(SnapshotProvider):
    """Persist world snapshots as JSON files on the local file system.

    Layout::

        {path}/{world_id}/step_{step:06d}.json
    """

    def __init__(self, path: str = "./data/snapshots") -> None:
        self._root = Path(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _world_dir(self, world_id: str) -> Path:
        _validate_world_id(world_id)
        return self._root / world_id

    def _snapshot_path(self, world_id: str, step: int) -> Path:
        return self._world_dir(world_id) / f"step_{step:06d}.json"

    def _world_config_path(self, world_id: str) -> Path:
        return self._world_dir(world_id) / "world_config.json"

    def _world_map_path(self, world_id: str) -> Path:
        return self._world_dir(world_id) / "world_map.json"

    @staticmethod
    def _serialize(snapshot: WorldSnapshot) -> str:
        return dump_json(dataclasses.asdict(snapshot))

    @staticmethod
    def _deserialize(raw: str) -> WorldSnapshot:
        data = json.loads(raw)
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["pending_messages"] = [
            _message_from_dict(item)
            for item in data.get("pending_messages", [])
            if isinstance(item, dict)
        ]
        data["pending_broadcasts"] = [
            _broadcast_from_dict(item)
            for item in data.get("pending_broadcasts", [])
            if isinstance(item, dict)
        ]
        return WorldSnapshot(**{k: v for k, v in data.items() if k in _SNAPSHOT_FIELDS})

    # ------------------------------------------------------------------
    # SnapshotProvider interface
    # ------------------------------------------------------------------

    async def save(self, world_id: str, step: int, snapshot: WorldSnapshot) -> None:
        if snapshot.world_id != world_id or snapshot.step != step:
            raise ValueError(
                f"Snapshot metadata mismatch: expected world_id={world_id!r} step={step}, "
                f"got world_id={snapshot.world_id!r} step={snapshot.step}"
            )
        target = self._snapshot_path(world_id, step)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Offload the blocking file write off the event loop so a step's snapshot
        # persist does not stall in-flight concurrent LLM calls.
        await asyncio.to_thread(atomic_write_text, target, self._serialize(snapshot))
        logger.info(
            "snapshot_saved",
            extra={"world_id": world_id, "step": step, "path": str(target)},
        )

    async def load(self, world_id: str, step: int) -> Optional[WorldSnapshot]:
        # Read off the event loop too, as in save: a snapshot is tens of KB of JSON on an
        # HTTP/WS request path, and one synchronous parse is enough to stall running worlds.
        return await asyncio.to_thread(self._read_snapshot, self._snapshot_path(world_id, step))

    def _read_snapshot(self, target: Path) -> Optional[WorldSnapshot]:
        if not target.exists():
            return None
        return self._deserialize(target.read_text(encoding="utf-8"))

    async def list_steps(self, world_id: str) -> List[int]:
        return await asyncio.to_thread(self._scan_steps, self._world_dir(world_id))

    @staticmethod
    def _scan_steps(world_dir: Path) -> List[int]:
        if not world_dir.exists():
            return []

        step_files: list[tuple[int, Path]] = []
        for entry in world_dir.iterdir():
            match = _STEP_RE.match(entry.name)
            if match:
                step_files.append((int(match.group(1)), entry))

        step_files.sort(key=lambda pair: pair[0])
        return [step for step, _ in step_files]

    async def load_latest(self, world_id: str) -> Optional[WorldSnapshot]:
        steps = await self.list_steps(world_id)
        if not steps:
            return None
        return await self.load(world_id, steps[-1])

    async def delete_steps_after(self, world_id: str, step: int) -> None:
        steps = await self.list_steps(world_id)
        await asyncio.to_thread(
            self._unlink_steps, [self._snapshot_path(world_id, s) for s in steps if s > step]
        )
        logger.info(
            "snapshots_deleted_after_step",
            extra={"world_id": world_id, "after_step": step},
        )

    @staticmethod
    def _unlink_steps(paths: List[Path]) -> None:
        for path in paths:
            path.unlink(missing_ok=True)

    async def delete_world(self, world_id: str) -> None:
        # Removes every step_*.json plus world_config.json / world_map.json under
        # this world's dir.
        # ignore_errors: partial trees / concurrent unlinks must not raise from a
        # whole-world purge — the operation is irreversible and best-effort by design.
        await asyncio.to_thread(
            shutil.rmtree, self._world_dir(world_id), ignore_errors=True
        )
        logger.info("world_snapshots_deleted", extra={"world_id": world_id})

    async def save_world_config(self, world_id: str, data: dict[str, Any]) -> None:
        target = self._world_config_path(world_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(atomic_write_text, target, dump_json(data))
        logger.info(
            "world_config_saved",
            extra={"world_id": world_id, "path": str(target)},
        )

    async def load_world_config(self, world_id: str) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(read_json_file, self._world_config_path(world_id))

    async def save_world_map(self, world_id: str, data: dict[str, Any]) -> None:
        target = self._world_map_path(world_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(atomic_write_text, target, dump_json(data))
        logger.info(
            "world_map_frozen",
            extra={"world_id": world_id, "path": str(target)},
        )

    async def load_world_map(self, world_id: str) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(read_json_file, self._world_map_path(world_id))

    def _world_asset_path(self, world_id: str, rel_path: str) -> Optional[Path]:
        """Resolve a frozen asset, or None if the path tries to leave the world's dir.

        `rel_path` originates in a .tmj, which is authored content — it reaches this
        method as an untrusted relative path, so it is confined here rather than at
        each caller.
        """
        root = (self._world_dir(world_id) / "assets").resolve()
        target = (root / rel_path).resolve()
        return target if target.is_relative_to(root) else None

    async def save_world_asset(self, world_id: str, rel_path: str, data: bytes) -> None:
        target = self._world_asset_path(world_id, rel_path)
        if target is None:
            logger.error(
                "world_asset_path_rejected",
                extra={"world_id": world_id, "rel_path": rel_path},
            )
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)

    async def load_world_asset(self, world_id: str, rel_path: str) -> Optional[bytes]:
        target = self._world_asset_path(world_id, rel_path)
        if target is None or not target.is_file():
            return None
        return await asyncio.to_thread(target.read_bytes)


def _message_from_dict(data: dict) -> Message:
    """Deserialize a Message from snapshot JSON.

    Delivery is determined by recipients + location_scope + deliver_step; urgency is parsed
    leniently via parse_urgency.
    """
    from core.interfaces.urgency import parse_urgency
    raw_urgency = data.get("urgency")
    return Message(
        id=str(data["id"]),
        world_id=str(data["world_id"]),
        sender_id=str(data["sender_id"]),
        content=str(data["content"]),
        recipients=list(data["recipients"]) if data.get("recipients") is not None else None,
        location_scope=(
            str(data["location_scope"])
            if data.get("location_scope") is not None
            else None
        ),
        deliver_step=int(data["deliver_step"]),
        created_step=int(data["created_step"]),
        sender_name=str(data.get("sender_name", "")),
        intent=str(data.get("intent", "")),
        urgency=parse_urgency(raw_urgency),
        # Dropping either field changes delivery after a restore: a mindless body would be
        # treated as a person who forms relations, and execution members would receive their own
        # act again.
        sender_is_agent=bool(data.get("sender_is_agent", True)),
        actor_ids=tuple(data.get("actor_ids") or ()),
        metadata=dict(data.get("metadata", {})),
    )


def _broadcast_from_dict(data: dict) -> Broadcast:
    """Deserialize a pending Broadcast from snapshot JSON.

    severity / phenomenon are normalized from strings by ``Broadcast.__post_init__``, so they are
    passed through. An unknown broadcast_type falls back to WORLD_EVENT instead of raising:
    a failed restore would drop the broadcast, and this path exists so that pending broadcasts
    such as death notices survive a restore.
    """
    raw_type = data.get("broadcast_type")
    try:
        broadcast_type = BroadcastType(raw_type)
    except ValueError:
        logger.warning("broadcast_type_unknown", extra={"value": str(raw_type)})
        broadcast_type = BroadcastType.WORLD_EVENT
    return Broadcast(
        content=str(data.get("content", "")),
        source=str(data.get("source", "system")),
        broadcast_type=broadcast_type,
        deliver_step=int(data["deliver_step"]),
        location_scope=(
            str(data["location_scope"]) if data.get("location_scope") is not None else None
        ),
        severity=data.get("severity"),
        phenomenon=data.get("phenomenon"),
    )
