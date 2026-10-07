"""File-system agent store provider."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import shutil
from pathlib import Path
from typing import List, Optional

from core.factory import ComponentKind, ProviderFactory
from core.interfaces.agent_store import AgentRelation, AgentState, AgentStoreProvider
from core.logging import get_logger
from core.serialization import atomic_write_text, dump_json, read_json_file

logger = get_logger(__name__)

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _validate_id(value: str, label: str) -> None:
    if not _SAFE_ID_RE.match(value):
        raise ValueError(
            f"Invalid {label} {value!r}: must contain only alphanumerics, hyphens, and underscores"
        )


# Unknown fields in stored JSON are dropped on load: a field removed from the dataclass must
# not make every existing save file unloadable.
_AGENT_STATE_FIELDS = frozenset(f.name for f in dataclasses.fields(AgentState))
_AGENT_RELATION_FIELDS = frozenset(f.name for f in dataclasses.fields(AgentRelation))


def _state_from_data(data: dict) -> AgentState:
    return AgentState(**{k: v for k, v in data.items() if k in _AGENT_STATE_FIELDS})


def _relation_from_data(data: dict) -> AgentRelation:
    return AgentRelation(**{k: v for k, v in data.items() if k in _AGENT_RELATION_FIELDS})


@ProviderFactory.register("file", kind=ComponentKind.AGENT_STORE)
class FileAgentStore(AgentStoreProvider):
    """Persist agent states and relations as JSON files on the local file system.

    Layout::

        {path}/{world_id}/{agent_id}.json
        {path}/{world_id}/relations/{from_id}__{to_id}.json
    """

    def __init__(self, path: str = "./data/agents") -> None:
        self._root = Path(path)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _world_dir(self, world_id: str) -> Path:
        """Validate on every path build (same as ``FileSnapshotProvider._world_dir``).

        Ids arrive here straight from API path parameters, and these helpers are their only way
        into the filesystem; validating only in delete would leave every other read and write open.
        """
        _validate_id(world_id, "world_id")
        return self._root / world_id

    def _state_path(self, world_id: str, agent_id: str) -> Path:
        _validate_id(agent_id, "agent_id")
        return self._world_dir(world_id) / f"{agent_id}.json"

    def _initial_state_path(self, world_id: str, agent_id: str) -> Path:
        _validate_id(agent_id, "agent_id")
        return self._world_dir(world_id) / f"{agent_id}__initial.json"

    def _relations_dir(self, world_id: str) -> Path:
        return self._world_dir(world_id) / "relations"

    def _relation_path(self, world_id: str, from_id: str, to_id: str) -> Path:
        _validate_id(from_id, "from_id")
        _validate_id(to_id, "to_id")
        return self._relations_dir(world_id) / f"{from_id}__{to_id}.json"

    def _initial_relation_path(self, world_id: str, from_id: str, to_id: str) -> Path:
        _validate_id(from_id, "from_id")
        _validate_id(to_id, "to_id")
        return self._relations_dir(world_id) / f"{from_id}__{to_id}__initial.json"

    # ------------------------------------------------------------------
    # AgentStoreProvider — agent state
    # ------------------------------------------------------------------

    async def save_agent_state(
        self, world_id: str, agent_id: str, state: AgentState
    ) -> None:
        path = self._state_path(world_id, agent_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Offload blocking write off the event loop — save_agent_state fires during
        # concurrent cognition phases and must not freeze in-flight LLM calls.
        await asyncio.to_thread(atomic_write_text, path, dump_json(dataclasses.asdict(state)))
        logger.debug("agent_state_saved", extra={"world_id": world_id, "agent_id": agent_id})

    async def load_agent_state(
        self, world_id: str, agent_id: str
    ) -> Optional[AgentState]:
        path = self._state_path(world_id, agent_id)
        data = await asyncio.to_thread(read_json_file, path)
        if data is None:
            return None
        data.setdefault("vitality", 1.0)
        return _state_from_data(data)

    async def list_agent_ids(self, world_id: str) -> List[str]:
        return await asyncio.to_thread(self._scan_agent_ids, self._world_dir(world_id))

    @staticmethod
    def _scan_agent_ids(world_dir: Path) -> List[str]:
        if not world_dir.exists():
            return []
        return [
            p.stem
            for p in sorted(world_dir.glob("*.json"))
            if not p.stem.endswith("__initial")
        ]

    # ------------------------------------------------------------------
    # AgentStoreProvider — relations
    # ------------------------------------------------------------------

    async def save_relation(self, relation: AgentRelation) -> None:
        path = self._relation_path(relation.world_id, relation.from_id, relation.to_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # save_relation fires during the concurrent perception phase — keep the
        # blocking write off the event loop.
        await asyncio.to_thread(atomic_write_text, path, dump_json(dataclasses.asdict(relation)))

    async def load_relation(
        self, world_id: str, from_id: str, to_id: str
    ) -> Optional[AgentRelation]:
        path = self._relation_path(world_id, from_id, to_id)
        data = await asyncio.to_thread(read_json_file, path)
        return None if data is None else _relation_from_data(data)

    async def load_all_relations(
        self, world_id: str, agent_id: str
    ) -> List[AgentRelation]:
        # One thread hop for the whole glob/read/parse, not one per file: snapshots call this for
        # every agent every step, and relation files grow with the square of the cast size.
        return await asyncio.to_thread(
            self._read_relations, self._relations_dir(world_id), agent_id
        )

    @staticmethod
    def _read_relations(relations_dir: Path, agent_id: str) -> List[AgentRelation]:
        if not relations_dir.exists():
            return []
        relations: List[AgentRelation] = []
        for p in sorted(relations_dir.glob(f"{agent_id}__*.json")):
            if p.stem.endswith("__initial"):
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                relations.append(_relation_from_data(data))
            except Exception:
                logger.warning("relation_load_failed", extra={"path": str(p)})
        return relations

    async def clear_relations(self, world_id: str) -> None:
        await asyncio.to_thread(self._unlink_relations, self._relations_dir(world_id))

    @staticmethod
    def _unlink_relations(relations_dir: Path) -> None:
        if not relations_dir.exists():
            return
        for p in relations_dir.glob("*.json"):
            if p.stem.endswith("__initial"):
                continue
            p.unlink(missing_ok=True)

    async def save_initial_agent_state(
        self, world_id: str, agent_id: str, state: AgentState
    ) -> None:
        path = self._initial_state_path(world_id, agent_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(atomic_write_text, path, dump_json(dataclasses.asdict(state)))

    async def load_initial_agent_state(
        self, world_id: str, agent_id: str
    ) -> Optional[AgentState]:
        path = self._initial_state_path(world_id, agent_id)
        data = await asyncio.to_thread(read_json_file, path)
        return None if data is None else _state_from_data(data)

    async def save_initial_relation(self, relation: AgentRelation) -> None:
        path = self._initial_relation_path(relation.world_id, relation.from_id, relation.to_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(atomic_write_text, path, dump_json(dataclasses.asdict(relation)))

    async def load_all_initial_relations(
        self, world_id: str, agent_id: str
    ) -> List[AgentRelation]:
        return await asyncio.to_thread(
            self._read_initial_relations, self._relations_dir(world_id), agent_id
        )

    @staticmethod
    def _read_initial_relations(relations_dir: Path, agent_id: str) -> List[AgentRelation]:
        if not relations_dir.exists():
            return []
        relations: List[AgentRelation] = []
        for p in sorted(relations_dir.glob(f"{agent_id}__*__initial.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                relations.append(_relation_from_data(data))
            except Exception:
                logger.warning("initial_relation_load_failed", extra={"path": str(p)})
        return relations

    async def delete_world(self, world_id: str) -> None:
        # Purges states (agent_id.json), initial baselines (agent_id__initial.json),
        # and the relations/ subdir (current + initial) by removing the whole world tree.
        # ignore_errors: partial trees / concurrent unlinks must not raise from a
        # whole-world purge — the operation is irreversible and best-effort by design.
        await asyncio.to_thread(
            shutil.rmtree, self._world_dir(world_id), ignore_errors=True
        )
        logger.info("world_agent_store_deleted", extra={"world_id": world_id})
