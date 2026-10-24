"""File-backed brute-force vector store with append-only JSONL persistence.

A sibling of InMemoryVectorStore, not a subclass; they share only the pure functions in
`similarity`. Search scans every record (dense*w + sparse*w, top_k), fine at this scale.

Append-only because the same memory is updated constantly (retrieve's touch, decay): a full
rewrite would make every recall O(N), a line is O(1), and payload-only updates skip the vector.

Format: one .jsonl per collection. Line types:
  - `{"_meta": {"collection": ..., "dim": N}}`  first line (the sanitized file name is lossy)
  - `{"id":..., "dense":[...], "sparse":{...}, "payload":{...}}`  upsert (last write wins)
  - `{"id":..., "_payload": {...}}`  update_payload (new payload, keeps the stored vectors)
  - `{"id":..., "_del": true}`  delete tombstone
Loading replays the lines in order. Compaction (an atomic rewrite to meta + live records) happens
only on this process's first write to a collection: a read-only process (list/observe/web) that
rewrote would erase memories a concurrently running simulation appended in between.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar

from core.factory import ComponentKind, ProviderFactory
from core.interfaces.vector_store import SearchResult, VectorStoreProvider
from core.logging import get_logger
from core.serialization import dump_json
from providers.vector_store.similarity import rank_records

logger = get_logger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_T = TypeVar("_T")


@ProviderFactory.register("file", kind=ComponentKind.VECTOR_STORE)
class FileVectorStore(VectorStoreProvider):
    """Brute-force store with its own in-memory index, persisted as one append-only .jsonl per collection."""

    def __init__(self, path: str = "./data/vectors") -> None:
        # collection -> {id -> {"dense", "sparse", "payload"}}
        self._records: dict[str, dict[str, dict[str, Any]]] = {}
        self._dimensions: dict[str, int | None] = {}
        self._files: dict[str, Path] = {}
        self._compacted: set[str] = set()  # collections this process has compacted (it is their writer)
        self._write_locks: dict[str, asyncio.Lock] = {}
        self._dir = Path(path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._load_all()

    # --- reads (brute-force over the in-memory index) ----------------------
    async def search(
        self,
        collection: str,
        dense_vector: list[float],
        sparse_vector: dict[int, float] | None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[SearchResult]:
        return rank_records(
            self._records.get(collection, {}), dense_vector, sparse_vector, top_k=top_k,
            filters=filters, dense_weight=dense_weight, sparse_weight=sparse_weight,
        )

    async def list_all(self, collection: str) -> list[SearchResult]:
        return [
            SearchResult(id=rid, score=1.0, payload=deepcopy(rec["payload"]))
            for rid, rec in self._records.get(collection, {}).items()
        ]

    # --- writes (persist before committing the index) --------------------
    async def create_collection(self, collection: str, dimension: int) -> None:
        async def write() -> None:
            previous = self._dimensions.get(collection)
            self._dimensions[collection] = dimension
            try:
                await asyncio.to_thread(self._ensure_meta, collection)
            except Exception:
                if previous is None:
                    self._dimensions.pop(collection, None)
                else:
                    self._dimensions[collection] = previous
                raise
            self._records.setdefault(collection, {})

        await self._write_ordered(collection, write)

    async def upsert(
        self,
        collection: str,
        id: str,
        dense_vector: list[float],
        sparse_vector: dict[int, float] | None,
        payload: dict[str, Any],
    ) -> None:
        rec = {"dense": list(dense_vector), "sparse": dict(sparse_vector or {}), "payload": deepcopy(payload)}

        async def write() -> None:
            expected = self._dimensions.get(collection)
            if expected is not None and len(rec["dense"]) != expected:
                raise ValueError(
                    f"Dense vector dimension mismatch for collection '{collection}': "
                    f"expected {expected}, got {len(rec['dense'])}"
                )
            await self._persist_change(collection, id, rec, {"id": id, **rec})
            self._records.setdefault(collection, {})[id] = rec

        await self._write_ordered(collection, write)

    async def update_payload(self, collection: str, id: str, payload: dict[str, Any]) -> bool:
        saved_payload = deepcopy(payload)

        async def write() -> bool:
            record = self._records.get(collection, {}).get(id)
            if record is None:
                return False
            updated = {**record, "payload": saved_payload}
            await self._persist_change(collection, id, updated, {"id": id, "_payload": saved_payload})
            self._records[collection][id] = updated
            return True

        return await self._write_ordered(collection, write)

    async def delete(self, collection: str, id: str) -> None:
        async def write() -> None:
            await self._persist_change(collection, id, None, {"id": id, "_del": True})
            self._records.get(collection, {}).pop(id, None)

        await self._write_ordered(collection, write)

    async def _write_ordered(self, collection: str, write: Callable[[], Awaitable[_T]]) -> _T:
        lock = self._write_locks.setdefault(collection, asyncio.Lock())

        async def run() -> _T:
            async with lock:
                return await write()

        # Cancelling to_thread does not stop its thread. Finish the accepted write and its index
        # commit before releasing the lock, or a later write/delete can be undone on disk.
        task = asyncio.create_task(run())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()
            raise

    async def _persist_change(
        self,
        collection: str,
        record_id: str,
        record: dict[str, Any] | None,
        change: dict[str, Any],
    ) -> None:
        if collection not in self._compacted:
            snapshot = dict(self._records.get(collection, {}))
            if record is None:
                snapshot.pop(record_id, None)
            else:
                snapshot[record_id] = record
            self._compacted.add(collection)
            if await asyncio.to_thread(self._rewrite, collection, snapshot):
                return
        await asyncio.to_thread(self._append_with_meta, collection, change)

    # --- persistence -------------------------------------------------------
    def _file_for(self, collection: str) -> Path:
        fp = self._files.get(collection)
        if fp is None:
            # Readable prefix plus a short hash: sanitizing is lossy, and the hash keeps distinct
            # collections from sharing a file.
            safe = _UNSAFE.sub("_", collection)
            digest = hashlib.sha1(collection.encode("utf-8")).hexdigest()[:8]
            fp = self._dir / f"{safe}.{digest}.jsonl"
            self._files[collection] = fp
        return fp

    @staticmethod
    def _dumps(obj: dict[str, Any]) -> str:
        return dump_json(obj) + "\n"

    def _append(self, collection: str, obj: dict[str, Any]) -> None:
        try:
            with self._file_for(collection).open("a", encoding="utf-8") as f:
                f.write(self._dumps(obj))
        except Exception as exc:
            logger.error("vector_store_append_failed", extra={"collection": collection, "error": str(exc)})
            raise

    @staticmethod
    def _unlink_collections(paths: list[Path]) -> None:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "vector_store_delete_unlink_failed",
                    extra={"path": str(path), "error": str(exc)},
                )

    def _append_with_meta(self, collection: str, obj: dict[str, Any]) -> None:
        """Write the meta header and change while the caller holds the collection's write lock."""
        self._ensure_meta(collection)
        self._append(collection, obj)

    def _ensure_meta(self, collection: str) -> None:
        """Write the meta header if the file doesn't exist yet."""
        if not self._file_for(collection).exists():
            self._append(collection, {"_meta": {"collection": collection, "dim": self._dimensions.get(collection)}})

    def _load_all(self) -> None:
        for fp in self._dir.glob("*.jsonl"):
            try:
                collection: str | None = None
                dim: int | None = None
                records: dict[str, dict[str, Any]] = {}
                skipped = 0
                for line in fp.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    # Skip a bad line, never the file: a write cut short by a crash or a full disk
                    # leaves one, and dropping the whole collection would let the first write's
                    # compaction erase it from disk.
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    if not isinstance(obj, dict):
                        skipped += 1
                        continue
                    if "_meta" in obj:
                        collection = obj["_meta"].get("collection")
                        dim = obj["_meta"].get("dim")
                    elif obj.get("_del"):
                        records.pop(obj["id"], None)
                    elif "_payload" in obj:
                        existing = records.get(obj["id"])
                        if existing is not None:  # drop orphan payload lines whose upsert was tombstoned
                            existing["payload"] = dict(obj["_payload"])
                    elif "id" in obj:
                        # JSON stores sparse keys as strings; convert back to int or they never match query keys.
                        records[obj["id"]] = {
                            "dense": list(obj.get("dense", [])),
                            "sparse": {int(k): float(v) for k, v in (obj.get("sparse") or {}).items()},
                            "payload": dict(obj.get("payload", {})),
                        }
                if skipped:
                    logger.warning(
                        "vector_store_lines_skipped", extra={"file": str(fp), "lines": skipped}
                    )
                if collection is None:
                    logger.warning("vector_store_no_meta", extra={"file": str(fp)})
                    continue
                self._dimensions[collection] = dim
                self._records[collection] = records
                self._files[collection] = fp
                # Don't compact on load; that waits for this process's first write
                # (_compact_on_first_write) so a read-only process never overwrites a concurrent
                # writer's appends.
            except Exception as exc:  # noqa: BLE001 — one corrupt file must not break the store
                logger.warning("vector_store_load_failed", extra={"file": str(fp), "error": str(exc)})

    async def delete_world(self, world_id: str) -> None:
        # Drop every collection named f"{world_id}:…" (each per-agent stream of this world).
        # Unlink is best-effort: a missing file must not stall the in-memory purge.
        prefix = f"{world_id}:"
        seen: set[str] = set()
        for source in (self._records, self._files, self._write_locks):
            for collection in list(source):
                if collection.startswith(prefix):
                    seen.add(collection)
        for collection in seen:
            async def delete(collection: str = collection) -> None:
                path = self._files.get(collection)
                if path is not None:
                    await asyncio.to_thread(self._unlink_collections, [path])
                self._records.pop(collection, None)
                self._dimensions.pop(collection, None)
                self._files.pop(collection, None)
                self._compacted.discard(collection)

            await self._write_ordered(collection, delete)
            self._write_locks.pop(collection, None)
        logger.info(
            "world_vectors_deleted",
            extra={"world_id": world_id, "collections": len(seen)},
        )

    def _rewrite(self, collection: str, records: dict[str, dict[str, Any]]) -> bool:
        """Atomically rewrite the .jsonl as meta + ``records``. True on success."""
        fp = self._file_for(collection)
        tmp = fp.with_name(fp.name + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                f.write(self._dumps({"_meta": {"collection": collection, "dim": self._dimensions.get(collection)}}))
                for rid, rec in records.items():
                    f.write(self._dumps({"id": rid, **rec}))
            tmp.replace(fp)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("vector_store_rewrite_failed", extra={"collection": collection, "error": str(exc)})
            return False
