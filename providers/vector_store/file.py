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
from pathlib import Path
from typing import Any

from core.factory import ComponentKind, ProviderFactory
from core.interfaces.vector_store import SearchResult, VectorStoreProvider
from core.logging import get_logger
from core.serialization import dump_json
from providers.vector_store.similarity import rank_records

logger = get_logger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


@ProviderFactory.register("file", kind=ComponentKind.VECTOR_STORE)
class FileVectorStore(VectorStoreProvider):
    """Brute-force store with its own in-memory index, persisted as one append-only .jsonl per collection."""

    def __init__(self, path: str = "./data/vectors") -> None:
        # collection -> {id -> {"dense", "sparse", "payload"}}
        self._records: dict[str, dict[str, dict[str, Any]]] = {}
        self._dimensions: dict[str, int | None] = {}
        self._files: dict[str, Path] = {}
        self._compacted: set[str] = set()  # collections this process has compacted (it is their writer)
        self._compacting: dict[str, asyncio.Future[bool]] = {}  # rewrites still in a worker thread
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
            SearchResult(id=rid, score=1.0, payload=rec["payload"])
            for rid, rec in self._records.get(collection, {}).items()
        ]

    # --- writes (update the index, then append to disk) -------------------
    async def create_collection(self, collection: str, dimension: int) -> None:
        self._records.setdefault(collection, {})
        self._dimensions[collection] = dimension
        await asyncio.to_thread(self._ensure_meta, collection)

    async def upsert(
        self,
        collection: str,
        id: str,
        dense_vector: list[float],
        sparse_vector: dict[int, float] | None,
        payload: dict[str, Any],
    ) -> None:
        expected = self._dimensions.get(collection)
        if expected is not None and len(dense_vector) != expected:
            raise ValueError(
                f"Dense vector dimension mismatch for collection '{collection}': "
                f"expected {expected}, got {len(dense_vector)}"
            )
        rec = {"dense": list(dense_vector), "sparse": dict(sparse_vector or {}), "payload": dict(payload)}
        self._records.setdefault(collection, {})[id] = rec
        if await self._compact_on_first_write(collection):
            return  # the rewrite already includes this record (_records was updated first)
        await asyncio.to_thread(self._append_with_meta, collection, {"id": id, **rec})

    async def update_payload(self, collection: str, id: str, payload: dict[str, Any]) -> bool:
        record = self._records.get(collection, {}).get(id)
        if record is None:
            return False
        record["payload"] = dict(payload)
        if await self._compact_on_first_write(collection):
            return True  # the rewrite already includes this record
        # Append only the payload, not the vector: touch/decay come through here on every recall.
        await asyncio.to_thread(
            self._append_with_meta, collection, {"id": id, "_payload": record["payload"]}
        )
        return True

    async def delete(self, collection: str, id: str) -> None:
        self._records.get(collection, {}).pop(id, None)
        if await self._compact_on_first_write(collection):
            return  # the rewritten file no longer has this id, so no tombstone needed
        await asyncio.to_thread(self._append, collection, {"id": id, "_del": True})

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
        except Exception as exc:  # noqa: BLE001 — log write failures; don't crash the simulation
            logger.error("vector_store_append_failed", extra={"collection": collection, "error": str(exc)})

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
        """Write the meta header and the change in one thread hop, so another write can't land between them."""
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

    async def _compact_on_first_write(self, collection: str) -> bool:
        """Compact the collection on this process's first write to it.

        Returns True if the file was rewritten (including the change the caller just put in
        _records), so the caller needn't append. Returns False if the rewrite failed or was
        already done; the caller then appends as usual (a failed rewrite leaves the original
        file intact).

        The ``_compacted`` bookkeeping must stay on the event loop (no await between the next
        two lines): it is check-then-act, and in a worker thread two concurrent writes to one
        collection could both see "not compacted" and both rewrite the file. Only the expensive
        rewrite itself runs in a thread.

        The thread gets a snapshot taken here, never the live dict: writes keep landing on the loop
        while it runs. Those writes wait for the rewrite before appending, or ``tmp.replace`` would
        erase their lines from disk while memory still holds them."""
        if collection in self._compacted:
            pending = self._compacting.get(collection)
            if pending is not None:
                await asyncio.shield(pending)
            return False
        self._compacted.add(collection)
        snapshot = {rid: dict(rec) for rid, rec in self._records.get(collection, {}).items()}
        task = asyncio.ensure_future(asyncio.to_thread(self._rewrite, collection, snapshot))
        self._compacting[collection] = task
        task.add_done_callback(lambda _: self._compacting.pop(collection, None))
        return await asyncio.shield(task)

    async def delete_world(self, world_id: str) -> None:
        # Drop every collection named f"{world_id}:…" (each per-agent stream of this world).
        # Unlink is best-effort: a missing file must not stall the in-memory purge.
        prefix = f"{world_id}:"
        seen: set[str] = set()
        for source in (self._records, self._files):
            for collection in list(source):
                if collection.startswith(prefix):
                    seen.add(collection)
        # Unlink in one thread hop: a world has two collections per character, each up to several MB.
        await asyncio.to_thread(
            self._unlink_collections,
            [p for p in (self._files.get(c) for c in seen) if p is not None],
        )
        for collection in seen:
            self._records.pop(collection, None)
            self._dimensions.pop(collection, None)
            self._files.pop(collection, None)
            self._compacted.discard(collection)
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
