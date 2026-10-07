"""In-memory vector store."""

from __future__ import annotations

from typing import Any, Dict

from core.factory import ComponentKind, ProviderFactory
from core.interfaces.vector_store import SearchResult, VectorStoreProvider
from providers.vector_store.similarity import rank_records


@ProviderFactory.register("in_memory", kind=ComponentKind.VECTOR_STORE)
class InMemoryVectorStore(VectorStoreProvider):
    """Minimal vector store implementation."""

    def __init__(self) -> None:
        self._records: Dict[str, Dict[str, dict[str, Any]]] = {}
        self._dimensions: dict[str, int] = {}

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
        collection_records = self._records.setdefault(collection, {})
        collection_records[id] = {
            "dense": list(dense_vector),
            "sparse": dict(sparse_vector or {}),
            "payload": dict(payload),
        }

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

    async def update_payload(self, collection: str, id: str, payload: dict[str, Any]) -> bool:
        record = self._records.get(collection, {}).get(id)
        if record is None:
            return False
        record["payload"] = dict(payload)
        return True

    async def delete(self, collection: str, id: str) -> None:
        self._records.get(collection, {}).pop(id, None)

    async def list_all(self, collection: str) -> list[SearchResult]:
        collection_records = self._records.get(collection, {})
        return [
            SearchResult(id=record_id, score=1.0, payload=record["payload"])
            for record_id, record in collection_records.items()
        ]

    async def create_collection(self, collection: str, dimension: int) -> None:
        self._records.setdefault(collection, {})
        self._dimensions[collection] = dimension

    async def delete_world(self, world_id: str) -> None:
        prefix = f"{world_id}:"
        for collection in [c for c in self._records if c.startswith(prefix)]:
            del self._records[collection]
        for collection in [c for c in self._dimensions if c.startswith(prefix)]:
            del self._dimensions[collection]
