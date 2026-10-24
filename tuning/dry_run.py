"""Storage shadows for restoring and exercising a world without changing its baseline."""

from __future__ import annotations

import dataclasses
from copy import deepcopy
from typing import Any

from core.container import Container
from core.interfaces.vector_store import SearchResult, VectorStoreProvider
from providers.message.in_memory import InMemoryMessageProvider
from providers.vector_store.in_memory import InMemoryVectorStore
from providers.vector_store.similarity import matches_filters


class DryRunVectorStore(VectorStoreProvider):
    """Read baseline vectors, keeping replacements and deletions local to one dry-run."""

    def __init__(self, baseline: VectorStoreProvider) -> None:
        self._baseline = baseline
        self._shadow = InMemoryVectorStore()
        self._hidden: dict[str, set[str]] = {}
        self._payloads: dict[tuple[str, str], dict[str, Any]] = {}
        self._deleted_worlds: set[str] = set()

    def _world_deleted(self, collection: str) -> bool:
        return collection.partition(":")[0] in self._deleted_worlds

    async def upsert(
        self, collection: str, id: str, dense_vector: list[float],
        sparse_vector: dict[int, float] | None, payload: dict[str, Any],
    ) -> None:
        self._hidden.setdefault(collection, set()).add(id)
        self._payloads.pop((collection, id), None)
        await self._shadow.upsert(collection, id, dense_vector, sparse_vector, deepcopy(payload))

    async def search(
        self, collection: str, dense_vector: list[float], sparse_vector: dict[int, float] | None,
        top_k: int = 10, filters: dict[str, Any] | None = None,
        dense_weight: float = 0.7, sparse_weight: float = 0.3,
    ) -> list[SearchResult]:
        hidden = self._hidden.get(collection, set())
        baseline_filters = filters
        baseline_limit = top_k + len(hidden)
        if filters and any(c == collection for c, _ in self._payloads):
            # A changed payload can enter or leave the filter independently of its stored vector.
            baseline_filters = None
            baseline_limit = len(await self._baseline.list_all(collection))
        baseline = [] if self._world_deleted(collection) else await self._baseline.search(
            collection, dense_vector, sparse_vector, top_k=baseline_limit,
            filters=baseline_filters, dense_weight=dense_weight, sparse_weight=sparse_weight,
        )
        results = [self._view(collection, r) for r in baseline if r.id not in hidden]
        results = [r for r in results if matches_filters(r.payload, filters)]
        results.extend(await self._shadow.search(
            collection, dense_vector, sparse_vector, top_k=top_k, filters=filters,
            dense_weight=dense_weight, sparse_weight=sparse_weight,
        ))
        return deepcopy(sorted(results, key=lambda r: r.score, reverse=True)[:top_k])

    def _view(self, collection: str, result: SearchResult) -> SearchResult:
        return dataclasses.replace(result, payload=deepcopy(
            self._payloads.get((collection, result.id), result.payload)
        ))

    async def update_payload(self, collection: str, id: str, payload: dict[str, Any]) -> bool:
        if await self._shadow.update_payload(collection, id, deepcopy(payload)):
            return True
        if any(r.id == id for r in await self.list_all(collection)):
            self._payloads[(collection, id)] = deepcopy(payload)
            return True
        return False

    async def delete(self, collection: str, id: str) -> None:
        self._hidden.setdefault(collection, set()).add(id)
        self._payloads.pop((collection, id), None)
        await self._shadow.delete(collection, id)

    async def list_all(self, collection: str) -> list[SearchResult]:
        hidden = self._hidden.get(collection, set())
        baseline = [] if self._world_deleted(collection) else await self._baseline.list_all(collection)
        return [self._view(collection, r) for r in baseline if r.id not in hidden] + deepcopy(
            await self._shadow.list_all(collection)
        )

    async def create_collection(self, collection: str, dimension: int) -> None:
        await self._shadow.create_collection(collection, dimension)

    async def delete_world(self, world_id: str) -> None:
        self._deleted_worlds.add(world_id)
        await self._shadow.delete_world(world_id)


async def isolate_container(container: Container, world_id: str) -> Container:
    """Shadow the stores before restore: restore itself removes memories."""
    messages = InMemoryMessageProvider()
    for message in await container.message_provider.peek_pending(world_id):
        await messages.enqueue(deepcopy(message))
    return dataclasses.replace(
        container, vector_store=DryRunVectorStore(container.vector_store),
        message_provider=messages,
    )
