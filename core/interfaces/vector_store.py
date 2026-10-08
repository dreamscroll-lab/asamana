"""Vector store contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    """Similarity search result.

    ``score`` is the **fused** hybrid score (dense×w_d + sparse×w_s), scaled by the caller's
    weights: only for *ranking within one query*, never against an absolute threshold.

    ``dense_score`` is the raw cosine similarity ∈ [-1, 1]. A relevance floor must be applied to
    **this**: a floor on ``score`` moves with the weights and goes silently dead.
    ``sparse_score`` is the raw sparse overlap, for retrieval diagnostics.
    """

    id: str
    score: float
    payload: dict[str, Any]
    dense_score: float = 0.0
    sparse_score: float = 0.0


class VectorStoreProvider(ABC):
    """Abstract vector store."""

    @abstractmethod
    async def upsert(
        self,
        collection: str,
        id: str,
        dense_vector: list[float],
        sparse_vector: dict[int, float] | None,
        payload: dict[str, Any],
    ) -> None:
        """Store or replace one vector record."""

    @abstractmethod
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
        """Search records by vector similarity, restricted to records matching *filters*.

        ``filters`` scopes the **candidate set before ranking** — a filtered-out record can
        never occupy a top_k slot. Two predicate forms, chosen by the payload field's type:

        - payload value is a scalar → equality (``{"kind": "insight"}``)
        - payload value is a list   → **membership** (``{"related_agents": "agent-x"}`` matches
          any record whose ``related_agents`` list contains ``agent-x``)

        Membership makes identity-scoped recall ("what do I know about this person") a scope,
        not a post-filter, which would drop relevant records already truncated out of top_k.
        """

    @abstractmethod
    async def update_payload(self, collection: str, id: str, payload: dict[str, Any]) -> bool:
        """Replace one record's payload, keeping its stored vectors. False if no such record.

        For payload-only changes (last_accessed_step, decay_score): ``upsert`` would re-embed
        unchanged text. Callers fall back to a full ``upsert`` on False.
        """

    @abstractmethod
    async def delete(self, collection: str, id: str) -> None:
        """Delete one record if present."""

    @abstractmethod
    async def list_all(self, collection: str) -> list[SearchResult]:
        """Load all records in a collection."""

    @abstractmethod
    async def create_collection(self, collection: str, dimension: int) -> None:
        """Create a collection if it does not exist."""

    @abstractmethod
    async def delete_world(self, world_id: str) -> None:
        """Purge every collection belonging to *world_id*.

        Collections are per-agent-per-stream named ``f"{world_id}:{agent_id}:memory:{stream}"``,
        so this drops every collection whose name starts with ``f"{world_id}:"``.
        Called by whole-world deletion; irreversible.
        """
