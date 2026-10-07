"""Pure vector-similarity functions shared by the InMemory and File brute-force stores, so the
two can't drift apart.
"""

from __future__ import annotations

from math import sqrt
from typing import Any

from core.interfaces.vector_store import SearchResult


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def sparse_similarity(left: dict[int, float], right: dict[int, float]) -> float:
    """Cosine similarity of two sparse vectors, in [0, 1], on the same scale as dense cosine.

    Must be normalized, not a raw dot product over shared keys: a query with a proper noun can
    score well above 1.0 while dense cosine tops out at 1.0, so `fused = dense*w_d +
    sparse*w_s` would stop being a weighted average. It would be unbounded, scale with query
    length and be incomparable across streams, making any absolute threshold meaningless.
    """
    if not left or not right:
        return 0.0
    dot = sum(left[key] * right[key] for key in set(left) & set(right))
    if dot == 0.0:
        return 0.0
    left_norm = sqrt(sum(v * v for v in left.values()))
    right_norm = sqrt(sum(v * v for v in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def matches_filters(payload: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Filter semantics depend on the payload field's type (contract in
    VectorStoreProvider.search):

    - scalar field -> equality (``{"kind": "insight"}``)
    - list field -> membership (``{"related_agents": "agent-x"}`` matches any record whose
      related_agents contains agent-x)

    Membership filters make identity-scoped questions ("what I know about someone") a
    constraint applied before ranking. Filtering after ranking would lose matches to the top_k
    cut, since similarity ranking knows nothing about identity.
    """
    if not filters:
        return True
    for key, value in filters.items():
        field = payload.get(key)
        if isinstance(field, list):
            if value not in field:
                return False
        elif field != value:
            return False
    return True


def hybrid_score(
    record: dict[str, Any],
    dense_vector: list[float],
    sparse_vector: dict[int, float] | None,
    dense_weight: float,
    sparse_weight: float,
) -> tuple[float, float, float]:
    """Return ``(fused, dense, sparse)``.

    The raw scores come back with the fused one because fused depends on the weights and is only
    good for ranking within one query. An absolute relevance floor must use dense (plain cosine
    in [-1, 1], independent of weights and store implementation).
    """
    dense = cosine_similarity(record["dense"], dense_vector)
    sparse = sparse_similarity(record["sparse"], sparse_vector or {})
    return dense * dense_weight + sparse * sparse_weight, dense, sparse


def rank_records(
    records: dict[str, dict[str, Any]],
    dense_vector: list[float],
    sparse_vector: dict[int, float] | None,
    *,
    top_k: int,
    filters: dict[str, Any] | None,
    dense_weight: float,
    sparse_weight: float,
) -> list[SearchResult]:
    """Score every record passing ``filters`` and return the best ``top_k``; the search shared by
    the File and InMemory stores."""
    results = []
    for record_id, record in records.items():
        if not matches_filters(record["payload"], filters):
            continue
        fused, dense, sparse = hybrid_score(record, dense_vector, sparse_vector, dense_weight, sparse_weight)
        results.append(SearchResult(
            id=record_id, score=fused, payload=record["payload"], dense_score=dense, sparse_score=sparse,
        ))
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k]
