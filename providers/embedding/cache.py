"""Bounded LRU for single-text embeddings, shared by the networked providers.

Embeddings are deterministic for a given (text, model, dimension, …), so identical
single-text calls — chiefly repeated retrieval queries — are served from here instead of
re-hitting the endpoint. Zero output change; purely avoids duplicate round-trips.
Mostly-unique write content self-evicts, so it never grows unbounded.
"""

from __future__ import annotations

from collections import OrderedDict

from core.interfaces.embedding import EmbeddingResult

_MAX_ENTRIES = 512


def _copy(result: EmbeddingResult) -> EmbeddingResult:
    # Copy on both store and fetch: a caller mutating a returned vector would otherwise corrupt the
    # cache silently for every later hit.
    return EmbeddingResult(
        dense=list(result.dense),
        sparse=dict(result.sparse) if result.sparse is not None else None,
    )


class EmbeddingCache:
    """LRU keyed by the text itself."""

    def __init__(self, max_entries: int = _MAX_ENTRIES) -> None:
        self._entries: "OrderedDict[str, EmbeddingResult]" = OrderedDict()
        self._max = max_entries

    def get(self, text: str) -> EmbeddingResult | None:
        cached = self._entries.get(text)
        if cached is None:
            return None
        self._entries.move_to_end(text)
        return _copy(cached)

    def put(self, text: str, result: EmbeddingResult) -> None:
        self._entries[text] = _copy(result)
        if len(self._entries) > self._max:
            self._entries.popitem(last=False)
