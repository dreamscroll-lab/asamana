"""Vector store provider implementations.

Two brute-force stores (File for persistence, InMemory for tests) sharing the pure functions in
`similarity`.

Chroma is deliberately not supported: it silently drops sparse vectors (breaking hybrid search)
and its metadata only accepts scalars, so it can't express membership filters on
`related_agents`, which identity-scoped recall ("what I know about someone") depends on.
"""

from providers.vector_store.file import FileVectorStore
from providers.vector_store.in_memory import InMemoryVectorStore

__all__ = ["FileVectorStore", "InMemoryVectorStore"]
