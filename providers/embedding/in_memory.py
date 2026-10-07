"""In-memory embedding provider."""

from __future__ import annotations

import hashlib
import random

from core.factory import ComponentKind, ProviderFactory
from core.interfaces.embedding import EmbeddingProvider, EmbeddingResult


@ProviderFactory.register("in_memory", kind=ComponentKind.EMBEDDING)
class InMemoryEmbeddingProvider(EmbeddingProvider):
    """Cheap deterministic embeddings for tests.

    Uses MD5-seeded Gaussian RNG so the same text always yields the same
    vector while different texts yield distinct vectors.  Sparse is always
    None — the provider does not model keyword weights.
    """

    def __init__(self, dimension: int = 1024) -> None:
        self._dimension = dimension

    async def embed(self, text: str) -> EmbeddingResult:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        dense = [rng.gauss(0, 1) for _ in range(self._dimension)]
        return EmbeddingResult(dense=dense, sparse=None)

    @property
    def dimension(self) -> int:
        return self._dimension
