"""The embedding the deployment gets when it has no model keys."""

from __future__ import annotations

from core.interfaces.embedding import EmbeddingProvider, EmbeddingResult
from core.model_keys import ModelKeysMissing


class KeylessEmbeddingProvider(EmbeddingProvider):
    """Raises on every use, ``dimension`` included: there is no real value to report, and a guessed
    one would reach the vector store as a collection's dimension."""

    async def embed(self, text: str) -> EmbeddingResult:
        raise ModelKeysMissing()

    @property
    def dimension(self) -> int:
        raise ModelKeysMissing()
