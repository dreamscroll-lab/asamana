"""Embedding contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

#: Read by a networked embedding provider when no ``api_key_env`` is given. Separate from the LLM
#: key; when both are one account, set both to the same value.
DEFAULT_API_KEY_ENV = "EMBEDDING_API_KEY"


@dataclass
class EmbeddingResult:
    """Embedding output."""

    dense: list[float]
    sparse: dict[int, float] | None = None


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    #: The env var this provider reads its credential from; the container gates it only when set.
    #: ``None`` = hits no endpoint (e.g. ``InMemoryEmbeddingProvider``). Networked implementations
    #: must set it in ``__init__``: forgetting silently loses backpressure.
    api_key_env: str | None = None

    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dense vector dimension."""
