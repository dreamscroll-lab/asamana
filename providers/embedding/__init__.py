"""Embedding provider implementations."""

from providers.embedding.dashscope import DashScopeEmbeddingProvider
from providers.embedding.in_memory import InMemoryEmbeddingProvider
from providers.embedding.openai_compat import OpenAICompatEmbeddingProvider

# Both network providers import their SDKs inside the constructor, so importing them here is
# cheap and lets config select any of them.
__all__ = [
    "DashScopeEmbeddingProvider",
    "InMemoryEmbeddingProvider",
    "OpenAICompatEmbeddingProvider",
]
