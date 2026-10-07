"""Any OpenAI-compatible ``/embeddings`` endpoint, configured with an endpoint, key and model.

Dense only: hybrid search falls back to dense (stores treat ``sparse=None`` as normal). Sparse
vectors need the DashScope native endpoint (``providers/embedding/dashscope.py``).

Embedding and LLM endpoints differ even for one vendor, so this declares its own ``base_url``,
and its key defaults to ``EMBEDDING_API_KEY``, separate from the LLM key and not vendor-named.
"""

from __future__ import annotations

import os
from typing import Any

from core.factory import ComponentKind, ProviderFactory
from core.interfaces.embedding import (
    DEFAULT_API_KEY_ENV,
    EmbeddingProvider,
    EmbeddingResult,
)
from providers.embedding.cache import EmbeddingCache

_DEFAULT_TIMEOUT_SECONDS = 60.0


@ProviderFactory.register("openai_compat", kind=ComponentKind.EMBEDDING)
class OpenAICompatEmbeddingProvider(EmbeddingProvider):
    """Embeddings from any OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        dimension: int,
        base_url: str,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        **call_params: Any,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} is not set (required by the openai_compat embedding)")
        from openai import AsyncOpenAI

        # ``dimension`` is ours: the vector store's contract (a mismatched upsert raises).
        # ``call_params`` belong to the endpoint and pass through uninterpreted. To make the
        # endpoint return that dimension, set the vendor parameter ``dimensions: 1024`` in config.
        self.api_key_env = api_key_env
        self._model = model
        self._dimension = dimension
        self._call_params = dict(call_params)
        self._cache = EmbeddingCache()
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> EmbeddingResult:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        payload: dict[str, Any] = {}
        if self._call_params:
            payload["extra_body"] = dict(self._call_params)
        response = await self._client.embeddings.create(model=self._model, input=[text], **payload)
        # Raise rather than store an empty vector; callers handle embed failures under Rule 1.
        if len(response.data) != 1:
            raise ValueError(f"embedding endpoint returned {len(response.data)} vectors for 1 text")
        result = EmbeddingResult(dense=list(response.data[0].embedding))
        self._cache.put(text, result)
        return result
