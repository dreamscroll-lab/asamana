"""DashScope native text-embedding provider: dense plus learned sparse vectors for hybrid search.

The OpenAI-compatible endpoint (`/compatible-mode/v1/embeddings`) returns dense vectors only.
Sparse vectors require the native endpoint
(`/api/v1/services/embeddings/text-embedding/text-embedding` with
`parameters.output_type="dense&sparse"`), where text-embedding-v3/v4 return both a dense vector
and a learned sparse vector (token -> weight), with no local bge-m3 needed.

Required env:
  EMBEDDING_API_KEY (renamable via the ``api_key_env`` param). Embedding credentials are separate
  from the LLM's and not tied to a vendor name, so any embedding vendor uses this one variable.
  The endpoint defaults to the native URL; override it with ``base_url`` (own gateway / other
  region).

Returns the API's dense vector and the raw learned sparse vector unnormalized.
The file / in_memory stores fuse the two in hybrid_score; weights are in VECTOR_WEIGHTS.
"""

from __future__ import annotations

import asyncio
import os

from core.factory import ComponentKind, ProviderFactory
from core.interfaces.embedding import (
    DEFAULT_API_KEY_ENV,
    EmbeddingProvider,
    EmbeddingResult,
)
from core.logging import get_logger
from core.rate_gate import classify_rate_limit
from providers.embedding.cache import EmbeddingCache

logger = get_logger(__name__)

_DEFAULT_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
_DEFAULT_TIMEOUT_SECONDS = 60.0
# Raw httpx has none of the openai SDK's built-in retries, so transient failures are retried here:
# backpressure (429/503/529) and a connection that never got made. Waits 1s, 2s, 4s unless the
# endpoint sends Retry-After.
_TRANSIENT_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0
# A longer Retry-After (a spent quota) isn't waited out inside one call: the error goes up to the
# shared gate, which caps it and cools every caller down together. Don't lower this toward the
# backoff: a memory whose embedding fails is never written to the vector store, so a routine
# throttle must still be waited out and retried.
_MAX_INLINE_WAIT_SECONDS = 60.0


def _parse_embeddings(payload: dict) -> list[EmbeddingResult]:
    """Parse the native response into (dense, sparse) results.

    sparse_embedding looks like [{"index":int,"token":str,"value":float}, ...] and becomes
    {index: value} (raw).
    """
    out: list[EmbeddingResult] = []
    for e in payload.get("output", {}).get("embeddings", []):
        sparse = {int(it["index"]): float(it["value"]) for it in (e.get("sparse_embedding") or [])}
        out.append(EmbeddingResult(dense=list(e.get("embedding", [])), sparse=sparse or None))
    return out


@ProviderFactory.register("dashscope", kind=ComponentKind.EMBEDDING)
class DashScopeEmbeddingProvider(EmbeddingProvider):
    """DashScope native embedding: dense + learned sparse (default text-embedding-v3, 1024-d)."""

    def __init__(
        self,
        model: str = "text-embedding-v3",
        dimension: int = 1024,
        output_type: str = "dense&sparse",
        base_url: str = _DEFAULT_URL,
        api_key_env: str = DEFAULT_API_KEY_ENV,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} is not set (required by the dashscope embedding)")
        import httpx

        self.api_key_env = api_key_env
        self._url = base_url
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._model = model
        self._dimension = dimension
        self._output_type = output_type
        self._cache = EmbeddingCache()
        self._client = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_SECONDS)

    async def embed(self, text: str) -> EmbeddingResult:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        parsed = _parse_embeddings(await self._post({
            "model": self._model,
            "input": {"texts": [text]},
            "parameters": {"output_type": self._output_type, "dimension": self._dimension},
        }))
        # Quota or parameter errors come back as 200 with an error body, which parses to no
        # vector: raise rather than store an empty one. Callers handle embed failures under Rule 1
        # (only this item's recall is lost).
        if len(parsed) != 1:
            raise ValueError(f"dashscope embedding returned {len(parsed)} vectors for 1 text")
        result = parsed[0]
        self._cache.put(text, result)
        return result

    async def _post(self, body: dict) -> dict:
        import httpx

        attempt = 0
        while True:
            try:
                resp = await self._client.post(self._url, headers=self._headers, json=body)
                resp.raise_for_status()
                return resp.json()
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                if attempt == _TRANSIENT_RETRIES:
                    raise
                reason, retry_after = type(exc).__name__, None
            except httpx.HTTPStatusError as exc:
                limited, retry_after = classify_rate_limit(exc)
                if not limited or attempt == _TRANSIENT_RETRIES:
                    raise
                if retry_after is not None and retry_after > _MAX_INLINE_WAIT_SECONDS:
                    raise
                reason = str(exc.response.status_code)
            delay = retry_after or _BACKOFF_BASE_SECONDS * 2 ** attempt
            attempt += 1
            logger.warning(
                "embedding_retry",
                extra={"attempt": attempt, "reason": reason, "delay_s": delay},
            )
            await asyncio.sleep(delay)

    @property
    def dimension(self) -> int:
        return self._dimension
