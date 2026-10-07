"""Adaptive-backpressure wrapper for a networked embedding provider.

Embeddings bypass the LLM router, so they get their own gate: the embedding endpoint has its
own limits even when it bills to the same key as an LLM, so a 429 on one never throttles the
other. They pass ``reserve_rate=False`` (separate quota, no token counts), so they only wait
out the cooldown, without reserving RPM slots or feeding the chat TPM governor.

Applied at container assembly to embedding providers that name a credential; InMemory names none.
"""

from __future__ import annotations

from core.interfaces.embedding import EmbeddingProvider, EmbeddingResult
from core.rate_gate import RateGate, classify_rate_limit


class GatedEmbeddingProvider(EmbeddingProvider):
    """Wrap an EmbeddingProvider so its calls share the embedding endpoint's cooldown."""

    def __init__(self, inner: EmbeddingProvider, rate_gate: RateGate) -> None:
        self._inner = inner
        self._gate = rate_gate

    async def embed(self, text: str) -> EmbeddingResult:
        await self._gate.acquire(reserve_rate=False)
        try:
            result = await self._inner.embed(text)
        except BaseException as exc:
            self._penalize_if_backpressure(exc)
            raise
        self._gate.note_success()
        return result

    def _penalize_if_backpressure(self, exc: BaseException) -> None:
        limited, retry_after = classify_rate_limit(exc)
        if limited:
            self._gate.penalize(retry_after)

    @property
    def dimension(self) -> int:
        return self._inner.dimension
