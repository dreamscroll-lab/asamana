"""The LLM every scene gets when the deployment has no model keys."""

from __future__ import annotations

from core.interfaces.llm import LLMMessage, LLMProvider, LLMResponse
from core.model_keys import ModelKeysMissing


class KeylessLLMProvider(LLMProvider):
    """Raises on every call. The API refuses model-calling requests before they get here; this
    only keeps a path that slips past from failing somewhere less legible."""

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        *,
        json_mode: bool = False,
    ) -> LLMResponse:
        raise ModelKeysMissing()
