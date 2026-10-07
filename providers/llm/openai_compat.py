"""Transport for any OpenAI-compatible endpoint, shared by every vendor.

The caller supplies the URL and key; which vendor lives where belongs to
``providers.llm.catalog``. Keeping those apart means a new endpoint is a few lines of config,
not another provider class.
"""

from __future__ import annotations

from typing import Any

from core.interfaces.llm import LLMMessage, LLMProvider, LLMResponse
from core.logging import get_logger

logger = get_logger(__name__)


class OpenAICompatProvider(LLMProvider):
    """LLM provider for any OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
        call_params: dict[str, Any] | None = None,
        accepts_temperature: bool = True,
    ) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self._call_params = dict(call_params or {})
        self._accepts_temperature = accepts_temperature
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        *,
        json_mode: bool = False,
    ) -> LLMResponse:
        payload: dict[str, object] = {}
        if json_mode:
            # The endpoint REJECTS this with a 400 unless the prompt says "json" somewhere.
            # Assert it here rather than let the 400 come back looking like an LLM failure and
            # get swallowed by a caller's fallback — a caller that asks for JSON without asking
            # the model for JSON has a bug, and a bug is not a degradation (CLAUDE.md Rule 3).
            if not any("json" in m.content.lower() for m in messages):
                raise ValueError(
                    "json_mode requires the prompt to contain the word 'json' "
                    f"(scene prompt for model={self.model} does not)"
                )
            payload["response_format"] = {"type": "json_object"}
        # Call params pass through uninterpreted (see config.models.SceneLLM). Omit them when
        # empty: they are endpoint-specific, and an endpoint that doesn't know a field may 400.
        if self._call_params:
            payload["extra_body"] = dict(self._call_params)
        # Some endpoints fix the temperature and reject any explicit value (see
        # config.models.LLMProviderConfig); there the caller's temperature is simply not sent.
        if self._accepts_temperature:
            payload["temperature"] = temperature
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                max_tokens=max_tokens,
                **payload,
            )
        except Exception as exc:
            logger.warning(
                "openai_compat_call_failed",
                extra={"provider": "openai_compat", "model": self.model, "error": str(exc)},
            )
            raise

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=response.model,
            # getattr because the field names and whether they exist depend on the endpoint, and
            # the SDK doesn't guarantee them; endpoints without thinking get ""/0.
            thinking=getattr(choice.message, "reasoning_content", None) or "",
            thinking_tokens=getattr(
                getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0
            ) or 0,
        )
