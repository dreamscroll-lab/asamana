"""Mock LLM providers for tests and local development."""

from __future__ import annotations

from collections.abc import Iterator

from core.factory import ComponentKind, ProviderFactory
from core.interfaces.llm import LLMMessage, LLMProvider, LLMResponse


@ProviderFactory.register("mock", kind=ComponentKind.LLM)
class MockLLMProvider(LLMProvider):
    """Deterministic provider used by tests.

    Always returns ``fixed_response``.  Tracks every call in
    ``call_history`` for assertion in tests.
    """

    def __init__(
        self,
        model: str = "mock",
        fixed_response: str = "mock response",
        **_endpoint_params: object,
    ) -> None:
        # Accept and ignore endpoint params: the mock stands in for any endpoint, so it must
        # accept any endpoint's constructor args (``llm.providers.<name>.params`` is splatted into
        # every scene routed to that provider).
        self.model = model
        self.fixed_response = fixed_response
        self.call_history: list[list[LLMMessage]] = []

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        *,
        json_mode: bool = False,   # accepted and ignored: a canned reply is already well-formed
    ) -> LLMResponse:
        self.call_history.append(list(messages))
        return LLMResponse(
            content=self.fixed_response,
            input_tokens=0,
            output_tokens=0,
            model=self.model,
        )


class SequentialMockLLM(LLMProvider):
    """Returns responses in the order they are supplied, then falls back.

    Useful for testing multi-step cognition sequences where each LLM call
    must return a specific value.

    Example::

        llm = SequentialMockLLM(["decide", "summarize", "reflect"])
        # first complete() → "decide"
        # second complete() → "summarize"
        # subsequent calls → "default mock response"
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses: Iterator[str] = iter(responses)
        self.call_history: list[list[LLMMessage]] = []

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        *,
        json_mode: bool = False,   # accepted and ignored: a canned reply is already well-formed
    ) -> LLMResponse:
        self.call_history.append(list(messages))
        content = next(self._responses, "default mock response")
        return LLMResponse(content=content, input_tokens=0, output_tokens=0, model="mock")
