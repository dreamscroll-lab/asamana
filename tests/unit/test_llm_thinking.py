"""The thinking channel: what the model thought, whether it is recorded, and how much budget it used.

Thinking text and thinking token count are different things. The text is long and only read when
debugging, so it isn't persisted by default. The token count is a single integer, but it is the only
way to tell whether thinking is silently using up ``max_tokens``, so it is always recorded. Putting
both behind one switch means either bloated traces or tuning the budget blind.
"""

from __future__ import annotations

import pytest

import re
from pathlib import Path

from core.interfaces.llm import (
    MAX_TOKENS_MULTIPLIER, LLMMessage, LLMProvider, LLMResponse, LLMRouter, LLMScene, output_budget,
)
from core.interfaces.trace import LLMCallTrace
from providers.trace.in_memory import InMemoryTraceSink


class _ThinkingProvider(LLMProvider):
    """An endpoint that returns reasoning alongside the answer."""

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        *,
        json_mode: bool = False,
    ) -> LLMResponse:
        return LLMResponse(
            content='{"ok": true}',
            input_tokens=10,
            output_tokens=20,
            model="thinker",
            thinking="我先想了想这个主题意味着什么……",
            thinking_tokens=512,
        )


async def _run(capture: bool) -> LLMCallTrace:
    sink = InMemoryTraceSink()
    router = LLMRouter(
        {scene: _ThinkingProvider() for scene in LLMScene},
        trace_sink=sink,
        capture_thinking=capture,
    )
    await router.complete(LLMScene.WORLD_BUILDING, [LLMMessage(role="user", content="x")])
    return sink.llm_calls[0]


@pytest.mark.asyncio
async def test_thinking_text_is_not_recorded_by_default() -> None:
    """Thinking text isn't persisted by default: it is often several times longer than the answer, and
    thinking is enabled to improve the output, not to be read."""
    call = await _run(capture=False)

    assert call.thinking == ""
    assert call.response_content == '{"ok": true}'   # the answer is recorded as usual


@pytest.mark.asyncio
async def test_thinking_tokens_are_recorded_even_when_the_text_is_not() -> None:
    """The token count ignores the switch: it is the only way to tell whether thinking is eating the
    answer's budget. ``max_tokens`` is sized from the estimated output (see
    core.interfaces.llm.output_budget); thinking that consumes the margin truncates the JSON.
    """
    call = await _run(capture=False)

    assert call.thinking_tokens == 512


@pytest.mark.asyncio
async def test_thinking_text_is_recorded_when_asked_for() -> None:
    call = await _run(capture=True)

    assert call.thinking == "我先想了想这个主题意味着什么……"


@pytest.mark.asyncio
async def test_a_provider_without_thinking_is_byte_identical_to_before() -> None:
    """An endpoint without thinking support must be unaffected by this channel."""
    from providers.llm.mock import MockLLMProvider

    sink = InMemoryTraceSink()
    router = LLMRouter(
        {scene: MockLLMProvider() for scene in LLMScene},
        trace_sink=sink,
        capture_thinking=True,     # even when enabled
    )
    await router.complete(LLMScene.WORLD_BUILDING, [LLMMessage(role="user", content="x")])

    assert sink.llm_calls[0].thinking == ""
    assert sink.llm_calls[0].thinking_tokens == 0


def test_output_budget_scales_the_estimate_and_rounds_up_to_ten() -> None:
    assert output_budget(100) == 100 * MAX_TOKENS_MULTIPLIER
    assert output_budget(101) % 10 == 0
    assert output_budget(101) >= 101 * MAX_TOKENS_MULTIPLIER


def test_no_call_site_hard_codes_max_tokens() -> None:
    """A literal at a call site escapes MAX_TOKENS_MULTIPLIER: raising it would not reach that call."""
    literal = re.compile(r"max_tokens\s*=\s*\d|_MAX_TOKENS\s*=\s*\d")
    root = Path(__file__).resolve().parents[2]
    offenders = [
        f"{path.relative_to(root)}:{n}"
        for package in ("agent", "engine", "world", "tuning")
        for path in sorted((root / package).rglob("*.py"))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if literal.search(line)
    ]
    assert offenders == []
