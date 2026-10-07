"""Unit tests for the cross-scene dialogue progress metric (tuning/judge_dialogue_progress)."""

from __future__ import annotations

import pytest

from core.interfaces.llm import LLMProvider, LLMResponse, LLMRouter, LLMScene
from tuning.judge_dialogue_leak import DialogueCase
from tuning.judge_dialogue_progress import judge_one
from tuning.trace import InMemoryTraceSink


class _Mock(LLMProvider):
    def __init__(self, response: str = "", *, boom: bool = False) -> None:
        self.response, self.boom = response, boom
        self.seen: list[str] = []

    async def complete(self, messages, temperature=0.7, max_tokens=1000, **kw) -> LLMResponse:
        self.seen.append(messages[-1].content)
        if self.boom:
            raise RuntimeError("judge down")
        return LLMResponse(content=self.response, model="mock", input_tokens=1, output_tokens=1)


def _router(p: LLMProvider) -> LLMRouter:
    return LLMRouter({s: p for s in LLMScene}, trace_sink=InMemoryTraceSink())


def _case(step: int, line: str) -> DialogueCase:
    return DialogueCase(step=step, call_id=f"c{step}", name_a="甲", lane_a="", name_b="乙",
                        lane_b="", lines=[("甲", line), ("乙", "嗯。")])


@pytest.mark.asyncio
async def test_verdict_and_nth_are_recorded() -> None:
    m = _Mock('{"reason": "r", "new": "定下了时辰", "verdict": "advance"}')
    out = await judge_one(_router(m), [_case(1, "催一次")], _case(3, "定了寅时"),
                          judge_scene=LLMScene.WORLD_BUILDING)
    assert out["judged"] is True and out["verdict"] == "advance" and out["nth"] == 2
    assert out["new"] == "定下了时辰"


@pytest.mark.asyncio
async def test_every_prior_scene_reaches_the_prompt() -> None:
    """Judging whether anything new was said requires the full history. Omit one conversation and
    a repeat gets misread as progress."""
    m = _Mock('{"reason": "r", "new": "", "verdict": "restate"}')
    prior = [_case(1, "第一次催"), _case(2, "第二次催"), _case(3, "第三次催")]
    await judge_one(_router(m), prior, _case(4, "第四次催"), judge_scene=LLMScene.WORLD_BUILDING)
    prompt = m.seen[0]
    assert all(p.lines[0][1] in prompt for p in prior)
    assert "此前谈过 3 场" in prompt and "第 4 场" in prompt


@pytest.mark.asyncio
async def test_unreadable_verdict_becomes_unknown_not_restate() -> None:
    """An unrecognized grade must not default to restate. That would turn a judging failure into
    "it really was going in circles"."""
    m = _Mock('{"reason": "r", "new": "", "verdict": "有点进展"}')
    out = await judge_one(_router(m), [_case(1, "x")], _case(2, "y"),
                          judge_scene=LLMScene.WORLD_BUILDING)
    assert out["verdict"] == "unknown"


@pytest.mark.asyncio
async def test_judge_failure_marks_the_scene_unjudged() -> None:
    out = await judge_one(_router(_Mock(boom=True)), [_case(1, "x")], _case(2, "y"),
                          judge_scene=LLMScene.WORLD_BUILDING)
    assert out["judged"] is False and "verdict" not in out
