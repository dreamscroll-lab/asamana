"""Unit tests for the TALK cross-lane leak metric (tuning/judge_dialogue_leak).

Deterministic, no real LLM: a synthetic trace dir drives extraction, and a mock
provider drives judging (index resolution, out-of-range filtering, Rule 1 failure).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.interfaces.llm import LLMProvider, LLMResponse, LLMRouter, LLMScene
from tuning.judge_dialogue_leak import DialogueCase, extract_dialogues, judge_one
from tuning.trace import InMemoryTraceSink

WORLD = "w-leak"

_LANES = {
    "a": {"name": "甲", "text": "甲的人设。\n- 我对乙的认识：\n  - 甲独有的事。"},
    "b": {"name": "乙", "text": "乙的人设。\n- 我对甲的认识：\n  - 乙独有的事。"},
}


def _write_trace(tmp_path: Path, response: str, *, scene: str = "agent_action_narration",
                 lanes: dict | None = _LANES) -> str:
    d = tmp_path / WORLD
    d.mkdir(parents=True)
    rec = {
        "kind": "llm_call", "world_id": WORLD, "stage": "action", "scene": scene,
        "prompt_messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        "extra": {"dialogue_lanes": lanes} if lanes else {},
        "response_content": response, "step": 7, "call_id": "c1", "ok": True, "parse_ok": True,
    }
    (d / "step_000007.jsonl").write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(tmp_path)


class _Mock(LLMProvider):
    def __init__(self, response: str = "", *, boom: bool = False) -> None:
        self.response, self.boom = response, boom

    async def complete(self, messages, temperature=0.7, max_tokens=1000, **kw) -> LLMResponse:
        if self.boom:
            raise RuntimeError("judge down")
        return LLMResponse(content=self.response, model="mock", input_tokens=1, output_tokens=1)


def _router(provider: LLMProvider) -> LLMRouter:
    return LLMRouter({s: provider for s in LLMScene}, trace_sink=InMemoryTraceSink())


def _case() -> DialogueCase:
    return DialogueCase(step=7, call_id="c1", name_a="甲", lane_a="甲独有的事",
                        name_b="乙", lane_b="乙独有的事",
                        lines=[("甲", "你听说了吗？"), ("乙", "乙独有的事我知道。")])


def test_extraction_reads_both_lanes_and_named_lines(tmp_path) -> None:
    base = _write_trace(tmp_path, '{"dialogue": [{"speaker": 1, "line": "问一句"}, '
                                  '{"speaker": 2, "line": "答一句"}]}')
    cases = extract_dialogues(WORLD, trace_dir=base)
    assert len(cases) == 1
    c = cases[0]
    assert (c.name_a, c.name_b) == ("甲", "乙")
    assert "甲独有的事" in c.lane_a and "乙独有的事" in c.lane_b
    assert c.lines == [("甲", "问一句"), ("乙", "答一句")]


def test_narration_without_two_lanes_is_not_a_dialogue(tmp_path) -> None:
    """Narration and dialogue share a scene, so a record without A/B columns must not be counted as
    dialogue."""
    base = _write_trace(tmp_path, '{"outcome": "他做完了。"}', lanes=None)
    assert extract_dialogues(WORLD, trace_dir=base) == []


@pytest.mark.asyncio
async def test_index_resolves_to_the_line_it_points_at_on_both_axes() -> None:
    r = _router(_Mock('{"reason": "r", "divergence": {"what": "谁传的假令", "enacted": "voiced"}, '
                      '"crossings": [{"why": "只在甲栏", "index": 2}], '
                      '"concessions": [{"why": "认了对方那套", "index": 1}]}'))
    out = await judge_one(r, _case(), judge_scene=LLMScene.WORLD_BUILDING)
    assert out["judged"] is True
    assert out["crossings"] == [
        {"index": 2, "speaker": "乙", "line": "乙独有的事我知道。", "why": "只在甲栏"}]
    assert [c["index"] for c in out["concessions"]] == [1]
    assert out["divergence"] == {"what": "谁传的假令", "enacted": "voiced"}


@pytest.mark.asyncio
async def test_concessions_are_counted_even_when_nothing_crossed() -> None:
    """A concession brings no new information, so its overreach axis is always empty. Measuring the
    two axes separately is what keeps it from being missed."""
    r = _router(_Mock('{"reason": "r", "crossings": [], '
                      '"concessions": [{"why": "认下自己栏里说是对方干的事", "index": 2}]}'))
    out = await judge_one(r, _case(), judge_scene=LLMScene.WORLD_BUILDING)
    assert out["crossings"] == []
    assert len(out["concessions"]) == 1


@pytest.mark.asyncio
async def test_out_of_range_and_non_integer_indices_are_dropped() -> None:
    """Out-of-range or non-integer indices are dropped per IndexedRef, so a verdict is never pinned
    to a sentence that doesn't exist."""
    r = _router(_Mock('{"reason": "x", "crossings": [{"index": 9}, {"index": "第一句"}, '
                      '{"index": 1}], "concessions": []}'))
    out = await judge_one(r, _case(), judge_scene=LLMScene.WORLD_BUILDING)
    assert [l["index"] for l in out["crossings"]] == [1]


@pytest.mark.asyncio
async def test_judge_failure_marks_the_scene_unjudged_rather_than_clean() -> None:
    """Rule 1: a verdict that didn't happen doesn't mean the conversation was clean. The whole
    conversation is marked judged=False and excluded from aggregation."""
    out = await judge_one(_router(_Mock(boom=True)), _case(), judge_scene=LLMScene.WORLD_BUILDING)
    assert out["judged"] is False
    assert "crossings" not in out and "concessions" not in out


@pytest.mark.asyncio
async def test_unreadable_enacted_becomes_unknown_not_none() -> None:
    """An unrecognized grade must not default to "none". That would turn a judging failure into
    "there was no disagreement"."""
    r = _router(_Mock('{"reason": "r", "divergence": {"what": "x", "enacted": "谈崩了"}, '
                      '"crossings": [], "concessions": []}'))
    out = await judge_one(r, _case(), judge_scene=LLMScene.WORLD_BUILDING)
    assert out["divergence"]["enacted"] == "unknown"


@pytest.mark.asyncio
async def test_missing_divergence_block_degrades_to_unknown() -> None:
    r = _router(_Mock('{"reason": "r", "crossings": [], "concessions": []}'))
    out = await judge_one(r, _case(), judge_scene=LLMScene.WORLD_BUILDING)
    assert out["divergence"] == {"what": "", "enacted": "unknown"}
