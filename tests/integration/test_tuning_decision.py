"""Integration + unit tests for the decision (decide) validation suite."""

from __future__ import annotations

import json

import pytest

from core.interfaces.llm import LLMMessage, LLMResponse, LLMRouter, LLMScene
from providers.llm.mock import MockLLMProvider
from tuning.build_harness import run_build
from tuning.trace import InMemoryTraceSink
from tuning.validation_decision import _deterministic_checks, validate_decision

# Alpha = main, Beta = background.
_WORLD_RESPONSE = json.dumps({
    "world_name": "Decide World", "era_description": "E.",
    "core_tension": "A vs B.", "narrative_theme": "t.", "narrative_pitch": "A and B act now.",
    "world_time_config": {"start_year": 1},
    "key_figures": [
        {"name": "Alpha", "role": "r1", "importance": "main", "brief": "A."},
        {"name": "Beta", "role": "r2", "importance": "background", "brief": "B."},
    ],
    "initial_relations": [
        {"from": "Alpha", "to": "Beta", "trust": 0.6, "affection": 0.2, "labels": ["相识"],
         "reverse_trust": 0.6, "reverse_affection": 0.2, "reverse_labels": ["相识"]},
    ],
}, ensure_ascii=False)
_CAST_RESPONSE = json.dumps({"roles": [
    {"index": 1, "narrative_role": "r", "arc_summary": "a", "key_relationships": [2]},
    {"index": 2, "narrative_role": "r", "arc_summary": "a", "key_relationships": [1]},
]}, ensure_ascii=False)

_APPRAISAL_RESPONSE = json.dumps(
    {"emotion": "anticipation", "intensity": 0.5, "valence": 0.1, "reason": "静中思忖",
     "need_activation": {"self_actualization": 0.6}}, ensure_ascii=False)
_GOAL_RESPONSE = json.dumps({"goals": ["推敲未竟的方略", "理清下一步部署"]}, ensure_ascii=False)
# WORK (index 3) needs no target -> always valid, never triggers the fallback.
_DECIDE_RESPONSE = json.dumps({
    "selected_index": 3,
    "action_description": "我专注推敲未竟的方略，逐条落定部署",
    "inner_monologue": "趁此静时，把计议想周全要紧",
    "expected_outcome": "方略更完整，心里更有底",
    "estimated_steps": 2,
}, ensure_ascii=False)
_JUDGE_RESPONSE = json.dumps({
    "fields": {"score": 5, "rationale": "结构规范", "issues": []},
    "action_fit": {"score": 5, "rationale": "选得合理", "issues": []},
    "coherence": {"score": 5, "rationale": "前后一致", "issues": []},
    "overall": "合理",
}, ensure_ascii=False)


class _PromptAwareLLM:
    """AGENT_DECISION_MAIN serves both appraisal and decide (two expected JSON shapes); dispatch on prompt markers."""

    def __init__(self) -> None:
        self.model = "mock"
        self.call_history: list[list[LLMMessage]] = []

    async def complete(self, messages, temperature: float = 0.7, max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        self.call_history.append(list(messages))
        text = " ".join(m.content for m in messages)
        if "【我能做的】" in text or "selected_index" in text:
            resp = _DECIDE_RESPONSE
        elif "我要想清楚的" in text:
            resp = _GOAL_RESPONSE
        else:
            resp = _APPRAISAL_RESPONSE
        return LLMResponse(content=resp, input_tokens=0, output_tokens=0, model=self.model)

    async def stream(self, messages, temperature: float = 0.7):  # pragma: no cover
        yield ""


_SCENARIOS = {
    "scenarios": [
        {"name": "work", "criteria_focus": ["action_fit"],
         "description": "平和、有自身要务待推进", "expect": "Alpha 选 WORK 合理",
         "scenario": {
             "present": [{"agent": "Alpha", "sees": []}],
             "entities": [{"agent": "Alpha", "items": [
                 {"name": "未完成的卷宗", "takeable": True, "desc": "尚需整理"}]}],
             "memories": [{"agent": "Alpha", "factual": ["手头有一桩未竟之事"],
                           "insights": ["趁静把它做完"]}],
         }},
        {"name": "talk", "criteria_focus": ["action_fit"],
         "description": "盟友 Beta 在场", "expect": "Alpha 与 Beta 当面交谈合理",
         "scenario": {
             "present": [{"agent": "Alpha", "sees": ["Beta"]}],
             "memories": [{"agent": "Alpha", "experiential": ["与 Beta 共事一向愉快"]}],
         }},
    ],
}


@pytest.mark.asyncio
async def test_validate_decision_writes_checks_judge_and_injects_memory(container, test_config, tmp_path) -> None:
    aware = _PromptAwareLLM()
    providers = {scene: aware for scene in LLMScene}
    providers[LLMScene.WORLD_BUILDING] = MockLLMProvider(fixed_response=_WORLD_RESPONSE)
    providers[LLMScene.CAST_DESIGN] = MockLLMProvider(fixed_response=_CAST_RESPONSE)
    container.llm_router = LLMRouter(providers)
    world_id, _ = await run_build(container, test_config, "decide-theme", template="changan_iso", sink=InMemoryTraceSink())

    scen_path = tmp_path / "decision.json"
    scen_path.write_text(json.dumps(_SCENARIOS, ensure_ascii=False), encoding="utf-8")

    summary = await validate_decision(
        container, test_config, world_id,
        scenarios_path=str(scen_path), trace_dir=str(tmp_path),
        judge_provider=MockLLMProvider(fixed_response=_JUDGE_RESPONSE),
    )

    base = tmp_path / world_id / "validation" / "decision"
    assert (base / "summary.json").exists() and (base / "summary.md").exists()
    for name in ("work", "talk"):
        for f in ("input", "agents", "prompt", "checks", "judge"):
            assert (base / name / f"{f}.json").exists(), f"{name}/{f}.json missing"

    assert summary["scenario_count"] == 2
    assert summary["all_deterministic_passed"] is True
    assert summary["criteria_avg"]["fields"] == 5

    # Alpha chose WORK (the mock decide) — captured as the action output.
    agents = json.loads((base / "work" / "agents.json").read_text(encoding="utf-8"))
    alpha = next(a for a in agents if a["agent_name"] == "Alpha")
    assert alpha["outputs"]["action_type"] == "work"

    # The injected memory reaches Alpha's inputs and the decision prompt.
    assert "手头有一桩未竟之事" in alpha["inputs"]["factual_memories"]
    prompt = json.loads((base / "work" / "prompt.json").read_text(encoding="utf-8"))
    assert prompt["decide_calls"]
    combined = " ".join(
        m["content"] for c in prompt["decide_calls"] for m in c["prompt"]
    )
    assert "手头有一桩未竟之事" in combined, "injected memory must reach the decision prompt"
    assert "未完成的卷宗" in combined, "injected entity must reach the decision prompt"


def test_deterministic_checks_catch_violations() -> None:
    """Synthetic bad actions must be caught (non-vacuous): non-canonical type / empty description /
    steps<1 / illegal MOVE destination."""
    per_agent = [{
        "agent_name": "Solo",
        "inputs": {"reachable_locations": ["hall"]},
        "outputs": {
            "action_type": "fly",                       # non-canonical
            "action_description": "",
            "inner_monologue": "",
            "estimated_steps": 0,                        # <1
            "target": {"location_id": "void"},
        },
    }, {
        "agent_name": "Mover",
        "inputs": {"reachable_locations": ["hall"]},
        "outputs": {
            "action_type": "move", "action_description": "我去别处",
            "inner_monologue": "走", "estimated_steps": 1,
            "target": {"location_id": "faraway"},        # not adjacent
        },
    }]
    det = _deterministic_checks(per_agent)
    assert det["passed"] is False
    fv = " ".join(det["field_violations"])
    assert "action_type 非规范" in fv
    assert "action_description 为空" in fv
    assert "estimated_steps 非法" in fv
    assert "inner_monologue 为空" in fv
    assert "不在可达地点" in fv


def test_deterministic_checks_catch_physical_missing_kind() -> None:
    """PHYSICAL bound to a target whose kind is outside the derived domain -> must be caught.

    The target kind is derived by decision.py's binding branches (agent / the environment's
    entity_type / object), not classified by the LLM. So this isn't testing whether the model
    classified correctly. It guards the binding branches: if one breaks and leaves None or an
    out-of-domain kind, execution routing (co-location check, effects on people) goes wrong."""
    per_agent = [{
        "agent_name": "Striker",
        "inputs": {},
        "outputs": {
            "action_type": "physical", "action_description": "我夺下他手中的刀",
            "inner_monologue": "先下手", "estimated_steps": 1,
            # A bound target with no kind only happens when a binding branch is broken.
            "target": {"acts_on": [{"kind": None, "id": "长刀"}]},
        },
    }]
    det = _deterministic_checks(per_agent)
    assert det["passed"] is False
    assert any("目标类别缺失/非法" in v for v in det["field_violations"])


def test_deterministic_checks_pass_on_clean_action() -> None:
    per_agent = [{
        "agent_name": "Solo",
        "inputs": {"reachable_locations": ["hall"]},
        "outputs": {
            "action_type": "work", "action_description": "我专注做一件事",
            "inner_monologue": "趁静做完", "estimated_steps": 2, "target": {},
        },
    }]
    assert _deterministic_checks(per_agent)["passed"] is True
