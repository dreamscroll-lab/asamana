"""Integration + unit tests for the perception_emotion validation suite (LLM + judge)."""

from __future__ import annotations

import json

import pytest

from core.interfaces.llm import LLMRouter, LLMScene
from providers.llm.mock import MockLLMProvider
from tuning.build_harness import run_build
from tuning.trace import InMemoryTraceSink
from tuning.validation_perception_emotion import (
    _deterministic_checks,
    validate_perception_emotion,
)

# Alpha = main, Beta = background.
_WORLD_RESPONSE = json.dumps({
    "world_name": "Emo World", "era_description": "E.",
    "core_tension": "A vs B.", "narrative_theme": "t.", "narrative_pitch": "A and B act now.",
    "world_time_config": {"start_year": 1},
    "key_figures": [
        {"name": "Alpha", "role": "r1", "importance": "main", "brief": "A."},
        {"name": "Beta", "role": "r2", "importance": "background", "brief": "B."},
    ],
    "initial_relations": [
        {"from": "Alpha", "to": "Beta", "trust": 0.6, "affection": 0.0, "labels": ["相识"],
         "reverse_trust": 0.6, "reverse_affection": 0.0, "reverse_labels": ["相识"]},
    ],
}, ensure_ascii=False)
_CAST_RESPONSE = json.dumps({"roles": [
    {"index": 1, "narrative_role": "r", "arc_summary": "a", "key_relationships": [2]},
    {"index": 2, "narrative_role": "r", "arc_summary": "a", "key_relationships": [1]},
]}, ensure_ascii=False)

# Production appraisal output (what AGENT_DECISION_* scenes return in this test):
# emotion + per-need activation.
_EMOTION_RESPONSE = json.dumps(
    {"emotion": "fear", "intensity": 0.7, "valence": -0.6, "reason": "心头骤紧，戒备顿生",
     "need_activation": {"safety": 0.9, "social": 0.2}},
    ensure_ascii=False,
)
# Judge output.
_JUDGE_RESPONSE = json.dumps({
    "fields": {"score": 5, "rationale": "字段规范", "issues": []},
    "semantic": {"score": 4, "rationale": "贴合威胁", "issues": []},
    "functional": {"score": 5, "rationale": "本能反应", "issues": []},
    "overall": "合理的即时恐惧",
}, ensure_ascii=False)

_SCENARIOS = {
    "scenarios": [
        {"name": "threat", "criteria_focus": ["semantic", "functional"],
         "description": "acute threat to Alpha", "expect": "fear, negative, high",
         "scenario": {"external_goals": [
             {"to": "Alpha", "text": "有人持刃逼近", "urgency": "critical", "drive_type": "threat"}
         ]}},
        {"name": "message", "criteria_focus": ["semantic"],
         "description": "urgent message to Alpha", "expect": "alarm",
         "scenario": {"messages": [
             {"to": "Alpha", "from": "Beta", "content": "速速戒备", "urgency": "high"}
         ]}},
    ],
}


@pytest.mark.asyncio
async def test_validate_perception_emotion_writes_checks_and_judge(
    container, test_config, tmp_path
) -> None:
    providers = {scene: MockLLMProvider(fixed_response=_EMOTION_RESPONSE) for scene in LLMScene}
    providers[LLMScene.WORLD_BUILDING] = MockLLMProvider(fixed_response=_WORLD_RESPONSE)
    providers[LLMScene.CAST_DESIGN] = MockLLMProvider(fixed_response=_CAST_RESPONSE)
    container.llm_router = LLMRouter(providers)
    world_id, _ = await run_build(container, test_config, "emo-theme", template="changan_iso", sink=InMemoryTraceSink())

    scen_path = tmp_path / "perception_emotion.json"
    scen_path.write_text(json.dumps(_SCENARIOS, ensure_ascii=False), encoding="utf-8")

    summary = await validate_perception_emotion(
        container, test_config, world_id,
        scenarios_path=str(scen_path), trace_dir=str(tmp_path),
        judge_provider=MockLLMProvider(fixed_response=_JUDGE_RESPONSE),
    )

    base = tmp_path / world_id / "validation" / "perception_emotion"
    assert (base / "summary.json").exists() and (base / "summary.md").exists()
    for name in ("threat", "message"):
        for f in ("input", "agents", "prompt", "checks", "judge"):
            assert (base / name / f"{f}.json").exists(), f"{name}/{f}.json missing"

    assert summary["scenario_count"] == 2
    # Valid emotion JSON → field checks pass.
    assert summary["all_deterministic_passed"] is True
    # Judge (mock) → the criteria scores are surfaced.
    assert summary["criteria_avg"]["fields"] == 5
    assert summary["criteria_avg"]["functional"] == 5

    # Alpha (main) is reached with a signal → an emotion LLM call is captured.
    prompt = json.loads((base / "threat" / "prompt.json").read_text(encoding="utf-8"))
    assert prompt["emotion_calls"], "expected at least one captured emotion call"

    # The judge actually scored the produced emotion.
    judge = json.loads((base / "threat" / "judge.json").read_text(encoding="utf-8"))
    assert judge["fields"]["score"] == 5

    # need_activation is captured per agent, and the deterministic check validated it.
    agents = json.loads((base / "threat" / "agents.json").read_text(encoding="utf-8"))
    alpha = next(a for a in agents if a["agent_name"] == "Alpha")
    assert alpha["outputs"]["need_activation"].get("safety") == 0.9
    checks = json.loads((base / "threat" / "checks.json").read_text(encoding="utf-8"))
    assert checks["activation_issues"] == []


def test_deterministic_checks_catch_bad_fields() -> None:
    """Synthetic offending raw outputs must be flagged (the checks aren't vacuous)."""
    per_agent = [
        {"agent_name": "A", "inputs": {}, "outputs": {"emotion": {"primary": "fear"}}},
        {"agent_name": "B", "inputs": {}, "outputs": {"emotion": None}},
    ]
    raw_by_agent = {
        # valence out of range + polarity mismatch (fear with positive valence) + long reason
        "a1": json.dumps({"emotion": "fear", "intensity": 0.5, "valence": 1.8,
                          "reason": "这是一段明显超过二十字上限的冗长理由用来触发长度报警检查项"},
                         ensure_ascii=False),
        # not JSON
        "a2": "totally not json",
    }
    det = _deterministic_checks(per_agent, raw_by_agent)
    assert det["passed"] is False
    issues = " ".join(det["field_issues"])
    assert "越界" in issues and "极性矛盾" in issues and "过长" in issues
    assert "非合法 JSON" in issues
    assert det["none_output_agents"] == ["B"]


def test_deterministic_checks_allow_omitted_emotion() -> None:
    """Omitting emotion is a valid 'no new reaction' output — not a field issue."""
    per_agent = [{"agent_name": "A", "inputs": {}, "outputs": {"emotion": None}}]
    raw_by_agent = {"a1": json.dumps({"need_activation": {"safety": 0.4}}, ensure_ascii=False)}
    det = _deterministic_checks(per_agent, raw_by_agent)
    assert det["passed"] is True
    assert det["field_issues"] == []


def test_deterministic_checks_pass_on_clean_output() -> None:
    per_agent = [{"agent_name": "A", "inputs": {}, "outputs": {"emotion": {"primary": "joy"}}}]
    raw_by_agent = {"a1": json.dumps(
        {"emotion": "joy", "intensity": 0.4, "valence": 0.5, "reason": "暖意涌上心头"},
        ensure_ascii=False,
    )}
    det = _deterministic_checks(per_agent, raw_by_agent)
    assert det["passed"] is True
    assert det["field_issues"] == []
