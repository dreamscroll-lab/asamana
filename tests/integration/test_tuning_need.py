"""Integration + unit tests for the need_engine (motivation) validation suite."""

from __future__ import annotations

import json

import pytest

from agent.need import NeedType
from core.interfaces.llm import LLMRouter, LLMScene
from providers.llm.mock import MockLLMProvider
from tuning.build_harness import run_build
from tuning.trace import InMemoryTraceSink
from tuning.validation_need import _deterministic_checks, validate_need

# Alpha = main, Beta = background.
_WORLD_RESPONSE = json.dumps({
    "world_name": "Need World", "era_description": "E.",
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

_APPRAISAL_RESPONSE = json.dumps(
    {"emotion": "fear", "intensity": 0.6, "valence": -0.5, "reason": "心生戒备",
     "need_activation": {"safety": 0.7}}, ensure_ascii=False)
# thought-first lite-CoT: the thought field is grounding-only (not parsed); only goals are read.
_GOAL_RESPONSE = json.dumps(
    {"thought": "我此刻最怕的是局势失控，刚听到风声，手头还没了结的事不重复",
     "goals": ["先稳住局势、观察四周", "联络可信之人"]},
    ensure_ascii=False,
)
_JUDGE_RESPONSE = json.dumps({
    "fields": {"score": 5, "rationale": "结构规范", "issues": []},
    "ranking": {"score": 4, "rationale": "排序合理", "issues": []},
    "goals": {"score": 5, "rationale": "目标具体", "issues": []},
    "overall": "动机合理",
}, ensure_ascii=False)

_SCENARIOS = {
    "scenarios": [
        {"name": "override", "criteria_focus": ["ranking"],
         "description": "high threat pressure → safety override", "expect": "dominant=safety",
         "scenario": {"external_goals": [
             {"to": "Alpha", "text": "立刻自保", "urgency": "high", "drive_type": "threat", "need": "safety"}
         ]}},
        {"name": "internal", "criteria_focus": ["ranking"],
         "description": "no external pressure", "expect": "internal dominant",
         "scenario": {"messages": [
             {"to": "Alpha", "from": "Beta", "content": "一起商议", "urgency": "normal"}
         ]}},
    ],
}


@pytest.mark.asyncio
async def test_validate_need_writes_checks_and_judge(container, test_config, tmp_path) -> None:
    providers = {scene: MockLLMProvider(fixed_response=_APPRAISAL_RESPONSE) for scene in LLMScene}
    providers[LLMScene.WORLD_BUILDING] = MockLLMProvider(fixed_response=_WORLD_RESPONSE)
    providers[LLMScene.CAST_DESIGN] = MockLLMProvider(fixed_response=_CAST_RESPONSE)
    providers[LLMScene.NEED_GOAL_GENERATION] = MockLLMProvider(fixed_response=_GOAL_RESPONSE)
    container.llm_router = LLMRouter(providers)
    world_id, _ = await run_build(container, test_config, "need-theme", template="changan_iso", sink=InMemoryTraceSink())

    scen_path = tmp_path / "need.json"
    scen_path.write_text(json.dumps(_SCENARIOS, ensure_ascii=False), encoding="utf-8")

    summary = await validate_need(
        container, test_config, world_id,
        scenarios_path=str(scen_path), trace_dir=str(tmp_path),
        judge_provider=MockLLMProvider(fixed_response=_JUDGE_RESPONSE),
    )

    base = tmp_path / world_id / "validation" / "need"
    assert (base / "summary.json").exists() and (base / "summary.md").exists()
    for name in ("override", "internal"):
        for f in ("input", "agents", "prompt", "checks", "judge"):
            assert (base / name / f"{f}.json").exists(), f"{name}/{f}.json missing"

    assert summary["scenario_count"] == 2
    assert summary["all_deterministic_passed"] is True
    assert summary["criteria_avg"]["fields"] == 5

    # Override: the high-urgency threat pressure with related_need=safety drives Alpha's dominant.
    agents = json.loads((base / "override" / "agents.json").read_text(encoding="utf-8"))
    alpha = next(a for a in agents if a["agent_name"] == "Alpha")
    assert alpha["outputs"]["dominant_need"] == "safety"
    # Short-term goals are a retained FIFO queue: related_need may be None (a wildcard: build-time
    # goals, deferred intents, or a duplicate of an existing goal that kept its original tag) or
    # a canonical need; the queue need not be bound to the current dominant. The real assertion of
    # the override scenario is that dominant is raised to safety (line above).
    _canon = {nt.value for nt in NeedType}
    assert alpha["outputs"]["goal_entities"]
    for g in alpha["outputs"]["goal_entities"]:
        assert g["related_need"] is None or g["related_need"] in _canon

    checks = json.loads((base / "override" / "checks.json").read_text(encoding="utf-8"))
    assert checks["override_violations"] == [] and checks["field_violations"] == []

    # The goal-generation call is captured. Its prompt is [system, user]; concat both for the
    # substring checks (as test_decision._decision_prompt does).
    prompt = json.loads((base / "override" / "prompt.json").read_text(encoding="utf-8"))
    assert prompt["goal_calls"]
    gp = "\n".join(m["content"] for m in prompt["goal_calls"][0]["prompt"])
    assert "我的价值观：" in gp                  # values injected
    assert "我此刻的情绪：" in gp                 # emotion as backdrop
    assert "我刚刚感知到：" in gp                  # labeled signal block header
    assert "不是一锤定音的大计" in gp              # intention wording (want/do)
    assert "短期目标 ≠ 长期目标" in gp             # short-term != long-term reverse-constraint
    assert "当下真正在驱动我的是这些" in gp        # current dominant + signals lead, long-term is backdrop
    assert "只是在场" in gp                         # mere presence shouldn't spawn an aggressive goal against them
    assert '"thought"' in gp                        # think-first lite-CoT field comes before goals

    # The `internal` scenario injects a message → the channel/source-labeled signal shows up.
    iprompt = json.loads((base / "internal" / "prompt.json").read_text(encoding="utf-8"))
    igp = "\n".join(m["content"] for m in iprompt["goal_calls"][0]["prompt"])
    assert "收到来自" in igp


def test_deterministic_checks_catch_violations() -> None:
    """Synthetic offending outputs must be flagged (the checks aren't vacuous)."""
    per_agent = [{
        "agent_name": "Solo",
        "inputs": {"external_goals": [
            {"urgency": "critical", "drive_type": "threat", "related_need": "safety", "text": "保命"}
        ]},
        "outputs": {
            "dominant_need": "esteem",  # should be safety (override) → override violation
            "scores": {"safety": 0.8, "esteem": 0.5},  # subset of needs is fine (no missing check)
            # 7 items -> over cap (6), a count violation; g1 repeated -> a duplicate violation.
            "short_term_goals": ["g1", "g2", "g3", "g4", "g5", "g6", "g1"],
            "goal_entities": [{"text": "g1", "related_need": "bogus"}],  # non-canonical need -> field violation
            "external_goals": [],
        },
    }]
    det = _deterministic_checks(per_agent)
    assert det["passed"] is False
    assert det["override_violations"]  # dominant should have been overridden to safety
    fv = " ".join(det["field_violations"])
    assert "数量" in fv and "重复" in fv and "related_need" in fv


def test_deterministic_checks_pass_on_clean_internal() -> None:
    """No external pressure → dominant must equal argmax(scores); clean output passes."""
    per_agent = [{
        "agent_name": "Solo",
        "inputs": {"external_goals": []},
        "outputs": {
            "dominant_need": "social",
            "scores": {"physiological": 0.2, "safety": 0.3, "social": 0.7, "esteem": 0.4, "self_actualization": 0.1},
            "short_term_goals": ["主动与人交流"],
            "goal_entities": [{"text": "主动与人交流", "related_need": "social"}],
            "external_goals": [],
        },
    }]
    det = _deterministic_checks(per_agent)
    assert det["passed"] is True
