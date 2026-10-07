"""Integration + unit tests for the world_pressure validation suite."""

from __future__ import annotations

import json

import pytest

from core.interfaces.llm import LLMRouter, LLMScene
from providers.llm.mock import MockLLMProvider
from tuning.build_harness import run_build
from tuning.trace import InMemoryTraceSink
from tuning.validation import _deterministic_checks, validate_world_pressure

_WORLD_RESPONSE = json.dumps({
    "world_name": "Val World", "era_description": "E.",
    "core_tension": "A vs B.", "narrative_theme": "t.", "narrative_pitch": "A and B act now.",
    "world_time_config": {"start_year": 1},
    "key_figures": [
        {"name": "Alpha", "role": "r1", "importance": "main", "brief": "A."},
        {"name": "Beta", "role": "r2", "importance": "main", "brief": "B."},
    ],
    "initial_relations": [
        {"from": "Alpha", "to": "Beta", "trust": 0.5, "affection": 0.0, "labels": ["相识"],
         "reverse_trust": 0.5, "reverse_affection": 0.0, "reverse_labels": ["相识"]},
    ],
}, ensure_ascii=False)
_CAST_RESPONSE = json.dumps({"roles": [
    {"index": 1, "narrative_role": "r", "arc_summary": "a", "key_relationships": [2]},
    {"index": 2, "narrative_role": "r", "arc_summary": "a", "key_relationships": [1]},
]}, ensure_ascii=False)

# per-agent world_pressure response: one goal with a NON-canonical drive_type.
_PRESSURE_RESPONSE = json.dumps({
    "goals": [{"drive_type": "panic", "urgency": "high", "text": "立即逃离", "source": 0}]
}, ensure_ascii=False)
_JUDGE_RESPONSE = json.dumps({
    "boundary": {"score": 4, "rationale": "ok", "issues": []},
    "reasonableness": {"score": 3, "rationale": "ok", "issues": []},
    "fields": {"score": 2, "rationale": "bad drive_type", "issues": ["panic 非规范"]},
    "overall": "测试总评",
}, ensure_ascii=False)

_SCENARIOS = {
    "scenarios": [
        {"name": "broadcast_global", "description": "全局广播", "criteria_focus": ["reasonableness", "fields"],
         "expect": "应有 threat", "scenario": {"broadcasts": [{"content": "喊杀骤起", "severity": "high"}]}},
        {"name": "broadcast_scoped_empty", "description": "无人地点广播", "criteria_focus": ["boundary"],
         "expect": "全员空", "scenario": {"broadcasts": [
             {"content": "偏僻库房闷响", "severity": "high", "location_scope": "@empty_location"}]}},
    ]
}


@pytest.mark.asyncio
async def test_validate_world_pressure_writes_reports_and_runs_checks(
    container, test_config, tmp_path
) -> None:
    providers = {scene: MockLLMProvider(fixed_response="{}") for scene in LLMScene}
    providers[LLMScene.WORLD_BUILDING] = MockLLMProvider(fixed_response=_WORLD_RESPONSE)
    providers[LLMScene.CAST_DESIGN] = MockLLMProvider(fixed_response=_CAST_RESPONSE)
    providers[LLMScene.WORLD_PRESSURE] = MockLLMProvider(fixed_response=_PRESSURE_RESPONSE)
    container.llm_router = LLMRouter(providers)

    world_id, _ = await run_build(container, test_config, "val-theme", template="changan_iso", sink=InMemoryTraceSink())

    scen_path = tmp_path / "scenarios.json"
    scen_path.write_text(json.dumps(_SCENARIOS, ensure_ascii=False), encoding="utf-8")

    summary = await validate_world_pressure(
        container, test_config, world_id,
        scenarios_path=str(scen_path), trace_dir=str(tmp_path),
        # dedicated judge LLM injected as a mock (avoids a real qwen3.7-max call)
        judge_provider=MockLLMProvider(fixed_response=_JUDGE_RESPONSE),
    )

    base = tmp_path / world_id / "validation" / "world_pressure"
    assert (base / "summary.json").exists() and (base / "summary.md").exists()
    for name in ("broadcast_global", "broadcast_scoped_empty"):
        for f in ("input", "agents", "prompt", "checks", "judge"):
            assert (base / name / f"{f}.json").exists(), f"{name}/{f}.json missing"

    assert summary["scenario_count"] == 2
    assert summary["criteria_avg"]["fields"] == 2  # judge mock value

    # deterministic field check flagged the non-canonical drive_type (global broadcast → agents evaluated).
    checks_g = json.loads((base / "broadcast_global" / "checks.json").read_text(encoding="utf-8"))
    assert any("panic" in i for i in checks_g["field_issues"])

    # A broadcast scoped to an empty location reaches no agent, so nobody is evaluated → no
    # goals → the location hard boundary passes structurally.
    checks_e = json.loads((base / "broadcast_scoped_empty" / "checks.json").read_text(encoding="utf-8"))
    assert checks_e["location_boundary_empty"]["passed"] is True
    assert checks_e["location_boundary_empty"]["offending_agents"] == []


def test_deterministic_checks_flags_location_and_semantic_leaks() -> None:
    """The checks fire when given offending data (synthetic, no LLM)."""
    resolved = {
        "broadcasts": [{"content": "偏僻库房闷响", "severity": "high", "location_scope": "void"}],
        "messages": [{"to": "李世民", "from": "李渊", "content": "即刻入宫觐见"}],
    }
    per_agent = [{
        "agent_name": "李建成",
        "inputs": {"location_id": "hall", "injected_messages": []},
        "outputs": {"external_goals": [
            {"drive_type": "threat", "urgency": "high", "text": "利用李世民入宫之机控制局势"}]},
    }]
    det = _deterministic_checks({"name": "x"}, resolved, per_agent, raw_responses=[])

    # 李建成 at hall, broadcast scoped to "void" (no one there) but 李建成 produced a goal → offender.
    assert det["location_boundary_empty"]["passed"] is False
    assert "李建成" in det["location_boundary_empty"]["offending_agents"]
    # 李建成's goal references 「入宫」 from a message addressed to 李世民 → semantic leak.
    assert det.get("semantic_leaks")
    assert any("入宫" in s for s in det["semantic_leaks"])
    assert det["passed"] is False


def test_semantic_leak_ignores_public_role_words() -> None:
    """A role title (e.g. 太子) is public identity — sharing it is not a leak.

    丁 is not co-located with anyone but references 太子 (丙's role); the message it
    never received also mentions 太子. The role gram must be suppressed via the global
    public-terms set, so no leak is flagged.
    """
    resolved = {"messages": [{"to": "甲", "from": "信使", "content": "太子车驾将至，速决"}]}
    per_agent = [
        {"agent_name": "丙", "inputs": {"location_id": "hall", "co_located": [], "injected_messages": []},
         "outputs": {"external_goals": []}, "role": "太子"},
        {"agent_name": "丁", "inputs": {"location_id": "court", "injected_messages": [],
                                        "co_located": [{"name": "丙", "role": "太子", "trust": 0.5}]},
         "outputs": {"external_goals": [{"drive_type": "event", "urgency": "normal", "text": "查明太子安危"}]}},
    ]
    det = _deterministic_checks({"name": "x"}, resolved, per_agent, raw_responses=[])
    assert not det.get("semantic_leaks")  # "太子" is a public role, not a leak
