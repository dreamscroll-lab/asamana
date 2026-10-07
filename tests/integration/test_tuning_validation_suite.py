"""The scaffold every stage validation suite runs on (tuning/validation_suite)."""

from __future__ import annotations

import importlib
import json

import pytest

from core.interfaces.llm import LLMRouter, LLMScene
from providers.llm.mock import MockLLMProvider
from tuning.build_harness import run_build
from tuning.trace import InMemoryTraceSink
from tuning.validation_suite import STEP_LEAK, aggregate, summary_md, text_leak

_WORLD_RESPONSE = json.dumps({
    "world_name": "Suite World", "era_description": "E.",
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

_SUITES = [
    ("tuning.validation", "validate_world_pressure", "world_pressure"),
    ("tuning.validation_action", "validate_action", "action"),
    ("tuning.validation_decision", "validate_decision", "decision"),
    ("tuning.validation_event", "validate_event", "event"),
    ("tuning.validation_feedback", "validate_feedback", "feedback"),
    ("tuning.validation_interrupt", "validate_interrupt", "interrupt"),
    ("tuning.validation_long_term_goal", "validate_long_term_goal", "long_term_goal"),
    ("tuning.validation_memory_maintain", "validate_memory_maintain", "memory_maintain"),
    ("tuning.validation_memory_retrieve", "validate_memory_retrieve", "memory_retrieve"),
    ("tuning.validation_memory_write", "validate_memory_write", "memory_write"),
    ("tuning.validation_need", "validate_need", "need"),
    ("tuning.validation_perception_emotion", "validate_perception_emotion", "perception_emotion"),
    ("tuning.validation_relation", "validate_relation", "relation"),
]


@pytest.mark.parametrize("text", ["第3步", "3个步骤", "过了 12 步", "step=12", "two steps later"])
def test_step_leak_catches_code_layer_steps(text: str) -> None:
    assert STEP_LEAK.search(text)


@pytest.mark.parametrize("text", ["三步并作两步", "footsteps echoed", "他退了半步"])
def test_step_leak_spares_narrative_wording(text: str) -> None:
    assert not STEP_LEAK.search(text)


def test_text_leak_reports_ids_before_steps() -> None:
    assert text_leak("agent-7 到了", agent_ids={"agent-7"}) == "泄漏 agent id「agent-7」"
    assert text_leak("去 loc_gate", agent_ids=(), location_ids={"loc_gate"}) == "泄漏 location id「loc_gate」"
    assert text_leak("第3步", agent_ids=()) is not None
    assert text_leak("清晨入宫", agent_ids={"agent-7"}, location_ids={"loc_gate"}) is None


def test_aggregate_averages_only_scored_rows() -> None:
    rows = [
        {"scores": {"a": 4, "b": 0}, "deterministic_passed": True},
        {"scores": {"a": 2}, "deterministic_passed": False},
    ]
    summary = aggregate("w", rows, ("a", "b"))
    assert summary["criteria_avg"] == {"a": 3.0, "b": 0}
    assert summary["all_deterministic_passed"] is False


def test_summary_md_lays_out_suite_columns_and_issues() -> None:
    summary = aggregate("w", [{
        "name": "s1", "scores": {"a": 4}, "deterministic_passed": True,
        "overall": "好|坏", "issues": ["问题一"],
    }], ("a",))
    md = summary_md(summary, title="t", avg_label="均值", columns=[("a", lambda r: r["scores"]["a"])])
    assert "| 场景 | a | 确定性 | 总评 |" in md
    assert "| s1 | 4 | ✅ | 好/坏 |" in md
    assert "- 问题一" in md


@pytest.mark.asyncio
@pytest.mark.parametrize(("module_name", "entry", "stage"), _SUITES)
async def test_every_suite_turns_a_crashed_scenario_into_a_failed_row(
    container, test_config, tmp_path, monkeypatch, module_name: str, entry: str, stage: str,
) -> None:
    providers = {scene: MockLLMProvider(fixed_response="{}") for scene in LLMScene}
    providers[LLMScene.WORLD_BUILDING] = MockLLMProvider(fixed_response=_WORLD_RESPONSE)
    providers[LLMScene.CAST_DESIGN] = MockLLMProvider(fixed_response=_CAST_RESPONSE)
    container.llm_router = LLMRouter(providers)
    world_id, _ = await run_build(
        container, test_config, "suite-theme", template="changan_iso", sink=InMemoryTraceSink()
    )

    module = importlib.import_module(module_name)

    async def _crash(*_args: object, **_kwargs: object) -> dict:
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "_run_scenario", _crash)
    scen_path = tmp_path / "scenarios.json"
    scen_path.write_text(json.dumps({"scenarios": [{"name": "crash", "scenario": {}}]}), encoding="utf-8")

    summary = await getattr(module, entry)(
        container, test_config, world_id,
        scenarios_path=str(scen_path), trace_dir=str(tmp_path),
        judge_provider=MockLLMProvider(fixed_response="{}"),
    )

    row = summary["scenarios"][0]
    assert summary["scenario_count"] == 1
    assert row["deterministic_passed"] is False and "boom" in row["overall"]
    assert set(summary["criteria_avg"].values()) == {0}
    md = (tmp_path / world_id / "validation" / stage / "summary.md").read_text(encoding="utf-8")
    assert "| crash |" in md
