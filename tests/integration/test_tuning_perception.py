"""Integration + unit tests for the perception validation suite (rule-based, no LLM)."""

from __future__ import annotations

import json

import pytest

from core.interfaces.llm import LLMRouter, LLMScene
from providers.llm.mock import MockLLMProvider
from tuning.build_harness import run_build
from tuning.trace import InMemoryTraceSink
from tuning.validation_perception import (
    _build_tuning,
    _deterministic_checks,
    _diff_selections,
    validate_perception,
)

# Alpha = main, Beta = background → a real main/bg split for the threshold A/B diff.
_WORLD_RESPONSE = json.dumps({
    "world_name": "Perc World", "era_description": "E.",
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

_SCENARIOS = {
    "knob_sets": {"baseline": {}, "low_bg": {"threshold_bg": 0.15}},
    "scenarios": [
        {"name": "thr", "criteria_focus": ["threshold"], "description": "weak global ambient",
         "expect": "main keeps, bg drops (baseline); low_bg → bg keeps",
         "scenario": {"ambient": [{"content": "远处微光摇曳", "strength": 0.2}]}},
        {"name": "caps", "criteria_focus": ["caps"], "description": "7 messages > cap",
         "expect": "Alpha message count capped",
         "scenario": {"messages": [
             {"to": "Alpha", "from": "Beta", "content": f"消息内容第{i}条", "urgency": "normal"}
             for i in range(7)
         ]}},
        {"name": "dedup", "criteria_focus": ["dedup"], "description": "duplicate ambient",
         "expect": "collapses to one",
         "scenario": {"ambient": [{"content": "完全相同的话", "strength": 0.5},
                                  {"content": "完全相同的话", "strength": 0.5}]}},
    ],
}


@pytest.mark.asyncio
async def test_validate_perception_writes_checks_and_diff(container, test_config, tmp_path) -> None:
    providers = {scene: MockLLMProvider(fixed_response="{}") for scene in LLMScene}
    providers[LLMScene.WORLD_BUILDING] = MockLLMProvider(fixed_response=_WORLD_RESPONSE)
    providers[LLMScene.CAST_DESIGN] = MockLLMProvider(fixed_response=_CAST_RESPONSE)
    container.llm_router = LLMRouter(providers)
    world_id, _ = await run_build(container, test_config, "perc-theme", template="changan_iso", sink=InMemoryTraceSink())

    scen_path = tmp_path / "perception.json"
    scen_path.write_text(json.dumps(_SCENARIOS, ensure_ascii=False), encoding="utf-8")

    summary = await validate_perception(
        container, test_config, world_id,
        scenarios_path=str(scen_path), trace_dir=str(tmp_path),
    )

    base = tmp_path / world_id / "validation" / "perception"
    assert (base / "summary.json").exists() and (base / "summary.md").exists()
    for name in ("thr", "caps", "dedup"):
        for f in ("input", "selection", "checks", "diff", "tuning"):
            assert (base / name / f"{f}.json").exists(), f"{name}/{f}.json missing"
        assert not (base / name / "judge.json").exists()  # judge off → no file

    assert summary["scenario_count"] == 3
    assert summary["knob_sets"] == ["baseline", "low_bg"]
    # The real selection logic obeys its own rules → every deterministic check passes.
    assert summary["all_deterministic_passed"] is True

    # caps: Alpha is main (message cap 5); 7 injected → ≤ 5 selected.
    sel = json.loads((base / "caps" / "selection.json").read_text(encoding="utf-8"))
    alpha = next(a for a in sel["baseline"] if a["agent_name"] == "Alpha")
    msg_selected = [i for i in alpha["selected"] if i["source"] == "message"]
    assert len(msg_selected) <= 5

    # A/B diff: lowering threshold_bg to 0.15 lets the background agent (Beta) keep the
    # 0.2 ambient it dropped at baseline → a non-empty diff for the `thr` scenario.
    diff = json.loads((base / "thr" / "diff.json").read_text(encoding="utf-8"))
    assert diff.get("low_bg", {}).get("Beta", {}).get("added")


def test_deterministic_checks_catch_violations() -> None:
    """Synthetic offending selections must be flagged (the checks aren't vacuous)."""
    tuning = _build_tuning(None)  # production defaults: threshold_main 0.15, message cap_main 5
    agents = [{
        "agent_id": "a1", "agent_name": "Solo", "is_main_character": True,
        "collected": [], "selected": [
            # below-threshold item slipped in → threshold violation
            {"source": "ambient", "content": "弱", "signal_strength": 0.05, "importance": "LOW", "related_agents": []},
            # duplicate (source, content) → dedup violation
            {"source": "message", "content": "同", "signal_strength": 0.9, "importance": "HIGH", "related_agents": []},
            {"source": "message", "content": "同", "signal_strength": 0.9, "importance": "HIGH", "related_agents": []},
        ],
    }, {
        "agent_id": "a2", "agent_name": "Bg", "is_main_character": False,
        "collected": [], "selected": [
            # two ambient items for a background agent (ambient cap_bg = 1) → cap violation
            {"source": "ambient", "content": "甲", "signal_strength": 0.9, "importance": "HIGH", "related_agents": []},
            {"source": "ambient", "content": "乙", "signal_strength": 0.9, "importance": "HIGH", "related_agents": []},
        ],
    }]
    det = _deterministic_checks(agents, tuning)
    assert det["passed"] is False
    assert det["threshold_violations"] and det["dedup_violations"] and det["cap_violations"]


def test_diff_selections_added_removed() -> None:
    base = [{"agent_id": "a1", "agent_name": "A", "selected": [
        {"source": "ambient", "content": "keep", "importance": "LOW"},
        {"source": "message", "content": "gone", "importance": "MEDIUM"},
    ]}]
    var = [{"agent_id": "a1", "agent_name": "A", "selected": [
        {"source": "ambient", "content": "keep", "importance": "LOW"},
        {"source": "broadcast", "content": "new", "importance": "HIGH"},
    ]}]
    d = _diff_selections(base, var)
    assert any("new" in x for x in d["A"]["added"])
    assert any("gone" in x for x in d["A"]["removed"])
