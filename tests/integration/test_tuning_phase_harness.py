"""Integration test: plan_step dry-run harness restores a world and captures plans."""

from __future__ import annotations

import json

import pytest

from core.interfaces.llm import LLMRouter, LLMScene
from providers.llm.mock import MockLLMProvider
from tuning.build_harness import run_build
from tuning.fixtures import load_fixture
from tuning.phase_harness.plan_step import run_plan_step
from tuning.phase_harness.world_pressure import run_world_pressure
from tuning.trace import InMemoryTraceSink

_WORLD_RESPONSE = json.dumps({
    "world_name": "Plan World",
    "era_description": "Era.",
    "core_tension": "A and B want different things.",
    "narrative_theme": "Tension.",
    "narrative_pitch": "A and B are poised to act.",
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

_CAST_RESPONSE = json.dumps({
    "roles": [
        {"index": 1, "narrative_role": "r", "arc_summary": "a", "key_relationships": [2]},
        {"index": 2, "narrative_role": "r", "arc_summary": "a", "key_relationships": [1]},
    ]
}, ensure_ascii=False)


@pytest.mark.asyncio
async def test_run_plan_step_restores_and_captures_plans(container, test_config) -> None:
    providers = {scene: MockLLMProvider(fixed_response="{}") for scene in LLMScene}
    providers[LLMScene.WORLD_BUILDING] = MockLLMProvider(fixed_response=_WORLD_RESPONSE)
    providers[LLMScene.CAST_DESIGN] = MockLLMProvider(fixed_response=_CAST_RESPONSE)
    # A parseable decision (selected_index=3 → WORK) so plan_step yields a real
    # decision; "{}" would parse to no selection → action=None (agent skipped).
    providers[LLMScene.AGENT_DECISION_MAIN] = MockLLMProvider(
        fixed_response='{"selected_index": 3, "action_description": "整理书卷", "estimated_steps": 1}'
    )
    container.llm_router = LLMRouter(providers)

    # Build + persist (in-memory stores), then dry-run plan_step on the same container.
    world_id, _ = await run_build(container, test_config, "plan-theme", template="changan_iso", sink=InMemoryTraceSink())

    sink = InMemoryTraceSink()
    step = await run_plan_step(container, test_config, world_id, sink=sink)

    assert step == 1  # built-but-not-run world → first cognitive step
    assert sink.phases, "expected one plan trace per agent"
    assert all(p.phase == "step.plan" and p.step == 1 for p in sink.phases)
    assert all(p.world_id == world_id for p in sink.phases)

    view = sink.phases[0].outputs
    assert {"perception", "motivation", "decision"} <= set(view)
    assert "action_type" in view["decision"]


@pytest.mark.asyncio
async def test_run_world_pressure_records_per_agent_and_writes_fixture(
    container, test_config, tmp_path
) -> None:
    providers = {scene: MockLLMProvider(fixed_response="{}") for scene in LLMScene}
    providers[LLMScene.WORLD_BUILDING] = MockLLMProvider(fixed_response=_WORLD_RESPONSE)
    providers[LLMScene.CAST_DESIGN] = MockLLMProvider(fixed_response=_CAST_RESPONSE)
    container.llm_router = LLMRouter(providers)

    world_id, _ = await run_build(container, test_config, "pressure-theme", template="changan_iso", sink=InMemoryTraceSink())

    # Inject a broadcast scenario so pressure is not signal-gated to empty.
    scenario = {"broadcasts": [{"content": "远处传来喊杀声", "severity": "high"}]}
    sink = InMemoryTraceSink()
    step = await run_world_pressure(
        container, test_config, world_id, scenario=scenario, sink=sink, trace_dir=str(tmp_path)
    )

    assert step == 1
    assert sink.phases, "expected one world_pressure trace per active agent"
    assert all(p.phase == "stage.world_pressure" and p.step == 1 for p in sink.phases)
    assert all("external_goals" in p.outputs for p in sink.phases)
    # the injected scenario is echoed in the captured inputs
    assert sink.phases[0].inputs["injected_broadcasts"][0]["content"] == "远处传来喊杀声"
    # the LLM was actually invoked (signal present) and captured
    assert any(c.scene == "world_pressure" for c in sink.llm_calls)

    # both fixtures frozen: scenario (stable re-runs) + external_goals (downstream input).
    assert load_fixture(str(tmp_path), world_id, "scenario") is not None
    ext = load_fixture(str(tmp_path), world_id, "external_goals")
    assert ext is not None and ext["step"] == 1 and "goals" in ext


@pytest.mark.asyncio
async def test_world_pressure_scenario_injects_messages_and_ambient(
    container, test_config, tmp_path
) -> None:
    providers = {scene: MockLLMProvider(fixed_response="{}") for scene in LLMScene}
    providers[LLMScene.WORLD_BUILDING] = MockLLMProvider(fixed_response=_WORLD_RESPONSE)
    providers[LLMScene.CAST_DESIGN] = MockLLMProvider(fixed_response=_CAST_RESPONSE)
    container.llm_router = LLMRouter(providers)
    world_id, _ = await run_build(container, test_config, "sig-theme", template="changan_iso", sink=InMemoryTraceSink())

    # Inbox message addressed by name + a global ambient event.
    scenario = {
        "messages": [{"to": "Alpha", "from": "Beta", "content": "速来", "urgency": "high"}],
        "ambient": [{"content": "空气中弥漫血腥味", "strength": 0.8}],
    }
    sink = InMemoryTraceSink()
    await run_world_pressure(
        container, test_config, world_id, scenario=scenario, sink=sink, trace_dir=str(tmp_path)
    )

    by_name = {p.outputs["agent_name"]: p for p in sink.phases}
    # Alpha's inbox carries the injected message (resolved by name → id).
    assert by_name["Alpha"].inputs["injected_messages"][0]["content"] == "速来"
    assert by_name["Alpha"].inputs["injected_messages"][0]["from"] == "Beta"
    # Ambient with no target is global → present for every agent.
    assert all(p.inputs["injected_ambient"][0]["content"] == "空气中弥漫血腥味" for p in sink.phases)
    # signal present → the world-level LLM was invoked.
    assert any(c.scene == "world_pressure" for c in sink.llm_calls)
