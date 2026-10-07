"""Integration test: run_build captures phase products + LLM calls."""

from __future__ import annotations

import json

import pytest

from core.interfaces.llm import LLMRouter, LLMScene
from providers.llm.mock import MockLLMProvider
from tuning.build_harness import run_build
from tuning.trace import InMemoryTraceSink

_WORLD_RESPONSE = json.dumps({
    "world_name": "Trace World",
    "era_description": "Era.",
    "core_tension": "A and B want different things.",
    "narrative_theme": "Unspoken weight.",
    "narrative_pitch": "A and B are about to choose.",
    "world_time_config": {"start_year": 1},
    "key_figures": [
        {"name": "Alpha", "role": "r1", "importance": "main", "brief": "A."},
        {"name": "Beta", "role": "r2", "importance": "main", "brief": "B."},
    ],
    "initial_relations": [
        {"from": "Alpha", "to": "Beta", "trust": 0.5, "affection": 0.0, "labels": ["相识"]},
    ],
}, ensure_ascii=False)

_CAST_RESPONSE = json.dumps({
    "roles": [
        {"index": 1, "narrative_role": "r", "arc_summary": "a", "key_relationships": [2]},
        {"index": 2, "narrative_role": "r", "arc_summary": "a", "key_relationships": [1]},
    ]
}, ensure_ascii=False)


@pytest.mark.asyncio
async def test_run_build_records_phases_and_llm_calls(container, test_config) -> None:
    providers = {scene: MockLLMProvider(fixed_response="{}") for scene in LLMScene}
    providers[LLMScene.WORLD_BUILDING] = MockLLMProvider(fixed_response=_WORLD_RESPONSE)
    providers[LLMScene.CAST_DESIGN] = MockLLMProvider(fixed_response=_CAST_RESPONSE)
    container.llm_router = LLMRouter(providers)

    sink = InMemoryTraceSink()
    world_id, world = await run_build(container, test_config, "trace-theme", template="changan_iso", sink=sink)

    # world_id flows through to the built world and to every captured trace.
    assert world.world_id == world_id

    phase_names = {p.phase for p in sink.phases}
    assert phase_names == {
        "world_build.template_selection",
        "world_build.theme_analysis",
        "world_build.cast_design",
        "world_build.agent_generation",
        "world_build.initialization",
    }
    for p in sink.phases:
        assert p.world_id == world_id
        assert p.outputs  # dict (theme/cast) or non-empty list (agent_generation/initialization)

    # The initialization phase carries each agent's step-0 state, incl. goals.
    init = next(p for p in sink.phases if p.phase == "world_build.initialization")
    assert isinstance(init.outputs, list) and init.outputs
    assert all("short_term_goals" in a for a in init.outputs)

    # Theme analysis product carries the structured world name back.
    theme = next(p for p in sink.phases if p.phase == "world_build.theme_analysis")
    assert theme.outputs["world_name"] == "Trace World"

    # Every build-stage LLM call is captured with a non-empty prompt.
    scenes = {c.scene for c in sink.llm_calls}
    assert {"world_building", "cast_design", "persona_generation"} <= scenes
    for call in sink.llm_calls:
        assert call.world_id == world_id
        assert call.prompt_messages and call.prompt_messages[0]["content"]
