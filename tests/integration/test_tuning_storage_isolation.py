"""Dry-runs must leave in-flight production writes untouched, including during restore."""

from __future__ import annotations

from dataclasses import asdict
import json

import pytest

from agent.memory_types import MemoryStream
from core.interfaces.relation import AgentRelation
from providers.llm.mock import MockLLMProvider
from tuning.build_harness import run_build
from tuning.phase_harness.common import restore_traced
from tuning.phase_harness.need import run_need_engine
from tuning.phase_harness.perception_emotion import run_perception_emotion
from tuning.phase_harness.plan_step import run_plan_step
from tuning.trace import InMemoryTraceSink
from tuning.stages import all_stages


@pytest.mark.parametrize("entry", ["restore", "need_harness", "emotion", "plan", *[s.key for s in all_stages()]])
async def test_dry_run_preserves_production_relation_and_memory(
    mock_build_container, test_config, tmp_path, entry: str,
) -> None:
    container = mock_build_container
    world_id, world = await run_build(
        container, test_config, "theme", template="changan_iso", sink=InMemoryTraceSink(),
    )
    agent_id = next(iter(world.agents))
    relation = AgentRelation(
        world_id=world_id, from_id=agent_id, to_id="in-flight-person",
        trust_objective=0.8, affection_objective=0.3, updated_step=1,
        to_name="Only in the active step",
    )
    live_relations = world.agents[agent_id].relation_system
    live_relations.restore([*await live_relations.all_relations(), relation])
    collection = f"{world_id}:{agent_id}:memory:{MemoryStream.FACTUAL.value}"
    await container.vector_store.upsert(
        collection, "in-flight-memory", [0.0] * container.embedding.dimension, None,
        {"created_step": 1, "stored_content": "The active step has not committed yet."},
    )
    before_relations = [
        asdict(r) for a in world.agents.values() for r in await a.relation_system.all_relations()
    ]
    before_steps = await container.snapshot.list_steps(world_id)
    before_memories = [asdict(r) for r in await container.vector_store.list_all(collection)]
    sink = InMemoryTraceSink()
    if entry == "restore":
        isolated, restored, _, _ = await restore_traced(container, test_config, world_id, sink)
        assert await restored.agents[agent_id].relation_system.load_existing(relation.to_id) is None
        assert "in-flight-memory" not in {
            r.id for r in await isolated.vector_store.list_all(collection)
        }
        assert restored.agents
    elif entry == "need_harness":
        await run_need_engine(container, test_config, world_id, scenario={}, sink=sink)
    elif entry == "emotion":
        await run_perception_emotion(container, test_config, world_id, scenario={}, sink=sink)
    elif entry == "plan":
        await run_plan_step(container, test_config, world_id, sink=sink)
    else:
        spec = next(s for s in all_stages() if s.key == entry)
        scenarios = tmp_path / "scenarios.json"
        scenarios.write_text(json.dumps({"scenarios": []}), encoding="utf-8")
        kwargs = {"judge_provider": MockLLMProvider(fixed_response="{}")} if spec.judged else {}
        await spec.runner(
            container, test_config, world_id, scenarios_path=str(scenarios),
            trace_dir=str(tmp_path), **kwargs,
        )
    assert [
        asdict(r) for a in world.agents.values() for r in await a.relation_system.all_relations()
    ] == before_relations
    assert await container.snapshot.list_steps(world_id) == before_steps
    assert [asdict(r) for r in await container.vector_store.list_all(collection)] == before_memories
