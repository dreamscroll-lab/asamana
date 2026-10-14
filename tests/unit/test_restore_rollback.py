"""Restore resumes at the last committed step, so writes from a step that never committed (a
crash, or a run stopped while waiting on an actor) must not survive into the rerun."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agent.memory_types import MemoryStream
from core.container import Container
from engine.application import NarrativeApplication
from world.initializer import WorldInitializer


@pytest.fixture()
def container(mock_build_container: Container) -> Container:
    return mock_build_container


async def _memory_steps(container: Container, agent) -> list[int]:
    steps: list[int] = []
    for stream in MemoryStream:
        for record in await container.vector_store.list_all(agent.memory_system._namespace(stream)):  # noqa: SLF001
            steps.append(int(record.payload.get("created_step", 0) or 0))
    return steps


@pytest.mark.asyncio
async def test_restore_drops_what_an_uncommitted_step_wrote(container, test_config) -> None:
    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")
    await application.run_world(world.world_id, steps=2)
    agent = next(iter(world.agents.values()))
    committed = await container.snapshot.load(world.world_id, 2)

    # Step 3 starts writing, then never commits.
    await agent.memory_system.seed_factual_memory(current_step=3, raw_content="半步之中写下的记忆")
    relation = next(iter(await container.agent_store.load_all_relations(world.world_id, agent.agent_id)))
    await container.agent_store.save_relation(replace(relation, trust_objective=0.01, updated_step=3))
    stray = replace(relation, to_id="someone-met-mid-step", to_name="路人", labels=[], updated_step=3)
    await container.agent_store.save_relation(stray)

    restored = await WorldInitializer(container).restore(world.world_id)

    assert restored.current_step == 2
    assert all(step <= 2 for step in await _memory_steps(container, restored.agents[agent.agent_id]))
    relations = {
        f"{r.from_id}->{r.to_id}": r
        for agent_id in restored.agents
        for r in await container.agent_store.load_all_relations(world.world_id, agent_id)
    }
    assert set(relations) == set(committed.state.relations)
    key = f"{relation.from_id}->{relation.to_id}"
    assert relations[key].trust_objective == committed.state.relations[key]["trust_objective"]


@pytest.mark.asyncio
async def test_reset_still_keeps_backstory_memories(container, test_config) -> None:
    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")
    agent_id = next(iter(world.agents))
    seeds = await _memory_steps(container, world.agents[agent_id])
    await application.run_world(world.world_id, steps=2)

    reset = await application.reset_session(world.world_id)

    assert sorted(await _memory_steps(container, reset.agents[agent_id])) == sorted(seeds)
