"""Restore resumes at the last committed step, from that step's snapshot alone: nothing a step
that never committed (a crash, or a run stopped while waiting on an actor) did may survive into
the rerun, and everything the agent carried at the commit point must come back."""

from __future__ import annotations


import pytest

from agent.agent import snapshot_agent_state
from agent.goals import GoalEntity, GoalStatus
from agent.memory_types import MemoryStream
from agent.need import NeedType
from agent.personality import EmotionType
from agent.state_codec import agent_state_from_record
from core.container import Container
from core.interfaces.condition import BodyCondition
from interaction.application import NarrativeApplication
from world.initializer import WorldInitializer
from providers.snapshot.file import FileSnapshotProvider
from tests.unit.confirmed_world import confirmed_world


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
    world = await confirmed_world(application, "玄武门之变", template="changan_iso")
    await application.run_world(world.world_id, steps=2)
    agent = next(iter(world.agents.values()))
    committed = await container.snapshot.load(world.world_id, 2)

    # Step 3 starts: a memory is written and the agent's relations change, then it never commits.
    await agent.memory_system.seed_factual_memory(current_step=3, raw_content="半步之中写下的记忆")
    relation = (await agent.relation_system.all_relations())[0]
    await agent.relation_system.set_trust_affection(relation.to_id, trust=0.01, affection=None, step=3)
    await agent.relation_system.record_contact("someone-met-mid-step", step=3)

    restored = await WorldInitializer(container).restore(world.world_id)

    assert restored.current_step == 2
    assert all(step <= 2 for step in await _memory_steps(container, restored.agents[agent.agent_id]))
    relations = {
        f"{r.from_id}->{r.to_id}": r
        for restored_agent in restored.agents.values()
        for r in await restored_agent.relation_system.all_relations()
    }
    assert set(relations) == set(committed.state.relations)
    key = f"{relation.from_id}->{relation.to_id}"
    assert relations[key].trust_objective == committed.state.relations[key]["trust_objective"]


@pytest.mark.asyncio
async def test_reset_still_keeps_backstory_memories(container, test_config) -> None:
    application = NarrativeApplication(container, test_config)
    world = await confirmed_world(application, "玄武门之变", template="changan_iso")
    agent_id = next(iter(world.agents))
    seeds = await _memory_steps(container, world.agents[agent_id])
    await application.run_world(world.world_id, steps=2)

    reset = await application.reset_session(world.world_id)

    assert sorted(await _memory_steps(container, reset.agents[agent_id])) == sorted(seeds)


@pytest.mark.asyncio
async def test_an_empty_committed_relation_table_restores_no_relations(container, test_config) -> None:
    application = NarrativeApplication(container, test_config)
    world = await confirmed_world(application, "玄武门之变", template="changan_iso")
    committed = await container.snapshot.load(world.world_id, 0)
    committed.state.relations = {}
    await container.snapshot.save(world.world_id, 0, committed)

    restored = await WorldInitializer(container).restore(world.world_id)

    for agent in restored.agents.values():
        assert await agent.relation_system.all_relations() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["missing_character", "missing_field"])
async def test_restore_rejects_an_incomplete_snapshot(container, test_config, damage) -> None:
    application = NarrativeApplication(container, test_config)
    world = await confirmed_world(application, "玄武门之变", template="changan_iso")
    committed = await container.snapshot.load(world.world_id, 0)
    aid = next(iter(world.agents))
    if damage == "missing_character":
        committed.state.agents.pop(aid)
    else:
        committed.state.agents[aid].pop("emotion")
    await container.snapshot.save(world.world_id, 0, committed)

    with pytest.raises(ValueError, match="every character|incomplete"):
        await WorldInitializer(container).restore(world.world_id)


@pytest.mark.asyncio
async def test_restore_brings_back_everything_the_agent_carried(container, test_config, tmp_path) -> None:
    """The test of a snapshot: the scene it restores is the scene it saved. Nested state (emotion,
    goals with their lifecycle, body condition) and the memory id counter survive a trip through a
    file snapshot."""
    application = NarrativeApplication(container, test_config)
    world = await confirmed_world(application, "玄武门之变", template="changan_iso")
    aid = next(iter(world.agents))
    agent = world.agents[aid]

    agent.personality.update_emotion(
        primary=EmotionType.ANGER, intensity=0.8, valence=-0.6, triggered_by="被当众斥责",
    )
    agent.personality.set_condition(BodyCondition(description="左臂中箭", confined=False))
    state = agent.personality.state
    state.short_term_goal_entities = [GoalEntity(
        id="stg-1", text="说服尉迟恭", goal_type="short_term", status=GoalStatus.ACTIVE,
        related_need=NeedType.SAFETY, created_step=0, due_step=6, last_evaluated_step=0,
        progress_summary="fail:1;对方迟疑。",
    )]
    state.long_term_goal_entities = [GoalEntity(
        id="ltg-1", text="登上大位", goal_type="long_term", status=GoalStatus.ACTIVE,
        related_need=NeedType.SAFETY, created_step=0, progress_summary="已有眉目",
    )]
    state.long_term_goals = ["登上大位"]
    agent.personality.restore_state(state)
    agent.memory_system._seq = 42  # noqa: SLF001
    before = agent.committed_state()

    committed = await container.snapshot.load(world.world_id, 0)
    committed.state.agents[aid] = snapshot_agent_state(agent)
    files = FileSnapshotProvider(str(tmp_path))
    await files.save(world.world_id, 0, committed)
    await container.snapshot.save(world.world_id, 0, await files.load(world.world_id, 0))

    restored = await WorldInitializer(container).restore(world.world_id)

    assert restored.agents[aid].committed_state() == before


@pytest.mark.asyncio
async def test_failed_snapshot_write_restores_the_previous_commit(container, test_config, monkeypatch) -> None:
    application = NarrativeApplication(container, test_config)
    world = await confirmed_world(application, "玄武门之变", template="changan_iso")
    committed = await container.snapshot.load(world.world_id, 0)
    aid = next(iter(world.agents))

    async def fail_save(*args, **kwargs) -> None:
        raise OSError("Snapshot write unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(container.snapshot, "save", fail_save)
        await application.run_world(world.world_id, steps=1)
    assert world.agents[aid].personality.state.step == 1

    restored = await WorldInitializer(container).restore(world.world_id)

    assert restored.current_step == 0
    assert restored.agents[aid].committed_state() == agent_state_from_record(committed.state.agents[aid])


@pytest.mark.asyncio
async def test_a_snapshot_that_cannot_be_built_is_not_committed(container, test_config, monkeypatch) -> None:
    """A step whose snapshot is missing a part doesn't commit; the run goes on and the next step's
    snapshot carries everything forward."""
    application = NarrativeApplication(container, test_config)
    world = await confirmed_world(application, "玄武门之变", template="changan_iso")
    peek = container.message_provider.peek_pending
    calls = {"n": 0}

    async def fail_once(world_id: str):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("Message store unavailable")
        return await peek(world_id)

    monkeypatch.setattr(container.message_provider, "peek_pending", fail_once)
    await application.run_world(world.world_id, steps=2)

    assert await container.snapshot.list_steps(world.world_id) == [0, 2]
