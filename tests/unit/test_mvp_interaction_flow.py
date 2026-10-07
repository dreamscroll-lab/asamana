from __future__ import annotations

import pytest

from agent.decision import ActionType
from core.container import Container
from core.interfaces.action import EntityStateChange
from engine.application import NarrativeApplication
from interaction.replayer import Replayer
from interaction.world_manager import WorldManager
from world import WorldBuilder
from world.initializer import WorldInitializer
from world.models import WorldEntity, WorldEntityType
from worlds.tiled import TiledWorldConfig


@pytest.fixture()
def container(mock_build_container: Container) -> Container:
    """MVP flow tests build full worlds: alias the shared mock-build container."""
    return mock_build_container


@pytest.mark.asyncio
async def test_world_builder_persists_real_step_zero_snapshot(container) -> None:
    builder = WorldBuilder(container)

    world = await builder.build(
        theme="玄武门之变",
        world_id="world-init",
        world_configs=[TiledWorldConfig(template="changan_iso")],
    )
    snapshot = await container.snapshot.load(world.world_id, 0)

    assert snapshot is not None
    assert snapshot == world.step_zero_snapshot
    assert snapshot.metadata["phase"] == "initialization"
    assert snapshot.metadata["schedule"] == {"step": 0, "batches": []}
    assert snapshot.metadata["messages"]["step"] == 0
    assert set(snapshot.metadata["messages"]["inboxes"]) == set(world.agents)
    assert snapshot.metadata["environment"]["body_locations"]
    assert set(snapshot.metadata["environment"]["body_locations"].values()) <= set(
        world.world_config.get_places()
    )
    assert len(snapshot.agent_summaries) == len(world.agents)
    assert snapshot.event_summaries == []
    assert set(snapshot.agent_states) == set(world.agents)
    first_state = next(iter(snapshot.agent_states.values()))
    assert {
        "agent_id",
        "agent_name",
        "current_location",
        "activity_status",
        "current_action",
        "action_remaining_steps",
    } <= set(first_state)
    assert snapshot.agent_relations

    stored_states = [
        await container.agent_store.load_agent_state(world.world_id, agent_id)
        for agent_id in await container.agent_store.list_agent_ids(world.world_id)
    ]
    assert len(stored_states) == len(world.agents)
    assert any(state is not None and state.current_location for state in stored_states)


@pytest.mark.asyncio
async def test_application_runtime_produces_replayable_history(container, test_config) -> None:
    application = NarrativeApplication(container, test_config)
    manager = WorldManager(
        snapshot_provider=container.snapshot,
    )

    world = await application.build_world("玄武门之变", template="changan_iso")
    background_agents = [agent for agent in world.agent_list() if not agent.is_main_character]
    assert background_agents
    assert application.get_session(world.world_id).runtime._executor_registry.get_executor(
        ActionType.REST
    ) is not None

    scheduler_plan = application.get_session(world.world_id).runtime._scheduler.plan(
        world.agent_list(),
        step=2,
        in_progress_at_step_start=set(),
    )
    assert any(batch.phase == "background" for batch in scheduler_plan.batches)

    await application.run_world(world.world_id, steps=2)

    replayer = manager.create_replayer(world.world_id)
    steps = await replayer.list_steps()
    assert steps == [0, 1, 2]
    replay_steps = [await replayer.get_step(step) for step in steps]

    assert [event.step for event in replay_steps] == [0, 1, 2]
    assert replay_steps[0].world_time.label.startswith(world.clock_config.era_name)
    step_one_snapshot = await container.snapshot.load(world.world_id, 1)
    assert step_one_snapshot is not None
    # A snapshot round-trips to disk with both clocks kept apart: the world's own name for the
    # moment and the machine coordinate. The stub analysis sets no hours_per_step, so a step is
    # 2 hours and step 1 is 6:00 + 2h = 8 a.m. (spoken 12-hour form, no minutes on the hour).
    # "六月初一" comes from the changan_iso map; era and year are separate fields so the year can
    # advance.
    assert step_one_snapshot.time_label == "武德九年，六月初一，上午八点"
    assert step_one_snapshot.clock.startswith("step=0001 ")
    # Runtime step snapshots carry relations in the same shape as the step-0 snapshot.
    step_zero_snapshot = await container.snapshot.load(world.world_id, 0)
    assert step_one_snapshot.agent_relations
    assert set(next(iter(step_one_snapshot.agent_relations.values()))) == set(
        next(iter(step_zero_snapshot.agent_relations.values()))
    )
    assert replay_steps[0].actions
    assert replay_steps[0].agent_states
    assert await manager.latest_snapshot(world.world_id) == await container.snapshot.load(
        world.world_id,
        2,
    )


@pytest.mark.asyncio
async def test_runtime_snapshots_expose_all_agent_states_to_replay(container, test_config) -> None:
    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")
    await application.run_world(world.world_id, steps=1)

    snapshot = await container.snapshot.load(world.world_id, 1)
    assert snapshot is not None
    assert set(snapshot.agent_states) == set(world.agents)

    event = await Replayer(
        world.world_id,
        container.snapshot,
    ).get_step(1)
    assert set(event.agent_states) == set(world.agents)
    for summary in event.agent_states.values():
        assert summary.vitality is not None
        assert 0.0 <= summary.vitality <= 1.0


@pytest.mark.asyncio
async def test_restore_recovers_live_environment_entity_state(container, test_config) -> None:
    """A restored world recovers item state/ownership that drifted during the run,
    rather than reverting items to their step-0 seed positions."""
    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")

    # An item enters the world mid-run and is picked up + altered by an agent.
    runtime = application.get_session(world.world_id).runtime
    environment = runtime._environment
    holder_id = next(iter(world.agents))
    holder_location = environment.get_body_location(holder_id)
    environment.register_entity(
        WorldEntity(
            entity_id="seed_edict",
            name="敕令",
            entity_type=WorldEntityType.ITEM,
            state="sealed",
            presence_ref=holder_location,
            is_takeable=True,
        )
    )
    environment.change_entity_state(
        EntityStateChange(entity_id="seed_edict", new_state="unsealed", owner_id=holder_id),
    )

    # Persist a step so the live environment lands in a snapshot.
    await application.run_world(world.world_id, steps=1)

    # Restore from storage into a fresh world (new environment from step-0 seeds).
    initializer = WorldInitializer(container)
    restored = await initializer.restore(world.world_id)

    edict = restored.environment.find_item("seed_edict")
    assert edict is not None, "mid-run entity should be re-registered on restore"
    assert edict.state == "unsealed"
    assert edict.owner_id == holder_id
    assert edict.location_id is None
    assert [e.entity_id for e in restored.environment.get_items_of(holder_id)] == ["seed_edict"]


@pytest.mark.asyncio
async def test_restore_session_rehydrates_event_throttle(container, test_config) -> None:
    """restore_session replays fired events into the EventSystem so a restored world
    keeps its window quota instead of resetting it."""
    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")
    await application.run_world(world.world_id, steps=1)

    # Inject a fired event into the latest snapshot's event log to stand in for a
    # world event that the run would have produced.
    latest = await container.snapshot.load(world.world_id, 1)
    latest.events_this_step.append(
        {
            "id": "evt-restore", "step": 1, "narrative_desc": "城门骤变",
            "is_positive": None, "dispatched_to": ["broadcast"],
            "authored_by": "system", "metadata": {},
        }
    )
    await container.snapshot.save(world.world_id, 1, latest)

    # Fresh application instance forces a real restore (no in-memory session reuse).
    restored_app = NarrativeApplication(container, test_config)
    await restored_app.restore_session(world.world_id)

    event_system = restored_app.get_session(world.world_id).runtime._event_system
    assert any(ev.id == "evt-restore" for ev in event_system.list_events())


@pytest.mark.asyncio
async def test_listing_worlds_does_not_read_the_whole_history(container, test_config) -> None:
    """The world list reads two snapshots, so read volume doesn't grow with world age.

    The frontend polls this list for every world, and snapshot reads are synchronous JSON parsing
    that would stall running worlds' LLM calls. Identity is fixed in step 0 at build time; progress
    needs only the last step.
    """
    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")
    await application.run_world(world.world_id, steps=3)

    loaded: list[int] = []
    original_load = container.snapshot.load

    async def counting_load(world_id: str, step: int):
        loaded.append(step)
        return await original_load(world_id, step)

    container.snapshot.load = counting_load
    try:
        manager = WorldManager(
            snapshot_provider=container.snapshot
        )
        meta = await manager.get_world(world.world_id)
    finally:
        container.snapshot.load = original_load

    assert sorted(loaded) == [0, 3], f"只该读 step 0 与最后一步,实际读了 {sorted(loaded)}"
    # Reading less must not lose data: main characters and progress are still correct.
    assert meta is not None
    assert meta.current_step == 3
    assert meta.main_agent_names == sorted(
        agent.personality.soul.name
        for agent in world.agent_list()
        if agent.is_main_character
    )


@pytest.mark.asyncio
async def test_restoring_again_never_displaces_a_live_session(container, test_config) -> None:
    """Restore must not replace an existing session; replacing it also drops the re-entry guard.

    Two restores of one world would install a new session with task=None, and start_run's guard
    checks session.task, so a second, unstoppable run loop writes to the same step numbers.
    Reproduced here with sequential calls (restore's per-world lock covers the concurrent path).
    """
    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")

    restored_app = NarrativeApplication(container, test_config)
    await restored_app.restore_session(world.world_id)
    restored_app.start_run(world.world_id, steps=1)
    session = restored_app.get_session(world.world_id)

    await restored_app.restore_session(world.world_id)

    assert restored_app.get_session(world.world_id) is session, "恢复顶掉了正在跑的 session"
    with pytest.raises(ValueError, match="already running"):
        restored_app.start_run(world.world_id, steps=1)
    await session.task


@pytest.mark.asyncio
async def test_a_never_run_world_still_restores_its_in_flight_messages(
    container, test_config
) -> None:
    """A built-but-never-run world (last snapshot = step 0) must also restore in-flight messages.

    "Last persisted snapshot" and "which step the clock starts from" are separate questions: the
    latter skips step 0, the former must not. One number for both clears a never-run world's queue.
    """
    from core.interfaces.message import Message

    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")

    agent_id = next(iter(world.agents))
    await container.message_provider.enqueue(Message(
        id="msg-at-build", world_id=world.world_id, sender_id=agent_id,
        content="密信：按计行事", recipients=[agent_id],
        location_scope=None, deliver_step=5, created_step=0,
    ))
    # Rewrite the step-0 snapshot to carry this in-flight message (the queue was empty at build time).
    step0 = await container.snapshot.load(world.world_id, 0)
    step0.pending_messages = await container.message_provider.peek_pending(world.world_id)
    await container.snapshot.save(world.world_id, 0, step0)
    assert await container.snapshot.list_steps(world.world_id) == [0]

    await container.message_provider.clear()          # new process: the in-memory queue is gone
    restored = NarrativeApplication(container, test_config)
    await restored.restore_session(world.world_id)

    pending = await container.message_provider.peek_pending(world.world_id)
    assert [m.id for m in pending] == ["msg-at-build"]


@pytest.mark.asyncio
async def test_a_death_notice_survives_a_restore(container, test_config) -> None:
    """A death notice is queued for the next step, so it crosses the snapshot boundary and must be
    persisted.

    remove_body takes the body out on the step of death, so the notice is the survivors' only way
    to learn of it. Covers publish, snapshot, restore in a new process, then collect on the due step.
    """
    from core.interfaces.perception import Broadcast, BroadcastType

    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")
    await application.run_world(world.world_id, steps=1)

    channel = application.get_session(world.world_id).runtime._broadcast_channel
    channel.publish(Broadcast(
        content="宫中传出消息：那人已死。",
        source="system",
        broadcast_type=BroadcastType.WORLD_EVENT,
        deliver_step=3,        # future: still pending after step 2
    ))

    await application.run_world(world.world_id, steps=1)
    latest = await container.snapshot.load(world.world_id, 2)
    assert [b.content for b in latest.pending_broadcasts] == ["宫中传出消息：那人已死。"]

    # New process: the channel belongs to the runtime, and a fresh BroadcastChannel is empty.
    restored = NarrativeApplication(container, test_config)
    await restored.restore_session(world.world_id)
    restored_channel = restored.get_session(world.world_id).runtime._broadcast_channel

    due = restored_channel.collect(step=3)
    assert [b.content for b in due] == ["宫中传出消息：那人已死。"]
    assert due[0].deliver_step == 3


@pytest.mark.asyncio
async def test_restore_session_rehydrates_in_flight_messages(container, test_config) -> None:
    """A message sent before the save but due for delivery after it survives a
    cross-process restore. Only InMemoryMessageProvider exists, so the queue is lost
    on restart unless restore re-enqueues the snapshot's pending_messages."""
    from core.interfaces.message import Message

    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")

    agent_id = next(iter(world.agents))
    in_flight = Message(
        id="msg-in-flight",
        world_id=world.world_id,
        sender_id=agent_id,
        content="密信：按计行事",
        recipients=[agent_id],
        location_scope=None,
        deliver_step=10,  # future: stays pending after the run
        created_step=0,
    )
    await application.get_session(world.world_id).runtime._message_system.publish(in_flight)

    # Run a step so the pending message lands in the persisted snapshot.
    await application.run_world(world.world_id, steps=1)
    latest = await container.snapshot.load(world.world_id, 1)
    assert any(m.id == "msg-in-flight" for m in latest.pending_messages)

    # Simulate a process restart: the in-memory queue is gone.
    await container.message_provider.clear()
    assert await container.message_provider.peek_pending(world.world_id) == []

    restored_app = NarrativeApplication(container, test_config)
    await restored_app.restore_session(world.world_id)

    pending = await container.message_provider.peek_pending(world.world_id)
    assert [m.id for m in pending] == ["msg-in-flight"]
    assert pending[0].deliver_step == 10

    # Idempotent: restoring again must not re-enqueue in-flight messages. (The same app already
    # holds a session here and reuses it.)
    await NarrativeApplication(container, test_config).restore_session(world.world_id)
    pending_again = await container.message_provider.peek_pending(world.world_id)
    assert [m.id for m in pending_again] == ["msg-in-flight"]


def test_goal_entity_from_dict_preserves_lifecycle_metadata() -> None:
    """Restoring a persisted goal dict keeps related_need / created_step /
    last_evaluated_step / progress_summary instead of resetting them."""
    from agent.goals import GoalStatus
    from agent.need import NeedType
    from world.initializer import _goal_entity_from_dict

    ent = {
        "id": "stg-7",
        "text": "稳住东宫局势",
        "goal_type": "short_term",
        "status": "active",
        "related_need": NeedType.SAFETY.value,
        "created_step": 12,
        "last_evaluated_step": 18,
        "progress_summary": "fail:2;仍未达成。",
    }
    goal = _goal_entity_from_dict(ent, fallback_id="restored-stg-0")
    assert goal.id == "stg-7"
    assert goal.status == GoalStatus.ACTIVE
    assert goal.related_need == NeedType.SAFETY
    assert goal.created_step == 12
    assert goal.last_evaluated_step == 18
    assert goal.progress_summary == "fail:2;仍未达成。"

    # An unknown related_need degrades to None rather than raising.
    degraded = _goal_entity_from_dict(
        {"id": "x", "text": "t", "related_need": "not_a_need"}, fallback_id="fb"
    )
    assert degraded.related_need is None


def test_goal_entity_from_dict_origin_round_trip_and_legacy_default() -> None:
    """origin round-trips; existing saves (no such field) and unknown values fall back to COGNITIVE.

    COGNITIVE rather than RESIDUE is the conservative choice: misreading a goal as unfinished business
    would give an old plan unearned protection from eviction.
    """
    from agent.goals import GoalOrigin
    from world.initializer import _goal_entity_from_dict

    restored = _goal_entity_from_dict(
        {"id": "x", "text": "我答应了要去查此事", "origin": "residue"}, fallback_id="fb"
    )
    assert restored.origin == GoalOrigin.RESIDUE

    legacy = _goal_entity_from_dict({"id": "x", "text": "t"}, fallback_id="fb")
    assert legacy.origin == GoalOrigin.COGNITIVE

    unknown = _goal_entity_from_dict(
        {"id": "x", "text": "t", "origin": "not_an_origin"}, fallback_id="fb"
    )
    assert unknown.origin == GoalOrigin.COGNITIVE


@pytest.mark.asyncio
async def test_restore_preserves_persisted_goal_metadata(container, test_config) -> None:
    """Integration: goal lifecycle metadata written to the store survives a
    full restore — _apply_stored_state must not silently drop it."""
    from agent.need import NeedType

    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")
    await application.run_world(world.world_id, steps=1)

    agent_id = next(iter(world.agents))
    stored = await container.agent_store.load_agent_state(world.world_id, agent_id)
    stored.short_term_goal_entities = [
        {
            "id": "stg-meta",
            "text": "巩固兵权",
            "goal_type": "short_term",
            "status": "active",
            "related_need": NeedType.SAFETY.value,
            "created_step": 9,
            "last_evaluated_step": 14,
            "progress_summary": "fail:3;屡试未果。",
        }
    ]
    await container.agent_store.save_agent_state(world.world_id, agent_id, stored)

    initializer = WorldInitializer(container)
    restored = await initializer.restore(world.world_id)
    goals = restored.agents[agent_id].personality.state.short_term_goal_entities
    meta = next(g for g in goals if g.id == "stg-meta")
    assert meta.related_need == NeedType.SAFETY
    assert meta.created_step == 9
    assert meta.last_evaluated_step == 14
    assert meta.progress_summary == "fail:3;屡试未果。"


@pytest.mark.asyncio
async def test_reset_session_rebuilds_in_memory_world_at_step_zero(container, test_config) -> None:
    """reset_session must reset the *live* in-memory agents to the step-0
    baseline, not just the persisted store. Otherwise the session runs step-N minds
    on a step-0 clock."""
    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")
    await application.run_world(world.world_id, steps=2)

    agent_id = next(iter(world.agents))
    # The run advanced persisted state past step 0.
    advanced = await container.agent_store.load_agent_state(world.world_id, agent_id)
    assert advanced.updated_step > 0

    reset_world = await application.reset_session(world.world_id)

    # In-memory agents (the ones the fresh session will actually run) are at step 0.
    live_agent = application.get_session(world.world_id).world.agents[agent_id]
    assert live_agent.personality.state.step == 0
    assert reset_world.agents[agent_id].personality.state.step == 0
    # The session clock starts at step 0 too, and the world's current step with it.
    assert application.get_session(world.world_id).runtime._clock.current_step == 0
    assert reset_world.current_step == 0
    # Persisted state was rolled back to the baseline.
    rolled_back = await container.agent_store.load_agent_state(world.world_id, agent_id)
    assert rolled_back.updated_step == 0
    # No post-step-0 snapshots remain.
    assert await container.snapshot.list_steps(world.world_id) == [0]


@pytest.mark.asyncio
async def test_restore_sets_world_current_step_to_resume_step(container, test_config) -> None:
    """A restored world carries current_step = the latest snapshot (its resume
    step), the single authoritative world step the clock and planning derive from —
    agents keep no counter of their own that could drift after restore."""
    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")
    await application.run_world(world.world_id, steps=3)
    latest = max(await container.snapshot.list_steps(world.world_id))
    assert latest == 3

    initializer = WorldInitializer(container)
    restored = await initializer.restore(world.world_id)
    assert restored.current_step == latest
    # A freshly built world resumes at step 0.
    assert world.current_step == 0


@pytest.mark.asyncio
async def test_reset_session_purges_run_traces_but_keeps_build_traces(container, test_config) -> None:
    """Traces rewind with the clock. Reset restarts the clock at 1 and trace sinks
    append, so a surviving step trace from the previous run would be merged into the
    next run's step of the same number (the read side groups by step). Build traces
    stay — reset replays the world, it does not rebuild it."""
    from core.interfaces.trace import LLMCallTrace, StepTrace
    from providers.trace import InMemoryTraceSink

    sink = InMemoryTraceSink()
    container.trace_sink = sink  # tests run with observability off (NullTraceSink)
    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")

    def _trace(step: int | None) -> LLMCallTrace:
        return LLMCallTrace(
            world_id=world.world_id, stage="world_init" if step is None else "decision",
            scene="s", prompt_messages=[], response_content="r", temperature=0.7,
            max_tokens=10, input_tokens=1, output_tokens=1, model="m", latency_ms=1.0,
            timestamp="t", step=step,
        )

    sink.record_llm_call(_trace(None))          # world build
    sink.record_llm_call(_trace(1))             # run
    sink.record_step(StepTrace(world.world_id, 1, {"label": "晨"}, 5.0, "ts"))

    await application.reset_session(world.world_id)

    assert [c.step for c in sink.read_calls(world.world_id)] == [None]
    assert sink.read_step_summaries(world.world_id) == []


@pytest.mark.asyncio
async def test_reset_session_purges_runtime_memories_but_keeps_seeds(container, test_config) -> None:
    """Reset drops memories accumulated during the run (created_step > 0) while
    preserving the build-time backstory seeds (created_step <= 0)."""
    from agent.memory import memory_collection_name
    from agent.memory_types import MemoryStream

    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")

    agent_id = next(iter(world.agents))
    mem = application.get_session(world.world_id).world.agents[agent_id].memory_system
    await mem.seed_factual_memory(current_step=-3, raw_content="开战前的旧事", triggered_by="seed")
    await mem.seed_factual_memory(current_step=7, raw_content="运行期发生的事", triggered_by="run")

    collection = memory_collection_name(world.world_id, agent_id, MemoryStream.FACTUAL)
    before = await container.vector_store.list_all(collection)
    steps_before = {int(r.payload.get("created_step", 0)) for r in before}
    assert 7 in steps_before and -3 in steps_before

    await application.reset_session(world.world_id)

    after = await container.vector_store.list_all(collection)
    steps_after = [int(r.payload.get("created_step", 0)) for r in after]
    assert all(s <= 0 for s in steps_after), f"runtime memory survived reset: {steps_after}"
    assert -3 in steps_after, "build-time seed must survive reset"


@pytest.mark.asyncio
async def test_reset_session_drops_runtime_created_relations(container, test_config) -> None:
    """A relation created during the run (a directed pair absent at step 0) is
    deleted on reset, leaving only the initial baseline relations."""
    from core.interfaces.agent_store import AgentRelation

    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")

    from_id = next(iter(world.agents))
    initial = await container.agent_store.load_all_relations(world.world_id, from_id)
    initial_targets = {r.to_id for r in initial}
    assert "ghost-target" not in initial_targets

    await container.agent_store.save_relation(
        AgentRelation(
            world_id=world.world_id,
            from_id=from_id,
            to_id="ghost-target",
            trust_objective=0.5,
            affection_objective=0.0,
            updated_step=5,
            labels=["路人"],
        )
    )
    assert "ghost-target" in {
        r.to_id for r in await container.agent_store.load_all_relations(world.world_id, from_id)
    }

    await application.reset_session(world.world_id)

    after = await container.agent_store.load_all_relations(world.world_id, from_id)
    assert "ghost-target" not in {r.to_id for r in after}
    assert {r.to_id for r in after} == initial_targets


@pytest.mark.asyncio
async def test_build_world_auto_registers_in_shared_catalog(container, test_config) -> None:
    """build_world records the world in the catalog as a build-time side effect, so
    it is enumerable without a separate register step. A manager sharing the same
    catalog lists it with the build's display hints."""
    from world import WorldCatalog

    catalog = WorldCatalog()  # in-memory: no disk write in tests
    application = NarrativeApplication(container, test_config, catalog=catalog)
    world = await application.build_world("玄武门之变", template="changan_iso")

    entry = catalog.get(world.world_id)
    assert entry is not None
    assert entry["theme"] == "玄武门之变"
    assert entry["world_name"] == world.analysis.world_name

    manager = WorldManager(
        snapshot_provider=container.snapshot,
        catalog=catalog,
    )
    listed = await manager.list_worlds()
    meta = next((m for m in listed if m.world_id == world.world_id), None)
    assert meta is not None
    assert meta.world_name == world.analysis.world_name


@pytest.mark.asyncio
async def test_build_world_without_catalog_does_not_register(container, test_config) -> None:
    """With no catalog injected, build_world must not register anywhere (so tests and
    catalog-less embeddings never write ./data/worlds.json)."""
    application = NarrativeApplication(container, test_config)  # no catalog
    world = await application.build_world("玄武门之变", template="changan_iso")
    # Session is live and runnable even though nothing was catalogued.
    assert application.has_session(world.world_id)
