from __future__ import annotations

import asyncio

import pytest

from agent.agent import Agent, AgentStepPlan
from agent.decision import ActionType, AgentAction, DecisionEngine, DecisionStatus
from core.interfaces.action import ActionResult, ActionTarget, Observed, Ref
from agent.memory import MemorySystem
from agent.motivation import ExternalDriveType, ExternalGoal
from agent.need import NeedEngine, NeedEvaluation, NeedState, NeedType, build_innate_needs
from agent.personality import ActionStatus, PersonalityLayer, SoulLayer, activity_status_for
from agent.relation import RelationSystem
from core.interfaces.message import Message
from core.interfaces.trace import LLMCallTrace, Stage
from core.interfaces.urgency import Urgency
from engine.broadcast import BroadcastChannel
from engine.clock import GlobalClock, WorldTimeConfig
from engine.directory import LiveWorldDirectory
from engine.environment import SALIENT_AMBIENT_STRENGTH, IN_TRANSIT, EnvironmentSystem
from engine.event import EventSettings
from engine.executors import build_default_registry
from engine.executors.base import ActionExecutionState, ActionExecutor
from engine.execution_processor import ExecutionProcessor
from engine.executors.registry import ActionExecutorRegistry
from engine.message_system import MessageSystem
from engine.runtime import NarrativeRuntime
from engine.scheduler import AgentScheduler
from providers.trace.in_memory import InMemoryTraceSink
from core.interfaces.place import Place
from worlds.tiled import TiledWorldConfig
from tests.unit.bus_tap import tap


def _build_agent(container, *, world_id: str, agent_id: str, name: str, is_main_character: bool) -> Agent:
    return Agent(
        world_id=world_id,
        agent_id=agent_id,
        personality=PersonalityLayer(
            soul=SoulLayer(
                name=name,
                role="court_official",
                agent_id=agent_id,
                core_traits=["careful"],
                core_values=["order"],
                hard_constraints=[],
                # Static needs live in the soul: fill in the 5 Maslow baselines so runtime steps
                # have needs to compete.
                innate_needs=build_innate_needs([], []),
            )
        ),
        decision_engine=DecisionEngine(container.llm_router),
        llm_router=container.llm_router,
        memory_system=MemorySystem(
            container.llm_router,
            container.embedding,
            container.vector_store,
            world_id=world_id,
            agent_id=agent_id,
        ),
        need_engine=NeedEngine(),
        relation_system=RelationSystem(container.agent_store, world_id=world_id, agent_id=agent_id),
        agent_store=container.agent_store,
        is_main_character=is_main_character,
    )


def _set_work_decision(container) -> None:
    """Make the mock decision LLM return a parseable WORK selection so agents act.

    The default mock returns unparseable text → decide() returns None (no
    decision → agent skipped). Tests that exercise the runtime step proper need
    agents to actually act, so they pin a valid selection (selected_index=3 → WORK).
    """
    from core.interfaces.llm import LLMScene

    container.llm_router.get(LLMScene.AGENT_DECISION_MAIN).fixed_response = (
        '{"selected_index": 3, "action_description": "批阅文书", "estimated_steps": 1}'
    )


class _SignalPressure:
    """Test double for WorldPressureEvaluator: emits one external goal for any agent the
    world put a signal in front of this step (an inbox message, a perceivable broadcast, an
    ambient trace where it stands).

    In production the cadence gate wakes an agent on ``pending_external_goals``, which the LLM
    pressure evaluator fills from those signals. With a mock LLM nothing fills it and signalled
    agents never wake; this reproduces the path deterministically through the real channel.
    """

    async def evaluate(self, *, agents, agent_inboxes, broadcasts, world_time_label, agent_spatials):
        from agent.motivation import ExternalDriveType, ExternalGoal
        from core.interfaces.action import Urgency
        from engine.broadcast import BroadcastChannel

        out: dict[str, list] = {}
        for agent_id, agent in agents.items():
            spatial = agent_spatials.get(agent_id) if agent_spatials else None
            location = agent.personality.state.current_location or (spatial.location_id if spatial else "")
            ambient = list(spatial.ambient_events) if spatial is not None else []
            if agent_inboxes.get(agent_id) or BroadcastChannel.for_location(broadcasts, location) or ambient:
                out[agent_id] = [ExternalGoal(
                    text="respond to what just happened",
                    source_id="world",
                    urgency=Urgency.NORMAL,
                    drive_type=ExternalDriveType.EVENT,
                )]
        return out


def _build_runtime(
    container,
    *,
    world_id: str,
    registry: "ActionExecutorRegistry | None" = None,
    trace_sink=None,
) -> tuple[NarrativeRuntime, EnvironmentSystem, MessageSystem]:
    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    message_system = MessageSystem(
        container.message_provider,
        world_id=world_id,
    )
    broadcast_channel = BroadcastChannel()
    # Default to a real registry, as in production: there's no missing-executor fallback, and an
    # empty registry would just skip every agent.
    if registry is None:
        registry = build_default_registry(
            container.llm_router, LiveWorldDirectory.from_agents({}, environment),
        )
    event_settings = EventSettings(check_interval=2, max_events_per_window=1)
    runtime = NarrativeRuntime(
        world_id=world_id,
        clock=GlobalClock(WorldTimeConfig(start_hour=6, seconds_per_step=60)),
        scheduler=AgentScheduler(),
        environment=environment,
        message_system=message_system,
        event_settings=event_settings,
        snapshot_provider=container.snapshot,
        agent_store=container.agent_store,
        event_bus=container.event_bus,
        broadcast_channel=broadcast_channel,
        directory=LiveWorldDirectory.from_agents({}, environment),
        executor_registry=registry,
        pressure_evaluator=_SignalPressure(),
        trace_sink=trace_sink,
        llm_router=container.llm_router,
    )
    return runtime, environment, message_system


@pytest.mark.asyncio
async def test_a_step_hands_agents_a_named_body_not_a_bare_id(container) -> None:
    """After a full step, the co-present body without cognition must reach the perception packet
    with name, age and blurb.

    Guards against a missed assembly call: identity comes in two halves (immutable from the
    directory, mutable from the live object), and dropping either doesn't error. The decision list
    just prints "#1 某人", losing the blurb the judge rules the errand on. Only a full step shows it.
    """
    from core.interfaces.llm import LLMScene
    from world.models import NpcSeed

    world_id = "world-npc"
    runtime, environment, _ = _build_runtime(container, world_id=world_id)
    # Use a location that really exists on the map: spawn_npc requires a real place (see its
    # docstring), place_agent doesn't. The difference is intentional: test scaffolding may stand a
    # person in a place that doesn't exist, but a body that walks and can be sent on errands may
    # not.
    where = environment.space.all_place_ids()[0]
    environment.place_agent(agent_id="agent-1", location_id=where)
    assert environment.spawn_npc(
        NpcSeed(name="Wang Er", gender="male", age=34, description="fast on foot; cannot read"),
        location_id=where,
    )
    npc_id = environment.all_npcs()[0].npc_id
    agents = [_build_agent(
        container, world_id=world_id, agent_id="agent-1", name="Li Shimin", is_main_character=True,
    )]
    _set_work_decision(container)

    await runtime.run_step(agents)

    # Read the decision prompt itself rather than assembling a perception packet again: that
    # would only test the assembly function, not whether the full step calls it, and a missed call
    # is exactly this kind of bug.
    prompts = [
        m.content
        for call in container.llm_router.get(LLMScene.AGENT_DECISION_MAIN).call_history
        for m in call
    ]
    menu = "\n".join(prompts)
    assert "Wang Er" in menu                 # recognizable
    assert "34" in menu                      # age
    assert "fast on foot" in menu            # what the judge rules on
    assert npc_id not in menu                # no ids in the narrative layer
    assert npc_id not in environment.spatial_for(agent_id="agent-1").visible_agent_ids


@pytest.mark.asyncio
async def test_narrative_runtime_runs_one_step_and_persists_contract_state(container) -> None:
    published = tap(container.event_bus)
    world_id = "world-1"
    runtime, environment, _ = _build_runtime(container, world_id=world_id)
    environment.place_agent(agent_id="agent-1", location_id="palace")
    environment.place_agent(agent_id="agent-2", location_id="palace")

    agents = [
        _build_agent(container, world_id=world_id, agent_id="agent-1", name="Li Shimin", is_main_character=True),
        _build_agent(container, world_id=world_id, agent_id="agent-2", name="Li Jiancheng", is_main_character=False),
    ]
    _set_work_decision(container)

    result = await runtime.run_step(agents)
    event = published()[0]
    snapshot = await container.snapshot.load(world_id, 1)
    main_state = await container.agent_store.load_agent_state(world_id, "agent-1")
    bg_state = await container.agent_store.load_agent_state(world_id, "agent-2")
    action_records = event["actions"]

    assert result.step == 1
    assert result.world_time.startswith("step=0001 ")
    assert result.delivered_message_count == 0
    assert result.scheduled_agent_ids == ["agent-1", "agent-2"]
    assert len(result.action_summaries) == 2
    assert result.event_summaries == []
    assert snapshot is not None
    # The snapshot carries the clock on BOTH channels, in the one shape the live payload uses
    # — not the label in `world_time` plus a second copy in `metadata`, which would make the
    # same name mean the machine clock live and prose in replay.
    assert snapshot.clock.startswith("step=0001 ")  # code layer: the map parses its hour
    assert snapshot.time_label and "step=" not in snapshot.time_label  # narrative layer
    assert "time_label" not in snapshot.metadata and "world_time" not in snapshot.metadata
    assert snapshot.metadata["schedule"] == {
        "step": 1,
        "batches": [
            {
                "phase": "main",
                "concurrency_rule": "shared-read snapshot; commit after main batch",
                "agent_ids": ["agent-1"],
            },
            {
                "phase": "background",
                "concurrency_rule": "shared-read snapshot; commit after background batch",
                "agent_ids": ["agent-2"],
            },
        ],
    }
    assert snapshot.metadata["messages"] == {
        "step": 1,
        "delivered": [],
        "inboxes": {"agent-1": [], "agent-2": []},
        "undelivered": [],
    }
    assert snapshot.metadata["environment"]["body_locations"] == {"agent-1": "palace", "agent-2": "palace"}
    assert snapshot.agent_summaries == action_records
    assert snapshot.event_summaries == []
    assert main_state is not None
    assert main_state.dominant_need == "social"
    assert main_state.current_location == "palace"
    assert bg_state is not None
    assert bg_state.current_location == "palace"
    assert [record["agent_id"] for record in action_records] == ["agent-1", "agent-2"]
    # Born-zero (the helper pins estimated_steps=1): the same-step sweep folds each completion
    # into its begin record, which is re-stamped "settled" — one beat, opened and closed.
    assert [record["phase"] for record in action_records] == ["settled", "settled"]
    assert action_records[0]["visible_agent_ids"] == ["agent-2"]
    assert action_records[1]["visible_agent_ids"] == ["agent-1"]
    assert action_records[0]["message_ids"] == []
    assert action_records[1]["message_ids"] == []
    assert event["type"] == "step"
    assert event["world_id"] == world_id
    assert event["step"] == 1
    assert event["schedule"] == snapshot.metadata["schedule"]
    assert event["messages"] == snapshot.metadata["messages"]
    assert event["events"] == []
    assert event["environment"] == snapshot.metadata["environment"]


@pytest.mark.asyncio
async def test_runtime_fallback_move_updates_environment_and_agent_location(container) -> None:
    world_id = "world-move"
    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    message_system = MessageSystem(
        container.message_provider,
        world_id=world_id,
    )
    directory = LiveWorldDirectory.from_agents({}, environment)
    registry = build_default_registry(container.llm_router, directory)
    broadcast_channel = BroadcastChannel()
    runtime = NarrativeRuntime(
        world_id=world_id,
        clock=GlobalClock(WorldTimeConfig(start_hour=6, seconds_per_step=60)),
        scheduler=AgentScheduler(),
        environment=environment,
        message_system=message_system,
        event_settings=EventSettings(check_interval=2, max_events_per_window=0),
        snapshot_provider=container.snapshot,
        agent_store=container.agent_store,
        event_bus=container.event_bus,
        broadcast_channel=broadcast_channel,
        directory=directory,
        executor_registry=registry,
        llm_router=container.llm_router,
    )
    environment.place_agent(agent_id="agent-1", location_id="donggong")
    agent = _build_agent(
        container,
        world_id=world_id,
        agent_id="agent-1",
        name="Li Shimin",
        is_main_character=True,
    )

    async def plan_move(*, step, spatial, inbox, broadcasts, **_kw):
        need = NeedEvaluation(
            dominant_need=NeedType.SAFETY,
            scores={NeedType.SAFETY: 1.0},
            active_needs=[NeedState(NeedType.SAFETY, "safety", 1.0)],
            short_term_goals=["reach the gate"],
            long_term_goals=[],
            prompt_context="safety",
        )
        return AgentStepPlan(
            agent_id=agent.agent_id,
            step=1,
            spatial=spatial,
            inbox=list(inbox),
            broadcasts=list(broadcasts),
            need_evaluation=need,
            action=AgentAction(
                agent_id=agent.agent_id,
                step=1,
                action_type=ActionType.MOVE,
                action_description="Move to Taiji Palace",
                # Donggong → Chongren Ward is a one-step (adjacent) move, so it starts
                # and arrives within this single run_step.
                target=ActionTarget(acts_on=[Ref.place("chongren_fang")]),
            ),
        )

    agent.plan_step = plan_move

    result = await runtime.run_step([agent])
    snapshot = await container.snapshot.load(world_id, 1)

    # action_summaries use outcome (the 3p bystander channel: who, where, what, result). Empty
    # directory → "某人" fallback. A MOVE starts → IN_TRANSIT on its step, and the same-step sweep
    # completes and arrives; complete() writes the outcome (origin, natural duration, destination).
    # Chongren Ward is Donggong's one-step neighbour on the shipped map. Not Taiji Palace:
    # connections derived from the ground show that edge crosses a palace wall, so the walk goes
    # round by Xuanwu Gate and costs two steps (worlds/connections.py).
    assert result.action_summaries == ["某人从东宫出发，赶了约27分钟的路，抵达崇仁坊。"]
    assert environment.get_body_location(agent.agent_id) == "chongren_fang"
    assert agent.personality.state.current_location == "chongren_fang"
    assert snapshot is not None
    assert snapshot.agent_states[agent.agent_id]["current_location"] == "chongren_fang"
    # The trip opened and closed in one step, yet the origin still saw him leave: the completion
    # supersedes the opening only where the completion itself is seen.
    (record,) = snapshot.actions_this_step
    assert record["phase"] == "settled"
    observed_at = {o["location_id"]: o["text"] for o in record["observations"]}
    assert "离开此地" in observed_at["donggong"]
    assert "抵达崇仁坊" in observed_at["chongren_fang"]
    carried = environment._carry_annotations
    assert any("离开此地" in a.content for a in carried["donggong"])


def test_born_zero_completion_supersedes_the_opening_only_where_it_is_seen(container) -> None:
    """Merging a same-step completion keeps the opening's observations at places the completion
    doesn't reach, and drops them where it does (a one-step WORK's "着手" is replaced, not doubled)."""
    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    directory = LiveWorldDirectory.from_agents({}, environment)
    processor = ExecutionProcessor(
        executor_registry=build_default_registry(container.llm_router, directory),
        environment=environment,
        message_system=MessageSystem(container.message_provider, world_id="world-merge"),
        directory=directory,
    )
    environment.place_agent(agent_id="agent-1", location_id="chongren_fang")
    state = ActionExecutionState.create(
        target=ActionTarget(), action_type=ActionType.MOVE, initiator_id="agent-1",
        participant_ids=["agent-1"], purpose="move", started_step=1, estimated_steps=1,
        opening_outcome="",
    )
    begin = {
        "execution_id": state.execution_id, "agent_id": "agent-1",
        "observations": [
            {"location_id": "donggong", "text": "甲离开此地。", "strength": None},
            {"location_id": "chongren_fang", "text": "甲着手动身。", "strength": None},
        ],
    }
    records = [begin]
    completion = ActionResult(
        action=AgentAction(agent_id="agent-1", step=1, action_type=ActionType.MOVE),
        expected_outcome="", outcome="甲抵达崇仁坊。", succeeded=True,
        observations=[Observed(location_id="chongren_fang", text="甲抵达崇仁坊。")],
    )

    processor.merge_completion_into_records(state, [completion], records, {})

    assert [(o["location_id"], o["text"]) for o in records[0]["observations"]] == [
        ("donggong", "甲离开此地。"),
        ("chongren_fang", "甲抵达崇仁坊。"),
    ]


@pytest.mark.asyncio
async def test_runtime_interrupts_executor_owned_action(container) -> None:
    world_id = "world-interrupt"
    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    message_system = MessageSystem(
        container.message_provider,
        world_id=world_id,
    )
    directory = LiveWorldDirectory.from_agents({}, environment)
    registry = build_default_registry(container.llm_router, directory)
    broadcast_channel = BroadcastChannel()
    runtime = NarrativeRuntime(
        world_id=world_id,
        clock=GlobalClock(WorldTimeConfig(start_hour=6, seconds_per_step=60)),
        scheduler=AgentScheduler(),
        environment=environment,
        message_system=message_system,
        event_settings=EventSettings(check_interval=2, max_events_per_window=0),
        snapshot_provider=container.snapshot,
        agent_store=container.agent_store,
        event_bus=container.event_bus,
        broadcast_channel=broadcast_channel,
        directory=directory,
        executor_registry=registry,
        llm_router=container.llm_router,
    )
    agent = _build_agent(
        container,
        world_id=world_id,
        agent_id="agent-1",
        name="Li Shimin",
        is_main_character=True,
    )
    state = ActionExecutionState.create(
        target=ActionTarget(),
        action_type=ActionType.REST,
        initiator_id=agent.agent_id,
        participant_ids=[agent.agent_id],
        purpose="rest",
        started_step=1,
        opening_outcome="开始",
        estimated_steps=8,
    )
    registry.add_active(state)
    agent.personality.begin_action(
        step=1,
        description="rest",
        activity_status=activity_status_for(ActionType.REST),
        estimated_steps=8,
    )

    # Drive the real production apply path directly (the decide step is covered by
    # test_agent_interrupt / test_runtime_interrupt_eval). Asserts the executor
    # interrupt tears down the ongoing action and writes back a failed result.
    await runtime._interrupts._apply_interrupt(
        agent_id=agent.agent_id,
        # Only the thought crosses into the executor — the triggering signals stopped at the
        # decision (see ActionExecutor.interrupt).
        thought="我得去看看",
        step=2,
        agents={agent.agent_id: agent},
    )

    assert not registry.is_agent_active(agent.agent_id)
    assert agent.personality.state.last_action_succeeded is False


def test_scheduler_excludes_step_start_in_progress_regardless_of_live_status(container) -> None:
    """The step-start IN_PROGRESS snapshot is the single source of truth for the
    decision gate. An agent in it is excluded even though its live status already
    flipped to COMPLETED this step (natural completion / interrupt teardown both
    land here); an agent absent from it is scheduled normally."""
    scheduler = AgentScheduler()
    a = _build_agent(container, world_id="w", agent_id="a", name="A", is_main_character=True)
    b = _build_agent(container, world_id="w", agent_id="b", name="B", is_main_character=True)
    # `a` was IN_PROGRESS at step start but completed during the step → live COMPLETED.
    a.personality.update_action_result(step=1, action="x", result="done", succeeded=True)
    assert a.personality.state.action_status == ActionStatus.COMPLETED

    plan = scheduler.plan([a, b], step=2, in_progress_at_step_start={"a"})

    scheduled = [aid for batch in plan.batches for aid in batch.agent_ids()]
    assert scheduled == ["b"]


@pytest.mark.asyncio
async def test_multi_step_action_defers_redecision_to_settle_step(container) -> None:
    """A 2-step action occupies step 1 (start) + step 2 (final tick). The agent must
    NOT re-decide on its completion step (step 2); its next decision is deferred to
    step 3 so it first perceives the feedback of the just-finished action."""
    from core.interfaces.llm import LLMScene

    world_id = "world-defer"
    registry = build_default_registry(
        container.llm_router, LiveWorldDirectory.from_agents({}, EnvironmentSystem(TiledWorldConfig(template="changan_iso")))
    )
    runtime, environment, _ = _build_runtime(container, world_id=world_id, registry=registry)
    environment.place_agent(agent_id="agent-1", location_id="palace")
    agent = _build_agent(
        container, world_id=world_id, agent_id="agent-1", name="Li Shimin", is_main_character=True
    )
    # Pin a 2-step WORK selection (estimated_steps=2).
    container.llm_router.get(LLMScene.AGENT_DECISION_MAIN).fixed_response = (
        '{"selected_index": 3, "action_description": "批阅文书", "estimated_steps": 2}'
    )

    s1 = await runtime.run_step([agent])
    assert "agent-1" in s1.scheduled_agent_ids
    st1 = await container.agent_store.load_agent_state(world_id, "agent-1")
    assert st1.action_status == ActionStatus.IN_PROGRESS.value  # multi-step started

    s2 = await runtime.run_step([agent])
    assert "agent-1" not in s2.scheduled_agent_ids  # final tick completes, but no re-decision
    st2 = await container.agent_store.load_agent_state(world_id, "agent-1")
    assert st2.action_status != ActionStatus.IN_PROGRESS.value  # action actually completed

    s3 = await runtime.run_step([agent])
    assert "agent-1" in s3.scheduled_agent_ids  # settle step passed → back in the loop


@pytest.mark.asyncio
async def test_interrupted_agent_re_decides_in_the_same_step(container) -> None:
    """An interrupted agent re-decides the same step, unlike natural completion (test above).

    Natural completion means his own action used this step; an interrupt means something more
    urgent stopped him. Freezing him a step would cancel that urgency and lose this step's
    pending_external_goals and perception appraisal, neither recoverable. It's safe: the interrupt's
    feedback has already landed synchronously in _apply_interrupt.
    """
    from core.interfaces.llm import LLMScene

    world_id = "world-interrupt-redecide"
    runtime, environment, message_system = _build_runtime(container, world_id=world_id)
    environment.place_agent(agent_id="agent-1", location_id="palace")
    environment.place_agent(agent_id="agent-2", location_id="palace")
    agent = _build_agent(container, world_id=world_id, agent_id="agent-1",
                         name="Li Shimin", is_main_character=True)
    other = _build_agent(container, world_id=world_id, agent_id="agent-2",
                         name="Li Jiancheng", is_main_character=False)

    container.llm_router.get(LLMScene.AGENT_DECISION_MAIN).fixed_response = (
        '{"selected_index": 3, "action_description": "批阅文书", "estimated_steps": 6}'
    )
    s1 = await runtime.run_step([agent, other])
    assert "agent-1" in s1.scheduled_agent_ids
    assert agent.personality.state.action_status == ActionStatus.IN_PROGRESS

    # An urgent message delivered this step: interrupt Path 1. The main character's interrupt
    # weighing uses the LLM, so pin it to "interrupt" here rather than depend on whatever a mock
    # happens to return.
    await message_system.publish(Message(
        id="urgent-1", world_id=world_id, sender_id="agent-2",
        content="宫门有变", recipients=["agent-1"], location_scope=None,
        created_step=1, deliver_step=2, urgency=Urgency.CRITICAL,
    ))

    weighed: list[str] = []

    async def _always_stop(**_kwargs):
        weighed.append("stop")
        return True, "我必须立刻动身"

    agent.evaluate_interrupt = _always_stop  # type: ignore[method-assign]
    torn = runtime._executor_registry.get_active_for_agent("agent-1").execution_id

    s2 = await runtime.run_step([agent, other])

    assert weighed                                   # interrupt weighing did happen
    still = runtime._executor_registry.get_active_for_agent("agent-1")
    assert still is None or still.execution_id != torn   # the original execution was torn down
    # ...and he's back in the decision loop the same step.
    assert "agent-1" in s2.scheduled_agent_ids
    assert agent.personality.state.last_decision_step == 2


@pytest.mark.asyncio
async def test_a_multi_step_action_is_marked_at_its_opening_only(container) -> None:
    """A multi-step span is marked only on the starting step, with the remaining count taken
    from the execution itself, not the decision's self-reported estimated_steps (an estimate,
    usually wrong). Later steps render normally, whether interrupted or completed; readers can tell
    it's the same action continuing, and audit doesn't need to chain across steps."""
    from core.interfaces.llm import LLMScene
    from providers.trace.in_memory import InMemoryTraceSink

    world_id = "world-span-mark"
    sink = InMemoryTraceSink()
    container.llm_router._trace_sink = sink  # noqa: SLF001
    runtime, environment, _ms = _build_runtime(container, world_id=world_id, trace_sink=sink)
    environment.place_agent(agent_id="agent-1", location_id="palace")
    agent = _build_agent(container, world_id=world_id, agent_id="agent-1",
                         name="Li Shimin", is_main_character=True)
    # The decision claims 6 steps; the execution sets the remainder by its own rules, and the mark
    # must use the latter.
    container.llm_router.get(LLMScene.AGENT_DECISION_MAIN).fixed_response = (
        '{"selected_index": 3, "action_description": "批阅文书", "estimated_steps": 6}'
    )
    await runtime.run_step([agent])

    dec = next(c for c in sink.llm_calls if c.stage == Stage.DECISION.value and c.step == 1)
    exec_state = runtime._executor_registry.get_active_for_agent("agent-1")  # noqa: SLF001
    assert dec.extra["verdict"] == "executed"
    assert dec.extra["spans_steps"] == exec_state.remaining_steps + 1

    await runtime.run_step([agent])   # the continuation step: no new decision, so no new span mark
    assert not [c for c in sink.llm_calls
                if c.stage == Stage.DECISION.value and c.step == 2]


@pytest.mark.asyncio
async def test_narrative_runtime_delivers_targeted_messages_without_llm_event_approval(container) -> None:
    published = tap(container.event_bus)
    world_id = "world-1"
    runtime, environment, message_system = _build_runtime(container, world_id=world_id)
    environment.place_agent(agent_id="agent-1", location_id="palace")
    environment.place_agent(agent_id="agent-2", location_id="palace")

    agents = [
        _build_agent(container, world_id=world_id, agent_id="agent-1", name="Li Shimin", is_main_character=True),
        _build_agent(container, world_id=world_id, agent_id="agent-2", name="Li Jiancheng", is_main_character=False),
    ]

    _set_work_decision(container)
    await message_system.publish(
        Message(
            id="direct-1",
            world_id=world_id,
            sender_id="agent-2",
            content="Meet me at dawn",
            recipients=["agent-1"],
            location_scope=None,
            created_step=1,
            deliver_step=2,
        )
    )

    first_step = await runtime.run_step(agents)
    assert first_step.event_summaries == []
    assert published()[0]["step"] == 1

    second_step = await runtime.run_step(agents)
    second_event = published()[0]
    second_snapshot = await container.snapshot.load(world_id, 2)
    main_state = await container.agent_store.load_agent_state(world_id, "agent-1")
    bg_state = await container.agent_store.load_agent_state(world_id, "agent-2")

    assert second_step.step == 2
    assert second_step.delivered_message_count == 1
    assert second_step.scheduled_agent_ids == ["agent-1", "agent-2"]
    assert len(second_step.action_summaries) == 2
    assert second_step.event_summaries == []
    assert second_snapshot is not None
    messages_metadata = second_snapshot.metadata["messages"]
    actions_by_id = {record["agent_id"]: record for record in second_event["actions"]}
    assert second_snapshot.event_summaries == []
    assert messages_metadata["step"] == 2
    assert len(messages_metadata["delivered"]) == 1
    assert messages_metadata["undelivered"] == []
    assert "direct-1" in messages_metadata["inboxes"]["agent-1"]
    assert "direct-1" not in messages_metadata["inboxes"]["agent-2"]
    assert len(messages_metadata["inboxes"]["agent-1"]) == 1
    assert len(messages_metadata["inboxes"]["agent-2"]) == 0
    # Lossless transport: delivered content is byte-identical to what was sent
    delivered_dict = messages_metadata["delivered"][0]
    assert delivered_dict["id"] == "direct-1"
    assert delivered_dict["content"] == "Meet me at dawn"
    assert delivered_dict["recipients"] == ["agent-1"]
    assert any(
        memory.raw_content.startswith("[消息]")
        for memory in agents[0].memory_system._entries.values()
    )
    # A received letter counts as contact but moves no value: only LLM-judged interactions do.
    relation = await agents[0].relation_system.load_existing("agent-2")
    assert relation is not None and relation.interaction_count == 1
    assert relation.trust_objective == 0.5
    assert main_state is not None
    assert main_state.current_location == "palace"
    assert bg_state is not None
    assert bg_state.current_location == "palace"
    assert "direct-1" in actions_by_id["agent-1"]["message_ids"]
    assert "direct-1" not in actions_by_id["agent-2"]["message_ids"]
    assert len(actions_by_id["agent-1"]["message_ids"]) == 1
    assert actions_by_id["agent-2"]["message_ids"] == []
    assert second_event["type"] == "step"
    assert second_event["world_id"] == world_id
    assert second_event["step"] == 2
    assert second_event["messages"] == messages_metadata
    assert second_event["events"] == second_snapshot.event_summaries
    assert second_event["schedule"]["batches"][0]["agent_ids"] == ["agent-1"]
    assert second_event["schedule"]["batches"][1]["agent_ids"] == ["agent-2"]


# ---------------------------------------------------------------------------
# TALK cannot double-book an agent already IN a multi-step execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_talk_to_active_agent_is_rejected_without_double_booking(container) -> None:
    """When agent-b is already in an active multi-step execution (e.g. REST),
    agent-a's TALK targeting agent-b must be rejected with succeeded=False,
    and the registry must still contain exactly one state for agent-b."""
    world_id = "world-ar1"
    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    environment.place_agent(agent_id="agent-a", location_id="palace")
    environment.place_agent(agent_id="agent-b", location_id="palace")

    directory = LiveWorldDirectory.from_agents({}, environment)
    registry = build_default_registry(container.llm_router, directory)

    message_system = MessageSystem(
        container.message_provider, world_id=world_id
    )
    broadcast_channel = BroadcastChannel()
    runtime = NarrativeRuntime(
        world_id=world_id,
        clock=GlobalClock(WorldTimeConfig(start_hour=6, seconds_per_step=60)),
        scheduler=AgentScheduler(),
        environment=environment,
        message_system=message_system,
        event_settings=EventSettings(check_interval=99, max_events_per_window=0),
        snapshot_provider=container.snapshot,
        agent_store=container.agent_store,
        event_bus=container.event_bus,
        broadcast_channel=broadcast_channel,
        directory=directory,
        executor_registry=registry,
        llm_router=container.llm_router,
    )

    agent_a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="A", is_main_character=True)
    agent_b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="B", is_main_character=False)

    # Pre-register agent-b as IN_PROGRESS on a multi-step REST.
    rest_state = ActionExecutionState.create(
        target=ActionTarget(),
        action_type=ActionType.REST,
        initiator_id="agent-b",
        participant_ids=["agent-b"],
        purpose="resting",
        started_step=1,
        opening_outcome="开始",
        estimated_steps=5,
    )
    registry.add_active(rest_state)
    agent_b.personality.begin_action(
        step=1, description="resting",
        activity_status=activity_status_for(ActionType.REST),
        estimated_steps=5,
    )

    # Force agent-a to decide TALK targeting agent-b via a fixed LLM response.
    import json
    from providers.llm.mock import SequentialMockLLM
    from core.interfaces.llm import LLMRouter, LLMScene
    # person_indices points at agent-b, present (the only other person on the list). TALK must
    # actually bind a person for "the target is busy" to arise: an empty list is rejected while
    # parsing decide (talk_without_target), producing no action.
    talk_response = json.dumps({
        "selected_index": 0,  # TALK
        "person_indices": [1],
        "action_description": "找B说话。",
        "inner_monologue": "需要和B交流。",
        "estimated_steps": 1,
    })
    fixed_llm = SequentialMockLLM([talk_response] * 20)
    agent_a.decision_engine._llm_router = LLMRouter({scene: fixed_llm for scene in LLMScene})  # noqa: SLF001

    await runtime.run_step([agent_a, agent_b])

    # agent-a's TALK must have failed (target was busy).
    actions = await container.snapshot.load_latest(world_id)
    assert actions is not None
    agent_a_record = next(
        (r for r in actions.actions_this_step if r["agent_id"] == "agent-a"), None
    )
    assert agent_a_record is not None
    assert agent_a_record["succeeded"] is False

    # Registry must still hold exactly one active state (agent-b's REST, unchanged).
    active_states = registry.all_active()
    assert len(active_states) == 1
    assert active_states[0].action_type == ActionType.REST
    assert active_states[0].initiator_id == "agent-b"


# ---------------------------------------------------------------------------
# World-level concurrent arbitration (ExecutionArbiter.arbitrate)
# ---------------------------------------------------------------------------

def _arb_runtime(container, world_id, agents_dict, environment, *, trace_sink=None):
    directory = LiveWorldDirectory.from_agents(agents_dict, environment)
    registry = build_default_registry(container.llm_router, directory, seconds_per_step=3600)
    ms = MessageSystem(container.message_provider, world_id=world_id)
    bc = BroadcastChannel()
    return NarrativeRuntime(
        world_id=world_id,
        clock=GlobalClock(WorldTimeConfig(start_hour=6, seconds_per_step=3600)),
        scheduler=AgentScheduler(),
        environment=environment,
        message_system=ms,
        event_settings=EventSettings(check_interval=99, max_events_per_window=0),
        snapshot_provider=container.snapshot,
        agent_store=container.agent_store,
        event_bus=container.event_bus,
        broadcast_channel=bc,
        directory=directory,
        executor_registry=registry,
        trace_sink=trace_sink,
        llm_router=container.llm_router,
    )


def _plan_for(agent, action, environment, *, step=1, decision_status=DecisionStatus.ACTED,
              external_goals=(), foiled_misses=0):
    spatial = environment.spatial_for(agent_id=agent.agent_id, step=step)
    need = NeedEvaluation(
        dominant_need=NeedType.SAFETY,
        scores={NeedType.SAFETY: 1.0},
        active_needs=[NeedState(NeedType.SAFETY, "safety", 1.0)],
        short_term_goals=[], long_term_goals=[],
        prompt_context="",
    )
    return AgentStepPlan(
        agent_id=agent.agent_id, step=step, spatial=spatial,
        inbox=[], broadcasts=[], need_evaluation=need, action=action,
        decision_status=decision_status,
        consumed_external_goals=list(external_goals),
        foiled_misses=foiled_misses,
    )


def _trace_call(world_id: str, *, agent_id: str, stage: Stage, step: int = 1) -> LLMCallTrace:
    """A trace for one call at some stage; this test only looks at adopted / reject_reason /
    extra."""
    return LLMCallTrace(
        world_id=world_id, stage=stage.value, scene=f"{stage.value}_scene",
        prompt_messages=[{"role": "user", "content": "p"}], response_content="r",
        temperature=0.7, max_tokens=100, input_tokens=10, output_tokens=4,
        model="mock", latency_ms=1.0, timestamp="t", agent_id=agent_id, step=step,
    )


def _pressure(urgency: Urgency, text: str = "即刻动身面圣，否则以抗旨论处") -> ExternalGoal:
    """An external pressure item: preemption only reads urgency; other fields just need valid
    values."""
    return ExternalGoal(
        text=text, source_id="agent-sovereign", urgency=urgency,
        drive_type=ExternalDriveType.AUTHORITY, related_need=NeedType.SAFETY,
    )


@pytest.mark.asyncio
async def test_arbitrate_enrollment_pulls_target_into_passive_join(container) -> None:
    """Multi-step joining: the initiator starts and the collaborator passively joins (names not ids,
    natural duration not steps)."""
    world_id = "world-arb-enroll"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                       action_description="商议对策", target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]), estimated_steps=2)
    work = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.WORK,
                       action_description="批阅文书", estimated_steps=1)
    planned = [("main", a, _plan_for(a, talk, env)), ("bg", b, _plan_for(b, work, env))]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    a_aa, b_aa = arb["agent-a"], arb["agent-b"]
    # Initiator: starts AA
    assert a_aa.ongoing_execution_id is not None and a_aa.is_passive_join is False
    # Start step surfaces the executor-authored opening narrative (names the partner, no step leak).
    assert "李建成" in a_aa.action_result.outcome
    assert "步" not in a_aa.action_result.outcome and "steps" not in a_aa.action_result.outcome
    # Collaborator: joins passively, its own WORK is void; the inviter is named, not given by id
    assert b_aa.is_passive_join is True
    assert "李世民" in b_aa.action_result.outcome
    assert "agent-a" not in b_aa.action_result.outcome


@pytest.mark.asyncio
async def test_conscripted_agent_defers_decided_intent_into_foiled_buffer(container) -> None:
    """When B is pulled into someone's TALK and its decided action is discarded, that intent goes
    into B's foiled buffer and survives to the next step's decision (the perception signals behind
    it last only one step and can't be rebuilt).

    Not the short-term goal queue: the intent didn't happen, same as not_executed, and the queue
    would store the raw action_description, a concrete action the short-term goal prompt forbids,
    replaying it as "unfinished business" again and again."""
    world_id = "world-arb-defer"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                       action_description="商议对策", target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]), estimated_steps=2)
    work = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.WORK,
                       action_description="独自前往城防查探敌情", estimated_steps=1)
    planned = [("main", a, _plan_for(a, talk, env)), ("bg", b, _plan_for(b, work, env))]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    assert arb["agent-b"].is_passive_join is True            # B's WORK is discarded
    records = await rt._commit_execution(planned, arb, agents)  # noqa: SLF001

    # B's discarded intent is in the foiled buffer, available for the next decision
    foiled = b.memory_system.recent_foiled_attempts(1)
    assert any("独自前往城防查探敌情" in t for t in foiled)
    # and not in the short-term goal queue: a concrete action isn't a short-term goal (see
    # docstring)
    assert b.personality.state.short_term_goal_entities == []
    # Nor in persistent memory (being foiled is a non-event; written there it would be recalled via
    # embedding over and over)
    assert b.memory_system._entries == {}       # noqa: SLF001

    # The discarded WORK must never leak into B's record of what it actually did this step: the
    # passive-join record is the TALK it joined, not its own decision (plan.action is only used for
    # defer, never in the executed record).
    b_rec = next(r for r in records if r["agent_id"] == "agent-b")
    assert b_rec["action_type"] == ActionType.TALK
    assert "独自前往城防查探敌情" not in b_rec["summary"]
    assert "李世民" in b_rec["outcome"]


@pytest.mark.asyncio
async def test_conscription_leaves_its_two_marks_on_the_step_traces(container) -> None:
    """Being conscripted leaves two different marks in audit: motivation is void (marked unadopted),
    the decision isn't (only tagged as conscripted).

    The motivation output was never enqueued; a judge reading it as fact would rule "proposed a goal
    but didn't pursue it". The decision survives to the next step and shouldn't be hidden; the judge
    only needs to waive the decision↔action consistency check. The initiator gets neither mark.
    """
    world_id = "world-arb-motivation-mark"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    sink = InMemoryTraceSink()
    for aid in ("agent-a", "agent-b"):
        for stage in (Stage.MOTIVATION, Stage.DECISION):
            sink.record_llm_call(_trace_call(world_id, agent_id=aid, stage=stage))
    rt = _arb_runtime(container, world_id, agents, env, trace_sink=sink)

    talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                       action_description="商议对策", target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]), estimated_steps=2)
    work = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.WORK,
                       action_description="批阅文书", estimated_steps=1)
    planned = [("main", a, _plan_for(a, talk, env)), ("bg", b, _plan_for(b, work, env))]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    assert arb["agent-b"].is_passive_join is True
    await rt._commit_execution(planned, arb, agents)  # noqa: SLF001

    marked = {c.agent_id: (c.adopted, c.reject_reason) for c in sink.llm_calls
              if c.stage == Stage.MOTIVATION.value}
    assert marked["agent-b"] == (False, "conscripted")
    assert marked["agent-a"] == (None, ""), "发起方没被征召,他那次不该被判未采纳"
    # The decision isn't void (it survives to the next step via defer_decided_intent), but what the
    # world did with it must be stamped. One verdict field answers it, so audit doesn't have to
    # infer the same thing from side signals (is there an execution_id, an action call, is talk_role
    # addressee); each of those gets some edge case wrong.
    decisions = {c.agent_id: c.extra for c in sink.llm_calls if c.stage == Stage.DECISION.value}
    assert decisions["agent-b"] == {"verdict": "conscripted"}
    # Initiator: this action spans two steps, with the remainder taken from the execution (not
    # the decision's estimated_steps).
    assert decisions["agent-a"] == {"verdict": "executed", "spans_steps": 2}


@pytest.mark.asyncio
async def test_executor_start_crash_is_marked_a_failure_not_a_verdict(container) -> None:
    """start() raising means the machinery broke, not the world's ruling on this intent: stamp
    start_failed.

    Stamping a verdict (especially "he didn't act") would make audit read a crash as a choice.
    """
    world_id = "world-arb-start-crash"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    agents = {"agent-a": a}
    sink = InMemoryTraceSink()
    sink.record_llm_call(_trace_call(world_id, agent_id="agent-a", stage=Stage.DECISION))
    rt = _arb_runtime(container, world_id, agents, env, trace_sink=sink)

    work = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.WORK,
                       action_description="批阅文书", estimated_steps=1)
    planned = [("main", a, _plan_for(a, work, env))]

    async def boom(*args, **kwargs):
        raise RuntimeError("executor exploded")
    rt._executor_registry.get_executor(ActionType.WORK).start = boom  # noqa: SLF001

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    assert "agent-a" not in arb, "崩掉的行动不该留下裁决条目"
    await rt._commit_execution(planned, arb, agents)  # noqa: SLF001

    decisions = {c.agent_id: c.extra for c in sink.llm_calls if c.stage == Stage.DECISION.value}
    assert decisions["agent-a"] == {"verdict": "start_failed"}


@pytest.mark.asyncio
async def test_carry_skips_adjudication_failed_null_step(container) -> None:
    """A failed adjudication is a null step (infrastructure failure, nothing happened in the world):
    its outcome is never carried to bystanders, or they'd perceive an event that doesn't exist
    (viewpoint contract: null steps produce no ambient)."""
    from engine.clock import WorldTime
    world_id = "world-carry-null"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")  # actor
    env.place_agent(agent_id="agent-b", location_id="palace")  # bystander
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    rt = _arb_runtime(container, world_id, {"agent-a": a, "agent-b": b}, env)

    # The bystander perception channel reads the record's observation (public WORK → observation ==
    # outcome).
    rt._carry_step_observations(  # noqa: SLF001
        agent_records=[
            {"agent_id": "agent-a", "action_type": ActionType.WORK, "location_id": "palace",
             "outcome": "李世民完成了批阅文书。", "observations": [{"location_id": "palace", "text": "李世民完成了批阅文书。"}],
             "adjudication_failed": False},
            {"agent_id": "agent-a", "action_type": ActionType.WORK, "location_id": "palace",
             "outcome": "「密谋」一事，一时未能确知是否做成。",
             "observations": [{"location_id": "palace", "text": "「密谋」一事，一时未能确知是否做成。"}], "adjudication_failed": True},
        ],
        tick_records=[],
    )
    env.begin_step(step=2, world_time=WorldTime(step=2, elapsed_seconds=7200))
    ambient = "".join(ev.content for ev in env.spatial_for(agent_id="agent-b").ambient_events)
    assert "批阅文书" in ambient        # a normal observation is carried
    assert "未能确知" not in ambient     # the null step is skipped, not carried


@pytest.mark.asyncio
async def test_the_one_acted_upon_does_not_read_a_third_person_copy_of_it(container) -> None:
    """The person struck already has a first-person memory from the effect; an environment
    observation "someone restrained Chang He" on top would make him remember the same thing twice.
    Bystanders still see it, and since the memory concerns the person struck, he's in related."""
    from engine.clock import WorldTime
    world_id = "world-carry-acted-upon"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    for aid in ("agent-a", "agent-b", "agent-c"):
        env.place_agent(agent_id=aid, location_id="palace")
    agents = {
        aid: _build_agent(container, world_id=world_id, agent_id=aid, name=name, is_main_character=True)
        for aid, name in (("agent-a", "尉迟敬德"), ("agent-b", "常何"), ("agent-c", "李世民"))
    }
    rt = _arb_runtime(container, world_id, agents, env)

    rt._carry_step_observations(  # noqa: SLF001
        agent_records=[{
            "agent_id": "agent-a", "action_type": ActionType.PHYSICAL, "location_id": "palace",
            "participant_ids": ["agent-a"], "acted_upon": ["agent-b"],
            "observations": [{"location_id": "palace", "text": "尉迟敬德挥槊制服了常何。", "strength": 0.7}],
            "adjudication_failed": False,
        }],
        tick_records=[],
    )
    env.begin_step(step=2, world_time=WorldTime(step=2, elapsed_seconds=7200))

    victim = [ev.content for ev in env.spatial_for(agent_id="agent-b").ambient_events]
    [seen] = [ev for ev in env.spatial_for(agent_id="agent-c").ambient_events if "制服" in ev.content]
    assert not any("制服" in c for c in victim)
    assert "agent-b" in seen.agent_actor_ids


def test_acted_upon_lists_landed_effects_but_not_listeners(container) -> None:
    """acted_upon comes from landed effects, same source as overheard_by; listeners belong in
    overheard_by and aren't counted as acted upon."""
    from core.interfaces.action import ActionResult, TargetAgentEffect

    world_id = "world-acted-upon-record"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    agents = {"agent-a": _build_agent(container, world_id=world_id, agent_id="agent-a", name="甲",
                                      is_main_character=True)}
    rt = _arb_runtime(container, world_id, agents, env)
    es = ActionExecutionState.create(
        target=ActionTarget(), action_type=ActionType.PHYSICAL, initiator_id="agent-a",
        participant_ids=["agent-a"], purpose="x", started_step=1, estimated_steps=1, opening_outcome="",
    )
    result = ActionResult(
        action=AgentAction(agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL, action_description="x"),
        expected_outcome="", outcome="x", succeeded=True,
        target_effects=[TargetAgentEffect(agent_id="agent-b", factual_memory="被打"),
                        TargetAgentEffect(agent_id="agent-c", factual_memory="听见", overheard=True)],
    )

    record = rt._processor.completion_record(es, result, agents)  # noqa: SLF001

    assert record["acted_upon"] == ["agent-b"]
    assert record["overheard_by"] == ["agent-c"]


@pytest.mark.asyncio
async def test_collaborative_action_ambient_carried_once_for_bystander(container) -> None:
    """A joint action (TALK) produces one record per participant, but they share the same
    third-person observation and member set: bystanders should perceive it once, not N times (N
    writes into bystander memory, each embedded)."""
    from engine.clock import WorldTime
    world_id = "world-carry-dedup"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")  # participant
    env.place_agent(agent_id="agent-b", location_id="palace")  # participant
    env.place_agent(agent_id="agent-c", location_id="palace")  # bystander (not a participant)
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    rt = _arb_runtime(container, world_id, {"agent-a": a, "agent-b": b}, env)

    transcript = "在大殿，李世民与李建成的交谈：\n李世民：兄长安好。"  # outcome: full transcript (authoritative god view)
    shared = "在大殿，李世民与李建成在交谈。"                         # observation: bystander-level (no content)
    rt._carry_step_observations(  # noqa: SLF001
        agent_records=[
            {"agent_id": "agent-a", "action_type": ActionType.TALK, "location_id": "palace",
             "outcome": transcript, "observations": [{"location_id": "palace", "text": shared}],
             "adjudication_failed": False,
             "participant_ids": ["agent-a", "agent-b"]},
            {"agent_id": "agent-b", "action_type": ActionType.TALK, "location_id": "palace",
             "outcome": transcript, "observations": [{"location_id": "palace", "text": shared}],
             "adjudication_failed": False,
             "participant_ids": ["agent-a", "agent-b"]},
        ],
        tick_records=[],
    )
    env.begin_step(step=2, world_time=WorldTime(step=2, elapsed_seconds=7200))
    ambient_all = "".join(ev.content for ev in env.spatial_for(agent_id="agent-c").ambient_events)
    # Information asymmetry: bystanders perceive only the reduced observation and never the
    # transcript.
    assert "兄长安好" not in ambient_all, "对话转录泄漏给了旁观者"
    events = [
        ev for ev in env.spatial_for(agent_id="agent-c").ambient_events
        if "李世民与李建成在交谈" in ev.content
    ]
    assert len(events) == 1, f"协作行动应只 carry 一次,旁观者实际感知 {len(events)} 次"


@pytest.mark.asyncio
async def test_covert_undetected_carries_nothing_detected_exposes_publicly(container) -> None:
    """COVERT has no private perception channel: the actor knows what it did from factual_memory,
    not perception.
    - not exposed → bystanders get no ambient (observation is empty, nothing carried);
    - exposed → bystanders get a public "[秘密行动暴露] …" ambient; the exposed actor filters itself out
      via actor_ids."""
    from engine.clock import WorldTime
    world_id = "world-covert-carry"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="vault")   # actor
    env.place_agent(agent_id="agent-b", location_id="vault")   # bystander
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    rt = _arb_runtime(container, world_id, {"agent-a": a, "agent-b": b}, env)

    # Not exposed: observation is empty (the covert executor's `if detected else ""`).
    rt._carry_step_observations(  # noqa: SLF001
        agent_records=[
            {"agent_id": "agent-a", "action_type": ActionType.COVERT, "location_id": "vault",
             "outcome": "在vault，李世民窃取了密信。", "observations": [],
             "adjudication_failed": False, "detected": False},
        ],
        tick_records=[],
    )
    env.begin_step(step=2, world_time=WorldTime(step=2, elapsed_seconds=7200))
    bystander_ambient = "".join(ev.content for ev in env.spatial_for(agent_id="agent-b").ambient_events)
    assert bystander_ambient == "", "未暴露 covert 不应给旁观者任何感知"

    # Exposed: the prefix and the strong-social-signal strength are authorized by the covert
    # executor in observations (see CovertExecutor._exposure_observations); the runtime only
    # delivers by location. The runtime and interrupt_coordinator must not each build their own
    # copy: that would duplicate the logic and bypass unified delivery.
    rt._carry_step_observations(  # noqa: SLF001
        agent_records=[
            {"agent_id": "agent-a", "action_type": ActionType.COVERT, "location_id": "vault",
             "outcome": "在vault，李世民窃取了密信，被撞见。",
             "observations": [{
                 "location_id": "vault",
                 "text": "[秘密行动暴露] 在vault，李世民窃取了密信，被撞见。",
                 "strength": SALIENT_AMBIENT_STRENGTH,
             }],
             "adjudication_failed": False, "detected": True},
        ],
        tick_records=[],
    )
    env.begin_step(step=3, world_time=WorldTime(step=3, elapsed_seconds=10800))
    assert any("[秘密行动暴露]" in ev.content for ev in env.spatial_for(agent_id="agent-b").ambient_events)
    assert env.spatial_for(agent_id="agent-a").ambient_events == [], "暴露者不应读到自己的暴露 ambient"


@pytest.mark.asyncio
async def test_arbitrate_acting_actor_cannot_be_enrolled(container) -> None:
    """A body already used this step (B doing its own WORK) can't also be conscripted into A's TALK
    (no double consumption). A vetoed TALK isn't an immediate failure result but an actor-only
    failed execution (born at zero): the failure is stored in extra["completed_result"] and lands as
    feedback in the same-step sweep. The rejection text uses names, not ids, and no steps."""
    world_id = "world-arb-double"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                       action_description="商议对策", target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]), estimated_steps=2)
    work = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.WORK,
                       action_description="批阅文书", estimated_steps=1)
    # B is arbitrated before A (acts first this step, its body is used)
    planned = [("bg", b, _plan_for(b, work, env)), ("main", a, _plan_for(a, talk, env))]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    b_aa = arb["agent-b"]
    # B did its own WORK and wasn't pulled into the TALK
    assert b_aa.is_passive_join is False
    assert b_aa.action_result.action.action_type == ActionType.WORK
    assert b_aa.action_result.action.action_description == "批阅文书"
    # A's TALK is vetoed → an actor-only failed execution (B isn't a participant, wasn't
    # conscripted).
    a_exec = rt._executor_registry.get_active_for_agent("agent-a")  # noqa: SLF001
    assert a_exec is not None
    assert a_exec.participant_ids == ["agent-a"]              # actor-only, B not conscripted
    rejection = a_exec.extra["completed_result"]
    assert rejection.succeeded is False
    assert "李建成" in rejection.outcome                       # name, not id
    assert "agent-b" not in rejection.outcome
    assert "步" not in rejection.outcome
    # B's WORK execution is registered separately (WORK type); B isn't in any TALK (no double
    # consumption).
    b_exec = rt._executor_registry.get_active_for_agent("agent-b")  # noqa: SLF001
    assert b_exec is not None and b_exec.action_type == ActionType.WORK
    assert "agent-a" not in b_exec.participant_ids


@pytest.mark.asyncio
async def test_enrolling_action_is_arbitrated_before_independent_ones_in_same_phase(container) -> None:
    """Within a phase, conscripting actions (TALK) are arbitrated before independent ones (see
    _initiative_order for why).

    B comes earlier in input order, but A's TALK needs B's body, so A is arbitrated first and gets
    it; B then joins passively and gives up its own COVERT. In the other order A would waste the
    step.
    """
    world_id = "world-arb-enroll-first"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="常何", is_main_character=True)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    covert = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.COVERT,
                         action_description="暗中巡视禁军队列", estimated_steps=1)
    talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                       action_description="低声询问守军是否已听令",
                       target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]), estimated_steps=2)
    # Same phase; B is earlier in input order (higher initiative), but A's action needs B's body.
    planned = [("main", b, _plan_for(b, covert, env)), ("main", a, _plan_for(a, talk, env))]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    # B is conscripted as a passive participant instead of doing its own COVERT
    assert arb["agent-b"].is_passive_join is True
    a_exec = rt._executor_registry.get_active_for_agent("agent-a")  # noqa: SLF001
    assert a_exec is not None and a_exec.action_type == ActionType.TALK
    assert "agent-b" in a_exec.participant_ids
    # No "the other side is busy" failed execution is left behind
    assert "未能如愿" not in arb["agent-a"].action_result.outcome


@pytest.mark.asyncio
async def test_enrolling_reorder_never_crosses_phases(container) -> None:
    """Reordering happens only within a phase: a background character's TALK must not jump ahead of
    a main character's independent action.

    Phase order follows first appearance, not alphabetical order; sorting by name would put
    "background" before "main" and invert initiative.
    """
    world_id = "world-arb-phase-guard"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="张婕妤", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    work = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.WORK,
                       action_description="批阅文书", estimated_steps=1)
    talk = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.TALK,
                       action_description="上前攀谈", target=ActionTarget(acts_on=[Ref.agent("agent-a")], claims=[Ref.agent("agent-a")]), estimated_steps=2)
    planned = [("main", a, _plan_for(a, work, env)), ("background", b, _plan_for(b, talk, env))]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    # The main character does its own thing and isn't pulled into a background character's
    # conversation
    assert arb["agent-a"].is_passive_join is False
    assert arb["agent-a"].action_result.action.action_type == ActionType.WORK
    assert arb["agent-a"].action_result.action.action_description == "批阅文书"
    b_exec = rt._executor_registry.get_active_for_agent("agent-b")  # noqa: SLF001
    assert b_exec is not None and b_exec.participant_ids == ["agent-b"]
    assert "李世民正埋头忙着手上的事" in b_exec.extra["completed_result"].outcome


@pytest.mark.asyncio
async def test_starved_intent_takes_initiative_across_phases(container) -> None:
    """An intent foiled up to the threshold jumps phase initiative: same setup as the previous test
    plus starvation, and the result flips.

    This is the only time phase is overridden, and the reason it can be: the ones that starve are
    the ones that sit late in the order for a long time, so not overriding phase would fix nothing.
    """
    from engine.arbiter import _FOILED_MISS_PRIORITY_THRESHOLD

    world_id = "world-arb-starved"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="张婕妤", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    work = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.WORK,
                       action_description="批阅文书", estimated_steps=1)
    talk = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.TALK,
                       action_description="上前攀谈", target=ActionTarget(acts_on=[Ref.agent("agent-a")], claims=[Ref.agent("agent-a")]), estimated_steps=2)
    planned = [
        ("main", a, _plan_for(a, work, env)),
        ("background", b, _plan_for(b, talk, env,
                                    foiled_misses=_FOILED_MISS_PRIORITY_THRESHOLD)),
    ]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    # The background character goes first and gets the body; the main character yields this step,
    # and its WORK intent retries via the foiled buffer.
    b_exec = rt._executor_registry.get_active_for_agent("agent-b")  # noqa: SLF001
    assert b_exec is not None and b_exec.action_type == ActionType.TALK
    assert "agent-a" in b_exec.participant_ids
    assert arb["agent-a"].is_passive_join is True


@pytest.mark.asyncio
async def test_deeper_starvation_goes_first_between_two_starved_agents(container) -> None:
    """When two starving agents compete for one body, the one starved longer goes first; a boolean
    flag would let the same one always win and the other starve forever."""
    from engine.arbiter import _FOILED_MISS_PRIORITY_THRESHOLD as N

    world_id = "world-arb-starved-tie"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    for aid in ("agent-a", "agent-b", "agent-c"):
        env.place_agent(agent_id=aid, location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="张婕妤", is_main_character=True)
    c = _build_agent(container, world_id=world_id, agent_id="agent-c", name="常何", is_main_character=True)
    agents = {"agent-a": a, "agent-b": b, "agent-c": c}
    rt = _arb_runtime(container, world_id, agents, env)

    def talk_to_c(actor):
        return AgentAction(agent_id=actor, step=1, action_type=ActionType.TALK,
                           action_description="上前攀谈",
                           target=ActionTarget(acts_on=[Ref.agent("agent-c")], claims=[Ref.agent("agent-c")]), estimated_steps=2)

    idle = AgentAction(agent_id="agent-c", step=1, action_type=ActionType.WORK,
                       action_description="批阅文书", estimated_steps=1)
    # A is earlier in input order, but B has starved longer.
    planned = [
        ("main", a, _plan_for(a, talk_to_c("agent-a"), env, foiled_misses=N)),
        ("main", b, _plan_for(b, talk_to_c("agent-b"), env, foiled_misses=N + 2)),
        ("main", c, _plan_for(c, idle, env)),
    ]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    b_exec = rt._executor_registry.get_active_for_agent("agent-b")  # noqa: SLF001
    assert b_exec is not None and "agent-c" in b_exec.participant_ids
    a_exec = rt._executor_registry.get_active_for_agent("agent-a")  # noqa: SLF001
    assert a_exec is not None and a_exec.participant_ids == ["agent-a"]
    assert "常何正与张婕妤交谈" in a_exec.extra["completed_result"].outcome


@pytest.mark.asyncio
async def test_rejection_quotes_the_first_person_intent(container) -> None:
    """action_description is the first-person original with its own punctuation; embedding it in the
    rejection sentence needs 「」 quotes and trailing punctuation stripped.

    Spliced in raw it renders "我本想我走到他面前…细节。，却因", and that sentence enters decide's
    foiled block.
    """
    world_id = "world-arb-quote"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="尉迟恭", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="房玄龄", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                       action_description="我走到房玄龄面前，压低声音与他确认接应路线的细节。",
                       target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]), estimated_steps=2)
    work = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.WORK,
                       action_description="批阅文书", estimated_steps=1)
    planned = [("bg", b, _plan_for(b, work, env)), ("main", a, _plan_for(a, talk, env))]

    await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    rejection = rt._executor_registry.get_active_for_agent("agent-a").extra["completed_result"]  # noqa: SLF001
    assert "尉迟恭本想做「我走到房玄龄面前，压低声音与他确认接应路线的细节」" in rejection.outcome
    assert "我本想做「我走到房玄龄面前，压低声音与他确认接应路线的细节」" in rejection.factual_memory
    for text in (rejection.outcome, rejection.factual_memory):
        assert "本想我走到" not in text
        assert "。，" not in text


def test_busy_reason_says_what_the_body_is_visibly_doing(container) -> None:
    """The rejection reason is what the rejected person sees up close: a conversation names the
    partner, resting says resting; a covert action is invisible to others and falls back to the
    vague phrase."""
    from engine.arbiter import _BUSY_UNSEEN

    world_id = "world-busy-reason"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    names = {"agent-a": "李世民", "agent-b": "常何", "agent-c": "尉迟恭", "agent-d": "侯君集"}
    agents = {}
    for aid, name in names.items():
        env.place_agent(agent_id=aid, location_id="palace")
        agents[aid] = _build_agent(container, world_id=world_id, agent_id=aid, name=name, is_main_character=True)
    rt = _arb_runtime(container, world_id, agents, env)

    def busy(action_type, *pids):
        rt._executor_registry.add_active(ActionExecutionState.create(  # noqa: SLF001
            target=ActionTarget(), action_type=action_type, initiator_id=pids[0],
            participant_ids=list(pids), purpose="x", started_step=1, estimated_steps=3,
            opening_outcome="",
        ))

    busy(ActionType.TALK, "agent-a", "agent-b")
    busy(ActionType.REST, "agent-c")
    busy(ActionType.COVERT, "agent-d")

    def reason(pid):
        return rt._arbiter._busy_reason(pid, {}, {})  # noqa: SLF001

    assert reason("agent-a") == "李世民正与常何交谈"
    assert reason("agent-c") == "尉迟恭正在歇息"
    assert reason("agent-d") == f"侯君集{_BUSY_UNSEEN}"


@pytest.mark.asyncio
async def test_a_body_admitted_this_step_is_busy_with_what_it_is_about_to_do(container) -> None:
    """Arbitration reads the world at the start of the step: if the target is setting off this step,
    the reason is that he's about to leave, not that he's gone."""
    world_id = "world-busy-this-step"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    for entity_id, links in (("palace", {"garden": 1}), ("garden", {"palace": 1})):
        env.space.register_place(Place(
            place_id=entity_id, name=entity_id, description=entity_id, connections=dict(links),
        ))
    for aid in ("agent-a", "agent-b"):
        env.place_agent(agent_id=aid, location_id="palace")
    mover = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    talker = _build_agent(container, world_id=world_id, agent_id="agent-b", name="常何", is_main_character=False)
    agents = {"agent-a": mover, "agent-b": talker}
    rt = _arb_runtime(container, world_id, agents, env)

    leave = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.MOVE, action_description="去园中",
                        estimated_steps=1, target=ActionTarget(acts_on=[Ref.place("garden")]))
    talk = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.TALK, action_description="拦下他说话",
                       estimated_steps=2, target=ActionTarget(acts_on=[Ref.agent("agent-a")], claims=[Ref.agent("agent-a")]))
    planned = [("main", mover, _plan_for(mover, leave, env)), ("bg", talker, _plan_for(talker, talk, env))]

    await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001

    rejection = rt._executor_registry.get_active_for_agent("agent-b").extra["completed_result"]  # noqa: SLF001
    assert rejection.not_executed
    assert "李世民正要动身离开" in rejection.outcome
    assert rt._executor_registry.get_active_for_agent("agent-a").action_type == ActionType.MOVE  # noqa: SLF001


def _two_rooms_with(container, world_id: str, names: dict[str, str]):
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    for entity_id, links in (("palace", {"garden": 1}), ("garden", {"palace": 1})):
        env.space.register_place(Place(
            place_id=entity_id, name=entity_id, description=entity_id, connections=dict(links),
        ))
    agents = {}
    for aid, name in names.items():
        env.place_agent(agent_id=aid, location_id="palace")
        agents[aid] = _build_agent(container, world_id=world_id, agent_id=aid, name=name, is_main_character=True)
    return env, agents, _arb_runtime(container, world_id, agents, env)


@pytest.mark.asyncio
async def test_an_invitation_cannot_hold_a_body_that_means_to_leave(container) -> None:
    """Within a tier, a solo leaving MOVE goes before TALK: even with TALK earlier in input order,
    the person leaves and the invitation fails."""
    env, agents, rt = _two_rooms_with(container, "world-leave-beats-invite", {"agent-a": "李世民", "agent-b": "常何"})
    leave = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.MOVE, action_description="去园中",
                        estimated_steps=1, target=ActionTarget(acts_on=[Ref.place("garden")]))
    talk = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.TALK, action_description="拦下他说话",
                       estimated_steps=2, target=ActionTarget(acts_on=[Ref.agent("agent-a")], claims=[Ref.agent("agent-a")]))
    planned = [
        ("main", agents["agent-b"], _plan_for(agents["agent-b"], talk, env)),
        ("main", agents["agent-a"], _plan_for(agents["agent-a"], leave, env)),
    ]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001

    assert arb["agent-a"].is_passive_join is False
    assert rt._executor_registry.get_active_for_agent("agent-a").action_type == ActionType.MOVE  # noqa: SLF001
    rejection = rt._executor_registry.get_active_for_agent("agent-b").extra["completed_result"]  # noqa: SLF001
    assert "李世民正要动身离开" in rejection.outcome


@pytest.mark.asyncio
async def test_a_carry_still_takes_a_body_that_means_to_leave(container) -> None:
    """Leaving only beats invitation: carrying someone off (COMPEL) still goes first and takes the
    leaver along."""
    env, agents, rt = _two_rooms_with(container, "world-compel-beats-leave", {"agent-a": "李世民", "agent-b": "尉迟敬德"})
    leave = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.MOVE, action_description="去园中",
                        estimated_steps=1, target=ActionTarget(acts_on=[Ref.place("garden")]))
    carry = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.MOVE, action_description="拽着他走",
                        estimated_steps=1, target=ActionTarget(acts_on=[Ref.place("garden")], claims=[Ref.agent("agent-a")]))
    planned = [
        ("main", agents["agent-a"], _plan_for(agents["agent-a"], leave, env)),
        ("main", agents["agent-b"], _plan_for(agents["agent-b"], carry, env)),
    ]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001

    assert arb["agent-a"].is_passive_join is True
    assert rt._executor_registry.get_active_for_agent("agent-b").participant_ids == ["agent-b", "agent-a"]  # noqa: SLF001


def test_every_action_type_has_a_visible_busy_phrase_or_is_deliberately_unseen() -> None:
    """A new action type without its phrase would silently fall back to the vague one; only COVERT
    belongs there."""
    from engine.arbiter import _BUSY_WITH

    uncovered = set(ActionType) - set(_BUSY_WITH) - {ActionType.TALK}
    assert uncovered == {ActionType.COVERT}


@pytest.mark.asyncio
async def test_critical_pressure_keeps_own_body_against_conscription(container) -> None:
    """Under CRITICAL external pressure, an agent claims its own body first and the conscripter
    fails (see _initiative_order for why).

    A's TALK would normally be arbitrated before B's independent action and take B; with
    defy-the-edict pressure on B the order flips: B does its own thing and A is rejected, with the
    reason being what B visibly looks like.
    """
    world_id = "world-arb-preempt"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="魏徵", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=True)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                       action_description="截住太子追问玄甲军虚实",
                       target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]), estimated_steps=2)
    own = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.WORK,
                      action_description="备好车马即刻动身面圣", estimated_steps=1)
    planned = [
        ("main", a, _plan_for(a, talk, env)),
        ("main", b, _plan_for(b, own, env, external_goals=[_pressure(Urgency.CRITICAL)])),
    ]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    assert arb["agent-b"].is_passive_join is False
    b_exec = rt._executor_registry.get_active_for_agent("agent-b")  # noqa: SLF001
    assert b_exec is not None and b_exec.action_type == ActionType.WORK
    # The conscripter fails: an actor-only failed execution, with the reason naming B (name, not id)
    a_exec = rt._executor_registry.get_active_for_agent("agent-a")  # noqa: SLF001
    assert a_exec is not None and a_exec.participant_ids == ["agent-a"]
    rejection = a_exec.extra["completed_result"]
    assert rejection.succeeded is False
    assert "李建成正埋头忙着手上的事" in rejection.outcome


@pytest.mark.asyncio
async def test_high_pressure_alone_does_not_preempt_conscription(container) -> None:
    """The preemption threshold is CRITICAL, not interrupt's HIGH; loosening it would make
    conscription fail broadly."""
    world_id = "world-arb-preempt-bar"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="魏徵", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=True)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                       action_description="截住太子追问玄甲军虚实",
                       target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]), estimated_steps=2)
    own = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.WORK,
                      action_description="备好车马即刻动身面圣", estimated_steps=1)
    planned = [
        ("main", a, _plan_for(a, talk, env)),
        ("main", b, _plan_for(b, own, env, external_goals=[_pressure(Urgency.HIGH)])),
    ]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    assert arb["agent-b"].is_passive_join is True
    a_exec = rt._executor_registry.get_active_for_agent("agent-a")  # noqa: SLF001
    assert a_exec is not None and "agent-b" in a_exec.participant_ids


@pytest.mark.asyncio
async def test_pressure_preempt_never_crosses_phases(container) -> None:
    """Preemption only works within a phase: however urgent a background character is, it can't take
    a body the main character is conscripting.

    Pressure is about stakes; phase is narrative tier. The two axes mustn't cross.
    """
    world_id = "world-arb-preempt-phase"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="张婕妤", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                       action_description="低声嘱托宫中传话",
                       target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]), estimated_steps=2)
    own = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.WORK,
                      action_description="连夜收拾细软准备出宫", estimated_steps=1)
    planned = [
        ("main", a, _plan_for(a, talk, env)),
        ("background", b, _plan_for(b, own, env, external_goals=[_pressure(Urgency.CRITICAL)])),
    ]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    assert arb["agent-b"].is_passive_join is True


@pytest.mark.asyncio
async def test_step_start_in_progress_body_cannot_be_conscripted(container) -> None:
    """A body that enters this step mid-action (its action finished in the start-of-step tick, left
    the active registry, and was excluded from planning) can't be conscripted by another action this
    step; otherwise one agent does two actions in a step and runs feedback twice. `is_agent_active`
    can't stop it (the execution is finished); in_progress_at_step_start seeds `consumed` instead. A
    was doing COVERT at step start (finishing this step, not in planned); B decides TALK→A: B's TALK
    is vetoed and A isn't conscripted."""
    world_id = "world-arb-inprogress"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="尉迟恭", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    # A is mid-action at step start (its COVERT finished in the tick and left active), so it's not
    # in planned_steps; only B is planned.
    talk = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.TALK,
                       action_description="呈递密报并恳请定夺", target=ActionTarget(acts_on=[Ref.agent("agent-a")], claims=[Ref.agent("agent-a")]), estimated_steps=1)
    planned = [("bg", b, _plan_for(b, talk, env))]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, {"agent-a"})  # noqa: SLF001
    # A wasn't conscripted: no arbitration entry (no passive join, no second feedback run).
    assert "agent-a" not in arb
    # B's TALK is vetoed as an actor-only failure: A is busy and not a participant.
    b_aa = arb["agent-b"]
    assert b_aa.is_passive_join is False
    b_exec = rt._executor_registry.get_active_for_agent("agent-b")  # noqa: SLF001
    assert b_exec is not None and b_exec.participant_ids == ["agent-b"]
    rejection = b_exec.extra["completed_result"]
    assert rejection.succeeded is False
    assert "正忙于他事" in rejection.outcome
    assert "李世民" in rejection.outcome and "agent-a" not in rejection.outcome  # name, not id


@pytest.mark.asyncio
async def test_denied_actor_stays_invitable_same_step(container) -> None:
    """A body whose own action was rejected did nothing; later arrivals can still engage it, and it
    mustn't count as "busy".

    A goes first and tries to engage C, who is busy, and is rejected; B then engages A. If A counted
    as taken, one failed contention would turn into two.
    """
    world_id = "world-arb-denied-free"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    for aid in ("agent-a", "agent-b", "agent-c"):
        env.place_agent(agent_id=aid, location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李建成", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李元吉", is_main_character=True)
    c = _build_agent(container, world_id=world_id, agent_id="agent-c", name="魏徵", is_main_character=True)
    agents = {"agent-a": a, "agent-b": b, "agent-c": c}
    rt = _arb_runtime(container, world_id, agents, env)

    a_talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                         action_description="寻魏徵过目名单",
                         target=ActionTarget(acts_on=[Ref.agent("agent-c")], claims=[Ref.agent("agent-c")]), estimated_steps=2)
    b_talk = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.TALK,
                         action_description="拦住建成抠定章程",
                         target=ActionTarget(acts_on=[Ref.agent("agent-a")], claims=[Ref.agent("agent-a")]), estimated_steps=2)
    planned = [("main", a, _plan_for(a, a_talk, env)), ("main", b, _plan_for(b, b_talk, env))]

    # C is mid-action at step start → A is rejected; that rejection mustn't make B fail too.
    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, {"agent-c"})  # noqa: SLF001

    b_exec = rt._executor_registry.get_active_for_agent("agent-b")  # noqa: SLF001
    assert b_exec is not None and set(b_exec.participant_ids) == {"agent-a", "agent-b"}
    assert arb["agent-b"].is_passive_join is False
    # A joins B's conversation and gets a join record, not a second "未能如愿".
    assert arb["agent-a"].is_passive_join is True
    assert "未能如愿" not in arb["agent-a"].action_result.outcome
    assert rt._executor_registry.get_active_for_agent("agent-a") is b_exec  # noqa: SLF001


@pytest.mark.asyncio
async def test_a_body_one_initiator_could_not_take_is_taken_by_the_next(container, caplog) -> None:
    """Two people in one step try to engage the same person: the first was himself taken by someone
    else, so his invitation is void; the later one engages normally.

    The invitee's own invitation also failed (his target was taken), and that failure doesn't use up
    his body: he's still engaged by the later one, with a single join record. Both dropped decisions
    carry over to the next step.
    """
    world_id = "world-arb-second-taker"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    ids = {"agent-a": "李元吉", "agent-b": "李世民", "agent-c": "李渊", "agent-d": "尉迟敬德"}
    for aid in ids:
        env.place_agent(agent_id=aid, location_id="palace")
    agents = {
        aid: _build_agent(container, world_id=world_id, agent_id=aid, name=name, is_main_character=True)
        for aid, name in ids.items()
    }
    rt = _arb_runtime(container, world_id, agents, env)

    def talk(actor, other, desc):
        return AgentAction(agent_id=actor, step=1, action_type=ActionType.TALK, action_description=desc,
                           target=ActionTarget(acts_on=[Ref.agent(other)], claims=[Ref.agent(other)]),
                           estimated_steps=1)

    a, b, c, d = (agents[k] for k in ids)
    # Initiative: Yuanji, under pressure, goes first; the rest share a tier, in input order Shimin →
    # Li Yuan → Yuchi.
    planned = [
        ("main", a, _plan_for(a, talk("agent-a", "agent-c", "向父皇哭诉求救"), env,
                              external_goals=[_pressure(Urgency.CRITICAL)])),
        ("main", b, _plan_for(b, talk("agent-b", "agent-c", "向父皇陈明元吉谋逆"), env)),
        ("main", c, _plan_for(c, talk("agent-c", "agent-b", "质问秦王为何兵戈相向"), env)),
        ("main", d, _plan_for(d, talk("agent-d", "agent-b", "向秦王禀报宫中情形"), env)),
    ]

    import logging

    caplog.set_level(logging.DEBUG, logger="engine.arbiter")
    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001

    registry = rt._executor_registry  # noqa: SLF001
    first = registry.get_active_for_agent("agent-a")
    second = registry.get_active_for_agent("agent-d")
    assert set(first.participant_ids) == {"agent-a", "agent-c"}
    assert set(second.participant_ids) == {"agent-d", "agent-b"}, "李渊约不到秦王,尉迟随后照常约走"
    assert arb["agent-c"].is_passive_join and arb["agent-c"].ongoing_execution_id == first.execution_id
    assert arb["agent-b"].is_passive_join and arb["agent-b"].ongoing_execution_id == second.execution_id
    # The Prince of Qin's own failure isn't recorded: one verdict per body per step, and his is the
    # join.
    assert "未能如愿" not in arb["agent-b"].action_result.outcome
    assert not any(e.extra.get("completed_result") for e in registry.all_active() if e.initiator_id == "agent-b")

    # The logs alone should let you check this step: initiative order, who blocked the Prince of
    # Qin, and why his rejection wasn't recorded.
    logs = {r.message: r for r in caplog.records}
    order = logs["initiative_order"].order
    assert [o["agent_id"] for o in order] == ["agent-a", "agent-b", "agent-c", "agent-d"]
    assert order[0]["preempting"] and not order[1]["preempting"]
    denied = [r for r in caplog.records if r.message == "admission_denied"]
    assert [(r.agent_id, r.blocked_by) for r in denied] == [("agent-b", {"agent-c": "agent-a"})]
    assert (logs["denial_superseded_by_enrollment"].agent_id,
            logs["denial_superseded_by_enrollment"].initiator) == ("agent-b", "agent-d")

    await rt._commit_execution(planned, arb, agents)  # noqa: SLF001
    assert any("质问秦王" in t for t in c.memory_system.recent_foiled_attempts(1))
    assert any("陈明元吉谋逆" in t for t in b.memory_system.recent_foiled_attempts(1))


@pytest.mark.asyncio
async def test_denied_actor_settles_when_nobody_invites_him(container) -> None:
    """If nobody engages him, the rejected action still settles as a failure on this step; deferring
    the record isn't skipping it."""
    world_id = "world-arb-denied-settle"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-c", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李建成", is_main_character=True)
    c = _build_agent(container, world_id=world_id, agent_id="agent-c", name="魏徵", is_main_character=True)
    agents = {"agent-a": a, "agent-c": c}
    rt = _arb_runtime(container, world_id, agents, env)

    a_talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                         action_description="寻魏徵过目名单",
                         target=ActionTarget(acts_on=[Ref.agent("agent-c")], claims=[Ref.agent("agent-c")]), estimated_steps=2)
    arb, _ = await rt._arbiter.arbitrate([("main", a, _plan_for(a, a_talk, env))], agents, 1, {"agent-c"})  # noqa: SLF001

    a_exec = rt._executor_registry.get_active_for_agent("agent-a")  # noqa: SLF001
    assert a_exec is not None and a_exec.participant_ids == ["agent-a"]
    rejection = a_exec.extra["completed_result"]
    assert rejection.succeeded is False and rejection.not_executed is True
    assert "魏徵正忙于他事" in rejection.outcome
    assert arb["agent-a"].is_passive_join is False


@pytest.mark.asyncio
async def test_denied_action_is_stamped_rejected(container) -> None:
    """An action ruled infeasible must be stamped rejected: the judge uses it to waive the
    decision↔action consistency check. Stamped executed, it would be judged "decided TALK but no
    conversation happened" (a broken chain / fabrication).

    The verdict must come from the completion result carried by the execution; the one on the
    first-step marker is always the default.
    """
    world_id = "world-arb-verdict-rejected"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-c", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李建成", is_main_character=True)
    c = _build_agent(container, world_id=world_id, agent_id="agent-c", name="魏徵", is_main_character=True)
    agents = {"agent-a": a, "agent-c": c}
    sink = InMemoryTraceSink()
    sink.record_llm_call(_trace_call(world_id, agent_id="agent-a", stage=Stage.DECISION))
    rt = _arb_runtime(container, world_id, agents, env, trace_sink=sink)

    talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                       action_description="寻魏徵过目名单",
                       target=ActionTarget(acts_on=[Ref.agent("agent-c")], claims=[Ref.agent("agent-c")]), estimated_steps=2)
    planned = [("main", a, _plan_for(a, talk, env))]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, {"agent-c"})  # noqa: SLF001
    await rt._commit_execution(planned, arb, agents)  # noqa: SLF001

    decisions = {c_.agent_id: c_.extra for c_ in sink.llm_calls if c_.stage == Stage.DECISION.value}
    assert decisions["agent-a"] == {"verdict": "rejected"}


@pytest.mark.asyncio
async def test_no_decision_agent_is_skipped_and_not_committed(container) -> None:
    """Decision unusable (plan.action=None) → the agent is skipped this step: no arbitration, no
    commit, no record; other agents act normally (Rule 1 fallback tier 1; the body stays idle and
    unused)."""
    world_id = "world-arb-skip"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    work = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.WORK,
                       action_description="批阅文书", estimated_steps=1)
    # agent-b's decision failed → plan carries action=None.
    planned = [("main", a, _plan_for(a, work, env)), ("bg", b, _plan_for(b, None, env))]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    assert "agent-a" in arb          # acted normally
    assert "agent-b" not in arb      # no decision → no arbitration entry

    records = await rt._commit_execution(planned, arb, agents)  # noqa: SLF001
    committed_ids = {r["agent_id"] for r in records}
    assert "agent-a" in committed_ids
    assert "agent-b" not in committed_ids   # skipped: no commit, no record


@pytest.mark.asyncio
async def test_one_agents_perception_failure_is_logged_and_spares_the_others(container, caplog) -> None:
    """Rule 4 per slot: one agent's perception error only logs a warning and doesn't affect other
    agents' perception in the same step."""
    import logging
    from types import SimpleNamespace

    from engine.message_system import MessageDelivery

    world_id = "world-perceive-fail"
    runtime, _, _ = _build_runtime(container, world_id=world_id)
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    perceived: list[str] = []

    async def broken(**kwargs) -> None:
        raise RuntimeError("perception store down")

    async def recorded(**kwargs) -> None:
        perceived.append("agent-b")

    a.perceive_step = broken  # type: ignore[method-assign]
    b.perceive_step = recorded  # type: ignore[method-assign]
    spatial = SimpleNamespace(location_id="palace")

    with caplog.at_level(logging.WARNING, logger="engine.runtime"):
        await runtime._perceive_all_agents(  # noqa: SLF001
            {"agent-a": a, "agent-b": b}, MessageDelivery(step=1),
            {"agent-a": spatial, "agent-b": spatial}, [], 1,
        )

    assert perceived == ["agent-b"]
    failures = [r for r in caplog.records if r.getMessage() == "agent_perception_failed"]
    assert [r.agent_id for r in failures] == ["agent-a"]


@pytest.mark.asyncio
async def test_no_action_agent_is_skipped_like_no_decision(container) -> None:
    """A deliberate NO_ACTION takes the same minimal path as a failed decision: no arbitration, no
    commit, no record, no state change. The difference exists only in the code layer
    (decision_status + log level); narratively and in state it's identical."""
    world_id = "world-arb-noaction"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    work = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.WORK,
                       action_description="批阅文书", estimated_steps=1)
    # agent-b deliberately does nothing this beat (act:false → NO_ACTION).
    planned = [("main", a, _plan_for(a, work, env)),
               ("bg", b, _plan_for(b, None, env, decision_status=DecisionStatus.NO_ACTION))]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    assert "agent-a" in arb
    assert "agent-b" not in arb      # NO_ACTION → no arbitration entry (skipped)

    records = await rt._commit_execution(planned, arb, agents)  # noqa: SLF001
    committed_ids = {r["agent_id"] for r in records}
    assert "agent-a" in committed_ids
    assert "agent-b" not in committed_ids   # no commit, no record, zero state change


@pytest.mark.asyncio
async def test_no_decision_agent_can_still_be_conscripted(container) -> None:
    """Conscription beats having no decision: an agent whose decision failed can still be pulled in
    as a collaborator by someone's multi-step action and passive-join, and commit doesn't crash on
    plan.action=None (it uses the arbitration stub's type/description)."""
    world_id = "world-arb-skip-join"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                       action_description="商议对策", target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]), estimated_steps=2)
    # agent-b's decision failed (action=None) but it is co-located and gets enrolled.
    planned = [("main", a, _plan_for(a, talk, env)), ("bg", b, _plan_for(b, None, env))]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    assert arb["agent-b"].is_passive_join is True          # conscription beats no-decision
    records = await rt._commit_execution(planned, arb, agents)  # noqa: SLF001
    b_rec = next(r for r in records if r["agent_id"] == "agent-b")
    assert "李世民" in b_rec["outcome"]                      # joined the initiator's action
    assert b_rec["action_type"] == ActionType.TALK          # record uses the joint action's type
    # No decision (action=None) → no intent to keep; neither the foiled buffer nor the goal queue
    # gets anything
    assert b.memory_system.recent_foiled_attempts(1) == []
    assert b.personality.state.short_term_goal_entities == []


@pytest.mark.asyncio
async def test_duration1_talk_completes_both_same_step(container, caplog) -> None:
    """Action duration rule: a TALK with duration≤1 completes on its step and both sides get
    feedback that step (the same-step completion pass), not the next. The registry empties that
    step, neither side stays IN_PROGRESS, and each gets one feedback with origin=finalize."""
    import logging
    world_id = "world-talk-d1"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    rt = _arb_runtime(container, world_id, {"agent-a": a, "agent-b": b}, env)

    talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                       action_description="商议对策", target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]), estimated_steps=1)

    async def plan_a(*, step, spatial, inbox, broadcasts, **_kw):
        return _plan_for(a, talk, env, step=step)

    async def plan_b(*, step, spatial, inbox, broadcasts, **_kw):
        return _plan_for(b, None, env, step=step)   # conscripted into the TALK

    a.plan_step = plan_a
    b.plan_step = plan_b

    with caplog.at_level(logging.INFO):
        await rt.run_step([a, b])

    # Same-step completion: the registry is empty this step and both sides are done (not
    # IN_PROGRESS).
    assert rt._executor_registry.all_active() == []  # noqa: SLF001
    assert a.personality.state.action_status != ActionStatus.IN_PROGRESS
    assert b.personality.state.action_status != ActionStatus.IN_PROGRESS
    # Each side gets one feedback this step.
    fb = [
        getattr(r, "agent_id", None)
        for r in caplog.records if r.getMessage() == "agent_feedback_recorded"
    ]
    assert sorted(fb) == ["agent-a", "agent-b"]


@pytest.mark.asyncio
async def test_duration2_talk_still_spans_two_steps(container) -> None:
    """Control case: a TALK with duration=2 isn't caught by the same-step sweep. At step 1 both are
    IN_PROGRESS with one registry entry (remaining_steps=1>0, not finished); at step 2 a normal tick
    takes it to 0 and it completes. Shows the remaining<=0 completion check is exact."""
    world_id = "world-talk-d2"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    rt = _arb_runtime(container, world_id, {"agent-a": a, "agent-b": b}, env)

    talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                       action_description="商议对策", target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]), estimated_steps=2)

    async def plan_a(*, step, spatial, inbox, broadcasts, **_kw):
        return _plan_for(a, talk, env, step=step)

    async def plan_b(*, step, spatial, inbox, broadcasts, **_kw):
        return _plan_for(b, None, env, step=step)

    a.plan_step = plan_a
    b.plan_step = plan_b

    await rt.run_step([a, b])
    # step 1: still in progress (not caught by the same-step completion pass).
    assert len(rt._executor_registry.all_active()) == 1  # noqa: SLF001
    assert a.personality.state.action_status == ActionStatus.IN_PROGRESS
    assert b.personality.state.action_status == ActionStatus.IN_PROGRESS

    # step 2: the initiator is mid-action and doesn't re-decide (plan_a must not be called again); a
    # normal tick completes it.
    a.plan_step = plan_a  # would restart the TALK; the scheduler must skip him
    await rt.run_step([a, b])
    assert rt._executor_registry.all_active() == []  # noqa: SLF001


@pytest.mark.asyncio
async def test_duration1_record_shows_completion_not_opening_marker(container) -> None:
    """Double-display guard: for a duration-1 action, begin produces an opening marker record and
    the sweep produces the completion; they must merge in place into one record with the completed
    outcome. The snapshot's actions_this_step has exactly one entry for the agent, with the
    completed outcome (the background WORK template "做完了"), not the opening marker ("着手"), and no
    duplicate."""
    world_id = "world-d1-display"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=False)
    rt = _arb_runtime(container, world_id, {"agent-a": a}, env)

    work = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.WORK,
                       action_description="批阅文书", estimated_steps=1)

    async def plan_a(*, step, spatial, inbox, broadcasts, **_kw):
        return _plan_for(a, work, env, step=step)

    a.plan_step = plan_a

    await rt.run_step([a])
    snapshot = await container.snapshot.load(world_id, 1)
    recs = [r for r in snapshot.actions_this_step if r["agent_id"] == "agent-a"]
    # Exactly one record (not "opening marker + completion").
    assert len(recs) == 1
    rec = recs[0]
    # The outcome is the completed form (background WORK template "做完了"), not the opening marker
    # "着手".
    assert "做完了" in rec["outcome"]
    assert "着手" not in rec["outcome"]
    # The registry empties this step and the agent leaves IN_PROGRESS (same-step completion).
    assert rt._executor_registry.all_active() == []  # noqa: SLF001
    assert a.personality.state.action_status != ActionStatus.IN_PROGRESS


@pytest.mark.asyncio
async def test_decision_monologue_reaches_record_and_survives_d1_fold(container) -> None:
    """The inner monologue (the decision's first-person reason) must leave the world with the action
    record; otherwise the feed can only show what was done and how it went, never why. A born-zero
    action's begin marker is cleared and updated in place, and duration-1 is the most common path,
    so this guards both: the runtime write and its survival through folding."""
    world_id = "world-monologue"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=False)
    rt = _arb_runtime(container, world_id, {"agent-a": a}, env)

    work = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.WORK,
                       action_description="批阅文书", estimated_steps=1,
                       inner_monologue="父皇明日要问，我今夜须把奏疏理清。")

    async def plan_a(*, step, spatial, inbox, broadcasts, **_kw):
        return _plan_for(a, work, env, step=step)

    a.plan_step = plan_a

    await rt.run_step([a])
    snapshot = await container.snapshot.load(world_id, 1)
    rec = next(r for r in snapshot.actions_this_step if r["agent_id"] == "agent-a")
    assert rec["inner_monologue"] == "父皇明日要问，我今夜须把奏疏理清。"
    # Background agents use the same LLM decision path (CLAUDE.md §5), so the monologue isn't split
    # by tier; agent-a here is is_main_character=False.
    assert rec["is_main_character"] is False


@pytest.mark.asyncio
async def test_conscripted_participant_record_carries_no_monologue(container) -> None:
    """The conscripted agent didn't make this decision this step (his own intent was deferred). The
    begin record's record_action is the initiator's action object, so copying its
    inner_monologue would print A's thoughts under B's name. The participant's field must be empty;
    the initiator's is filled as usual."""
    world_id = "world-monologue-join"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=True)
    rt = _arb_runtime(container, world_id, {"agent-a": a, "agent-b": b}, env)

    talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                       action_description="商议对策", target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]),
                       estimated_steps=1, inner_monologue="他若肯让步，今夜便无须动刀。")

    async def plan_a(*, step, spatial, inbox, broadcasts, **_kw):
        return _plan_for(a, talk, env, step=step)

    async def plan_b(*, step, spatial, inbox, broadcasts, **_kw):
        return _plan_for(b, None, env, step=step)   # conscripted into the TALK

    a.plan_step = plan_a
    b.plan_step = plan_b

    await rt.run_step([a, b])
    snapshot = await container.snapshot.load(world_id, 1)
    by_agent = {r["agent_id"]: r for r in snapshot.actions_this_step}
    assert by_agent["agent-a"]["inner_monologue"] == "他若肯让步，今夜便无须动刀。"
    assert by_agent["agent-b"]["inner_monologue"] == ""


@pytest.mark.asyncio
async def test_finalize_skips_self_targeted_effect(container) -> None:
    """Self-target guard: a self-targeted action's target_effect must not be applied back to
    the actor. Otherwise the actor gets its own feedback via finalize_ongoing_action and a second
    write as a target via apply_target_effect (vitality deducted twice / two memories).
    _finalize_execution must skip effects where effect.agent_id == the actor."""
    from core.interfaces.action import ActionResult, ActionTarget, TargetAgentEffect
    world_id = "world-selfguard"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=False)
    rt = _arb_runtime(container, world_id, {"agent-a": a}, env)

    exec_state = ActionExecutionState.create(
        target=ActionTarget(),
        action_type=ActionType.PHYSICAL, initiator_id="agent-a", participant_ids=["agent-a"],
        purpose="自伤", started_step=1, estimated_steps=1, opening_outcome="",
    )
    rt._executor_registry.add_active(exec_state)  # noqa: SLF001
    stub = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
                       action_description="自伤", target=ActionTarget(acts_on=[Ref.agent("agent-a")]))
    # The result points a target_effect at the actor itself (vitality_damage=0.5); the actor's own
    # result.vitality_damage=0.0.
    self_effect = TargetAgentEffect(agent_id="agent-a", factual_memory="x", vitality_damage=0.5)
    result = ActionResult(action=stub, expected_outcome="", outcome="在宫中，李世民自伤。",
                          succeeded=True, factual_memory="我做了一件事。", vitality_damage=0.0,
                          target_effects=[self_effect])

    async def _fake_complete(*args, **kwargs):
        return [result]

    rt._executor_registry.get_executor(ActionType.PHYSICAL).complete = _fake_complete  # type: ignore[method-assign]  # noqa: SLF001

    vitality_before = a.personality.state.vitality
    await rt._processor.finalize_executions_batch([exec_state], {"agent-a": a}, 1)  # noqa: SLF001
    # The self-targeted effect is skipped → vitality only reflects result.vitality_damage (0.0),
    # with no second 0.5 deduction.
    assert a.personality.state.vitality == vitality_before


def test_build_default_registry_covers_all_action_types(container) -> None:
    """Fail-fast invariant: every ActionType has an executor."""
    directory = LiveWorldDirectory.from_agents({}, EnvironmentSystem())
    registry = build_default_registry(container.llm_router, directory)
    for t in ActionType:
        assert registry.get_executor(t) is not None


# ---------------------------------------------------------------------------
# Rule 4 / Rule 1 boundary: one agent failing must not kill the step or run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_execution_survives_one_agent_plan_crash(container) -> None:
    """Planning gather degrades per slot: one agent's plan_step exception doesn't cancel sibling
    tasks or propagate; the failed agent degrades to a FAILED empty plan (action=None), keeping its
    entry (it can still be conscripted this step)."""
    world_id = "world-plan-crash"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)
    _set_work_decision(container)

    async def _boom(**kwargs):
        raise RuntimeError("simulated plan crash")

    a.plan_step = _boom  # type: ignore[method-assign]

    from engine.message_system import MessageDelivery
    plan = AgentScheduler().plan([a, b], step=1, in_progress_at_step_start=set())
    planned = await rt._plan_execution(  # noqa: SLF001
        plan=plan, agents=[a, b], in_progress_at_step_start=set(),
        deliveries=MessageDelivery(step=1), step=1,
        world_time_label="清晨", agent_spatials=None, broadcasts=[],
    )

    by_id = {p.agent_id: p for _phase, _agent, p in planned}
    assert set(by_id) == {"agent-a", "agent-b"}
    assert by_id["agent-a"].action is None
    assert by_id["agent-a"].decision_status is DecisionStatus.FAILED
    assert by_id["agent-b"].action is not None  # sibling task not cancelled


@pytest.mark.asyncio
async def test_commit_execution_survives_one_agent_commit_crash(container) -> None:
    """Commit gather degrades per slot: one agent's begin exception doesn't cancel siblings or
    propagate; the failed slot only loses its record."""
    world_id = "world-commit-crash"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    work_a = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.WORK,
                         action_description="巡视宫防", estimated_steps=1)
    work_b = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.WORK,
                         action_description="批阅文书", estimated_steps=1)
    planned = [("main", a, _plan_for(a, work_a, env)), ("bg", b, _plan_for(b, work_b, env))]
    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001

    async def _boom(**kwargs):
        raise OSError("disk full")

    a.begin_ongoing_step = _boom  # type: ignore[method-assign]

    records = await rt._commit_execution(planned, arb, agents)  # noqa: SLF001
    assert [r["agent_id"] for r in records] == ["agent-b"]
    assert b.personality.state.action_status == ActionStatus.IN_PROGRESS


@pytest.mark.asyncio
async def test_land_result_feedback_isolates_per_call_failures(container) -> None:
    """Per-call feedback boundary: a failed finalize writeback must not swallow the target effects
    of the same result, let alone propagate."""
    world_id = "world-feedback-crash"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    async def _boom(**kwargs):
        raise RuntimeError("embedding 429")

    a.finalize_ongoing_action = _boom  # type: ignore[method-assign]
    applied: list[str] = []

    async def _spy(effect, *, from_agent_id, step):
        applied.append(effect.agent_id)

    b.apply_target_effect = _spy  # type: ignore[method-assign]

    from core.interfaces.action import ActionResult, TargetAgentEffect
    action = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
                         action_description="出手制止", target=ActionTarget(acts_on=[Ref.agent("agent-b")]))
    result = ActionResult(
        action=action, expected_outcome="制止对方", outcome="李世民出手制止了李建成。",
        succeeded=True,
        target_effects=[TargetAgentEffect(agent_id="agent-b", factual_memory="我被制止了")],
    )
    await rt._processor.land_result_feedback(result, agents, step=1)  # noqa: SLF001 — mustn't raise
    assert applied == ["agent-b"]


@pytest.mark.asyncio
async def test_a_declared_product_reaches_the_world(container) -> None:
    """The executor only declares and the feedback layer applies (Executor/Feedback boundary); this
    is the middle of that chain.

    What gets created is reachable by its author from the next step on, so "I finished it" becomes
    physical evidence instead of a private memory.
    """
    from core.interfaces.action import ActionResult, EntitySpawn

    world_id = "world-work-product"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="常何", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    action = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.WORK,
                         action_description="拟定换防部署")
    result = ActionResult(
        action=action, expected_outcome="定下换防", outcome="常何写完了换防部署令。",
        succeeded=True,
        entity_spawns=[EntitySpawn(
            name="换防部署令", description="圈定亲信的名单", holder_id="agent-a", is_public=False,
        )],
    )
    await rt._processor.land_result_feedback(result, agents, step=1)  # noqa: SLF001

    assert "换防部署令" in [e.name for e in env.all_live_entities()]
    assert [e.name for e in env.spatial_for(agent_id="agent-a").visible_entities] == ["换防部署令"]
    # Private property stays private through the feedback layer.
    assert env.spatial_for(agent_id="agent-b").visible_entities == []


@pytest.mark.asyncio
async def test_executor_complete_failure_releases_participants(container) -> None:
    """executor.complete raising → null step: no result, no writeback, participants reset to idle
    (never stuck forever)."""
    world_id = "world-complete-crash"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    agents = {"agent-a": a}
    rt = _arb_runtime(container, world_id, agents, env)

    class _FailingExecutor:
        async def complete(self, *args, **kwargs):
            raise RuntimeError("adjudication infra down")

    rt._executor_registry.register(ActionType.WORK, _FailingExecutor())  # noqa: SLF001
    exec_state = ActionExecutionState.create(
        target=ActionTarget(),
        action_type=ActionType.WORK, initiator_id="agent-a", participant_ids=["agent-a"],
        purpose="巡视宫防", started_step=1, estimated_steps=2, opening_outcome="",
    )
    rt._executor_registry.add_active(exec_state)  # noqa: SLF001
    a.personality.update_action_status(status=ActionStatus.IN_PROGRESS, current_action="巡视宫防", remaining_steps=1)

    results_by_exec = await rt._processor.finalize_executions_batch([exec_state], agents, step=2)  # noqa: SLF001
    assert results_by_exec.get(exec_state.execution_id) == []
    assert a.personality.state.action_status == ActionStatus.IDLE


@pytest.mark.asyncio
async def test_snapshot_save_failure_does_not_kill_run(container) -> None:
    """A failed end-of-step snapshot write (Rule 6) logs an error and continues; the run must not
    stop."""
    world_id = "world-snapshot-crash"
    runtime, environment, _ = _build_runtime(container, world_id=world_id)
    environment.place_agent(agent_id="agent-1", location_id="palace")
    agents = [_build_agent(container, world_id=world_id, agent_id="agent-1", name="李世民", is_main_character=True)]
    _set_work_decision(container)

    class _FailingSnapshot:
        async def save(self, *args, **kwargs):
            raise OSError("disk full")

    runtime._snapshot_provider = _FailingSnapshot()  # noqa: SLF001

    results = await runtime.run(agents, total_steps=2)
    assert len(results) == 2  # both steps completed; neither was killed by the snapshot failure


# ---------------------------------------------------------------------------
# Conscription joins don't depend on iteration order + the dead can't be conscripted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conscription_joins_target_planned_before_initiator(container) -> None:
    """When the conscripted agent's entry comes before the initiator's and it has no action this
    step (skipped by the walk), the join must still land. Otherwise it's a participant by identity
    (in participant_ids) but still IDLE, starts a second action next step, and does two things at
    once."""
    world_id = "world-arb-join-order"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                       action_description="商议对策", target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]), estimated_steps=2)
    # B comes first (its decision failed; when the walk reaches it, it isn't claimed yet and is
    # skipped), then A starts TALK→B.
    planned = [
        ("bg", b, _plan_for(b, None, env, decision_status=DecisionStatus.FAILED)),
        ("main", a, _plan_for(a, talk, env)),
    ]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    assert "agent-b" in arb and arb["agent-b"].is_passive_join is True
    records = await rt._commit_execution(planned, arb, agents)  # noqa: SLF001
    assert {r["agent_id"] for r in records} == {"agent-a", "agent-b"}
    # The join lands: B and A are both IN_PROGRESS, so B won't be scheduled for a second action next
    # step.
    assert b.personality.state.action_status == ActionStatus.IN_PROGRESS


@pytest.mark.asyncio
async def test_compel_takes_a_body_whose_own_action_was_admitted_earlier_this_step(container) -> None:
    """The person seized comes before the initiator and his own action was already admitted this
    step: that admission is retracted and he gets only a join record.

    Keep that admission and he has both a record of an action that never happened and a place in
    someone else's execution: commit would pass it to begin_ongoing_step and send the observer an
    action that never took place.
    """
    world_id = "world-arb-compel-order"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    # Locations must really be registered in the spatial graph: with only place_agent, MOVE fails
    # early with "no path", the compulsory-conscription case is never reached, and the test asserts
    # something always true.
    for entity_id, links in (("palace", {"garden": 1}), ("garden", {"palace": 1})):
        env.space.register_place(Place(
            place_id=entity_id, name=entity_id, description=entity_id, connections=dict(links),
        ))
    for aid in ("agent-a", "agent-b"):
        env.place_agent(agent_id=aid, location_id="palace")
    # The victim is a main character → earlier in initiative; the carrier is a background character
    # → later.
    victim = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    carrier = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    agents = {"agent-a": victim, "agent-b": carrier}
    rt = _arb_runtime(container, world_id, agents, env)

    own = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.WORK,
                      action_description="伏案理事", target=ActionTarget(), estimated_steps=4)
    carry = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.MOVE,
                        action_description="拽着他走", estimated_steps=1,
                        target=ActionTarget(acts_on=[Ref.place("garden")], claims=[Ref.agent("agent-a")]))
    planned = [
        ("main", victim, _plan_for(victim, own, env)),
        ("bg", carrier, _plan_for(carrier, carry, env)),
    ]

    arb, seized = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001

    assert arb["agent-a"].is_passive_join is True, "他只该有入伙记录"
    move = rt._executor_registry.get_active_for_agent("agent-b")  # noqa: SLF001
    assert move is not None and move.participant_ids == ["agent-b", "agent-a"]
    assert arb["agent-a"].ongoing_execution_id == move.execution_id, (
        "他的条目必须指向挟带的那个执行体"
    )
    # His desk work never started: no "interrupted just as it began" record; the intent carries over
    # to the next step via defer_decided_intent when he joins.
    assert seized == []
    # The join wording follows the mode: compulsion must not be phrased as "invited", which would
    # launder force into consent, and this text lands in the god-view feed, bystander perception and
    # his own memory.
    assert "强制拉入" in arb["agent-a"].action_result.outcome
    assert "邀入" not in arb["agent-a"].action_result.outcome
    # The join record carries the target too (same as action_description): the trip heads for a
    # destination and his turn is taken.
    joined_target = arb["agent-a"].action_result.action.target
    assert joined_target.acted_on_place == "garden"
    assert [r.id for r in joined_target.claims] == ["agent-a"]
    # After commit he's IN_PROGRESS alongside the carrier and won't start a second action next step.
    await rt._commit_execution(planned, arb, agents)  # noqa: SLF001
    assert victim.personality.state.action_status == ActionStatus.IN_PROGRESS
    # The discarded intent is backfilled as usual: only the passive-join branch calls
    # defer_decided_intent. If he got his own record he'd go through begin_ongoing_step and the
    # intent would silently vanish, and the perception signals behind this decision last only one
    # step, so it could never be recovered.
    assert any("伏案理事" in t for t in victim.memory_system.recent_foiled_attempts(1))


@pytest.mark.asyncio
async def test_a_seizure_tears_the_old_action_down_where_it_happened(container) -> None:
    """The forced teardown must happen before the seizer sets off, so the victim's action ends
    where he was, not on the road he's dragged onto.

    The last line of MOVE's start() moves the person. With teardown after it, the interrupt record's
    location would be the IN_TRANSIT pseudo-location (which flows into snapshots and the read
    model), and the first-person memory's scene anchor would point at the road he was just dragged
    onto.
    """
    world_id = "world-arb-seize-place"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    for entity_id, links in (("palace", {"garden": 3}), ("garden", {"palace": 3})):
        env.space.register_place(Place(
            place_id=entity_id, name=entity_id, description=entity_id, connections=dict(links),
        ))
    for aid in ("agent-a", "agent-b", "agent-c"):
        env.place_agent(agent_id=aid, location_id="palace")
    victim = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李建成", is_main_character=True)
    carrier = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李世民", is_main_character=False)
    partner = _build_agent(container, world_id=world_id, agent_id="agent-c", name="李渊", is_main_character=True)
    agents = {"agent-a": victim, "agent-b": carrier, "agent-c": partner}
    rt = _arb_runtime(container, world_id, agents, env)

    # The victim is talking with a third person; what's taken is that conversation, which belongs to
    # both.
    talk = AgentAction(agent_id="agent-c", step=1, action_type=ActionType.TALK,
                       action_description="命他退往东宫偏殿静思", estimated_steps=4,
                       target=ActionTarget(acts_on=[Ref.agent("agent-a")]))
    talk_state = await rt._executor_registry.get_executor(ActionType.TALK).start(  # noqa: SLF001
        talk, 1, agents=agents, environment=env, message_system=rt._message_system,  # noqa: SLF001
    )
    rt._executor_registry.add_active(talk_state)  # noqa: SLF001

    carry = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.MOVE,
                        action_description="强行带他前往玄武门", estimated_steps=3,
                        target=ActionTarget(acts_on=[Ref.place("garden")], claims=[Ref.agent("agent-a")]))
    # The two talking have no action of their own this step, but do have entries, as in the real
    # runtime (every living agent gets one), so the conscription can be recorded.
    busy = dict(decision_status=DecisionStatus.NOT_SCHEDULED)
    planned = [
        ("main", victim, _plan_for(victim, None, env, **busy)),
        ("main", partner, _plan_for(partner, None, env, **busy)),
        ("bg", carrier, _plan_for(carrier, carry, env)),
    ]

    _arb, seized = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001

    assert env.get_body_location("agent-a") == IN_TRANSIT, "拆完之后他才被带上路"
    assert {r["agent_id"] for r in seized} == {"agent-a", "agent-c"}
    for record in seized:
        assert record["location_id"] == "palace", "这场谈话终结在事发地,不在途中"
        assert "palace" in record["outcome"]


@pytest.mark.asyncio
async def test_a_rejected_move_never_sets_out(container) -> None:
    """If the collaborator can't be recorded, the trip is rejected before start(), and the world
    isn't changed at all.

    There's no "set off, then roll back", so no rollback step to lose. Once an execution is
    discarded nothing holds it, and if the person were already IN_TRANSIT nobody could move him.
    """
    world_id = "world-arb-abort"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    for entity_id, links in (("palace", {"garden": 3}), ("garden", {"palace": 3})):
        env.space.register_place(Place(
            place_id=entity_id, name=entity_id, description=entity_id, connections=dict(links),
        ))
    for aid in ("agent-a", "agent-b"):
        env.place_agent(agent_id=aid, location_id="palace")
    carrier = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    victim = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    agents = {"agent-a": carrier, "agent-b": victim}
    rt = _arb_runtime(container, world_id, agents, env)

    carry = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.MOVE,
                        action_description="拽着他走", estimated_steps=3,
                        target=ActionTarget(acts_on=[Ref.place("garden")], claims=[Ref.agent("agent-b")]))
    # The person carried off has no plan entry → arbitration rules "no response" → the action is
    # rejected before start() runs.
    planned = [("main", carrier, _plan_for(carrier, carry, env))]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001

    assert env.get_body_location("agent-a") == "palace", "被拒的行程不该把他留在途中"
    assert env.get_body_location("agent-b") == "palace"


@pytest.mark.asyncio
async def test_a_conscription_is_narrated_to_bystanders_exactly_once(container) -> None:
    """A joint action is one world event, and bystanders should perceive it once.

    The initiating execution already authorized the bystander narration and named both sides (a TALK
    opens with "甲与乙开始聊了起来"). A second "乙被甲邀入" from the join record makes bystanders read
    the same event twice with no new information; carry-side dedup keys on location + member set + text, so
    it catches the same sentence but not two sentences about the same event.

    It's also a boundary: bystander narration is authorized by executors, item by item. Arbitration
    isn't an executor and shouldn't write an action's narration.
    """
    world_id = "world-arb-once"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    for aid in ("agent-a", "agent-b"):
        env.place_agent(agent_id=aid, location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                       action_description="商议对策", target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]),
                       estimated_steps=2)
    planned = [
        ("main", a, _plan_for(a, talk, env)),
        ("bg", b, _plan_for(b, None, env, decision_status=DecisionStatus.NOT_SCHEDULED)),
    ]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001

    assert arb["agent-b"].is_passive_join is True
    assert arb["agent-b"].action_result.observations == [], "入伙记录不另发旁观叙述"
    # The only one comes from the initiating execution and names both sides.
    initiator_obs = arb["agent-a"].action_result.observations
    assert len(initiator_obs) == 1
    assert "李世民" in initiator_obs[0].text and "李建成" in initiator_obs[0].text
    # The 1p memory channel stays empty too: begin markers aren't written to memory, so filling it
    # would be dead data.
    assert arb["agent-b"].action_result.factual_memory == ""


@pytest.mark.asyncio
async def test_conscription_rejects_dead_co_participant(container) -> None:
    """A body that died earlier in the step is still visible in the environment until end-of-step
    cleanup; conscripting a corpse must be rejected by arbitration (no response)."""
    world_id = "world-arb-dead-target"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    b.set_active(False)  # died earlier this step, not yet removed from the environment
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    talk = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.TALK,
                       action_description="上前搭话", target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]), estimated_steps=2)
    planned = [("main", a, _plan_for(a, talk, env)), ("bg", b, _plan_for(b, None, env))]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    assert "agent-b" not in arb  # the corpse gets no arbitration entry
    a_exec = rt._executor_registry.get_active_for_agent("agent-a")  # noqa: SLF001
    assert a_exec is not None and a_exec.participant_ids == ["agent-a"]
    rejection = a_exec.extra["completed_result"]
    assert rejection.succeeded is False
    assert "毫无回应" in rejection.outcome


@pytest.mark.asyncio
async def test_executor_start_failure_skips_agent(container) -> None:
    """executor.start raising means the same as the executor-None branch: skip the agent (stays
    idle, replans next step) without cancelling others in the batch or propagating."""
    world_id = "world-start-crash"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="李建成", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    class _FailingStart:
        def claim_bodies(self, *args, **kwargs):
            return []

        async def start(self, *args, **kwargs):
            raise RuntimeError("start infra down")

    rt._executor_registry.register(ActionType.WORK, _FailingStart())  # noqa: SLF001
    work_a = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.WORK,
                         action_description="巡视宫防", estimated_steps=1)
    rest_b = AgentAction(agent_id="agent-b", step=1, action_type=ActionType.REST,
                         action_description="小憩", estimated_steps=1)
    planned = [("main", a, _plan_for(a, work_a, env)), ("bg", b, _plan_for(b, rest_b, env))]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001
    assert "agent-a" not in arb                      # skipped, no invented execution
    assert a.personality.state.action_status == ActionStatus.IDLE
    assert "agent-b" in arb                          # siblings unaffected


@pytest.mark.asyncio
async def test_executor_tick_failure_skips_narrative_only(container) -> None:
    """executor.tick raising only loses this step's progress narration: the execution stays
    registered, the count has advanced, and the next step proceeds."""
    world_id = "world-tick-crash"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    agents = {"agent-a": a}
    rt = _arb_runtime(container, world_id, agents, env)

    class _FailingTick:
        async def tick(self, *args, **kwargs):
            raise RuntimeError("tick infra down")

    rt._executor_registry.register(ActionType.WORK, _FailingTick())  # noqa: SLF001
    exec_state = ActionExecutionState.create(
        target=ActionTarget(),
        action_type=ActionType.WORK, initiator_id="agent-a", participant_ids=["agent-a"],
        purpose="巡视宫防", started_step=1, estimated_steps=3, opening_outcome="",
    )
    rt._executor_registry.add_active(exec_state)  # noqa: SLF001
    a.personality.update_action_status(status=ActionStatus.IN_PROGRESS, current_action="巡视宫防", remaining_steps=2)

    records = await rt._processor.tick_ongoing_executions(agents, step=2)  # noqa: SLF001 — mustn't raise
    assert records == []
    assert rt._executor_registry.is_agent_active("agent-a")  # noqa: SLF001 — still registered
    assert exec_state.remaining_steps == 1                   # count advanced


@pytest.mark.asyncio
async def test_multistep_move_emits_transit_contract_in_step_event(container) -> None:
    """A weight-2 move exposes {from,to,elapsed,total} on the mover's step state.

    The render-side transit contract: while an agent is IN_TRANSIT, the step event carries its
    origin/destination + progress so a 2D renderer animates the walk ALONG the edge (see
    engine.runtime + AgentStateSummary). The edge is built here, a two-hour walk under the default
    registry's one-hour step, independent of any map's geometry.
    """
    published = tap(container.event_bus)
    world_id = "world-transit"
    runtime, environment, _ = _build_runtime(container, world_id=world_id)
    for entity_id, name, links in (
        ("near_gate", "近门", {"far_gate": 2 * 3600}),
        ("far_gate", "远门", {"near_gate": 2 * 3600}),
    ):
        environment.space.register_place(Place(
            place_id=entity_id, name=name, description=name, connections=dict(links),
        ))
    environment.place_agent(agent_id="agent-1", location_id="near_gate")
    agent = _build_agent(
        container, world_id=world_id, agent_id="agent-1", name="Li Shimin", is_main_character=True,
    )

    async def plan_move(*, step, spatial, inbox, broadcasts, **_kw):
        need = NeedEvaluation(
            dominant_need=NeedType.SAFETY,
            scores={NeedType.SAFETY: 1.0},
            active_needs=[NeedState(NeedType.SAFETY, "safety", 1.0)],
            short_term_goals=["reach the gate"],
            long_term_goals=[],
            prompt_context="safety",
        )
        return AgentStepPlan(
            agent_id=agent.agent_id, step=step, spatial=spatial,
            inbox=list(inbox), broadcasts=list(broadcasts), need_evaluation=need,
            action=AgentAction(
                agent_id=agent.agent_id, step=step, action_type=ActionType.MOVE,
                action_description="前往远门",
                target=ActionTarget(acts_on=[Ref.place("far_gate")]),
            ),
        )

    agent.plan_step = plan_move

    # Step 1: the move starts; the agent is still travelling → transit present.
    await runtime.run_step([agent])
    event = published()[0]
    transit = event["agent_states"]["agent-1"].get("transit")
    assert transit is not None, "in-transit agent must carry the transit contract"
    assert transit["from_location_id"] == "near_gate"
    assert transit["to_location_id"] == "far_gate"
    assert transit["total_steps"] == 2
    assert 0 <= transit["elapsed_steps"] < transit["total_steps"]  # not yet arrived
    assert environment.get_body_location("agent-1") == IN_TRANSIT

    # Step 2: arrival — no more transit, landed at the destination.
    agent.plan_step = plan_move  # ignored while the executor owns the agent
    await runtime.run_step([agent])
    event2 = published()[0]
    assert event2["agent_states"]["agent-1"].get("transit") is None
    assert event2["agent_states"]["agent-1"]["arrival"]["path"] == ["near_gate", "far_gate"]
    assert environment.get_body_location("agent-1") == "far_gate"
    # The step after landing carries no trip at all.
    await runtime.run_step([agent])
    assert "arrival" not in published()[0]["agent_states"]["agent-1"]


@pytest.mark.asyncio
async def test_a_trip_crossed_within_one_step_carries_its_route_as_an_arrival(container) -> None:
    """Three short hops under the default one-hour step land in one step, with no transit ever;
    the arrival gives the renderer the waypoints the engine reported him passing."""
    published = tap(container.event_bus)
    world_id = "world-arrival"
    runtime, environment, _ = _build_runtime(container, world_id=world_id)
    for entity_id, links in (
        ("gate", {"yard": 600}),
        ("yard", {"gate": 600, "hall": 600}),
        ("hall", {"yard": 600}),
    ):
        environment.space.register_place(Place(
            place_id=entity_id, name=entity_id, description=entity_id, connections=dict(links),
        ))
    environment.place_agent(agent_id="agent-1", location_id="gate")
    agent = _build_agent(
        container, world_id=world_id, agent_id="agent-1", name="Li Shimin", is_main_character=True,
    )

    async def plan_move(*, step, spatial, inbox, broadcasts, **_kw):
        return AgentStepPlan(
            agent_id=agent.agent_id, step=step, spatial=spatial,
            inbox=list(inbox), broadcasts=list(broadcasts),
            need_evaluation=NeedEvaluation(
                dominant_need=NeedType.SAFETY, scores={NeedType.SAFETY: 1.0},
                active_needs=[NeedState(NeedType.SAFETY, "safety", 1.0)],
                short_term_goals=[], long_term_goals=[], prompt_context="safety",
            ),
            action=AgentAction(
                agent_id=agent.agent_id, step=step, action_type=ActionType.MOVE,
                action_description="去大殿",
                target=ActionTarget(acts_on=[Ref.place("hall")]),
            ),
        )

    agent.plan_step = plan_move
    await runtime.run_step([agent])
    state = published()[0]["agent_states"]["agent-1"]
    assert state.get("transit") is None
    assert state["arrival"]["path"] == ["gate", "yard", "hall"]
    assert environment.get_body_location("agent-1") == "hall"


# ---------------------------------------------------------------------------
# Same-step two-phase completion: concurrent adjudication + deterministic grouped parallel
# settlement
# ---------------------------------------------------------------------------


def _add_work_exec(rt, agent_id: str):
    es = ActionExecutionState.create(
        target=ActionTarget(),
        action_type=ActionType.WORK, initiator_id=agent_id, participant_ids=[agent_id],
        purpose="巡视", started_step=1, estimated_steps=1, opening_outcome="",
    )
    rt._executor_registry.add_active(es)  # noqa: SLF001
    return es


class _BarrierExecutor(ActionExecutor):
    """Test double: complete() blocks on a shared gate until ``expected`` calls arrive, recording
    peak concurrency. Proves Phase A adjudicates concurrently (if serial, the first call never sees
    the gate fill, times out, and peak==1)."""

    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.inflight = 0
        self.peak = 0
        self._gate = asyncio.Event()

    async def start(self, *a, **k):  # pragma: no cover - start isn't used
        raise NotImplementedError

    async def tick(self, *a, **k):
        return []

    async def complete(self, state, step, **kwargs):
        from core.interfaces.action import ActionResult
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        if self.inflight >= self.expected:
            self._gate.set()
        try:
            await asyncio.wait_for(self._gate.wait(), timeout=2.0)
        except asyncio.TimeoutError:  # pragma: no cover - only when it degrades to serial
            pass
        self.inflight -= 1
        stub = AgentAction(agent_id=state.initiator_id, step=step,
                           action_type=state.action_type, action_description=state.purpose)
        return [ActionResult(action=stub, expected_outcome="", outcome="done",
                             observations=[Observed(location_id="palace", text="done")], succeeded=True)]


@pytest.mark.asyncio
async def test_independent_adjudication_runs_concurrently(container) -> None:
    """Executions that share no agents → fully concurrent adjudication: peak concurrency == N."""
    world_id = "world-parallel-adj"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    n = 4
    agents = {}
    for i in range(n):
        aid = f"agent-{i}"
        env.place_agent(agent_id=aid, location_id="palace")
        agents[aid] = _build_agent(container, world_id=world_id, agent_id=aid, name=f"A{i}", is_main_character=False)
    rt = _arb_runtime(container, world_id, agents, env)
    barrier = _BarrierExecutor(expected=n)
    rt._executor_registry.register(ActionType.WORK, barrier)  # noqa: SLF001

    async def _noop_land(result, agents_dict, step):  # isolate adjudication
        pass
    rt._processor.land_result_feedback = _noop_land  # type: ignore[method-assign]  # noqa: SLF001

    exec_states = [_add_work_exec(rt, f"agent-{i}") for i in range(n)]
    results = await rt._processor.finalize_executions_batch(exec_states, agents, step=1)  # noqa: SLF001

    assert barrier.peak == n  # all N executions are in complete() at once → truly concurrent
    assert all(len(results[es.execution_id]) == 1 for es in exec_states)


@pytest.mark.asyncio
async def test_adjudication_exception_isolated(container) -> None:
    """One execution's complete() raising doesn't cancel siblings; the failed one is a null step and
    its participants reset to idle."""
    world_id = "world-adj-isolation"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    ids = ("agent-a", "agent-b", "agent-c")
    agents = {}
    for aid in ids:
        env.place_agent(agent_id=aid, location_id="palace")
        agents[aid] = _build_agent(container, world_id=world_id, agent_id=aid, name=aid, is_main_character=False)
    rt = _arb_runtime(container, world_id, agents, env)
    from core.interfaces.action import ActionResult

    class _MixedExecutor(ActionExecutor):
        async def start(self, *a, **k):  # pragma: no cover
            raise NotImplementedError

        async def tick(self, *a, **k):
            return []

        async def complete(self, state, step, **kwargs):
            if state.initiator_id == "agent-b":
                raise RuntimeError("adjudication infra down")
            stub = AgentAction(agent_id=state.initiator_id, step=step,
                               action_type=state.action_type, action_description=state.purpose)
            return [ActionResult(action=stub, expected_outcome="", outcome="ok",
                                 observations=[Observed(location_id="palace", text="ok")], succeeded=True)]

    rt._executor_registry.register(ActionType.WORK, _MixedExecutor())  # noqa: SLF001

    async def _noop_land(result, agents_dict, step):
        pass
    rt._processor.land_result_feedback = _noop_land  # type: ignore[method-assign]  # noqa: SLF001

    exec_states = []
    for aid in ids:
        es = _add_work_exec(rt, aid)
        agents[aid].personality.update_action_status(
            status=ActionStatus.IN_PROGRESS, current_action="巡视", remaining_steps=0)
        exec_states.append(es)

    results = await rt._processor.finalize_executions_batch(exec_states, agents, step=1)  # noqa: SLF001

    assert len(results[exec_states[0].execution_id]) == 1   # agent-a fine
    assert results[exec_states[1].execution_id] == []        # agent-b failed → null step
    assert len(results[exec_states[2].execution_id]) == 1   # agent-c fine
    assert agents["agent-b"].personality.state.action_status == ActionStatus.IDLE


@pytest.mark.asyncio
async def test_same_target_apply_serialized_no_lost_update(container) -> None:
    """Two actions hit the same target in one step: grouping puts them together and lands them
    serially → vitality drops by the sum (no lost update). Verifies apply groups merge by touched
    set and land deterministically in series."""
    world_id = "world-same-target"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    for aid in ("agent-a", "agent-b", "victim"):
        env.place_agent(agent_id=aid, location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="A", is_main_character=False)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="B", is_main_character=False)
    victim = _build_agent(container, world_id=world_id, agent_id="victim", name="V", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b, "victim": victim}
    rt = _arb_runtime(container, world_id, agents, env)
    from core.interfaces.action import ActionResult, ActionTarget, TargetAgentEffect

    def _hit(actor: str) -> ActionResult:
        stub = AgentAction(agent_id=actor, step=1, action_type=ActionType.PHYSICAL, action_description="击打",
                           target=ActionTarget(acts_on=[Ref.agent("victim")]))
        return ActionResult(action=stub, expected_outcome="", outcome="击中", observations=[Observed(location_id="palace", text="击中")], succeeded=True,
                            target_effects=[TargetAgentEffect(agent_id="victim", factual_memory="被击", vitality_damage=0.3)])

    class _HitExecutor(ActionExecutor):
        async def start(self, *a, **k):  # pragma: no cover
            raise NotImplementedError

        async def tick(self, *a, **k):
            return []

        async def complete(self, state, step, **kwargs):
            return [_hit(state.initiator_id)]

    rt._executor_registry.register(ActionType.PHYSICAL, _HitExecutor())  # noqa: SLF001

    async def _noop_finalize(**kwargs):  # only test the target landing
        pass
    a.finalize_ongoing_action = _noop_finalize  # type: ignore[method-assign]
    b.finalize_ongoing_action = _noop_finalize  # type: ignore[method-assign]

    es_list = []
    for actor in ("agent-a", "agent-b"):
        es = ActionExecutionState.create(
            target=ActionTarget(),
            action_type=ActionType.PHYSICAL, initiator_id=actor, participant_ids=[actor],
            purpose="击打", started_step=1, estimated_steps=1, opening_outcome="")
        rt._executor_registry.add_active(es)  # noqa: SLF001
        es_list.append(es)

    v0 = victim.personality.state.vitality
    await rt._processor.finalize_executions_batch(es_list, agents, step=1)  # noqa: SLF001
    # Both 0.3 hits land (serial within the group, no lost update) → exactly -0.6
    assert victim.personality.state.vitality == pytest.approx(v0 - 0.6, abs=1e-6)


class _PairExecutor(ActionExecutor):
    """One execution yields a result per participant; with ``hit_partner`` the initiator's result
    has a target effect on the other."""

    def __init__(self, *, hit_partner: bool) -> None:
        self.hit_partner = hit_partner

    async def start(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    async def tick(self, *a, **k):
        return []

    async def complete(self, state, step, **kwargs):
        from core.interfaces.action import ActionResult, TargetAgentEffect

        initiator, partner = state.participant_ids

        def _result(agent_id, effects):
            stub = AgentAction(agent_id=agent_id, step=step, action_type=state.action_type,
                               action_description=state.purpose)
            return ActionResult(action=stub, expected_outcome="", outcome="done",
                                observations=[Observed(location_id="palace", text="done")],
                                succeeded=True, target_effects=effects)

        effects = [TargetAgentEffect(agent_id=partner, factual_memory="被推了一把")] if self.hit_partner else []
        return [_result(initiator, effects), _result(partner, [])]


async def _land_pair(container, world_id: str, *, hit_partner: bool) -> tuple[int, list[str]]:
    """Land a two-person execution and return (peak landing concurrency, landing order)."""
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    agents = {}
    for aid in ("agent-a", "agent-b"):
        env.place_agent(agent_id=aid, location_id="palace")
        agents[aid] = _build_agent(container, world_id=world_id, agent_id=aid, name=aid, is_main_character=False)
    rt = _arb_runtime(container, world_id, agents, env)
    rt._executor_registry.register(ActionType.WORK, _PairExecutor(hit_partner=hit_partner))  # noqa: SLF001

    both_in_flight = asyncio.Event()
    inflight = 0
    peak = 0
    landed: list[str] = []

    async def _barrier_land(result, agents_dict, step):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        if inflight == 2:
            both_in_flight.set()
        try:
            await asyncio.wait_for(both_in_flight.wait(), timeout=0.2)
        except asyncio.TimeoutError:
            pass
        landed.append(result.action.agent_id)
        inflight -= 1
    rt._processor.land_result_feedback = _barrier_land  # type: ignore[method-assign]  # noqa: SLF001

    es = ActionExecutionState.create(
        target=ActionTarget(),
        action_type=ActionType.WORK, initiator_id="agent-a", participant_ids=["agent-a", "agent-b"],
        purpose="一起巡视", started_step=1, estimated_steps=1, opening_outcome="",
    )
    rt._executor_registry.add_active(es)  # noqa: SLF001
    await rt._processor.finalize_executions_batch([es], agents, step=1)  # noqa: SLF001
    return peak, landed


@pytest.mark.asyncio
async def test_same_execution_disjoint_results_land_concurrently(container) -> None:
    """Two participants of one execution each touch only themselves → both results land at once
    without waiting on each other."""
    peak, landed = await _land_pair(container, "world-pair-disjoint", hit_partner=False)
    assert peak == 2
    assert sorted(landed) == ["agent-a", "agent-b"]


@pytest.mark.asyncio
async def test_same_execution_shared_key_results_land_in_order(container) -> None:
    """The initiator's result has a target effect on the other → the two share a conflict key and
    land serially; the target's own result lands first."""
    peak, landed = await _land_pair(container, "world-pair-shared", hit_partner=True)
    assert peak == 1
    assert landed == ["agent-b", "agent-a"]


def _landing_result(actor: str, *hit: str) -> ActionResult:
    from core.interfaces.action import TargetAgentEffect

    stub = AgentAction(agent_id=actor, step=1, action_type=ActionType.PHYSICAL, action_description="x")
    return ActionResult(action=stub, expected_outcome="", outcome="x", succeeded=True,
                        target_effects=[TargetAgentEffect(agent_id=t, factual_memory="x") for t in hit])


def test_landing_order_puts_effects_on_someone_after_his_own_result() -> None:
    """A stabs B, B stabs C, C just talks: each person's own result lands before effects on him,
    regardless of input order."""
    members = [("e1", _landing_result("A", "B")), ("e2", _landing_result("B", "C")),
               ("e3", _landing_result("C"))]
    order = [r.action.agent_id for _e, r in ExecutionProcessor._landing_order(members)]  # noqa: SLF001
    assert order == ["C", "B", "A"]


def test_landing_order_keeps_input_order_when_nobody_is_hit() -> None:
    members = [("e1", _landing_result("A")), ("e2", _landing_result("B"))]
    order = [r.action.agent_id for _e, r in ExecutionProcessor._landing_order(members)]  # noqa: SLF001
    assert order == ["A", "B"]


def test_landing_order_falls_back_to_input_order_on_mutual_hits() -> None:
    """Mutual stabbing in a cycle in one step: no deadlock, each lands exactly once, in input order
    within the cycle."""
    members = [("e1", _landing_result("A", "B")), ("e2", _landing_result("B", "A")),
               ("e3", _landing_result("C", "A"))]
    order = [r.action.agent_id for _e, r in ExecutionProcessor._landing_order(members)]  # noqa: SLF001
    assert order == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_adjudication_failure_never_reaches_the_observer(container, caplog) -> None:
    """A failed adjudication (judge LLM down / output truncated into unparsable JSON) is dropped
    entirely from the observer stream.

    Nothing happened in the world: the feedback layer already treats it as a null step and the
    ambient carry skips it; this guards the last exit, the god-view actions channel. It must be
    dropped, not flagged: the display only knows real failure and foiled, so succeeded=False would
    be drawn as a real defeat (the "未果" label + a red ✗ on the map). It appears in bulk whenever a
    vendor hiccups.
    """
    published = tap(container.event_bus)
    from core.interfaces.llm import LLMScene

    world_id = "world-adj-fail"
    runtime, environment, _ = _build_runtime(container, world_id=world_id)
    environment.place_agent(agent_id="agent-1", location_id="palace")
    agents = [
        _build_agent(container, world_id=world_id, agent_id="agent-1",
                     name="Li Shimin", is_main_character=True),
    ]
    _set_work_decision(container)
    # The adjudication scene returns non-JSON → extract_json raises → the judge returns None →
    # adjudication_failed=True. This is exactly what max_tokens truncation looks like in production,
    # not an artificial branch.
    container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = "not json at all"

    import logging
    with caplog.at_level(logging.WARNING, logger="engine.runtime"):
        result = await runtime.run_step(agents)
    event = published()[0]
    snapshot = await container.snapshot.load(world_id, 1)

    assert event["actions"] == [], "空步不得出现在叙事流/地图的 actions 通道"
    assert result.action_summaries == [], "空步不得出现在观察者摘要"
    # The snapshot shares the source (actions_this_step comes from the same filtered list) → replay
    # and event briefings are clean too.
    assert snapshot is not None and snapshot.actions_this_step == []
    # The step did run (agent scheduled, decision succeeded); it just left no observable event in
    # the world.
    assert result.scheduled_agent_ids == ["agent-1"]

    # Something removed from every observable surface must leave exactly one trace, or operators
    # only see "the world seems frozen" with no hint that a vendor is flaking. This assertion
    # protects that trace from being quietly removed.
    withheld = [r for r in caplog.records if r.message == "null_steps_withheld_from_observer"]
    assert len(withheld) == 1, "空步被抹掉时必须留下且只留下一条运维痕迹"
    assert withheld[0].dropped == 1 and withheld[0].agent_ids == ["agent-1"]


@pytest.mark.asyncio
async def test_executor_crash_on_a_born_zero_act_never_reaches_the_observer(
    container, caplog, monkeypatch,
) -> None:
    """complete() raising: no results at all, a null step; this step's opening record must not reach
    the observer stream either."""
    published = tap(container.event_bus)
    world_id = "world-exec-crash"
    runtime, environment, _ = _build_runtime(container, world_id=world_id)
    environment.place_agent(agent_id="agent-1", location_id="palace")
    agents = [
        _build_agent(container, world_id=world_id, agent_id="agent-1",
                     name="Li Shimin", is_main_character=True),
    ]
    _set_work_decision(container)

    async def crash(*args, **kwargs):
        raise RuntimeError("executor down")

    monkeypatch.setattr(runtime._executor_registry.get_executor(ActionType.WORK), "complete", crash)

    import logging
    with caplog.at_level(logging.WARNING, logger="engine.runtime"):
        result = await runtime.run_step(agents)
    event = published()[0]

    assert event["actions"] == []
    assert result.action_summaries == []
    withheld = [r for r in caplog.records if r.message == "null_steps_withheld_from_observer"]
    assert len(withheld) == 1 and withheld[0].agent_ids == ["agent-1"]


@pytest.mark.asyncio
async def test_a_missing_executor_releases_its_participants(container, monkeypatch) -> None:
    """No judge = no result, the same null step as complete() raising: participants must be reset,
    or once the execution leaves the registry they stay IN_PROGRESS forever and never re-enter the
    decision loop."""
    world_id = "world-exec-missing"
    runtime, environment, _ = _build_runtime(container, world_id=world_id)
    environment.place_agent(agent_id="agent-1", location_id="palace")
    agent = _build_agent(container, world_id=world_id, agent_id="agent-1",
                         name="Li Shimin", is_main_character=True)
    exec_state = ActionExecutionState.create(
        target=ActionTarget(), action_type=ActionType.WORK, initiator_id=agent.agent_id,
        participant_ids=[agent.agent_id], purpose="审阅手诏",
        started_step=1, estimated_steps=1, opening_outcome="",
    )
    runtime._executor_registry.add_active(exec_state)  # noqa: SLF001
    agent.personality.update_action_status(
        status=ActionStatus.IN_PROGRESS, current_action="审阅手诏", remaining_steps=0,
    )
    monkeypatch.setattr(runtime._executor_registry, "get_executor", lambda _t: None)  # noqa: SLF001

    results = await runtime._processor._adjudicate_execution(  # noqa: SLF001
        exec_state, {agent.agent_id: agent}, step=1,
    )

    assert results == []
    assert agent.personality.state.action_status == ActionStatus.IDLE


@pytest.mark.asyncio
async def test_elsewhere_observations_are_routed_by_the_single_owner(container) -> None:
    """When one step is visible in several places, the "elsewhere" copy takes the same delivery
    path as the record's own.

    Contract: the executor only declares (ActionResult.observations, one per location) and the
    runtime routes them. An executor calling environment.record_carry_observation directly would bypass the
    dedup, null-step filter and IN_TRANSIT filter here, which crosses the boundary.

    The critical case is the second part: on the first step of a multi-hop move the record's
    location is IN_TRANSIT (he's mid-edge). If IN_TRANSIT were checked on the record rather than
    on each landing location, the whole record would be dropped along with the trace it declares at
    the origin, in exactly the case that needs it most.
    """
    from engine.clock import WorldTime
    world_id = "world-elsewhere"
    _, environment, _ = _build_runtime(container, world_id=world_id)
    runtime, environment, _ = _build_runtime(container, world_id=world_id)
    environment.place_agent(agent_id="mover", location_id="palace")
    environment.place_agent(agent_id="stayer", location_id="palace")

    runtime._carry_step_observations(  # noqa: SLF001
        agent_records=[{
            "agent_id": "mover", "action_type": ActionType.MOVE,
            "location_id": IN_TRANSIT,
            "observations": [
                # He's already mid-edge: this one can't be carried (nobody is "on the road")...
                {"location_id": IN_TRANSIT, "text": "在途中，甲赶路。"},
                # ...but those he left at the origin did see him go: same step, two places, two
                # different lines.
                {"location_id": "palace", "text": "在palace，甲离开此地，往garden方向去了。"},
            ],
            "adjudication_failed": False,
        }],
        tick_records=[],
    )
    environment.begin_step(step=2, world_time=WorldTime(step=2, elapsed_seconds=7200))

    stayed = "".join(e.content for e in environment.spatial_for(agent_id="stayer").ambient_events)
    assert "离开此地" in stayed, "记录落在 IN_TRANSIT 不该连累它声明的别处痕迹"
    assert "赶路" not in stayed, "IN_TRANSIT 那份本身仍不可 carry（没人在「途中」）"

    # Self-exclusion: the actor shouldn't read third-person text about itself as an external signal.
    assert "离开此地" not in "".join(
        e.content for e in environment.spatial_for(agent_id="mover").ambient_events
    )


@pytest.mark.asyncio
async def test_setting_out_moves_the_anchor_with_the_body(container) -> None:
    """The person is moved on the step he sets off, so the anchor must move too, or the rest of the
    step's cognition says he's still at the start.

    Donggong to Taiji Palace goes round by Xuanwu Gate: about 46 and 20 minutes, two 40-minute
    steps, so the first ``start()`` leaves him at the gate. Perception runs before the execution
    phase; without updating the anchor it stays at Donggong, and the maintenance-phase long-term
    goal review reads it.
    """
    world_id = "world-set-out"
    seconds_per_step = 40 * 60
    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    message_system = MessageSystem(container.message_provider, world_id=world_id)
    directory = LiveWorldDirectory.from_agents({}, environment)
    broadcast_channel = BroadcastChannel()
    runtime = NarrativeRuntime(
        world_id=world_id,
        clock=GlobalClock(WorldTimeConfig(start_hour=6, seconds_per_step=seconds_per_step)),
        scheduler=AgentScheduler(),
        environment=environment,
        message_system=message_system,
        event_settings=EventSettings(check_interval=2, max_events_per_window=0),
        snapshot_provider=container.snapshot,
        agent_store=container.agent_store,
        event_bus=container.event_bus,
        broadcast_channel=broadcast_channel,
        directory=directory,
        executor_registry=build_default_registry(
            container.llm_router, directory, seconds_per_step=seconds_per_step,
        ),
        llm_router=container.llm_router,
    )
    environment.place_agent(agent_id="agent-1", location_id="donggong")
    agent = _build_agent(container, world_id=world_id, agent_id="agent-1",
                         name="Li Jiancheng", is_main_character=True)

    async def plan_move(*, step, spatial, inbox, broadcasts, **_kw):
        return AgentStepPlan(
            agent_id=agent.agent_id, step=1, spatial=spatial,
            inbox=list(inbox), broadcasts=list(broadcasts),
            need_evaluation=NeedEvaluation(
                dominant_need=NeedType.SAFETY, scores={NeedType.SAFETY: 1.0},
                active_needs=[NeedState(NeedType.SAFETY, "safety", 1.0)],
                short_term_goals=[], long_term_goals=[], prompt_context="safety",
            ),
            action=AgentAction(
                agent_id=agent.agent_id, step=1, action_type=ActionType.MOVE,
                action_description="去太极宫",
                target=ActionTarget(acts_on=[Ref.place("taiji_palace")]),
            ),
        )

    agent.plan_step = plan_move
    await runtime.run_step([agent])

    where = environment.get_body_location(agent.agent_id)
    assert where == "xuanwu_gate", "两步的路，第一拍应当停在途经的玄武门"
    assert agent._situation.location_view == environment.location_view(where)


# ─────────────────────────────────────────────────────────────────────────────
# Authoritative history: which records qualify for the lurker's material track
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_happenings_track_admits_only_what_was_authorized(container) -> None:
    """Only the executor's authorization counts; nothing is inferred from outcome.

    Inferring has only two options, both silent failures: a list by type (new types leak by default)
    or text matching (breaks when a template changes). So this reads only ``happening``; anything
    undeclared stays out, even with a full outcome (a private message's text, a non-event, an
    unexposed covert action's intent).
    """
    world_id = "world-auth-admit"
    rt, env, _ = _build_runtime(container, world_id=world_id)

    rt._record_step_happenings(  # noqa: SLF001
        agent_records=[
            # Declared: others only see him at his desk; the result exists only in this layer.
            {"agent_id": "worker", "action_type": ActionType.WORK, "location_id": "hall",
             "outcome": "在hall，甲拟就奏章，力陈三策。",
             "happening": "在hall，甲拟就奏章，力陈三策。",
             "observations": [{"location_id": "hall", "text": "在hall，甲伏案忙了许久。"}],
             "adjudication_failed": False},
            # The next three all have an outcome but no declaration; none should get in.
            # Message: the outcome holds the full text.
            {"agent_id": "sender", "action_type": ActionType.SEND_MESSAGE, "location_id": "hall",
             "outcome": "在hall，乙向不在场的丙传话：「今夜子时」。", "observations": [],
             "adjudication_failed": False},
            # Non-event: nothing happened in the world.
            {"agent_id": "foiled", "action_type": ActionType.TALK, "location_id": "hall",
             "outcome": "丁本想做「找戊说话」，却因对方不在场未能如愿。", "observations": [],
             "not_executed": True, "adjudication_failed": False},
            # Unexposed covert action: the outcome states its intent.
            {"agent_id": "spy", "action_type": ActionType.COVERT, "location_id": "hall",
             "outcome": "在hall，己正秘密进行「翻找遗嘱」。", "observations": [],
             "adjudication_failed": False, "detected": False},
        ],
        tick_records=[],
        step=1,
    )

    material = [c for _, c in env.recent_happenings("hall", since_step=0)]
    assert material == ["在hall，甲拟就奏章，力陈三策。"]
    joined = "".join(material)
    assert "今夜子时" not in joined, "私信原文不得进材料轨"
    assert "未能如愿" not in joined, "非事件不得被当成情报"
    assert "翻找遗嘱" not in joined, "未暴露 covert 的秘密意图绝不得交给另一个潜伏者"


@pytest.mark.asyncio
async def test_happenings_track_drops_an_exposed_covert_it_adds_nothing(container) -> None:
    """An exposed lurker stays out of the observation track: it adds no information but is a
    real leak path.

    A lurker here already got the exposure ambient, and a covert observation is that outcome with a
    prefix. But if the judge writes extra findings into another lurker's outcome, they'd flow
    straight into this person's adjudication context: "outcome only describes visible traces" is a
    prompt convention, and this check is the code guarantee.
    """
    world_id = "world-auth-exposed"
    rt, env, _ = _build_runtime(container, world_id=world_id)

    rt._record_step_happenings(  # noqa: SLF001
        agent_records=[
            {"agent_id": "spy", "action_type": ActionType.COVERT, "location_id": "hall",
             "outcome": "在hall，己在帘后窥探，被人瞧见了。",
             # Covert actions never declare this layer (see CovertExecutor), exposed or not.
             "observations": [{"location_id": "hall",
                               "text": "[秘密行动暴露] 在hall，己在帘后窥探，被人瞧见了。"}],
             "adjudication_failed": False, "detected": True},
        ],
        tick_records=[],
        step=1,
    )
    assert env.recent_happenings("hall", since_step=0) == [], (
        "旁观者已看到同样多的事,见闻轨不该再存一份"
    )


@pytest.mark.asyncio
async def test_happenings_track_excludes_participants_and_overhearers(container) -> None:
    """The exclusion set is "people who already hold this": participants and those who overheard
    it.

    Listeners are kept out of participant_ids for turn-economy reasons, not cognitive ones:
    overhearing already gave them a first-person memory, so this material isn't something they
    otherwise couldn't get.
    """
    world_id = "world-auth-exclude"
    rt, env, _ = _build_runtime(container, world_id=world_id)

    rt._record_step_happenings(  # noqa: SLF001
        agent_records=[
            {"agent_id": "talker-a", "action_type": ActionType.TALK, "location_id": "hall",
             "outcome": "在hall，甲对乙说：「三日后动手」。",
             "happening": "在hall，甲对乙说：「三日后动手」。",
             "observations": [{"location_id": "hall", "text": "在hall，甲乙谈了一场。"}],
             "participant_ids": ["talker-a", "talker-b"],
             "overheard_by": ["listener"],
             "adjudication_failed": False},
        ],
        tick_records=[],
        step=1,
    )

    assert env.recent_happenings("hall", since_step=0, exclude_ids=("outsider",))
    assert env.recent_happenings("hall", since_step=0, exclude_ids=("talker-b",)) == []
    assert env.recent_happenings("hall", since_step=0, exclude_ids=("listener",)) == []


@pytest.mark.asyncio
async def test_happenings_track_rejects_pseudo_places(container) -> None:
    """Travelers would treat IN_TRANSIT as one shared room, and UNPLACED isn't a place at all. Both
    must be excluded."""
    from engine.environment import UNPLACED

    world_id = "world-auth-pseudo"
    rt, env, _ = _build_runtime(container, world_id=world_id)

    rt._record_step_happenings(  # noqa: SLF001
        agent_records=[
            {"agent_id": "mover", "action_type": ActionType.MOVE, "location_id": IN_TRANSIT,
             "outcome": "在途中，甲赶路。",
             "observations": [{"location_id": "hall", "text": "在hall，甲离开此地。"}],
             "adjudication_failed": False},
            {"agent_id": "ghost", "action_type": ActionType.WORK, "location_id": UNPLACED,
             "outcome": "不知何处，乙做了件事。",
             "observations": [{"location_id": "hall", "text": "在hall，乙做了件事。"}],
             "adjudication_failed": False},
        ],
        tick_records=[],
        step=1,
    )

    assert env.recent_happenings(IN_TRANSIT, since_step=0) == []
    assert env.recent_happenings(UNPLACED, since_step=0) == []


@pytest.mark.asyncio
async def test_covert_material_never_includes_the_current_step(container) -> None:
    """Guards the ordering invariant.

    All of a step's adjudication runs before ``_carry_step_observations`` (in run_step: interrupts →
    tick completion → same-step sweep → this call), and that order is what keeps a lurker's material
    window from seeing its own step. Moving the carry earlier (it only needs the two record lists,
    so it looks like a harmless refactor) would let COVERT read lines produced in its own step,
    cheating on time, and no other test would fail. This one is that gate.
    """
    world_id = "world-auth-timing"
    rt, env, _ = _build_runtime(container, world_id=world_id)

    record = {
        "agent_id": "worker", "action_type": ActionType.WORK, "location_id": "hall",
        "outcome": "在hall，甲把那件事办成了。",
        "happening": "在hall，甲把那件事办成了。",
        "observations": [{"location_id": "hall", "text": "在hall，甲忙了一阵。"}],
        "adjudication_failed": False,
    }
    # Adjudication happens before the write → the window has nothing from this step yet.
    assert env.recent_happenings("hall", since_step=0) == []
    rt._record_step_happenings(agent_records=[record], tick_records=[], step=5)  # noqa: SLF001
    # It exists only after the write, belonging to step 5: only a lurk starting at step 6
    # (since_step=5) can reach it.
    assert env.recent_happenings("hall", since_step=5) == [(5, "在hall，甲把那件事办成了。")]
    assert env.recent_happenings("hall", since_step=6) == []


@pytest.mark.asyncio
async def test_happenings_track_merges_one_execution_into_one_entry(container) -> None:
    """A joint action is one event, and its holders are the union across records.

    Each participant produces a record sharing the same outcome, but "who already holds it" isn't
    symmetric across them: listeners are attached only to the initiator's record (SocialExecutor
    attaches them once so they don't receive it twice). Writing per record would store two identical
    events, one missing the listener from its holders; that listener could later lurk and take back
    a transcript he already heard, which is what the exclusion set exists to prevent.
    """
    world_id = "world-auth-merge"
    rt, env, _ = _build_runtime(container, world_id=world_id)

    shared_outcome = "在hall，甲对乙说：「三日后动手」。"
    shared_obs = [{"location_id": "hall", "text": "在hall，甲乙谈了一场。"}]
    rt._record_step_happenings(  # noqa: SLF001
        agent_records=[
            # The initiator's record carries the listener
            {"agent_id": "talker-a", "action_type": ActionType.TALK, "location_id": "hall",
             "outcome": shared_outcome, "happening": shared_outcome, "observations": shared_obs,
             "participant_ids": ["talker-a", "talker-b"], "overheard_by": ["listener"],
             "adjudication_failed": False},
            # The target's doesn't: the same event with one person missing from the list
            {"agent_id": "talker-b", "action_type": ActionType.TALK, "location_id": "hall",
             "outcome": shared_outcome, "happening": shared_outcome, "observations": shared_obs,
             "participant_ids": ["talker-a", "talker-b"],
             "adjudication_failed": False},
        ],
        tick_records=[],
        step=1,
    )

    assert [c for _, c in env.recent_happenings("hall", since_step=0)] == [shared_outcome], (
        "一场协作只该留下一条"
    )
    assert env.recent_happenings("hall", since_step=0, exclude_ids=("listener",)) == [], (
        "旁听者已经持有它,绝不能因为另一条记录没登记他而漏掉"
    )


@pytest.mark.asyncio
async def test_happenings_needs_both_gates_not_just_the_type(container) -> None:
    """Each of the two gates blocks something different; drop either and things leak.

    The whitelist is by type and can't block the one step of an allowed type that must not be
    given: a TALK's complete step holds the dialogue transcript, while its interrupt step holds
    the interrupter's private thought (the line ``format_interrupt_reason_3p`` writes into outcome),
    which the membrane lists as privileged ("the reason an action was interrupted"). TALK is
    whitelisted, so by type alone that thought would flow into another person's adjudication
    context; what stops it is the executor not authorizing it.
    """
    world_id = "world-two-gates"
    rt, env, _ = _build_runtime(container, world_id=world_id)

    rt._record_step_happenings(  # noqa: SLF001
        agent_records=[
            # Type is whitelisted, but this step isn't authorized: the interrupt record has no
            # happening
            {"agent_id": "talker-a", "action_type": ActionType.TALK, "location_id": "hall",
             "outcome": "在hall，甲乙的交谈被打断了。甲心道：我得赶紧脱身。",
             "observations": [{"location_id": "hall", "text": "在hall，甲乙的交谈中断了。"}],
             "participant_ids": ["talker-a", "talker-b"],
             "adjudication_failed": False},
        ],
        tick_records=[],
        step=1,
    )
    assert env.recent_happenings("hall", since_step=0) == [], (
        "类型过了白名单不等于这一拍该给 —— 中断拍藏的是私念,不是可探的内容"
    )


@pytest.mark.asyncio
async def test_a_finished_run_leaves_the_event_being_generated_alone(container) -> None:
    """Event generation spans steps: started at step N, consumed by _consume_ready at step N+k.
    `run()` returning isn't a shutdown.

    Single-stepping (the director's "inject one, step once, look") runs one step per run. If cleanup
    cleared in-flight generation, this path would never reach consumption, and each check_interval
    would waste a round of gate + generation LLM calls. It's cleared by the world actually shutting
    down (aclose), not by every run returning.
    """
    runtime, _env, _msgs = _build_runtime(container, world_id="w-event-lifecycle")
    inflight = asyncio.create_task(asyncio.sleep(0.05))
    runtime._event_system._inflight = inflight

    await runtime.run([], total_steps=1)
    assert runtime._event_system._inflight is inflight, "run 收尾丢掉了在途生成"

    await runtime.aclose()
    assert runtime._event_system._inflight is None, "停机时才清在途生成"
