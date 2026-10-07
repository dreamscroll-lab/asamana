"""Contract tests for death handling (vitality=0).

Covers:
- Agent._trigger_death: flips is_active, records death_cause, and does NOT write a
  self-death memory for the dead.
- DeathHandler.apply_vitality_decay: decay that kills fills in the cause of death.
- DeathHandler.process_new_deaths (the single world-side sink): final save + removal from the
  environment + release of reusable items + a global death notice with the cause.
- restore: vitality is the source of truth for alive/dead — vitality=0 restores as
  is_active=False (no resurrection on restart).
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from agent.agent import Agent
from agent.decision import DecisionEngine
from agent.memory import MemorySystem
from agent.need import NeedEngine
from core.interfaces.action import ActionTarget, Ref
from engine.executors.registry import ActionExecutorRegistry
from agent.personality import (
    EmotionState,
    EmotionType,
    PersonalityLayer,
    SoulLayer,
    StateLayer,
)
from agent.relation import RelationSystem
from core.interfaces.llm import LLMRouter, LLMScene
from core.interfaces.perception import BroadcastType
from engine.broadcast import BroadcastChannel
from engine.clock import GlobalClock, WorldTimeConfig
from engine.directory import LiveWorldDirectory
from engine.environment import IN_TRANSIT, EnvironmentSystem
from engine.event import EventSettings
from engine.message_system import MessageSystem
from engine.runtime import NarrativeRuntime
from engine.scheduler import AgentScheduler
from providers.llm.mock import MockLLMProvider
from world.initializer import _apply_stored_state
from world.models import EntityPresence, WorldEntity, WorldEntityType
from worlds.tiled import TiledWorldConfig

WORLD_ID = "world-1"


def _make_agent(container, *, agent_id: str = "agent-target", name: str = "李建成", vitality: float = 1.0) -> Agent:
    router = LLMRouter({scene: MockLLMProvider() for scene in LLMScene})
    soul = SoulLayer(name=name, agent_id=agent_id, role="太子", core_traits=("谨慎",), core_values=("家族",))
    state = StateLayer(
        agent_id=agent_id, step=1,
        emotion=EmotionState(primary=EmotionType.NEUTRAL, intensity=0.3, valence=0.0),
        current_location="palace",
        vitality=vitality,
    )
    return Agent(
        world_id=WORLD_ID,
        agent_id=agent_id,
        personality=PersonalityLayer(soul=soul, state=state),
        decision_engine=DecisionEngine(router),
        llm_router=router,
        memory_system=MemorySystem(
            router, container.embedding, container.vector_store,
            world_id=WORLD_ID, agent_id=agent_id,
        ),
        need_engine=NeedEngine(),
        relation_system=RelationSystem(container.agent_store, world_id=WORLD_ID, agent_id=agent_id),
        agent_store=container.agent_store,
        is_main_character=False,
    )


def _build_runtime(
    container, environment: EnvironmentSystem, *, registry: ActionExecutorRegistry | None = None,
) -> NarrativeRuntime:
    message_system = MessageSystem(container.message_provider, world_id=WORLD_ID)
    broadcast_channel = BroadcastChannel()
    directory = LiveWorldDirectory.from_agents({}, environment)
    event_settings = EventSettings(check_interval=2, max_events_per_window=1)
    # executor_registry must be passed at construction — the runtime's subsystems
    # (DeathHandler / ExecutionProcessor / …) capture it there. Swapping
    # runtime._executor_registry post-hoc would leave them pointing at the old one.
    return NarrativeRuntime(
        world_id=WORLD_ID,
        clock=GlobalClock(WorldTimeConfig(start_hour=6, seconds_per_step=60)),
        scheduler=AgentScheduler(),
        environment=environment,
        message_system=message_system,
        event_settings=event_settings,
        snapshot_provider=container.snapshot,
        agent_store=container.agent_store,
        event_bus=container.event_bus,
        broadcast_channel=broadcast_channel,
        directory=directory,
        executor_registry=registry,
        llm_router=container.llm_router,
    )


# ---------------------------------------------------------------------------
# Agent._trigger_death
# ---------------------------------------------------------------------------


def test_trigger_death_sets_inactive_and_cause_without_self_memory(container) -> None:
    agent = _make_agent(container, vitality=0.0)
    agent._trigger_death(5, cause="饥馁力竭而亡。")
    assert agent.is_active is False
    assert agent.death_cause == "饥馁力竭而亡。"
    # After death there's no experiencing subject: never write a death memory for the dead.
    assert agent.memory_system._entries == {}


# ---------------------------------------------------------------------------
# DeathHandler.apply_vitality_decay fills in the cause when decay kills
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decay_lethal_sets_cause(container) -> None:
    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    runtime = _build_runtime(container, environment)
    agent = _make_agent(container, vitality=0.0005)  # one step of passive decay hits zero
    await runtime._deaths.apply_vitality_decay([agent], step=3)
    assert agent.personality.is_alive is False
    assert agent.is_active is False
    # death_cause carries its own final punctuation (see TargetAgentEffect.death_cause); identity
    # isn't here, it's added when the notice is assembled — the assembler must not add another
    # period, or it produces "…油尽灯枯。。".
    assert agent.death_cause == "油尽灯枯。"


@pytest.mark.asyncio
async def test_starvation_decay_lethal_sets_starvation_cause(container) -> None:
    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    runtime = _build_runtime(container, environment)
    agent = _make_agent(container, vitality=0.002)  # base decay alone doesn't kill; the extra hunger share does
    agent.personality._state.need_intensities["physiological"] = 1.0  # noqa: SLF001
    await runtime._deaths.apply_vitality_decay([agent], step=3)
    assert agent.is_active is False
    assert agent.death_cause == "饥馁力竭而亡。"


def test_engine_changes_vitality_only_through_the_agent() -> None:
    """Vitality has a single entry point, ``Agent.apply_vitality_damage``: bypassing it drains
    health without flipping is_active."""
    from pathlib import Path

    offenders = [
        str(path)
        for path in Path("engine").rglob("*.py")
        if "personality.apply_vitality_damage(" in path.read_text(encoding="utf-8")
        or "._trigger_death(" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# DeathHandler.process_new_deaths — the single world-side sink
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_new_deaths_full_convergence(container) -> None:
    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    runtime = _build_runtime(container, environment)

    agent = _make_agent(container, name="李建成", vitality=1.0)
    environment.place_agent(agent_id=agent.agent_id, location_id="donggong")
    # The dead holds a reusable item (owner-held: no location_id).
    environment.register_entity(WorldEntity(
        entity_id="seal", name="太子印", entity_type=WorldEntityType.ITEM,
        state="intact",
        presence=EntityPresence.HELD, presence_ref=agent.agent_id,
        is_takeable=True, is_public=True,
    ))

    # Alive at step start; then some lethal source kills him (simulated directly as the final
    # death state here).
    pre_step_active = {agent.agent_id}
    agent.personality.apply_vitality_damage(1.0)
    agent._trigger_death(7, cause="死于刀兵。")   # carries its own final punctuation

    await runtime._deaths.process_new_deaths([agent], pre_step_active, step=7)

    # 1) Removed from the environment: no position record, gone from others' view/locations.
    assert environment.get_body_location(agent.agent_id) == "unknown"
    assert agent.agent_id not in environment.bodies_at("donggong")

    # 2) Items are released at the place of death and can be picked up again.
    seal = environment.get_entity("seal")
    assert seal.owner_id is None
    assert seal.location_id == "donggong"
    assert seal.is_public is True

    # 3) Global death notice with cause, deliver_step = next step (published during execution,
    # after this step's perception → perceived next step).
    broadcasts = runtime._broadcast_channel.peek_pending()
    assert len(broadcasts) == 1
    bc = broadcasts[0]
    assert bc.broadcast_type == BroadcastType.WORLD_EVENT
    assert bc.location_scope is None
    assert bc.deliver_step == 8  # died at step 7 → perceived at step 8
    # Location + identity tag + verbatim cause, with exactly one period — the role lets people who
    # didn't know the dead still tell who it was.
    assert bc.content == "在东宫，李建成（太子）死于刀兵。"


@pytest.mark.asyncio
async def test_death_broadcast_says_en_route_while_items_fall_at_the_origin(container) -> None:
    """Death in transit: the notice truthfully says "途中" and items drop at the origin (items
    on a pseudo-location can't be found from any room)."""
    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    runtime = _build_runtime(container, environment)
    agent = _make_agent(container, name="李建成", vitality=1.0)
    environment.place_agent(agent_id=agent.agent_id, location_id="donggong")
    environment.move_body(body_id=agent.agent_id, location_id=IN_TRANSIT)  # departed, origin=donggong
    environment.register_entity(WorldEntity(
        entity_id="seal", name="太子印", entity_type=WorldEntityType.ITEM, state="intact",
        presence=EntityPresence.HELD, presence_ref=agent.agent_id, is_takeable=True,
    ))
    agent.personality.apply_vitality_damage(1.0)
    agent._trigger_death(7, cause="油尽灯枯。")

    await runtime._deaths.process_new_deaths([agent], {agent.agent_id}, step=7)

    assert runtime._broadcast_channel.peek_pending()[0].content == "在途中，李建成（太子）油尽灯枯。"
    assert environment.get_entity("seal").location_id == "donggong"


@pytest.mark.asyncio
async def test_death_broadcast_omits_the_place_when_there_is_none(container) -> None:
    """If the location can't be resolved, omit it — don't leave a "在此处，" pointing nowhere."""
    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    runtime = _build_runtime(container, environment)
    agent = _make_agent(container, name="李建成", vitality=1.0)   # never placed in the world
    agent.personality.apply_vitality_damage(1.0)
    agent._trigger_death(7, cause="油尽灯枯。")

    await runtime._deaths.process_new_deaths([agent], {agent.agent_id}, step=7)

    assert runtime._broadcast_channel.peek_pending()[0].content == "李建成（太子）油尽灯枯。"


@pytest.mark.asyncio
async def test_process_new_deaths_ignores_still_alive_and_already_dead(container) -> None:
    """Only handle new deaths, "alive at step start ∧ dead at step end": the living are untouched
    and those already dead at step start aren't processed twice."""
    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    runtime = _build_runtime(container, environment)

    alive = _make_agent(container, agent_id="alive", name="活着", vitality=1.0)
    already_dead = _make_agent(container, agent_id="old-dead", name="先前已死", vitality=0.0)
    already_dead.set_active(False)
    environment.place_agent(agent_id="alive", location_id="palace")

    # pre_step_active only holds the living (alive at step start); the already dead aren't in it.
    await runtime._deaths.process_new_deaths([alive, already_dead], {"alive"}, step=2)

    # No new deaths → no death notice.
    assert runtime._broadcast_channel.peek_pending() == []
    assert environment.get_body_location("alive") == "palace"


# ---------------------------------------------------------------------------
# restore: vitality is the source of truth for alive/dead
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cross-step death notice delivery: a notice published during execution must survive until
# survivors perceive it next step
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_death_broadcast_survives_into_next_step_perception(container) -> None:
    """A death notice is published during step N's execution (deliver_step=N+1),
    after step N's perception; it must be collected at step N+1 and perceived by survivors. Don't
    clear the whole channel at the start of each step: that wipes the notice before step N+1's
    perception — it would survive only in snapshot metadata (looking "broadcast") while no agent
    ever perceives it.
    """
    from core.interfaces.perception import Broadcast
    from core.interfaces.severity import Severity

    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    runtime = _build_runtime(container, environment)  # clock starts at 0; the first run_step runs step 1

    survivor = _make_agent(container, agent_id="survivor", name="魏徵", vitality=1.0)
    environment.place_agent(agent_id="survivor", location_id="palace")

    # Simulate a global death notice "published during the previous step's (step 0) execution":
    # deliver_step=1, waiting to be perceived at step 1.
    runtime._broadcast_channel.publish(Broadcast(
        content="李建成死于刀兵。",
        source="system",
        broadcast_type=BroadcastType.WORLD_EVENT,
        deliver_step=1,
        location_scope=None,
        severity=Severity.HIGH,
    ))

    # Capture the broadcasts survivors perceive this step.
    perceived_broadcasts: list = []
    original_perceive = survivor.perceive_step

    async def _spy_perceive(*args, broadcasts, **kwargs):
        perceived_broadcasts.extend(broadcasts)
        return await original_perceive(*args, broadcasts=broadcasts, **kwargs)

    survivor.perceive_step = _spy_perceive  # type: ignore[method-assign]

    await runtime.run_step([survivor])

    # Cross-step delivery: survivors really read it in step 1's perception, and it's dequeued
    # (consumed means removed).
    assert any("李建成" in b.content for b in perceived_broadcasts)
    assert runtime._broadcast_channel.peek_pending() == []


@pytest.mark.asyncio
async def test_restore_derives_inactive_from_zero_vitality(container) -> None:
    dead = _make_agent(container, agent_id="agent-dead", vitality=1.0)
    dead.personality.apply_vitality_damage(1.0)  # vitality → 0
    dead.set_active(False)
    await dead.persist_state(dead.personality.state)
    stored = await container.agent_store.load_agent_state(WORLD_ID, "agent-dead")
    assert stored is not None and stored.vitality == 0.0

    fresh = _make_agent(container, agent_id="agent-dead", vitality=1.0)
    assert fresh.is_active is True  # active by default
    _apply_stored_state(fresh, stored)
    assert fresh.is_active is False  # derived from vitality=0: no resurrection


# ---------------------------------------------------------------------------
# Runtime._teardown_executions_of_dead — death is a hard stop: corpses don't come back and
# survivors don't hang
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_death_tears_down_move_and_never_resurrects_corpse(container) -> None:
    """An agent dying in transit: its MOVE execution is torn down (no more tick/complete), and the
    movement interrupt placement guard doesn't put the corpse back in the world; carried items drop
    at the origin (not the IN_TRANSIT pseudo-location)."""
    from agent.personality import ActionStatus
    from core.interfaces.action import ActionTarget, ActionType
    from engine.environment import IN_TRANSIT
    from engine.executors import build_default_registry
    from engine.executors.base import ActionExecutionState

    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))

    agent = _make_agent(container, name="李建成", vitality=1.0)
    environment.place_agent(agent_id=agent.agent_id, location_id="palace")
    environment.move_body(body_id=agent.agent_id, location_id=IN_TRANSIT)  # departed, origin=palace
    environment.register_entity(WorldEntity(
        entity_id="seal", name="太子印", entity_type=WorldEntityType.ITEM,
        state="intact",
        presence=EntityPresence.HELD, presence_ref=agent.agent_id,
        is_takeable=True, is_public=True,
    ))
    agents_dict = {agent.agent_id: agent}
    registry = build_default_registry(
        container.llm_router, LiveWorldDirectory.from_agents(agents_dict, environment),
    )
    runtime = _build_runtime(container, environment, registry=registry)
    exec_state = ActionExecutionState.create(
        target=ActionTarget(),
        action_type=ActionType.MOVE, initiator_id=agent.agent_id,
        participant_ids=[agent.agent_id], purpose="赶往东宫",
        started_step=6, estimated_steps=4, opening_outcome="",
    )
    exec_state.extra["origin"] = "palace"
    exec_state.extra["destination"] = "palace"
    runtime._executor_registry.add_active(exec_state)  # noqa: SLF001
    agent.personality.update_action_status(
        status=ActionStatus.IN_PROGRESS, current_action="赶往东宫", remaining_steps=3,
    )

    pre_step_active = {agent.agent_id}
    agent.personality.apply_vitality_damage(1.0)
    agent._trigger_death(7, cause="死于刀兵。")   # carries its own final punctuation

    await runtime._deaths.process_new_deaths([agent], pre_step_active, step=7)

    # Execution torn down: later steps won't tick/complete a "dead man walking".
    assert runtime._executor_registry.all_active() == []  # noqa: SLF001
    # The corpse wasn't put back into the world by interrupt placement.
    assert environment.get_body_location(agent.agent_id) == "unknown"
    assert agent.agent_id not in environment.bodies_at("palace")
    # Action lifecycle fields reset (code layer); next step he's no longer treated as a body in
    # transit.
    assert agent.personality.state.action_status == ActionStatus.IDLE
    # Items of someone dying in transit drop at the origin, not IN_TRANSIT (which resolves to no
    # room — the items would vanish).
    seal = environment.get_entity("seal")
    assert seal.owner_id is None
    assert seal.location_id == "palace"


@pytest.mark.asyncio
async def test_death_teardown_releases_surviving_talk_partner(container) -> None:
    """One side dies in a multi-party TALK: the execution is torn down and survivors must not hang
    on IN_PROGRESS forever (they'd never re-enter the decision loop and the world silently
    degrades)."""
    from agent.personality import ActionStatus
    from core.interfaces.action import ActionType
    from engine.executors import build_default_registry
    from engine.executors.base import ActionExecutionState

    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))

    dead = _make_agent(container, agent_id="agent-dead", name="李建成", vitality=1.0)
    survivor = _make_agent(container, agent_id="agent-live", name="魏征", vitality=1.0)
    environment.place_agent(agent_id=dead.agent_id, location_id="palace")
    environment.place_agent(agent_id=survivor.agent_id, location_id="palace")
    agents_dict = {dead.agent_id: dead, survivor.agent_id: survivor}
    registry = build_default_registry(
        container.llm_router, LiveWorldDirectory.from_agents(agents_dict, environment),
    )
    runtime = _build_runtime(container, environment, registry=registry)
    exec_state = ActionExecutionState.create(
        target=ActionTarget(),
        action_type=ActionType.TALK, initiator_id=dead.agent_id,
        participant_ids=[dead.agent_id, survivor.agent_id], purpose="密谈",
        started_step=6, estimated_steps=4, opening_outcome="",
    )
    runtime._executor_registry.add_active(exec_state)  # noqa: SLF001
    for a in (dead, survivor):
        a.personality.update_action_status(
            status=ActionStatus.IN_PROGRESS, current_action="密谈", remaining_steps=3,
        )

    pre_step_active = {dead.agent_id}
    dead.personality.apply_vitality_damage(1.0)
    dead._trigger_death(7, cause="毒发身亡")

    await runtime._deaths.process_new_deaths([dead, survivor], pre_step_active, step=7)

    assert runtime._executor_registry.all_active() == []  # noqa: SLF001
    # The survivor gets an interrupt wrap-up or fallback reset; either way no longer IN_PROGRESS —
    # next step it enters decision normally.
    assert survivor.personality.state.action_status != ActionStatus.IN_PROGRESS
    assert survivor.is_active is True
    # The corpse is reset too (code layer).
    assert dead.personality.state.action_status == ActionStatus.IDLE


@pytest.mark.asyncio
async def test_death_teardown_narrates_the_death_as_the_cause(container) -> None:
    """The one place ``cause`` is produced — so the one place it can rot unnoticed.

    An interrupt has two shapes. Normally a participant breaks off on his own, and the
    executor names him from ``interrupted_agent_id`` — no cause needed. A death is the other
    shape: nobody chose anything, the reason is a fact about the world, and only the caller
    knows it. That is what ``cause`` carries, and it must reach the narrative.

    ``reason`` must stay EMPTY here: it is the interrupter's perceived signals (a prompt
    input), and a dead man perceived nothing — he IS the cause.
    """
    from agent.personality import ActionStatus
    from core.interfaces.action import ActionType, AgentAction
    from engine.executors import build_default_registry

    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    dead = _make_agent(container, agent_id="agent-dead", name="李建成", vitality=1.0)
    survivor = _make_agent(container, agent_id="agent-live", name="魏征", vitality=1.0)
    environment.place_agent(agent_id=dead.agent_id, location_id="palace")
    environment.place_agent(agent_id=survivor.agent_id, location_id="palace")
    agents_dict = {dead.agent_id: dead, survivor.agent_id: survivor}
    registry = build_default_registry(
        container.llm_router, LiveWorldDirectory.from_agents(agents_dict, environment),
    )
    runtime = _build_runtime(container, environment, registry=registry)
    talk = runtime._executor_registry.get_executor(ActionType.TALK)  # noqa: SLF001
    # Open the conversation through the executor itself, not by hand-assembling its state:
    # SocialExecutor.start is what records who the partner is, and an interrupt that cannot
    # see the partner silently produces no result for him at all.
    exec_state = await talk.start(
        AgentAction(
            agent_id=dead.agent_id, step=6, action_type=ActionType.TALK,
            action_description="密谈", target=ActionTarget(acts_on=[Ref.agent(survivor.agent_id)], claims=[Ref.agent(survivor.agent_id)]),
            estimated_steps=4,
        ),
        6, agents=agents_dict, environment=environment, message_system=None,
    )
    runtime._executor_registry.add_active(exec_state)  # noqa: SLF001
    for a in (dead, survivor):
        a.personality.update_action_status(
            status=ActionStatus.IN_PROGRESS, current_action="密谈", remaining_steps=3,
        )

    # Watch what the teardown actually hands the executor, and what comes back out.
    seen: dict = {}
    real_interrupt = talk.interrupt

    async def _spy(state, step, **kwargs):  # noqa: ANN001, ANN003
        seen["cause"] = kwargs.get("cause")
        seen["kwargs"] = kwargs
        seen["results"] = await real_interrupt(state, step, **kwargs)
        return seen["results"]

    talk.interrupt = _spy  # type: ignore[method-assign]

    dead.personality.apply_vitality_damage(1.0)
    dead._trigger_death(7, cause="毒发身亡")
    await runtime._deaths.process_new_deaths([dead, survivor], {dead.agent_id}, step=7)

    assert seen["cause"] == "李建成已经失去生命力"
    # The executor is never handed raw trigger signals — the decision consumes them (and a
    # dead man perceived nothing regardless: he IS the cause).
    assert "reason" not in seen["kwargs"]
    survivor_result = next(
        r for r in seen["results"] if r.action.agent_id == survivor.agent_id
    )
    # It reaches the narrative: the survivor's record says WHY the conversation stopped.
    assert "李建成已经失去生命力" in survivor_result.outcome
    assert "中断" in survivor_result.outcome
    assert survivor.personality.state.action_status != ActionStatus.IN_PROGRESS
    # The survivor's ending reaches the step's display stream, or the feed shows a conversation
    # that never ends; the dead get none.
    records = runtime._processor.take_forced_records()  # noqa: SLF001
    assert [(r["agent_id"], r["phase"]) for r in records] == [(survivor.agent_id, "interrupt")]
    assert runtime._processor.take_forced_records() == []  # noqa: SLF001


def test_someone_who_dies_this_step_is_known_dead_before_anyone_plans() -> None:
    """The dead set can't stay frozen at perception time — vitality decay and lethal actions both
    come after it.

    Perception (where the cached set comes from) runs before the decay and execution phases, so
    people who died this step aren't in it; yet relation rendering in both planning and feedback
    reads it. Without a refresh, someone who just died is still rendered as alive in the same step.
    """
    from engine.runtime import NarrativeRuntime

    dead = SimpleNamespace(is_active=False, _dead_agent_ids=frozenset(), note_deaths=None)
    alive = SimpleNamespace(is_active=True, _dead_agent_ids=frozenset(), note_deaths=None)
    for stub in (dead, alive):
        stub.note_deaths = lambda ids, s=stub: setattr(s, "_dead_agent_ids", ids)

    NarrativeRuntime._sync_dead_ids(  # type: ignore[arg-type]
        SimpleNamespace(), {"dead": dead, "alive": alive},
    )

    # The living see this death notice; the dead's own copy is updated too (rendering doesn't
    # distinguish alive from dead).
    assert alive._dead_agent_ids == frozenset({"dead"})
    assert dead._dead_agent_ids == frozenset({"dead"})
