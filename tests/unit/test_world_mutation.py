"""WorldMutationChannel contract tests: the private effect channel.

Each mutation kind (entity / made thing / vitality / relocation) changes the world correctly,
and every time it actually reaches someone's cognition. The second half is guarded by the
per-kind exhaustive tripwire ``test_no_mutation_kind_can_change_the_world_unnoticed``: a new
mutation kind must provide a sample in that table or the test fails. Delivery isn't left to each
kind's discretion.

Another group pins the per-author permission table: the LLM editor (``Author.SYSTEM``) may only
make things, and may only touch unheld things lying on the ground.

Three more fixed boundaries: a kill must flip is_active (or you get a walking corpse); a
relocation must write both copies of the position truth (or others see him here while he thinks
he's there); and a dead agent's third-person text is left to death handling (or onlookers
perceive the same event twice).
"""

from __future__ import annotations

import typing

import pytest

from agent.agent import Agent
from agent.decision import DecisionEngine
from agent.memory import MemorySystem
from agent.need import NeedEngine
from agent.personality import (
    ActionStatus,
    EmotionState,
    EmotionType,
    PersonalityLayer,
    SoulLayer,
    StateLayer,
)
from agent.relation import RelationSystem
from core.interfaces.action import ActionTarget, ActionType, EntityStateChange, ErrandOrder
from core.interfaces.llm import LLMRouter, LLMScene
from engine.directory import LiveWorldDirectory
from engine.environment import EnvironmentSystem
from engine.execution_processor import ExecutionProcessor
from engine.executors.base import ActionExecutionState
from engine.executors.registry import ActionExecutorRegistry
from engine.injection import Author
from engine.message_system import MessageSystem
from engine.world_mutation import (
    ConditionMutation,
    EntityMutation,
    Mutation,
    SpawnMutation,
    VitalityEffect,
    RelocateMutation,
    VitalityMutation,
    WorldMutationChannel,
)
from providers.llm.mock import MockLLMProvider
from world.models import EntityPresence, NpcSeed, WorldEntity, WorldEntityType
from core.interfaces.place import Place

WORLD_ID = "world-1"


def _make_agent(container, *, agent_id: str, name: str, location: str, vitality: float = 1.0) -> Agent:
    router = LLMRouter({scene: MockLLMProvider() for scene in LLMScene})
    soul = SoulLayer(name=name, agent_id=agent_id, role="臣", core_traits=("谨慎",), core_values=("家族",))
    state = StateLayer(
        agent_id=agent_id, step=1,
        emotion=EmotionState(primary=EmotionType.NEUTRAL, intensity=0.3, valence=0.0),
        current_location=location,
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


def _location(entity_id: str, name: str) -> Place:
    return Place(
        place_id=entity_id, name=name, )


def _channel(environment: EnvironmentSystem, container, *, registry=None):
    registry = registry or ActionExecutorRegistry()
    message_system = MessageSystem(container.message_provider, world_id=WORLD_ID)
    directory = LiveWorldDirectory.from_agents({}, environment)
    processor = ExecutionProcessor(
        executor_registry=registry,
        environment=environment,
        message_system=message_system,
        directory=directory,
    )
    return WorldMutationChannel(
        environment=environment,
        seconds_per_step=3600,
        processor=processor,
    ), registry


def _env_with_two_rooms() -> EnvironmentSystem:
    environment = EnvironmentSystem()
    environment.space.register_place(_location("palace", "太极宫"))
    environment.space.register_place(_location("market", "西市"))
    return environment


def _ambient_at(
    environment: EnvironmentSystem, location: str, *, observer: str = "onlooker", step: int = 9,
) -> list[str]:
    """What someone standing at ``location`` perceives on the NEXT step.

    Observations are carried, not live: ``begin_step`` is what flips the carry buffer into
    the active layer, so a test that reads before it sees nothing. The observer must also
    actually BE there — ``spatial_for`` resolves ambient off the perceiver's own location,
    so an unplaced id lands in "unknown" and perceives an empty room forever.
    """
    from engine.clock import GlobalClock, WorldTimeConfig

    environment.place_agent(agent_id=observer, location_id=location)
    clock = GlobalClock(WorldTimeConfig(start_hour=6, seconds_per_step=60))
    environment.begin_step(step=step, world_time=clock.tick())
    spatial = environment.spatial_for(agent_id=observer, step=step)
    return [ev.content for ev in spatial.ambient_events]


# ---------------------------------------------------------------------------
# Invariant: every mutation carries its own observable description
# ---------------------------------------------------------------------------


async def _cognitive_trace(
    environment: EnvironmentSystem, agent: Agent, *, step: int,
) -> list[str]:
    """Everything this change left in anyone's cognition: ambient at every location + the affected agent's memory.

    Whether anyone knows can only be asked from the cognition side: a world state changed correctly
    that nobody notices is exactly the failure this module prevents.
    """
    from engine.clock import GlobalClock, WorldTimeConfig

    environment.begin_step(step=step + 1, world_time=GlobalClock(WorldTimeConfig()).tick())
    traces = [
        ev.content
        for location in environment.space.all_place_ids()
        for ev in environment._step_annotations.get(location, [])
    ]
    await agent.memory_system.drain_writes()
    traces += [
        m.stored_content
        for pair in agent.memory_system.sample_recent_events(current_step=step)
        for m in pair if m is not None
    ]
    return traces


def _spy_on_memory_situation(agent: Agent) -> list:
    """Capture the ``situation`` each ``record_event`` call receives.

    Assert the time/place anchor at write time directly, not the wording in ``stored_content``: that
    is LLM-rewritten and has no header under the mock provider, so asserting it would test the mock.
    """
    stamps: list = []
    original = agent.memory_system.record_event

    async def _spy(*args, **kwargs):
        stamps.append(kwargs.get("situation"))
        return await original(*args, **kwargs)

    agent.memory_system.record_event = _spy  # type: ignore[method-assign]
    return stamps


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", typing.get_args(Mutation), ids=lambda k: k.__name__)
async def test_no_mutation_kind_can_change_the_world_unnoticed(container, kind) -> None:
    """After every mutation kind lands, someone must know it happened: a per-kind exhaustive tripwire.

    `observation` is required on the author side, but delivery is hand-written per kind, so a kind
    can change the world without reaching anyone. A new mutation kind must provide a sample in the
    table below (otherwise parametrizing over ``typing.get_args(Mutation)`` raises KeyError), and
    the sample must actually be perceived or remembered by someone.

    ``VitalityMutation`` uses WOUND, not KILL: a kill is the one explicit silence (left to death
    handling's world-wide notice) and has its own test.
    """
    environment = _env_with_two_rooms()
    environment.register_entity(WorldEntity(
        entity_id="seal", name="兵符", entity_type=WorldEntityType.ITEM,
        is_takeable=True, state="intact",
        presence=EntityPresence.AT_LOCATION, presence_ref="palace",
    ))
    agent = _make_agent(container, agent_id="a1", name="李世民", location="palace")
    environment.place_agent(agent_id="a1", location_id="palace")
    channel, _ = _channel(environment, container)

    samples: dict[type, Mutation] = {
        EntityMutation: EntityMutation(observation="兵符裂作两半。", entity_id="seal", new_state="broken"),
        SpawnMutation: SpawnMutation(
            observation="宫门上多了一张告示。", location_id="palace", name="告示", entity_type="landmark",
        ),
        VitalityMutation: VitalityMutation(
            observation="李世民肩头中了一箭。", body_id="a1", effect=VitalityEffect.WOUND,
        ),
        RelocateMutation: RelocateMutation(
            observation="李世民凭空出现在西市。", body_id="a1", location_id="market",
        ),
        ConditionMutation: ConditionMutation(
            observation="李世民被人反绑了双手。", body_id="a1", description="双手被反绑",
        ),
    }

    assert await channel.apply(samples[kind], step=4, agents={"a1": agent}, author=Author.DIRECTOR) is not None
    assert await _cognitive_trace(environment, agent, step=4), (
        f"{kind.__name__} 改了世界却没有任何人知道"
    )


def test_observation_is_a_required_field_on_every_mutation() -> None:
    """A mutation without an observable description can't be constructed at the type level. This
    is the only constraint in the module enforced by types rather than discipline. If the world
    changes and nobody knows, beliefs come unmoored from the world."""
    for kind, kwargs in (
        (EntityMutation, {"entity_id": "e"}),
        (SpawnMutation, {"location_id": "palace", "name": "信", "entity_type": "item"}),
        (VitalityMutation, {"body_id": "a", "effect": VitalityEffect.WOUND}),
        (RelocateMutation, {"body_id": "a", "location_id": "palace"}),
        (ConditionMutation, {"body_id": "a", "description": "双手被反绑"}),
    ):
        with pytest.raises(TypeError):
            kind(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ENTITY
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_mutation_changes_state_and_is_perceived_where_it_stood(container) -> None:
    environment = _env_with_two_rooms()
    environment.register_entity(WorldEntity(
        entity_id="gate", name="玄武门", entity_type=WorldEntityType.LANDMARK,
        state="intact",
        presence=EntityPresence.AT_LOCATION, presence_ref="palace",
    ))
    channel, _ = _channel(environment, container)

    applied = await channel.apply(
        EntityMutation(observation="玄武门轰然洞开。", entity_id="gate", new_state="broken"),
        step=1, agents={}, author=Author.DIRECTOR,
    )

    assert applied is not None
    assert environment.get_entity("gate").state == "broken"
    assert "玄武门轰然洞开。" in _ambient_at(environment, "palace")


@pytest.mark.asyncio
async def test_entity_mutation_can_destroy_and_removes_it_from_perception(container) -> None:
    environment = _env_with_two_rooms()
    environment.register_entity(WorldEntity(
        entity_id="seal", name="兵符", entity_type=WorldEntityType.ITEM,
        is_takeable=True,
        presence=EntityPresence.AT_LOCATION, presence_ref="palace",
    ))
    channel, _ = _channel(environment, container)

    applied = await channel.apply(
        EntityMutation(observation="兵符在火中化为灰烬。", entity_id="seal", destroyed=True),
        step=1, agents={}, author=Author.DIRECTOR,
    )

    assert applied is not None
    assert environment.get_entity("seal").is_destroyed is True
    assert environment.get_items_at("palace") == []


@pytest.mark.asyncio
async def test_entity_mutation_on_a_held_item_is_perceived_where_its_holder_stands(container) -> None:
    """Something held in someone's hands needs a scope too, or even the holder won't know it changed.

    ``location_id`` is a derived view of ``presence``, and HELD things don't have one. Computing
    scope from it gives nothing, so nobody perceives the change: the world moves on while every
    belief stays put, which is exactly the invariant at the top of this module.
    """
    environment = _env_with_two_rooms()
    environment.register_entity(WorldEntity(
        entity_id="sword", name="佩剑", entity_type=WorldEntityType.ITEM,
        is_takeable=True, state="intact",
        presence=EntityPresence.HELD, presence_ref="a1",
    ))
    environment.place_agent(agent_id="a1", location_id="market")
    channel, _ = _channel(environment, container)

    applied = await channel.apply(
        EntityMutation(observation="佩剑寸寸断裂。", entity_id="sword", new_state="broken"),
        step=1, agents={}, author=Author.DIRECTOR,
    )

    assert applied is not None
    assert environment.get_entity("sword").state == "broken"
    # The holder is present too and isn't self-filtered (what changed is the thing, not a third-person description of his own action).
    assert "佩剑寸寸断裂。" in _ambient_at(environment, "market", observer="a1")


@pytest.mark.asyncio
async def test_entity_mutation_on_unknown_entity_changes_nothing(container) -> None:
    environment = _env_with_two_rooms()
    channel, _ = _channel(environment, container)
    applied = await channel.apply(
        EntityMutation(observation="x", entity_id="nope"), step=1, agents={}, author=Author.DIRECTOR,
    )
    assert applied is None


# ---------------------------------------------------------------------------
# VITALITY
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lethal_vitality_mutation_flips_is_active_not_just_vitality(container) -> None:
    """The core kill boundary: ``Agent.is_active`` is a plain bool field, not derived from vitality.

    Death handling compares pre_step_active with is_active. Zeroing vitality without flipping the
    flag gives a walking corpse with vitality=0 that is still scheduled and still acts.
    """
    environment = _env_with_two_rooms()
    agent = _make_agent(container, agent_id="a1", name="李建成", location="palace")
    channel, _ = _channel(environment, container)

    applied = await channel.apply(
        VitalityMutation(observation="李建成中箭倒地,气绝。", body_id="a1", effect=VitalityEffect.KILL),
        step=7, agents={"a1": agent}, author=Author.DIRECTOR,
    )

    assert applied is not None
    assert agent.personality.state.vitality == 0.0
    assert agent.is_active is False                      # ← walking-corpse guard
    assert agent.death_cause == "李建成中箭倒地,气绝。"    # death handling uses this to compose the death notice


@pytest.mark.asyncio
async def test_lethal_mutation_leaves_the_death_broadcast_to_the_death_reaper(container) -> None:
    """A kill carries no ambient: death handling sends a world-wide death notice with the same
    cause, and another entry here would make onlookers perceive the event twice."""
    environment = _env_with_two_rooms()
    agent = _make_agent(container, agent_id="a1", name="李建成", location="palace")
    channel, _ = _channel(environment, container)

    await channel.apply(
        VitalityMutation(observation="李建成气绝。", body_id="a1", effect=VitalityEffect.KILL),
        step=7, agents={"a1": agent}, author=Author.DIRECTOR,
    )

    assert _ambient_at(environment, "palace") == []


@pytest.mark.asyncio
async def test_a_wound_that_happens_to_kill_also_defers_to_the_death_reaper(container) -> None:
    """Whether something kills isn't a property of the kind: a WOUND on someone near death is just as fatal.

    So "leave a dead agent's third person to death handling" lives at delivery and is decided by
    whether the affected agent is still alive this step, not hard-coded in the KILL branch. Tying it
    to the kind misses this case: onlookers would read both a "he was hit by an arrow" ambient and a
    world-wide death notice, perceiving the event twice.
    """
    environment = _env_with_two_rooms()
    agent = _make_agent(container, agent_id="a1", name="李建成", location="palace", vitality=0.2)
    environment.place_agent(agent_id="a1", location_id="palace")
    channel, _ = _channel(environment, container)

    applied = await channel.apply(
        VitalityMutation(observation="李建成又中一箭。", body_id="a1", effect=VitalityEffect.WOUND),
        step=3, agents={"a1": agent}, author=Author.DIRECTOR,
    )

    assert applied is not None
    assert agent.is_active is False          # 0.2 - 0.35 → dead
    assert _ambient_at(environment, "palace") == []


@pytest.mark.asyncio
async def test_nonlethal_vitality_mutation_is_perceived_by_bystanders(container) -> None:
    environment = _env_with_two_rooms()
    agent = _make_agent(container, agent_id="a1", name="李建成", location="palace")
    environment.place_agent(agent_id="a1", location_id="palace")
    channel, _ = _channel(environment, container)

    applied = await channel.apply(
        VitalityMutation(observation="李建成肩头中了一箭。", body_id="a1", effect=VitalityEffect.WOUND),
        step=3, agents={"a1": agent}, author=Author.DIRECTOR,
    )

    assert applied is not None
    assert agent.is_active is True
    assert agent.personality.state.vitality == pytest.approx(0.65)   # WOUND = 0.35
    assert "李建成肩头中了一箭。" in _ambient_at(environment, "palace")


@pytest.mark.asyncio
async def test_the_wounded_agent_remembers_being_wounded(container) -> None:
    """Someone wounded by the director remembers it; that's what going through the existing feedback-layer channel buys.

    Calling ``apply_vitality_damage`` directly would set the health right, but the victim would have
    no first-person record: an executor wounds you and you remember, the director wounds you and you
    don't: two physics for the same event. Vitality is agent-internal state and the feedback layer
    is its only writer; this test pins that decision.
    """
    environment = _env_with_two_rooms()
    agent = _make_agent(container, agent_id="a1", name="李建成", location="palace")
    environment.place_agent(agent_id="a1", location_id="palace")
    channel, _ = _channel(environment, container)

    await channel.apply(
        VitalityMutation(observation="李建成肩头中了一箭。", body_id="a1", effect=VitalityEffect.WOUND),
        step=3, agents={"a1": agent}, author=Author.DIRECTOR,
    )
    await agent.memory_system.drain_writes()

    remembered = [
        m.stored_content
        for pair in agent.memory_system.sample_recent_events(current_step=3)
        for m in pair if m is not None
    ]
    assert any("中了一箭" in text for text in remembered), remembered


@pytest.mark.asyncio
async def test_vitality_mutation_can_heal(container) -> None:
    environment = _env_with_two_rooms()
    agent = _make_agent(container, agent_id="a1", name="李建成", location="palace", vitality=0.4)
    channel, _ = _channel(environment, container)

    await channel.apply(
        VitalityMutation(observation="伤口以奇快的速度愈合。", body_id="a1", effect=VitalityEffect.HEAL),
        step=3, agents={"a1": agent}, author=Author.DIRECTOR,
    )

    assert agent.personality.state.vitality == pytest.approx(0.75)   # HEAL = -0.35


@pytest.mark.asyncio
async def test_vitality_mutation_on_a_dead_agent_is_refused(container) -> None:
    environment = _env_with_two_rooms()
    agent = _make_agent(container, agent_id="a1", name="李建成", location="palace")
    agent.set_active(False)
    channel, _ = _channel(environment, container)

    applied = await channel.apply(
        VitalityMutation(observation="x", body_id="a1", effect=VitalityEffect.WOUND), step=3, agents={"a1": agent}, author=Author.DIRECTOR,
    )
    assert applied is None


# ---------------------------------------------------------------------------
# RELOCATE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_relocate_writes_both_truths_about_where_someone_is(container) -> None:
    """Position has two copies of the truth: the environment side decides who sees whom, the agent side feeds its own prompt and snapshot.
    Writing only one gives "others see him at the east market while he thinks he's at the west market"."""
    environment = _env_with_two_rooms()
    agent = _make_agent(container, agent_id="a1", name="李世民", location="palace")
    environment.place_agent(agent_id="a1", location_id="palace")
    channel, _ = _channel(environment, container)

    applied = await channel.apply(
        RelocateMutation(
            observation="李世民出现在西市的人群中。", body_id="a1", location_id="market",
        ),
        step=5, agents={"a1": agent}, author=Author.DIRECTOR,
    )

    assert applied is not None
    assert environment.get_body_location("a1") == "market"           # world side
    assert agent.personality.state.current_location == "market"       # agent side
    assert environment.bodies_at("palace") == []
    assert environment.bodies_at("market") == ["a1"]


@pytest.mark.asyncio
async def test_relocate_reports_that_he_did_not_walk_there(container) -> None:
    """A move has to say how it happened, or the map will invent an explanation.

    The observer only gets a changed location_id and would walk the person along the pathfinding
    route: a teleport drawn as an ordinary walk, and joined with a later walk into a back-and-forth
    nobody can follow. The map can show a cut, but only if told.

    Only relocate declares it: the other two mutations don't change position.
    """
    environment = _env_with_two_rooms()
    agent = _make_agent(container, agent_id="a1", name="李世民", location="palace")
    environment.place_agent(agent_id="a1", location_id="palace")
    channel, _ = _channel(environment, container)

    applied = await channel.apply(
        RelocateMutation(
            observation="李世民出现在西市的人群中。", body_id="a1", location_id="market",
        ),
        step=5, agents={"a1": agent}, author=Author.DIRECTOR,
    )

    assert applied is not None
    assert applied.displaced == ("a1",)


@pytest.mark.asyncio
async def test_only_relocate_claims_a_displacement(container) -> None:
    """Wounding and destroying don't change position; declaring a move would make the map cut on someone who never moved."""
    environment = _env_with_two_rooms()
    agent = _make_agent(container, agent_id="a1", name="李建成", location="palace")
    environment.place_agent(agent_id="a1", location_id="palace")
    channel, _ = _channel(environment, container)

    applied = await channel.apply(
        VitalityMutation(
            observation="李建成肩头中了一箭。", body_id="a1", effect=VitalityEffect.WOUND,
        ),
        step=5, agents={"a1": agent}, author=Author.DIRECTOR,
    )

    assert applied is not None
    assert applied.displaced == ()


@pytest.mark.asyncio
async def test_relocate_is_perceived_at_both_ends(container) -> None:
    """A relocation has two viewpoints and people on both sides must know; the origin is the side most easily left silent.

    SpatialPerception has no cross-step diff, so vanishing from visible_agent_ids is silent.

    The origin line is the plainest departure, word for word MovementExecutor's unknown-direction
    branch: "suddenly vanished" would be the code interpreting for the director, and the director's
    line (him appearing at the destination) would be false at the origin.
    """
    environment = _env_with_two_rooms()
    agent = _make_agent(container, agent_id="a1", name="李世民", location="palace")
    environment.place_agent(agent_id="a1", location_id="palace")
    channel, _ = _channel(environment, container)

    environment.place_agent(agent_id="left_behind", location_id="palace")
    environment.place_agent(agent_id="witness", location_id="market")

    await channel.apply(
        RelocateMutation(observation="李世民凭空出现。", body_id="a1", location_id="market"),
        step=5, agents={"a1": agent}, author=Author.DIRECTOR,
    )

    from engine.clock import GlobalClock, WorldTimeConfig
    environment.begin_step(step=6, world_time=GlobalClock(WorldTimeConfig()).tick())

    def ambient(agent_id: str) -> list[str]:
        return [e.content for e in environment.spatial_for(agent_id=agent_id, step=6).ambient_events]

    assert ambient("witness") == ["李世民凭空出现。"]      # destination: someone sees him arrive
    assert ambient("left_behind") == ["李世民离开了此地。"]  # origin: someone sees him leave
    # The relocated agent doesn't read third-person text describing himself (execution-member self-filter).
    assert ambient("a1") == []


@pytest.mark.asyncio
async def test_the_relocated_agent_remembers_being_moved(container) -> None:
    """The relocated agent remembers it, including where he was before.

    Without this he'd be the only one in the world unaware he was moved (the destination line is
    self-filtered out, and he's no longer at the origin), standing somewhere unfamiliar next step
    with memories only of the last place. It's the same feedback-layer channel as "the man the
    director wounded remembers the arrow", and it shouldn't only half work.
    """
    environment = _env_with_two_rooms()
    agent = _make_agent(container, agent_id="a1", name="李世民", location="palace")
    environment.place_agent(agent_id="a1", location_id="palace")
    channel, _ = _channel(environment, container)

    await channel.apply(
        RelocateMutation(observation="李世民凭空出现在西市。", body_id="a1", location_id="market"),
        step=5, agents={"a1": agent}, author=Author.DIRECTOR,
    )
    await agent.memory_system.drain_writes()

    remembered = [
        m.stored_content
        for pair in agent.memory_system.sample_recent_events(current_step=5)
        for m in pair if m is not None
    ]
    assert any("凭空出现在西市" in text for text in remembered), remembered
    # The origin isn't made up; it's the fact just read from the environment, and it carries the "where I was" half of narrative continuity.
    assert any("太极宫" in text for text in remembered), remembered


@pytest.mark.asyncio
async def test_the_memory_is_stamped_with_where_and_when_he_now_is(container) -> None:
    """This memory's time/place header must be now, not the previous step.

    ``apply_target_effect`` stamps the memory's situation header from ``Agent._situation``, which assumes
    this step's perceive has already run. That holds on the executor feedback path (feedback runs at
    the end of the step) but not on the director path: injection deliberately lands before perceive
    (so the relocated agent is seen at the new location that same step). Without re-anchoring it
    would write "I'm at the Taiji Palace now… (he appears at the west market)": a self-contradicting
    memory that embedding keeps recalling.
    """
    from engine.clock import GlobalClock, WorldTimeConfig

    environment = _env_with_two_rooms()
    world_time = GlobalClock(WorldTimeConfig()).tick()
    environment.begin_step(step=5, world_time=world_time)
    agent = _make_agent(container, agent_id="a1", name="李世民", location="palace")
    environment.place_agent(agent_id="a1", location_id="palace")
    channel, _ = _channel(environment, container)
    stamps = _spy_on_memory_situation(agent)

    await channel.apply(
        RelocateMutation(observation="李世民凭空出现在西市。", body_id="a1", location_id="market"),
        step=5, agents={"a1": agent}, author=Author.DIRECTOR,
    )

    assert stamps, "当事人根本没记下这件事"
    assert stamps[0].location_view.name == "西市"          # not "太极宫", where he started
    assert stamps[0].time_label == world_time.time_label   # not the previous step's time


@pytest.mark.asyncio
async def test_a_mutation_that_moves_nobody_is_still_stamped_with_now(container) -> None:
    """The time half is wrong for all three kinds, so re-anchoring belongs to delivery, not to relocation alone.

    The man hit by an arrow didn't move, but injection still precedes perceive; without
    re-anchoring the memory lags a beat and lands on the previous step's time.
    """
    from engine.clock import GlobalClock, WorldTimeConfig

    environment = _env_with_two_rooms()
    clock = GlobalClock(WorldTimeConfig())
    clock.tick()                                   # previous step
    world_time = clock.tick()                      # this step
    environment.begin_step(step=5, world_time=world_time)
    agent = _make_agent(container, agent_id="a1", name="李建成", location="palace")
    environment.place_agent(agent_id="a1", location_id="palace")
    channel, _ = _channel(environment, container)
    stamps = _spy_on_memory_situation(agent)

    await channel.apply(
        VitalityMutation(
            observation="李建成肩头中了一箭。", body_id="a1", effect=VitalityEffect.WOUND,
        ),
        step=5, agents={"a1": agent}, author=Author.DIRECTOR,
    )

    assert stamps
    assert stamps[0].location_view.name == "太极宫"        # he didn't move
    assert stamps[0].time_label == world_time.time_label   # but the time must be this step's


@pytest.mark.asyncio
async def test_relocate_tears_down_the_execution_the_agent_was_in(container) -> None:
    """Someone moved in an instant can't carry on with what he was doing. Without tearing it down the
    execution keeps ticking and, on completion, treats him as still in the old place."""
    environment = _env_with_two_rooms()
    agent = _make_agent(container, agent_id="a1", name="李世民", location="palace")
    environment.place_agent(agent_id="a1", location_id="palace")
    registry = ActionExecutorRegistry()
    exec_state = ActionExecutionState.create(
        action_type=ActionType.WORK,
        initiator_id="a1",
        participant_ids=["a1"],
        started_step=1,
        estimated_steps=5,
        purpose="批阅奏章",
        opening_outcome="李世民着手批阅奏章。",
        target=ActionTarget(),
    )
    registry.add_active(exec_state)
    agent.personality.update_action_status(
        status=ActionStatus.IN_PROGRESS, current_action="批阅奏章", remaining_steps=4,
    )
    channel, _ = _channel(environment, container, registry=registry)

    await channel.apply(
        RelocateMutation(observation="李世民被带走了。", body_id="a1", location_id="market"),
        step=5, agents={"a1": agent}, author=Author.DIRECTOR,
    )

    assert registry.get_active_for_agent("a1") is None
    assert agent.personality.state.action_status == ActionStatus.IDLE


@pytest.mark.asyncio
async def test_relocate_to_an_unknown_place_changes_nothing(container) -> None:
    environment = _env_with_two_rooms()
    agent = _make_agent(container, agent_id="a1", name="李世民", location="palace")
    environment.place_agent(agent_id="a1", location_id="palace")
    channel, _ = _channel(environment, container)

    applied = await channel.apply(
        RelocateMutation(observation="x", body_id="a1", location_id="atlantis"),
        step=5, agents={"a1": agent}, author=Author.DIRECTOR,
    )

    assert applied is None
    assert environment.get_body_location("a1") == "palace"


# ---------------------------------------------------------------------------
# SPAWN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_puts_a_new_thing_on_the_ground_and_onlookers_see_it(container) -> None:
    environment = _env_with_two_rooms()
    onlooker = _make_agent(container, agent_id="a1", name="李世民", location="palace")
    environment.place_agent(agent_id="a1", location_id="palace")
    channel, _ = _channel(environment, container)

    outcome = await channel.apply(
        SpawnMutation(
            observation="宫门上多了一张告示。", location_id="palace", name="告示",
            entity_type="landmark", description="黄纸黑字", content="明日辰时开城门。",
        ),
        step=3, agents={"a1": onlooker}, author=Author.SYSTEM,
    )

    assert outcome is not None and outcome.reached == ("a1",)
    [made] = environment.get_items_at("palace")
    assert (made.name, made.content, made.is_takeable) == ("告示", "明日辰时开城门。", False)
    assert made.created_step == environment._current_step
    assert "宫门上多了一张告示。" in _ambient_at(environment, "palace", step=4)


@pytest.mark.asyncio
async def test_spawn_of_an_unknown_kind_changes_nothing(container) -> None:
    environment = _env_with_two_rooms()
    channel, _ = _channel(environment, container)

    outcome = await channel.apply(
        SpawnMutation(observation="…", location_id="palace", name="西市", entity_type="location"),
        step=3, agents={}, author=Author.SYSTEM,
    )

    assert outcome is None
    assert environment.get_items_at("palace") == []


# ---------------------------------------------------------------------------
# Permission table: the LLM editor may only make things, and only touch unheld things on the ground
# ---------------------------------------------------------------------------


def _env_with_ground_and_held_items() -> EnvironmentSystem:
    environment = _env_with_two_rooms()
    environment.register_entity(WorldEntity(
        entity_id="notice", name="告示", entity_type=WorldEntityType.LANDMARK, state="intact",
        presence=EntityPresence.AT_LOCATION, presence_ref="palace",
    ))
    environment.register_entity(WorldEntity(
        entity_id="letter", name="密信", entity_type=WorldEntityType.ITEM, is_takeable=True,
        state="sealed", presence=EntityPresence.HELD, presence_ref="a1",
    ))
    return environment


@pytest.mark.asyncio
async def test_system_author_can_alter_a_thing_lying_on_the_ground(container) -> None:
    environment = _env_with_ground_and_held_items()
    channel, _ = _channel(environment, container)

    outcome = await channel.apply(
        EntityMutation(
            observation="告示被人涂改了一行。", entity_id="notice", new_state="涂改",
            new_description="黄纸上多了墨痕", new_content="后日辰时开城门。",
        ),
        step=3, agents={}, author=Author.SYSTEM,
    )

    notice = environment.get_entity("notice")
    assert outcome is not None
    assert (notice.state, notice.description, notice.content) == ("涂改", "黄纸上多了墨痕", "后日辰时开城门。")


@pytest.mark.asyncio
async def test_system_author_can_destroy_a_thing_lying_on_the_ground(container) -> None:
    environment = _env_with_ground_and_held_items()
    channel, _ = _channel(environment, container)

    outcome = await channel.apply(
        EntityMutation(observation="告示被风卷走了。", entity_id="notice", destroyed=True),
        step=3, agents={}, author=Author.SYSTEM,
    )

    assert outcome is not None
    assert environment.get_entity("notice").is_destroyed


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", [
    EntityMutation(observation="密信化为灰烬。", entity_id="letter", destroyed=True),
    EntityMutation(observation="信上的字变了。", entity_id="letter", new_content="另一番话"),
    EntityMutation(observation="…", entity_id="no-such-thing", destroyed=True),
    VitalityMutation(observation="李世民中了一箭。", body_id="a1", effect=VitalityEffect.WOUND),
    RelocateMutation(observation="李世民出现在西市。", body_id="a1", location_id="market"),
    ConditionMutation(observation="李世民被人反绑了双手。", body_id="a1", description="双手被反绑"),
], ids=["destroy-held", "alter-held", "unknown", "vitality", "relocate", "condition"])
async def test_system_author_is_refused_everything_else(container, mutation) -> None:
    environment = _env_with_ground_and_held_items()
    agent = _make_agent(container, agent_id="a1", name="李世民", location="palace")
    environment.place_agent(agent_id="a1", location_id="palace")
    channel, _ = _channel(environment, container)
    before = (agent.personality.state.vitality, environment.get_body_location("a1"))

    outcome = await channel.apply(mutation, step=3, agents={"a1": agent}, author=Author.SYSTEM)

    assert outcome is None
    letter = environment.get_entity("letter")
    assert (letter.owner_id, letter.content, letter.is_destroyed) == ("a1", "", False)
    assert (agent.personality.state.vitality, environment.get_body_location("a1")) == before
    assert agent.personality.state.condition is None


@pytest.mark.asyncio
async def test_system_author_is_refused_a_thing_picked_up_after_the_plan_was_made(container) -> None:
    """Permission is checked when the change lands: if the thing was on the ground when the editor planned it but was picked up before landing, it can't be touched."""
    environment = _env_with_ground_and_held_items()
    channel, _ = _channel(environment, container)
    environment.register_entity(WorldEntity(
        entity_id="knife", name="短刀", entity_type=WorldEntityType.ITEM, is_takeable=True,
        presence=EntityPresence.AT_LOCATION, presence_ref="palace",
    ))
    environment.change_entity_state(EntityStateChange(entity_id="knife", owner_id="a1"))

    outcome = await channel.apply(
        EntityMutation(observation="短刀断了。", entity_id="knife", destroyed=True),
        step=3, agents={}, author=Author.SYSTEM,
    )

    assert outcome is None
    assert environment.get_entity("knife").owner_id == "a1"


@pytest.mark.asyncio
async def test_director_can_still_destroy_what_someone_holds(container) -> None:
    environment = _env_with_ground_and_held_items()
    channel, _ = _channel(environment, container)

    outcome = await channel.apply(
        EntityMutation(observation="密信化为灰烬。", entity_id="letter", destroyed=True),
        step=3, agents={}, author=Author.DIRECTOR,
    )

    assert outcome is not None
    assert environment.get_entity("letter").is_destroyed


# ---------------------------------------------------------------------------
# Conditions: both people and errand-runners can have them applied / cleared
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_condition_lands_on_an_agent_and_he_remembers_it(container) -> None:
    environment = _env_with_two_rooms()
    agent = _make_agent(container, agent_id="a1", name="李建成", location="palace")
    environment.place_agent(agent_id="a1", location_id="palace")
    channel, _ = _channel(environment, container)

    outcome = await channel.apply(
        ConditionMutation(observation="李建成被人反绑了双手。", body_id="a1", description="双手被反绑"),
        step=3, agents={"a1": agent}, author=Author.DIRECTOR,
    )
    await agent.memory_system.drain_writes()

    condition = agent.personality.state.condition
    assert outcome is not None and "a1" in outcome.reached
    assert condition is not None and condition.description == "双手被反绑"
    # For people, it needs outside help to clear (or they break free themselves); no time limit.
    assert condition.until_step is None
    remembered = [
        m.stored_content
        for pair in agent.memory_system.sample_recent_events(current_step=3)
        for m in pair if m is not None
    ]
    assert any("反绑" in text for text in remembered), remembered


@pytest.mark.asyncio
async def test_clearing_an_agent_condition_frees_him(container) -> None:
    environment = _env_with_two_rooms()
    agent = _make_agent(container, agent_id="a1", name="李建成", location="palace")
    environment.place_agent(agent_id="a1", location_id="palace")
    channel, _ = _channel(environment, container)
    await channel.apply(
        ConditionMutation(observation="李建成被人反绑了双手。", body_id="a1", description="双手被反绑"),
        step=3, agents={"a1": agent}, author=Author.DIRECTOR,
    )

    outcome = await channel.apply(
        ConditionMutation(observation="李建成的绳子被人割断了。", body_id="a1", description=""),
        step=4, agents={"a1": agent}, author=Author.DIRECTOR,
    )

    assert outcome is not None
    assert agent.personality.state.condition is None


@pytest.mark.asyncio
async def test_clearing_a_condition_nobody_has_is_nothing_happening(container) -> None:
    environment = _env_with_two_rooms()
    agent = _make_agent(container, agent_id="a1", name="李建成", location="palace")
    environment.place_agent(agent_id="a1", location_id="palace")
    npc_id = _npc(environment)
    channel, _ = _channel(environment, container)

    for body in ("a1", npc_id):
        outcome = await channel.apply(
            ConditionMutation(observation="绳子被割断了。", body_id=body, description=""),
            step=3, agents={"a1": agent}, author=Author.DIRECTOR,
        )
        assert outcome is None, body


# ---------------------------------------------------------------------------
# The errand-running tier (Npc): relocation / conditions
# ---------------------------------------------------------------------------


def _npc(environment: EnvironmentSystem, at: str = "palace") -> str:
    environment.spawn_npc(
        NpcSeed(name="王二", gender="男", age=34, description="跑得快、认得路"), location_id=at,
    )
    return environment.all_npcs()[-1].npc_id


@pytest.mark.asyncio
async def test_relocating_an_npc_keeps_its_errand_and_marks_it_displaced(container) -> None:
    environment = _env_with_two_rooms()
    environment.place_agent(agent_id="a1", location_id="palace")
    npc_id = _npc(environment)
    assert environment.assign_errand(ErrandOrder(npc_id, "market"), requester_id="a1")
    environment.get_npc(npc_id).errand.leg_remaining = 1   # halfway along an edge
    channel, _ = _channel(environment, container)

    outcome = await channel.apply(
        RelocateMutation(observation="王二被一队兵卒裹挟到了西市。", body_id=npc_id, location_id="market"),
        step=3, agents={}, author=Author.DIRECTOR,
    )

    errand = environment.get_npc(npc_id).errand
    assert outcome is not None
    assert environment.get_body_location(npc_id) == "market"
    # The errand isn't voided; only the unfinished edge is dropped along with the origin.
    assert errand is not None and errand.leg_remaining == 0
    assert environment.snapshot_state()["npc_displaced"] == [npc_id]
    # It isn't an agent, so it's not in the agent relocation list (that one feeds agent_states).
    assert outcome.displaced == ()
    assert _ambient_at(environment, "market", observer="w1", step=4) == ["王二被一队兵卒裹挟到了西市。"]


@pytest.mark.asyncio
async def test_relocating_an_npc_is_seen_leaving_where_it_stood(container) -> None:
    environment = _env_with_two_rooms()
    npc_id = _npc(environment)
    channel, _ = _channel(environment, container)

    await channel.apply(
        RelocateMutation(observation="王二出现在西市。", body_id=npc_id, location_id="market"),
        step=3, agents={}, author=Author.DIRECTOR,
    )

    assert _ambient_at(environment, "palace", observer="w1", step=4) == ["王二离开了此地。"]


def test_displaced_npcs_are_forgotten_at_the_next_step() -> None:
    from engine.clock import GlobalClock, WorldTimeConfig

    environment = _env_with_two_rooms()
    npc_id = _npc(environment)
    environment.displace_npc(npc_id, "market")
    environment.begin_step(step=4, world_time=GlobalClock(WorldTimeConfig()).tick())

    assert environment.snapshot_state()["npc_displaced"] == []


@pytest.mark.asyncio
async def test_condition_on_an_npc_wears_off_after_half_a_day(container) -> None:
    """It can't break free on its own: ``None`` (needs outside help) would be permanent for it, so it falls back to half a day, same as the physical path."""
    environment = _env_with_two_rooms()
    npc_id = _npc(environment)
    channel, _ = _channel(environment, container)

    outcome = await channel.apply(
        ConditionMutation(observation="王二被打晕在地。", body_id=npc_id, description="昏迷不醒"),
        step=3, agents={}, author=Author.DIRECTOR,
    )

    condition = environment.get_npc(npc_id).condition
    assert outcome is not None
    assert condition is not None and condition.description == "昏迷不醒"
    assert condition.until_step == 3 + 12   # half a day ÷ 3600 seconds per step

    await channel.apply(
        ConditionMutation(observation="王二醒了过来。", body_id=npc_id, description=""),
        step=4, agents={}, author=Author.DIRECTOR,
    )
    assert environment.get_npc(npc_id).condition is None


@pytest.mark.asyncio
@pytest.mark.parametrize("effect", list(VitalityEffect))
async def test_an_npc_has_no_vitality_to_wound_heal_or_kill(container, effect) -> None:
    environment = _env_with_two_rooms()
    npc_id = _npc(environment)
    channel, _ = _channel(environment, container)

    outcome = await channel.apply(
        VitalityMutation(observation="王二受了伤。", body_id=npc_id, effect=effect),
        step=3, agents={}, author=Author.DIRECTOR,
    )

    assert outcome is None
    assert environment.get_npc(npc_id) is not None
    assert environment.get_npc(npc_id).condition is None
    assert environment.get_body_location(npc_id) == "palace"


@pytest.mark.asyncio
@pytest.mark.parametrize("make", [
    lambda nid: RelocateMutation(observation="王二出现在西市。", body_id=nid, location_id="market"),
    lambda nid: ConditionMutation(observation="王二被绑住了。", body_id=nid, description="被绑住"),
], ids=["relocate", "condition"])
async def test_system_author_cannot_touch_an_npc(container, make) -> None:
    environment = _env_with_two_rooms()
    npc_id = _npc(environment)
    channel, _ = _channel(environment, container)

    outcome = await channel.apply(make(npc_id), step=3, agents={}, author=Author.SYSTEM)

    npc = environment.get_npc(npc_id)
    assert outcome is None
    assert npc is not None and npc.condition is None
    assert environment.get_body_location(npc_id) == "palace"
