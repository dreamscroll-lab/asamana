"""An actor's character inside the real step: the runtime asks the actor at the
decision and interrupt points, and every other stage runs as for anyone."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from types import SimpleNamespace

import pytest

from agent.decision import _ACTION_SPACE
from agent.personality import ActionStatus
from core.interfaces.action import ActionType, TargetAgentEffect
from core.interfaces.actor import (
    ActorAction,
    ActorAnswer,
    ActorAnswerRejected,
    ActorDecision,
    ActorInterruptAnswer,
    ActorInvitationAnswer,
    ActorDialogueTurn,
    ActorPace,
    ActorTurn,
    TurnKind,
)
from engine.run_control import StepWithdrawn
from core.interfaces.llm import LLMScene
from core.interfaces.message import Message
from core.interfaces.perception import Broadcast, BroadcastType
from core.interfaces.urgency import Urgency
from tests.unit.test_runtime_smoke import _build_agent, _build_runtime, _set_work_decision
from interaction.actor_view import step_view_for
from interaction.models import ActionSummary, MessageSummary, ObservationSummary, StepView
from tests.unit.step_view import beats, events

HERO = "agent-1"
OTHER = "agent-2"


def _setup(container, world_id: str, *, other_at: str = "garden", pace: ActorPace = ActorPace.EVERY_STEP):
    """The actor plays HERO. OTHER, an AI character, is elsewhere unless placed beside him."""
    runtime, environment, message_system = _build_runtime(container, world_id=world_id)
    environment.place_agent(agent_id=HERO, location_id="palace")
    environment.place_agent(agent_id=OTHER, location_id=other_at)
    agents = [
        _build_agent(container, world_id=world_id, agent_id=HERO, name="Li Shimin", is_main_character=True),
        _build_agent(container, world_id=world_id, agent_id=OTHER, name="Li Jiancheng", is_main_character=False),
    ]
    runtime.cast_actor(agents[0], pace=pace)
    _set_work_decision(container)
    return runtime, message_system, agents


def _work(steps: int = 1) -> ActorAction:
    return ActorAction(
        action_type=ActionType.WORK, action_description="亲自誊抄奏章", estimated_steps=steps,
    )


_IDLE = ActorDecision(idle=True)

_WORK_INDEX = next(i for i, c in enumerate(_ACTION_SPACE) if c.action_type is ActionType.WORK)
_TALK_INDEX = next(i for i, c in enumerate(_ACTION_SPACE) if c.action_type is ActionType.TALK)


def _read_as(container, **payload) -> None:
    """What the mock interpreter reads any words of the actor as."""
    container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = json.dumps({"reason": "", **payload})


async def _next_turn(runtime, step_task: "asyncio.Task") -> ActorTurn:
    for _ in range(2000):
        turn = runtime.actor.pending_turn
        if turn is not None:
            return turn
        if step_task.done():
            raise AssertionError("The step finished without asking the actor")
        await asyncio.sleep(0)
    raise AssertionError("No turn opened")


async def _step_answering(runtime, agents, *answers: "ActorAnswer"):
    """Run one step, answering each turn it opens in order."""
    task = asyncio.create_task(runtime.run_step(agents))
    turns = []
    for answer in answers:
        turns.append(await _next_turn(runtime, task))
        runtime.answer_actor(answer)
    result = await task
    return result, turns


async def _thread(container, runtime, world_id: str, kind: str | None = None) -> list[dict]:
    """The actor's thread so far: each committed step's, then the live one's."""
    thread: list[dict] = []
    for step in sorted(await container.snapshot.list_steps(world_id)):
        thread.extend((await container.snapshot.load(world_id, step)).actor_thread)
    thread.extend(runtime.actor_live_thread)
    return [e for e in thread if kind is None or e["kind"] == kind]


async def _publish_to_hero(message_system, world_id: str, *, step: int, text: str) -> None:
    await message_system.publish(Message(
        id=f"msg-{step}", world_id=world_id, sender_id=OTHER, content=text,
        recipients=[HERO], location_scope=None, created_step=step, deliver_step=step + 1,
        urgency=Urgency.NORMAL,
    ))


@pytest.mark.asyncio
async def test_the_step_waits_for_the_actor_and_executes_their_action(container) -> None:
    world_id = "world-actor-decide"
    runtime, _ms, agents = _setup(container, world_id)
    hero = agents[0]

    async def _never(**_kwargs):
        raise AssertionError("The actor's character must not consult the decision LLM")

    hero.decision_engine.decide = _never  # type: ignore[method-assign]

    result, turns = await _step_answering(runtime, agents, ActorDecision(action=_work()))

    assert [t.kind for t in turns] == [TurnKind.DECIDE]
    assert turns[0].agent_id == HERO and turns[0].step == 1
    assert turns[0].reason == ""   # asked at every step, not brought on by anything
    snapshot = await container.snapshot.load(world_id, 1)
    hero_actions = [r for r in beats(snapshot) if r["agent_id"] == HERO]
    assert [r["action_description"] for r in hero_actions] == ["亲自誊抄奏章"]
    assert any(r["agent_id"] == OTHER for r in beats(snapshot))
    assert HERO in result.scheduled_agent_ids
    # Feedback landed as for anyone.
    assert hero.personality.state.last_action is not None


@pytest.mark.asyncio
async def test_a_decision_leaves_the_goals_to_the_actor(container) -> None:
    """Nothing in the step writes goals for a performed character: the actor amends them in
    the mind."""
    runtime, _ms, agents = _setup(container, "world-actor-goals")
    hero = agents[0]
    hero.personality.set_long_term_goals(["登上大位"])
    before = list(hero.personality.state.short_term_goals)

    await _step_answering(runtime, agents, _IDLE)

    state = hero.personality.state
    assert state.long_term_goals == ["登上大位"]
    assert state.short_term_goals == before


@pytest.mark.asyncio
async def test_asked_only_when_something_reaches_the_character_at_that_pace(container) -> None:
    world_id = "world-actor-pace"
    runtime, message_system, agents = _setup(container, world_id, pace=ActorPace.WHEN_REACHED)

    # Nothing reached him: the world moves on without asking.
    result, turns = await _step_answering(runtime, agents)
    assert turns == [] and HERO not in result.scheduled_agent_ids

    await _publish_to_hero(message_system, world_id, step=1, text="今夜子时，玄武门见")
    _result, turns = await _step_answering(runtime, agents, _IDLE)

    assert turns[0].kind is TurnKind.DECIDE
    assert turns[0].reason == "有 1 件事发生"
    (perceived,) = await _thread(container, runtime, world_id, "perceived")
    assert [m["content"] for m in perceived["messages"]] == ["今夜子时，玄武门见"]


@pytest.mark.asyncio
async def test_asked_once_after_an_act_of_its_own_ends_at_that_pace(container) -> None:
    runtime, _ms, agents = _setup(container, "world-actor-pace-freed")
    await _step_answering(runtime, agents, ActorDecision(action=_work()))
    runtime.set_actor_pace(ActorPace.WHEN_REACHED)

    _result, turns = await _step_answering(runtime, agents, _IDLE)
    assert [t.kind for t in turns] == [TurnKind.DECIDE]   # free again: asked what next
    assert turns[0].reason == ""   # nothing reached him; the end of his own act is right above
    _result, turns = await _step_answering(runtime, agents)
    assert turns == []


@pytest.mark.asyncio
async def test_the_pace_changes_at_once(container) -> None:
    runtime, _ms, agents = _setup(container, "world-actor-pace-set", pace=ActorPace.WHEN_REACHED)
    _result, turns = await _step_answering(runtime, agents)
    assert turns == []
    runtime.set_actor_pace(ActorPace.EVERY_STEP)
    _result, turns = await _step_answering(runtime, agents, _IDLE)
    assert [t.kind for t in turns] == [TurnKind.DECIDE]


@pytest.mark.asyncio
async def test_asked_when_it_sees_someone_act_at_that_pace(container) -> None:
    """The actor gets what an AI character perceives: another's act seen in the room counts, as
    it would put an AI character before the world-pressure judgment."""
    runtime, _ms, agents = _setup(
        container, "world-actor-sees", other_at="palace", pace=ActorPace.WHEN_REACHED,
    )

    await _step_answering(runtime, agents)
    _result, turns = await _step_answering(runtime, agents, _IDLE)

    assert turns[0].kind is TurnKind.DECIDE


@pytest.mark.asyncio
async def test_what_the_actor_perceives_is_in_the_characters_view_of_the_step(container) -> None:
    """The actor's perception and its view of a step come from the same engine facts: something
    perceived with nothing behind it in the filtered StepView means two visibility rules have
    drifted apart."""
    world_id = "world-actor-view-agrees"
    runtime, message_system, agents = _setup(container, world_id, other_at="palace")

    await _step_answering(runtime, agents, _IDLE)
    await _publish_to_hero(message_system, world_id, step=1, text="今夜子时，玄武门见")
    await _step_answering(runtime, agents, _IDLE)

    async def seen(step: int) -> set[str]:
        view = step_view_for(StepView.from_snapshot(await container.snapshot.load(world_id, step)), HERO)
        texts: set[str] = set()
        for _scope, event in events(view):
            if isinstance(event, ActionSummary) and event.public is not None:
                texts.add(event.public.text)
            elif isinstance(event, ObservationSummary):
                texts.add(event.text)
            elif isinstance(event, MessageSummary):
                texts.add(event.content)
        return texts

    letters = {
        m["content"] for e in await _thread(container, runtime, world_id, "perceived") for m in e["messages"]
    }
    assert letters
    assert letters <= await seen(2)


@pytest.mark.asyncio
async def test_a_place_broadcast_stays_in_view_after_the_character_leaves(container) -> None:
    """Heard where it stood as the step began, kept by audience after it walks off in that step;
    the end-of-step place would lose it."""
    world_id = "world-actor-broadcast-left"
    runtime, environment, _ms = _build_runtime(container, world_id=world_id)
    environment.place_agent(agent_id=HERO, location_id="donggong")
    hero = _build_agent(container, world_id=world_id, agent_id=HERO, name="Li Shimin", is_main_character=True)
    runtime.cast_actor(hero, pace=ActorPace.EVERY_STEP)
    _set_work_decision(container)
    runtime._broadcast_channel.publish(Broadcast(  # noqa: SLF001
        content="东宫走水", source="system", broadcast_type=BroadcastType.WORLD_EVENT,
        deliver_step=1, location_scope="donggong",
    ))

    _result, turns = await _step_answering(runtime, [hero], ActorDecision(action=ActorAction(
        action_type=ActionType.MOVE, action_description="避往崇仁坊", destination_id="chongren_fang",
    )))

    (perceived,) = await _thread(container, runtime, world_id, "perceived")
    assert perceived["broadcasts"] == ["东宫走水"]
    assert environment.get_body_location(HERO) == "chongren_fang"
    view = step_view_for(StepView.from_snapshot(await container.snapshot.load(world_id, 1)), HERO)
    assert "东宫走水" in {e.content for _scope, e in events(view, "broadcast")}


@pytest.mark.asyncio
async def test_a_busy_character_can_carry_on(container) -> None:
    world_id = "world-actor-carry-on"
    runtime, message_system, agents = _setup(container, world_id)
    hero = agents[0]
    await _step_answering(runtime, agents, ActorDecision(action=_work(steps=6)))
    assert hero.personality.state.action_status == ActionStatus.IN_PROGRESS
    first = runtime._executor_registry.get_active_for_agent(HERO).execution_id

    await _publish_to_hero(message_system, world_id, step=1, text="殿下，东宫有异动")
    _result, turns = await _step_answering(runtime, agents, ActorInterruptAnswer(interrupt=False))

    assert [t.kind for t in turns] == [TurnKind.INTERRUPT]
    assert turns[0].reason == "有 1 件事发生"
    assert runtime._executor_registry.get_active_for_agent(HERO).execution_id == first


@pytest.mark.asyncio
async def test_breaking_off_leads_into_the_decision_stage(container) -> None:
    """Interrupt and decision stay two stages: the break-off goes through the interrupt
    coordinator, then the character decides like anyone torn down this step."""
    world_id = "world-actor-break-off"
    runtime, message_system, agents = _setup(container, world_id)
    await _step_answering(runtime, agents, ActorDecision(action=_work(steps=6)))
    await _publish_to_hero(message_system, world_id, step=1, text="殿下，速来")

    result, turns = await _step_answering(
        runtime, agents, ActorInterruptAnswer(interrupt=True), ActorDecision(action=_work()),
    )

    assert [t.kind for t in turns] == [TurnKind.INTERRUPT, TurnKind.DECIDE]
    assert HERO in result.scheduled_agent_ids
    snapshot = await container.snapshot.load(world_id, 2)
    assert [r for r in beats(snapshot) if r.get("interrupted_by") == HERO]


@pytest.mark.asyncio
async def test_a_break_off_outside_any_turn_lands_at_the_next_interrupt_stage(container) -> None:
    world_id = "world-actor-break-off-bare"
    runtime, _ms, agents = _setup(container, world_id)
    with pytest.raises(ActorAnswerRejected) as idle:
        runtime.actor_break_off()
    assert idle.value.reason == "nothing_to_break_off"
    await _step_answering(runtime, agents, ActorDecision(action=_work(steps=6)))
    runtime.set_actor_pace(ActorPace.WHEN_REACHED)

    runtime.actor_break_off()
    runtime.actor_break_off()   # asked once already: nothing more is recorded
    _result, turns = await _step_answering(runtime, agents, _IDLE)

    # Nothing reached him, so no INTERRUPT asks; his act ended, so he decides even at this pace.
    assert [t.kind for t in turns] == [TurnKind.DECIDE]
    snapshot = await container.snapshot.load(world_id, 2)
    assert [r for r in beats(snapshot) if r.get("interrupted_by") == HERO]
    kinds = [e["kind"] for e in await _thread(container, runtime, world_id)]
    assert kinds.count("break_off") == 1
    assert kinds[-4:] == ["break_off", "ended", "asked", "answered"]


@pytest.mark.asyncio
async def test_a_busy_character_with_nothing_happening_is_left_alone(container) -> None:
    runtime, _ms, agents = _setup(container, "world-actor-busy")

    await _step_answering(runtime, agents, ActorDecision(action=_work(steps=6)))
    _result, turns = await _step_answering(runtime, agents)

    assert turns == []


@pytest.mark.asyncio
async def test_withdrawing_a_decision_turn_ends_the_step(container) -> None:
    runtime, _ms, agents = _setup(container, "world-actor-abandon")
    task = asyncio.create_task(runtime.run_step(agents))
    await _next_turn(runtime, task)

    runtime.actor.abandon()

    with pytest.raises(StepWithdrawn):
        await task
    assert await container.snapshot.load("world-actor-abandon", 1) is None


@pytest.mark.asyncio
async def test_withdrawing_an_interrupt_turn_ends_the_step(container) -> None:
    world_id = "world-actor-abandon-interrupt"
    runtime, message_system, agents = _setup(container, world_id)
    await _step_answering(runtime, agents, ActorDecision(action=_work(steps=6)))
    await _publish_to_hero(message_system, world_id, step=1, text="殿下，速来")
    task = asyncio.create_task(runtime.run_step(agents))
    assert (await _next_turn(runtime, task)).kind is TurnKind.INTERRUPT

    runtime.actor.abandon()

    with pytest.raises(StepWithdrawn):
        await task
    assert await container.snapshot.load(world_id, 2) is None


@pytest.mark.asyncio
async def test_being_acted_upon_reaches_the_actor_with_the_next_step(container) -> None:
    """Whoever is acted on is left out of the onlookers' signals; an act landing on the character
    still asks it, once, in the step it perceives the act in. What landed on it is in the thread
    as it remembers it."""
    world_id = "world-actor-targeted"
    runtime, _ms, agents = _setup(container, world_id, pace=ActorPace.WHEN_REACHED)
    await _step_answering(runtime, agents)
    await agents[0].apply_target_effect(
        TargetAgentEffect(agent_id=HERO, factual_memory="李建成一把将我推开"), from_agent_id=OTHER, step=1,
    )

    _result, turns = await _step_answering(runtime, agents, _IDLE)

    assert turns[0].kind is TurnKind.DECIDE
    assert turns[0].reason == "有 1 件事发生"
    (landed,) = await _thread(container, runtime, world_id, "perceived")
    assert landed["acted_upon"] == ["李建成一把将我推开"]
    # Once: the step after has nothing new, so nobody is asked.
    await asyncio.wait_for(runtime.run_step(agents), timeout=10)
    assert runtime.actor.pending_turn is None


@pytest.mark.asyncio
async def test_being_named_to_listen_is_not_being_acted_upon(container) -> None:
    """A listener reaches the talk as something seen, never as an act on him: counted once, if
    at all, and never in the thread as what landed on him."""
    world_id = "world-actor-overheard"
    runtime, _ms, agents = _setup(container, world_id, pace=ActorPace.WHEN_REACHED)
    await _step_answering(runtime, agents)
    await agents[0].apply_target_effect(
        TargetAgentEffect(agent_id=HERO, factual_memory="我听见大哥与魏徵争执", overheard=True),
        from_agent_id=OTHER, step=1,
    )

    assert agents[0].personality.state.acted_upon == []
    _result, turns = await _step_answering(runtime, agents)
    assert turns == []
    assert await _thread(container, runtime, world_id, "perceived") == []


@pytest.mark.asyncio
async def test_what_landed_on_the_character_survives_a_restore(container) -> None:
    """It lands as one step commits and reaches the character the next: kept with its state,
    a restore in between must not lose it."""
    from world.initializer import _apply_stored_state

    _runtime, _ms, agents = _setup(container, "world-actor-acted-upon-restore")
    hero = agents[0]
    await hero.apply_target_effect(
        TargetAgentEffect(agent_id=HERO, factual_memory="李建成一把将我推开"), from_agent_id=OTHER, step=1,
    )
    await hero.persist_state(hero.personality.state)
    stored = await hero.agent_store.load_agent_state(hero.world_id, HERO)
    hero.personality.clear_acted_upon()

    _apply_stored_state(hero, stored)

    assert hero.personality.state.acted_upon == ["李建成一把将我推开"]


@pytest.mark.asyncio
async def test_a_letter_is_recorded_where_its_reader_stood(container) -> None:
    """The step record places a delivered letter where its reader perceived from that step, so a
    view of the step needs no second account of who perceived what."""
    world_id = "world-perceptions"
    runtime, environment, message_system = _build_runtime(container, world_id=world_id)
    environment.place_agent(agent_id=HERO, location_id="palace")
    environment.place_agent(agent_id=OTHER, location_id="palace")
    agents = [
        _build_agent(container, world_id=world_id, agent_id=HERO, name="Li Shimin", is_main_character=True),
        _build_agent(container, world_id=world_id, agent_id=OTHER, name="Li Jiancheng", is_main_character=False),
    ]
    _set_work_decision(container)
    await runtime.run_step(agents)
    await _publish_to_hero(message_system, world_id, step=1, text="速来")

    await runtime.run_step(agents)

    (letter,) = (await container.snapshot.load(world_id, 2)).events_of("message")
    assert letter["message_id"] == "msg-1"
    assert letter["receivers"] == [{"agent_id": HERO, "location_id": "palace"}]


@pytest.mark.asyncio
async def test_the_actors_long_term_goals_are_not_revised_by_the_engine(container) -> None:
    _runtime, _ms, agents = _setup(container, "world-actor-ltg")
    hero = agents[0]

    async def _never(**_kwargs):
        raise AssertionError("Long-term goals belong to the actor")

    hero.need_engine.revise_long_term_goals = _never  # type: ignore[method-assign]
    await hero.memory_system.seed_factual_memory(current_step=1, raw_content="与魏徵长谈")

    await hero.revise_long_term_goals(step=2)


@pytest.mark.asyncio
async def test_the_actors_character_does_not_reflect_on_its_own(container) -> None:
    """The character's judgments are the actor's insights, never formed by the engine."""
    _runtime, _ms, agents = _setup(container, "world-actor-reflect")
    reflected: list[str] = []

    def _engine(agent_id: str) -> SimpleNamespace:
        async def _reflect(step: int) -> None:
            reflected.append(agent_id)
        return SimpleNamespace(reflect=_reflect)

    for agent in agents:
        agent.reflection_engine = _engine(agent.agent_id)  # type: ignore[assignment]
        await agent.reflect(2)

    assert reflected == [OTHER]


@pytest.mark.asyncio
async def test_need_scoring_can_run_without_generating_goals(container) -> None:
    _runtime, _ms, agents = _setup(container, "world-actor-needs")
    hero = agents[0]

    async def _never(**_kwargs):
        raise AssertionError("Short-term goals belong to the actor")

    hero.need_engine._generate_short_term_goals = _never  # type: ignore[method-assign]  # noqa: SLF001

    evaluation = await hero.need_engine.run(
        current_step=1, personality=hero.personality, force_goal_update=True, generate_goals=False,
    )

    assert evaluation.dominant_need is not None


@pytest.mark.asyncio
async def test_the_actor_performs_one_character(container) -> None:
    """The actor is wired to its character once; every other character keeps AI cognition."""
    runtime, _ms, agents = _setup(container, "world-actor-one")

    assert agents[0].actor is not None and agents[1].actor is None
    assert runtime.actor.agent_id == HERO
    with pytest.raises(ValueError):
        runtime.cast_actor(agents[1], pace=ActorPace.EVERY_STEP)
    _result, turns = await _step_answering(runtime, agents, _IDLE)
    assert [t.agent_id for t in turns] == [HERO]


@pytest.mark.asyncio
async def test_a_decision_turn_is_briefed_on_whom_it_can_name(container) -> None:
    runtime, _ms, agents = _setup(container, "world-actor-briefing", other_at="palace")
    _result, (turn,) = await _step_answering(runtime, agents, _IDLE)

    assert turn.briefing is not None
    assert [(p.agent_id, p.present) for p in turn.briefing.people][:1] == [(OTHER, True)]


@pytest.mark.asyncio
async def test_an_action_the_character_cannot_take_is_refused_and_the_turn_stays_open(container) -> None:
    runtime, _ms, agents = _setup(container, "world-actor-refused")   # OTHER is in the garden
    task = asyncio.create_task(runtime.run_step(agents))
    await _next_turn(runtime, task)

    with pytest.raises(ActorAnswerRejected):
        runtime.answer_actor(ActorDecision(action=ActorAction(
            action_type=ActionType.TALK, action_description="找大哥说话", person_ids=(OTHER,),
        )))
    assert runtime.actor.pending_turn is not None and not task.done()

    runtime.answer_actor(ActorDecision(action=_work()))
    await task


@pytest.mark.asyncio
async def test_words_are_read_only_while_a_decision_turn_waits(container) -> None:
    world_id = "world-actor-read"
    runtime, _ms, agents = _setup(container, world_id)
    _read_as(container, decision="act", selected_index=_WORK_INDEX, action_description="亲自誊抄奏章")
    with pytest.raises(ValueError):
        await runtime.actor_say("抄奏章")

    task = asyncio.create_task(runtime.run_step(agents))
    turn = await _next_turn(runtime, task)
    said = await runtime.actor_say("抄奏章")
    assert (said.turn_id, said.words, said.reading, said.reason) == (
        turn.turn_id, "抄奏章", ActorDecision(action=_work()), None,
    )
    assert runtime.actor.pending_turn is not None   # reading answers nothing

    runtime.answer_actor(said.reading)
    await task
    with pytest.raises(ValueError):
        await runtime.actor_say("抄奏章")


@pytest.mark.asyncio
async def test_words_with_no_reading_say_why(container) -> None:
    runtime, _ms, agents = _setup(container, "world-actor-read-refused")   # OTHER is in the garden
    _read_as(container, decision="act", selected_index=_TALK_INDEX, action_description="找大哥说话",
             person_indices=[1])
    task = asyncio.create_task(runtime.run_step(agents))
    turn = await _next_turn(runtime, task)
    assert turn.briefing.people == ()   # nobody within reach to name

    refused = await runtime.actor_say("去找大哥当面说")
    _read_as(container, decision="unclear")
    unreadable = await runtime.actor_say("唉")

    assert (refused.reading, refused.reason) == (None, "talk_without_target")
    assert (unreadable.reading, unreadable.reason) == (None, "unreadable")
    runtime.answer_actor(_IDLE)
    await task


@pytest.mark.asyncio
async def test_the_thread_records_the_exchange_and_the_act_in_order(container) -> None:
    """Asked, said, answered, then the act from its start to how the character remembers its
    end; each committed step's part kept with that step's snapshot."""
    world_id = "world-actor-thread"
    runtime, _ms, agents = _setup(container, world_id)
    _read_as(container, decision="act", selected_index=_WORK_INDEX, action_description="亲自誊抄奏章")

    task = asyncio.create_task(runtime.run_step(agents))
    await _next_turn(runtime, task)
    said = await runtime.actor_say("抄奏章")
    runtime.answer_actor(said.reading)
    await task

    thread = (await container.snapshot.load(world_id, 1)).actor_thread
    assert [e["kind"] for e in thread] == ["asked", "said", "answered", "began", "ended"]
    asked, said_entry, answered, began, ended = thread
    assert said_entry["turn_id"] == answered["turn_id"] == asked["turn"]["turn_id"]
    assert answered["answer"]["action"]["action_description"] == "亲自誊抄奏章"
    assert (began["description"], began["total_steps"]) == ("亲自誊抄奏章", 1)
    assert ended["fact"] and ended["succeeded"] and ended["happened"]
    # Each entry names its step's moment in the world's own calendar, never by step.
    snapshot = await container.snapshot.load(world_id, 1)
    assert {e["time_label"] for e in thread} == {snapshot.time_label}
    assert snapshot.time_label
    assert runtime.actor_live_thread == []


@pytest.mark.asyncio
async def test_an_interrupt_turn_takes_no_words(container) -> None:
    world_id = "world-actor-interrupt-words"
    runtime, message_system, agents = _setup(container, world_id)
    await _step_answering(runtime, agents, ActorDecision(action=_work(steps=6)))
    await _publish_to_hero(message_system, world_id, step=1, text="殿下，速来")

    task = asyncio.create_task(runtime.run_step(agents))
    interrupt = await _next_turn(runtime, task)
    with pytest.raises(ValueError):
        await runtime.actor_say("去找大哥当面说")
    runtime.answer_actor(ActorInterruptAnswer(interrupt=False))
    await task

    assert interrupt.kind is TurnKind.INTERRUPT and interrupt.doing == "亲自誊抄奏章"


@pytest.mark.asyncio
async def test_a_multi_step_act_is_recorded_from_start_to_end(container) -> None:
    world_id = "world-actor-thread-multi"
    runtime, _ms, agents = _setup(container, world_id)
    await _step_answering(runtime, agents, ActorDecision(action=_work(steps=3)))
    await _step_answering(runtime, agents)
    await _step_answering(runtime, agents)

    act = [e for e in await _thread(container, runtime, world_id) if e["kind"] in ("began", "progressed", "ended")]
    assert [(e["kind"], e["step"]) for e in act] == [
        ("began", 1), ("progressed", 2), ("progressed", 3), ("ended", 3),
    ]
    assert act[0]["total_steps"] == 3
    assert [(e["elapsed_steps"], e["total_steps"]) for e in act[1:3]] == [(2, 3), (3, 3)]


@pytest.mark.asyncio
async def test_an_answer_is_in_the_thread_before_its_turn_closes(container) -> None:
    """Otherwise a reader of the thread would see the turn closed with no answer taken, until the
    step resumes."""
    world_id = "world-actor-answer-order"
    runtime, _ms, agents = _setup(container, world_id)
    events = container.event_bus.subscribe(world_id)
    await _step_answering(runtime, agents, _IDLE)

    seen = []
    while not events.empty():
        event = events.get_nowait()
        if event and event["type"] in ("actor_turn", "actor_entry"):
            seen.append(event["entry"]["kind"] if event["type"] == "actor_entry" else event["turn"] is not None)
    assert seen[:3] == ["asked", True, "answered"]
    assert seen[3] is False   # closed only after the answer was recorded


def test_the_step_loop_never_asks_who_drives_a_character() -> None:
    """Whether the AI or an actor drives a character is the character's own business: the step
    loop and the stages it runs call every character the same way."""
    from engine import interrupt_coordinator, scheduler, world_pressure
    from engine.runtime import NarrativeRuntime

    sources = [inspect.getsource(m) for m in (interrupt_coordinator, scheduler, world_pressure)] + [
        inspect.getsource(getattr(NarrativeRuntime, name))
        for name in ("run_step", "_plan_execution", "_plan_one_agent")
    ]
    for source in sources:
        assert not re.search(r"\bactor", source, re.IGNORECASE)


def _other_talks_to_hero(other) -> None:
    """OTHER, beside HERO, decides to talk with him (the only other person on his list)."""
    from core.interfaces.llm import LLMRouter
    from providers.llm.mock import SequentialMockLLM

    talk = json.dumps({
        "selected_index": _TALK_INDEX, "person_indices": [1], "action_description": "找二郎说话。",
        "inner_monologue": "有事相商。", "estimated_steps": 1,
    })
    other.decision_engine._llm_router = LLMRouter(  # noqa: SLF001
        {scene: SequentialMockLLM([talk] * 20) for scene in LLMScene}
    )


@pytest.mark.asyncio
async def test_an_invitation_asks_the_actor_and_a_refusal_turns_the_inviter_away(container) -> None:
    world_id = "world-actor-invite-refused"
    runtime, _ms, agents = _setup(container, world_id, other_at="palace")
    _other_talks_to_hero(agents[1])

    _result, turns = await _step_answering(
        runtime, agents, _IDLE, ActorInvitationAnswer(accept=False),
    )

    assert [t.kind for t in turns] == [TurnKind.DECIDE, TurnKind.INVITE]
    invite = turns[1]
    assert invite.invitation.inviter.agent_id == OTHER and invite.invitation.inviter.name
    assert (invite.invitation.action_type, invite.decided) == (ActionType.TALK, "")
    other_beat = next(r for r in beats(await container.snapshot.load(world_id, 1)) if r["agent_id"] == OTHER)
    assert other_beat["succeeded"] is False and "拒绝" in other_beat["outcome"]
    # Refusing leaves the character as it was: idle, nothing begun.
    assert agents[0].personality.state.action_status == ActionStatus.IDLE
    assert await _thread(container, runtime, world_id, "began") == []


@pytest.mark.asyncio
async def test_accepting_an_invitation_tells_the_actor_their_own_act_came_to_nothing(container) -> None:
    world_id = "world-actor-invite-accepted"
    runtime, _ms, agents = _setup(container, world_id, other_at="palace")
    # Same tier, so his TALK, which enlists someone, walks before HERO's solo act: had HERO
    # walked first, the invitation would find him busy and never be asked.
    agents[1].is_main_character = True
    _other_talks_to_hero(agents[1])

    _result, turns = await _step_answering(
        runtime, agents, ActorDecision(action=_work()), ActorInvitationAnswer(accept=True),
    )

    assert turns[1].kind is TurnKind.INVITE and turns[1].decided == "亲自誊抄奏章"
    acts = [e for e in await _thread(container, runtime, world_id) if e["kind"] in ("began", "ended")]
    own_began, own_ended, joined = acts[:3]
    assert own_began["description"] == "亲自誊抄奏章"
    assert own_ended["happened"] is False and "卷入他人的事" in own_ended["fact"]
    assert joined["kind"] == "began" and "邀入" in joined["description"]


@pytest.mark.asyncio
async def test_an_ai_invitee_is_not_asked(container) -> None:
    """Only a performed character answers an invitation; the AI always joins."""
    world_id = "world-actor-invite-ai"
    runtime, _ms, agents = _setup(container, world_id, other_at="palace")
    talk = ActorAction(
        action_type=ActionType.TALK, action_description="找大哥说话", person_ids=(OTHER,),
    )
    _result, turns = await _step_answering(
        runtime, agents, ActorDecision(action=talk), ActorDialogueTurn(line="有事相商", end=True),
    )

    assert [t.kind for t in turns] == [TurnKind.DECIDE, TurnKind.SPEAK]
    other_beat = next(r for r in beats(await container.snapshot.load(world_id, 1)) if r["agent_id"] == OTHER)
    assert other_beat["action_type"] == ActionType.TALK.value


@pytest.mark.asyncio
async def test_an_act_that_fails_to_start_is_told_and_the_actor_is_asked_again(container) -> None:
    world_id = "world-actor-start-failed"
    runtime, _ms, agents = _setup(container, world_id)
    executor = runtime._executor_registry.get_executor(ActionType.WORK)  # noqa: SLF001

    async def _broken(*_args, **_kwargs):
        raise RuntimeError("boom")

    executor.start = _broken  # type: ignore[method-assign]
    await _step_answering(runtime, agents, ActorDecision(action=_work()))

    acts = [e for e in await _thread(container, runtime, world_id) if e["kind"] in ("began", "ended")]
    assert [(e["kind"], e.get("fact"), e.get("happened")) for e in acts] == [
        ("began", None, None), ("ended", "", False),
    ]
    # Free again: even at this pace it is asked what next, not left waiting on the lost act.
    runtime.set_actor_pace(ActorPace.WHEN_REACHED)
    _result, turns = await _step_answering(runtime, agents, _IDLE)
    assert [t.kind for t in turns] == [TurnKind.DECIDE]


@pytest.mark.asyncio
async def test_an_act_that_fails_to_complete_is_told_as_ended_with_nothing_adjudicated(container) -> None:
    """The executor fails at completion: the character is released to idle, and the actor is told
    the act ended, or its card would read "in progress" for good."""
    world_id = "world-actor-complete-failed"
    runtime, _ms, agents = _setup(container, world_id)
    executor = runtime._executor_registry.get_executor(ActionType.WORK)  # noqa: SLF001

    async def _broken(*_args, **_kwargs):
        raise RuntimeError("boom")

    executor.complete = _broken  # type: ignore[method-assign]
    await _step_answering(runtime, agents, ActorDecision(action=_work()))

    acts = [e for e in await _thread(container, runtime, world_id) if e["kind"] in ("began", "ended")]
    assert [(e["kind"], e.get("fact"), e.get("happened")) for e in acts] == [
        ("began", None, None), ("ended", "", True),
    ]
    assert agents[0].personality.state.action_status == ActionStatus.IDLE


@pytest.mark.asyncio
async def test_accepting_while_idle_sets_nothing_aside(container) -> None:
    """With no act of its own this step, joining gives up nothing: no "came to nothing" line."""
    world_id = "world-actor-invite-idle"
    runtime, _ms, agents = _setup(container, world_id, other_at="palace")
    _other_talks_to_hero(agents[1])

    _result, turns = await _step_answering(runtime, agents, _IDLE, ActorInvitationAnswer(accept=True))

    assert turns[1].kind is TurnKind.INVITE and turns[1].decided == ""
    began = await _thread(container, runtime, world_id, "began")
    assert len(began) == 1 and "邀入" in began[0]["description"]
    assert all(e["happened"] for e in await _thread(container, runtime, world_id, "ended"))


@pytest.mark.asyncio
async def test_amendments_during_an_invitation_wait_for_the_next_step(container) -> None:
    """The invitation is asked after the step's plan is laid: landing goals then would be written
    over when the plan commits."""
    from core.interfaces.actor import ActorAmendments

    runtime, _ms, agents = _setup(container, "world-actor-invite-amend", other_at="palace")
    agents[1].is_main_character = True
    _other_talks_to_hero(agents[1])
    hero = agents[0]

    task = asyncio.create_task(runtime.run_step(agents))
    await _next_turn(runtime, task)
    runtime.answer_actor(ActorDecision(action=_work()))
    for _ in range(2000):
        turn = runtime.actor.pending_turn
        if turn is not None and turn.kind is TurnKind.INVITE:
            break
        await asyncio.sleep(0)
    landed = await runtime.amend_actor_mind(ActorAmendments(long_term_goals=("保全秦王府上下",)))
    runtime.answer_actor(ActorInvitationAnswer(accept=False))
    await task

    assert landed is False
    # He invites again next step; the amendment lands at its start.
    await _step_answering(runtime, agents, _IDLE, ActorInvitationAnswer(accept=False))
    assert list(hero.personality.state.long_term_goals) == ["保全秦王府上下"]


def _talk_to_other() -> ActorDecision:
    return ActorDecision(action=ActorAction(
        action_type=ActionType.TALK, action_description="找大哥说话", person_ids=(OTHER,),
    ))


def _other_replies(container, line: str) -> None:
    """OTHER's lines, and both sides' memories of the conversation."""
    container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = json.dumps({
        "thought": "", "line": line, "end": False,
        "fact": "我和他谈了一场。", "success": True, "relation": "neutral",
    }, ensure_ascii=False)


@pytest.mark.asyncio
async def test_a_conversation_with_the_performed_character_is_spoken_in_the_step_it_starts(container) -> None:
    world_id = "world-actor-talk-live"
    runtime, _ms, agents = _setup(container, world_id, other_at="palace")
    _other_replies(container, "何事？")

    _result, turns = await _step_answering(
        runtime, agents, _talk_to_other(), ActorDialogueTurn(line="大哥留步"), ActorDialogueTurn(line="改日再说", end=True),
    )

    assert [t.kind for t in turns] == [TurnKind.DECIDE, TurnKind.SPEAK, TurnKind.SPEAK]
    assert [t.line for t in turns[2].conversation.dialogue] == ["大哥留步", "何事？"]
    thread = await _thread(container, runtime, world_id)
    lines = [
        (e["dialogue_turn"]["speaker_id"], e["dialogue_turn"]["line"], e["end"])
        for e in thread if e["kind"] == "dialogue_turn"
    ]
    assert lines == [(HERO, "大哥留步", False), (OTHER, "何事？", False), (HERO, "改日再说", True)]
    # SPEAK turns leave their lines as their record, no asked/answered pair.
    assert [e["turn"]["kind"] for e in thread if e["kind"] == "asked"] == ["decide"]
    assert len([e for e in thread if e["kind"] == "answered"]) == 1
    # Spoken and remembered within the one step.
    assert [e["kind"] for e in thread if e["kind"] in ("began", "ended")] == ["began", "ended"]
    hero_beat = next(r for r in beats(await container.snapshot.load(world_id, 1)) if r["agent_id"] == HERO)
    assert [d["line"] for d in hero_beat["dialogue"]] == ["大哥留步", "何事？", "改日再说"]


@pytest.mark.asyncio
async def test_a_turn_too_long_is_refused_and_the_turn_stays_open(container) -> None:
    runtime, _ms, agents = _setup(container, "world-actor-talk-long", other_at="palace")
    task = asyncio.create_task(runtime.run_step(agents))
    await _next_turn(runtime, task)
    runtime.answer_actor(_talk_to_other())
    for _ in range(2000):
        turn = runtime.actor.pending_turn
        if turn is not None and turn.kind is TurnKind.SPEAK:
            break
        await asyncio.sleep(0)

    with pytest.raises(ActorAnswerRejected) as refused:
        runtime.answer_actor(ActorDialogueTurn(line="话" * (turn.conversation.max_turn_chars + 1)))
    assert refused.value.reason == "turn_too_long"
    assert runtime.actor.pending_turn is turn

    runtime.answer_actor(ActorDialogueTurn(line="", end=True))
    await task


@pytest.mark.asyncio
async def test_amendments_during_a_conversation_wait_for_the_next_step(container) -> None:
    """The conversation is spoken while the step's acts are adjudicated, before its feedback
    lands: the mind is not written mid-adjudication."""
    from core.interfaces.actor import ActorAmendments

    runtime, _ms, agents = _setup(container, "world-actor-talk-amend", other_at="palace")
    task = asyncio.create_task(runtime.run_step(agents))
    await _next_turn(runtime, task)
    runtime.answer_actor(_talk_to_other())
    for _ in range(2000):
        turn = runtime.actor.pending_turn
        if turn is not None and turn.kind is TurnKind.SPEAK:
            break
        await asyncio.sleep(0)

    landed = await runtime.amend_actor_mind(ActorAmendments(long_term_goals=("保全秦王府上下",)))
    runtime.answer_actor(ActorDialogueTurn(line="", end=True))
    await task

    assert landed is False
    await _step_answering(runtime, agents, _IDLE)
    assert list(agents[0].personality.state.long_term_goals) == ["保全秦王府上下"]


@pytest.mark.asyncio
async def test_a_refusal_is_remembered_by_the_inviter_and_ends_his_foiled_attempt(container) -> None:
    """A refusal happened to him, unlike a rejection: he remembers it, and it isn't kept as an
    attempt that never landed (which would also send him to the front of the queue to try again)."""
    from agent.memory_types import MemoryStream

    world_id = "world-actor-invite-remembered"
    runtime, _ms, agents = _setup(container, world_id, other_at="palace")
    other = agents[1]
    _other_talks_to_hero(other)

    await _step_answering(runtime, agents, _IDLE, ActorInvitationAnswer(accept=False))

    collection = other.memory_system._namespace(MemoryStream.FACTUAL)  # noqa: SLF001
    remembered = [
        record.payload.get("stored_content") for record in await container.vector_store.list_all(collection)
    ]
    assert any("拒绝" in (text or "") for text in remembered)
    assert other.memory_system.recent_foiled_attempts(1) == []
    assert other._foiled_misses == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_no_intent_is_put_in_the_actors_mouth(container) -> None:
    """An actor who states no expected outcome gets none: a guessed one would enter the
    character's memory as "我打算「…」" (MOVE writes the intent into its fact)."""
    runtime, _ms, agents = _setup(container, "world-actor-no-guessed-intent")
    executor = runtime._executor_registry.get_executor(ActionType.WORK)  # noqa: SLF001
    started = []
    start = executor.start

    async def _spy(action, *args, **kwargs):
        started.append(action)
        return await start(action, *args, **kwargs)

    executor.start = _spy  # type: ignore[method-assign]
    await _step_answering(runtime, agents, ActorDecision(action=_work()))

    hero_actions = [a for a in started if a.agent_id == HERO]
    assert hero_actions and hero_actions[0].expected_outcome == ""
