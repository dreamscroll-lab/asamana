"""An actor's character inside the real step: the runtime asks the actor at the
decision and interrupt points, and every other stage runs as for anyone."""

from __future__ import annotations

import asyncio
import inspect
import json
import re

import pytest

from agent.decision import _ACTION_SPACE
from agent.personality import ActionStatus
from core.interfaces.action import ActionType
from core.interfaces.actor import (
    ActorAction,
    ActorAnswerRejected,
    ActorDecision,
    ActorInterruptAnswer,
    ActorTurn,
    ActorWait,
    TriggerKind,
    TurnKind,
)
from engine.run_control import StepWithdrawn
from core.interfaces.llm import LLMScene
from core.interfaces.message import Message
from core.interfaces.urgency import Urgency
from tests.unit.test_runtime_smoke import _build_agent, _build_runtime, _set_work_decision
from tests.unit.step_view import beats

HERO = "agent-1"
OTHER = "agent-2"


def _setup(container, world_id: str, *, other_at: str = "garden"):
    """The actor plays HERO. OTHER, an AI character, is elsewhere unless placed beside him."""
    runtime, environment, message_system = _build_runtime(container, world_id=world_id)
    environment.place_agent(agent_id=HERO, location_id="palace")
    environment.place_agent(agent_id=OTHER, location_id=other_at)
    agents = [
        _build_agent(container, world_id=world_id, agent_id=HERO, name="Li Shimin", is_main_character=True),
        _build_agent(container, world_id=world_id, agent_id=OTHER, name="Li Jiancheng", is_main_character=False),
    ]
    runtime.cast_actor(agents[0])
    _set_work_decision(container)
    return runtime, message_system, agents


def _work(steps: int = 1) -> ActorAction:
    return ActorAction(
        action_type=ActionType.WORK, action_description="亲自誊抄奏章", estimated_steps=steps,
    )


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


async def _step_answering(runtime, agents, *answers: "ActorDecision | ActorInterruptAnswer"):
    """Run one step, answering each turn it opens in order."""
    task = asyncio.create_task(runtime.run_step(agents))
    turns = []
    for answer in answers:
        turns.append(await _next_turn(runtime, task))
        runtime.actor.submit(answer)
    result = await task
    return result, turns


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
    snapshot = await container.snapshot.load(world_id, 1)
    hero_actions = [r for r in beats(snapshot) if r["agent_id"] == HERO]
    assert [r["action_description"] for r in hero_actions] == ["亲自誊抄奏章"]
    assert any(r["agent_id"] == OTHER for r in beats(snapshot))
    assert HERO in result.scheduled_agent_ids
    # Feedback landed as for anyone.
    assert hero.personality.state.last_action is not None


@pytest.mark.asyncio
async def test_goals_come_from_the_actor(container) -> None:
    runtime, _ms, agents = _setup(container, "world-actor-goals")
    hero = agents[0]

    await _step_answering(runtime, agents, ActorDecision(
        wait=ActorWait(),
        long_term_goals=("登上大位",),
        short_term_goals=("探明东宫动静", "联络秦王府旧部"),
    ))

    state = hero.personality.state
    assert state.long_term_goals == ["登上大位"]
    assert state.short_term_goals == ["探明东宫动静", "联络秦王府旧部"]


@pytest.mark.asyncio
async def test_a_waiting_character_is_asked_again_when_a_message_reaches_them(container) -> None:
    world_id = "world-actor-wait"
    runtime, message_system, agents = _setup(container, world_id)

    await _step_answering(runtime, agents, ActorDecision(wait=ActorWait()))
    # Nothing reached him: the world moves on without asking.
    result, turns = await _step_answering(runtime, agents)
    assert turns == [] and HERO not in result.scheduled_agent_ids

    await _publish_to_hero(message_system, world_id, step=2, text="今夜子时，玄武门见")
    _result, turns = await _step_answering(runtime, agents, ActorDecision(wait=ActorWait()))

    assert turns[0].kind is TurnKind.DECIDE
    assert [(t.kind, t.text) for t in turns[0].triggers] == [(TriggerKind.MESSAGE, "今夜子时，玄武门见")]


@pytest.mark.asyncio
async def test_a_waiting_character_is_asked_again_when_it_sees_someone_act(container) -> None:
    """The actor gets what an AI character perceives: another's act seen in the room counts, as
    it would put an AI character before the world-pressure judgment."""
    runtime, _ms, agents = _setup(container, "world-actor-sees", other_at="palace")

    await _step_answering(runtime, agents, ActorDecision(wait=ActorWait()))
    _result, turns = await _step_answering(runtime, agents, ActorDecision(wait=ActorWait()))

    assert turns[0].kind is TurnKind.DECIDE
    assert TriggerKind.AMBIENT in {t.kind for t in turns[0].triggers}


@pytest.mark.asyncio
async def test_a_timed_wait_ends_on_its_own(container) -> None:
    runtime, _ms, agents = _setup(container, "world-actor-timed-wait")

    await _step_answering(runtime, agents, ActorDecision(wait=ActorWait(steps=2)))
    _result, turns = await _step_answering(runtime, agents)
    assert turns == []
    _result, turns = await _step_answering(runtime, agents, ActorDecision(action=_work()))
    assert [t.kind for t in turns] == [TurnKind.DECIDE]


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
        runtime, agents,
        ActorInterruptAnswer(interrupt=True, thought="此事更急"),
        ActorDecision(action=_work()),
    )

    assert [t.kind for t in turns] == [TurnKind.INTERRUPT, TurnKind.DECIDE]
    assert HERO in result.scheduled_agent_ids
    snapshot = await container.snapshot.load(world_id, 2)
    interrupted = [r for r in beats(snapshot) if r.get("interrupted_by") == HERO]
    assert interrupted and "此事更急" in interrupted[0]["outcome"]


@pytest.mark.asyncio
async def test_words_said_with_a_break_off_come_back_as_a_proposal(container) -> None:
    """They are read at the decision stage, against that step's lists, and the actor confirms."""
    world_id = "world-actor-break-off-then"
    runtime, message_system, agents = _setup(container, world_id)
    hero = agents[0]
    _read_as(container, decision="act", selected_index=_WORK_INDEX,
             action_description="亲自誊抄奏章", new_short_term_goals=["赶去玄武门"])
    await _step_answering(runtime, agents, ActorDecision(action=_work(steps=6)))
    await _publish_to_hero(message_system, world_id, step=1, text="殿下，速来")

    task = asyncio.create_task(runtime.run_step(agents))
    assert (await _next_turn(runtime, task)).kind is TurnKind.INTERRUPT
    runtime.actor.submit(ActorInterruptAnswer(interrupt=True, then="抄完这份就去玄武门"))
    turn = await _next_turn(runtime, task)

    assert turn.kind is TurnKind.DECIDE
    assert turn.proposal == ActorDecision(action=_work(), short_term_goals=("赶去玄武门",))
    runtime.actor.submit(turn.proposal)
    result = await task
    assert HERO in result.scheduled_agent_ids
    assert hero.personality.state.short_term_goals == ["赶去玄武门"]


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
async def test_being_targeted_asks_the_actor_next_step(container) -> None:
    runtime, _ms, agents = _setup(container, "world-actor-targeted")
    await _step_answering(runtime, agents, ActorDecision(wait=ActorWait()))
    runtime._note_acted_upon(  # noqa: SLF001
        {a.agent_id: a for a in agents}, [{"agent_id": OTHER, "initiator_id": OTHER, "acted_upon": [HERO]}], [],
    )

    _result, turns = await _step_answering(runtime, agents, ActorDecision(wait=ActorWait()))

    assert [t.kind for t in turns[0].triggers] == [TriggerKind.TARGETED]


def test_the_actors_own_acts_do_not_count_as_being_targeted(container) -> None:
    runtime, _ms, agents = _setup(container, "world-actor-own-act")
    # The partner's record of a talk the actor started may list the actor among those affected.
    runtime._note_acted_upon(  # noqa: SLF001
        {a.agent_id: a for a in agents}, [{"agent_id": OTHER, "initiator_id": HERO, "acted_upon": [HERO]}], [],
    )
    assert not runtime.actor.take_targeted()


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
        runtime.cast_actor(agents[1])
    _result, turns = await _step_answering(runtime, agents, ActorDecision(wait=ActorWait()))
    assert [t.agent_id for t in turns] == [HERO]


@pytest.mark.asyncio
async def test_a_decision_turn_briefs_the_actor_on_what_the_character_perceives(container) -> None:
    runtime, _ms, agents = _setup(container, "world-actor-briefing", other_at="palace")

    _result, turns = await _step_answering(runtime, agents, ActorDecision(wait=ActorWait()))

    briefing = turns[0].briefing
    assert briefing is not None and briefing.location_id == "palace"
    assert [(p.agent_id, p.present) for p in briefing.people][:1] == [(OTHER, True)]


@pytest.mark.asyncio
async def test_an_action_the_character_cannot_take_is_refused_and_the_turn_stays_open(container) -> None:
    runtime, _ms, agents = _setup(container, "world-actor-refused")   # OTHER is in the garden
    task = asyncio.create_task(runtime.run_step(agents))
    await _next_turn(runtime, task)

    with pytest.raises(ActorAnswerRejected):
        runtime.actor.submit(ActorDecision(action=ActorAction(
            action_type=ActionType.TALK, action_description="找大哥说话", person_ids=(OTHER,),
        )))
    assert runtime.actor.pending_turn is not None and not task.done()

    runtime.actor.submit(ActorDecision(action=_work()))
    await task


@pytest.mark.asyncio
async def test_words_are_read_only_while_a_turn_waits(container) -> None:
    runtime, _ms, agents = _setup(container, "world-actor-read")
    _read_as(container, decision="act", selected_index=_WORK_INDEX, action_description="亲自誊抄奏章")
    with pytest.raises(ValueError):
        await runtime.interpret_actor_answer("抄奏章")

    task = asyncio.create_task(runtime.run_step(agents))
    await _next_turn(runtime, task)
    reading = await runtime.interpret_actor_answer("抄奏章")
    assert reading == ActorDecision(action=_work())
    assert runtime.actor.pending_turn is not None   # reading answers nothing

    runtime.actor.submit(reading)
    await task
    with pytest.raises(ValueError):
        await runtime.interpret_actor_answer("抄奏章")


@pytest.mark.asyncio
async def test_a_reading_the_character_cannot_act_on_is_refused_with_its_reason(container) -> None:
    runtime, _ms, agents = _setup(container, "world-actor-read-refused")   # OTHER is in the garden
    _read_as(container, decision="act", selected_index=_TALK_INDEX, action_description="找大哥说话",
             person_indices=[1])
    task = asyncio.create_task(runtime.run_step(agents))
    turn = await _next_turn(runtime, task)
    assert turn.briefing.people == ()   # nobody within reach to name

    with pytest.raises(ActorAnswerRejected) as refused:
        await runtime.interpret_actor_answer("去找大哥当面说")
    assert refused.value.reason == "talk_without_target"

    runtime.actor.submit(ActorDecision(wait=ActorWait()))
    await task


@pytest.mark.asyncio
async def test_break_off_words_the_character_cannot_act_on_propose_nothing(container) -> None:
    world_id = "world-actor-break-off-refused"
    runtime, message_system, agents = _setup(container, world_id)   # OTHER is in the garden
    _read_as(container, decision="act", selected_index=_TALK_INDEX, action_description="找大哥说话",
             person_indices=[1])
    await _step_answering(runtime, agents, ActorDecision(action=_work(steps=6)))
    await _publish_to_hero(message_system, world_id, step=1, text="殿下，速来")

    _result, turns = await _step_answering(
        runtime, agents,
        ActorInterruptAnswer(interrupt=True, then="去找大哥当面说"),
        ActorDecision(wait=ActorWait()),
    )

    assert [t.kind for t in turns] == [TurnKind.INTERRUPT, TurnKind.DECIDE]
    assert turns[0].doing == "亲自誊抄奏章"
    assert turns[1].proposal is None


def test_the_step_loop_never_asks_who_drives_a_character() -> None:
    """Whether the AI or an actor drives a character is the character's own business: the step
    loop and the stages it runs call every character the same way."""
    from engine import interrupt_coordinator, scheduler, world_pressure
    from engine.runtime import NarrativeRuntime

    sources = [inspect.getsource(m) for m in (interrupt_coordinator, scheduler, world_pressure)] + [
        inspect.getsource(getattr(NarrativeRuntime, name))
        for name in ("run_step", "_plan_execution", "_plan_one_agent", "_note_acted_upon")
    ]
    for source in sources:
        assert not re.search(r"\bactor", source, re.IGNORECASE)
