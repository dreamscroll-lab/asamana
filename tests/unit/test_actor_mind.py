"""The character's mind as the actor follows it apart from turns, which the actor amends and
adds insights to. Amendments land at once while a turn is
open, and otherwise at the start of the next step."""

from __future__ import annotations

import asyncio

import pytest

from agent.memory_types import MemoryKind, MemoryStream
from core.interfaces.actor import (
    ActorAmendments,
    ActorAnswerRejected,
    ActorDecision,
    EmotionAmendment,
    RelationAmendment,
)
from tests.unit.test_actor_runtime import HERO, OTHER, _next_turn, _setup, _step_answering

_IDLE = ActorDecision(idle=True)


async def _insights(container, agent) -> list[str]:
    collection = agent.memory_system._namespace(MemoryStream.EXPERIENTIAL)  # noqa: SLF001
    return [
        record.payload.get("stored_content")
        for record in await container.vector_store.list_all(collection)
        if record.payload.get("kind") == MemoryKind.INSIGHT.value
    ]


_AMENDED = ActorAmendments(
    emotion=EmotionAmendment(emotion="anger", intensity=0.7, valence=-0.6),
    relations=(RelationAmendment(OTHER, trust=0.05),),
    long_term_goals=("保全秦王府上下",),
    short_term_goals=("今夜先见长孙无忌",),
    insights=("大哥终究容不下我",),
)


async def _assert_amended(container, hero) -> None:
    state = hero.personality.state
    assert (state.emotion.primary.value, state.emotion.intensity, state.emotion.valence) == ("anger", 0.7, -0.6)
    assert (await hero.relation_system.load_existing(OTHER)).trust_objective == 0.05
    assert list(state.long_term_goals) == ["保全秦王府上下"]
    assert list(state.short_term_goals) == ["今夜先见长孙无忌"]
    assert await _insights(container, hero) == ["大哥终究容不下我"]


async def _hero_with_a_relation(container, world_id: str):
    runtime, _ms, agents = _setup(container, world_id)
    hero = agents[0]
    await hero.relation_system.apply_interaction(OTHER, trust_delta=-0.1, affection_delta=-0.1, step=1)
    return runtime, agents, hero


@pytest.mark.asyncio
async def test_the_mind_shows_feeling_goals_and_relations(container) -> None:
    runtime, _agents, hero = await _hero_with_a_relation(container, "world-actor-mind")

    mind = await runtime.actor_mind()

    assert [r.agent_id for r in mind.relations] == [OTHER]
    assert mind.emotion == hero.personality.state.emotion.primary.value
    assert mind.pending is None


@pytest.mark.asyncio
async def test_amendments_saved_between_steps_land_at_the_next_step(container) -> None:
    runtime, agents, hero = await _hero_with_a_relation(container, "world-actor-mind-between")

    landed = await runtime.amend_actor_mind(_AMENDED)

    assert landed is False
    assert (await runtime.actor_mind()).pending == _AMENDED
    assert hero.personality.state.emotion.primary.value != "anger"
    assert await _insights(container, hero) == []

    _result, turns = await _step_answering(runtime, agents, _IDLE)

    assert turns
    await _assert_amended(container, hero)
    assert (await runtime.actor_mind()).pending is None


@pytest.mark.asyncio
async def test_amendments_saved_during_a_turn_land_at_once(container) -> None:
    runtime, agents, hero = await _hero_with_a_relation(container, "world-actor-mind-turn")
    task = asyncio.create_task(runtime.run_step(agents))
    await _next_turn(runtime, task)

    landed = await runtime.amend_actor_mind(_AMENDED)

    assert landed is True
    await _assert_amended(container, hero)
    runtime.answer_actor(_IDLE)
    await task


@pytest.mark.asyncio
async def test_a_later_save_overrides_an_earlier_one_and_insights_add_up(container) -> None:
    runtime, _agents, _hero = await _hero_with_a_relation(container, "world-actor-mind-merge")

    await runtime.amend_actor_mind(ActorAmendments(
        relations=(RelationAmendment(OTHER, trust=0.2, affection=0.3),), long_term_goals=("甲",),
        insights=("其一",),
    ))
    await runtime.amend_actor_mind(ActorAmendments(
        relations=(RelationAmendment(OTHER, trust=0.4),), insights=("其二",),
    ))

    pending = (await runtime.actor_mind()).pending
    assert pending.relations == (RelationAmendment(OTHER, trust=0.4, affection=0.3),)
    assert pending.long_term_goals == ("甲",)
    assert pending.insights == ("其一", "其二")


@pytest.mark.asyncio
@pytest.mark.parametrize("amendments, reason", [
    (ActorAmendments(relations=(RelationAmendment("agent-9", trust=0.5),)), "relation_not_listed"),
    (ActorAmendments(emotion=EmotionAmendment(emotion="自得", intensity=0.5, valence=0.2)), "unknown_emotion"),
])
async def test_only_what_the_mind_lists_can_be_amended(container, amendments, reason) -> None:
    runtime, _agents, _hero = await _hero_with_a_relation(container, f"world-actor-mind-{reason}")

    with pytest.raises(ActorAnswerRejected) as refused:
        await runtime.amend_actor_mind(amendments)

    assert refused.value.reason == reason
    assert (await runtime.actor_mind()).pending is None


def test_amendments_are_well_formed() -> None:
    with pytest.raises(ValueError):
        ActorAmendments(insights=("  ",))
    with pytest.raises(ValueError):
        RelationAmendment(HERO, trust=1.5)
    with pytest.raises(ValueError):
        ActorAmendments(short_term_goals=(" ",))


