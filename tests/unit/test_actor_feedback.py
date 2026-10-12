"""Feedback in actor mode: the engine writes it as for anyone, the next decision turn lists the
subjective part the actor hasn't seen, and the actor's amendments land at the decision stage."""

from __future__ import annotations

import asyncio

import pytest

from agent.memory_types import Memory, MemoryKind, MemoryStream
from core.interfaces.actor import (
    ActorAmendments,
    ActorAnswerRejected,
    ActorDecision,
    ActorWait,
    BriefMemoryKind,
    EmotionAmendment,
    MemoryAmendment,
    RelationAmendment,
)
from tests.unit.test_actor_runtime import HERO, OTHER, _next_turn, _setup, _step_answering

_WAIT = ActorDecision(wait=ActorWait(steps=1))


async def _felt(agent, memory_id: str, text: str, *, step: int, kind: MemoryKind = MemoryKind.EVENT) -> None:
    await agent.memory_system._persist(Memory(  # noqa: SLF001
        id=memory_id, stream=MemoryStream.EXPERIENTIAL, agent_id=agent.agent_id,
        raw_content=text, stored_content=text, created_step=step, kind=kind,
    ))


async def _stored_text(container, agent, memory_id: str) -> str | None:
    collection = agent.memory_system._namespace(MemoryStream.EXPERIENTIAL)  # noqa: SLF001
    for record in await container.vector_store.list_all(collection):
        if record.id == memory_id:
            return record.payload.get("stored_content")
    return None


@pytest.mark.asyncio
async def test_a_decision_turn_lists_the_feedback_the_actor_has_not_seen(container) -> None:
    runtime, _ms, agents = _setup(container, "world-actor-feedback-listed")
    hero = agents[0]
    await _felt(hero, "e-1", "我心里一阵发冷", step=1)
    await _felt(hero, "i-1", "大哥终究容不下我", step=1, kind=MemoryKind.INSIGHT)
    await hero.memory_system.seed_factual_memory(current_step=1, raw_content="大哥在殿上驳了我的奏议")
    await hero.relation_system.apply_interaction(OTHER, trust_delta=-0.1, affection_delta=-0.1, step=1)

    _result, turns = await _step_answering(runtime, agents, _WAIT)

    feedback = turns[0].feedback
    assert [(m.memory_id, m.kind) for m in feedback.memories] == [
        ("e-1", BriefMemoryKind.EXPERIENTIAL), ("i-1", BriefMemoryKind.INSIGHT),
    ]
    assert [r.agent_id for r in feedback.relations] == [OTHER]

    await _felt(hero, "e-2", "又被他刁难了一回", step=2)
    _result, turns = await _step_answering(runtime, agents, _WAIT)
    assert [m.memory_id for m in turns[0].feedback.memories] == ["e-2"]


@pytest.mark.asyncio
async def test_amendments_land_with_the_decision(container) -> None:
    runtime, _ms, agents = _setup(container, "world-actor-feedback-amended")
    hero = agents[0]
    await _felt(hero, "e-1", "我心里一阵发冷", step=1)
    await _felt(hero, "e-2", "我觉得自己被看轻了", step=1)
    await hero.relation_system.apply_interaction(OTHER, trust_delta=-0.1, affection_delta=-0.1, step=1)

    await _step_answering(runtime, agents, ActorDecision(
        wait=ActorWait(),
        amendments=ActorAmendments(
            emotion=EmotionAmendment(emotion="anger", intensity=0.7, valence=-0.6),
            memories=(MemoryAmendment("e-1", "我面上不动声色，心里已有了计较"), MemoryAmendment("e-2", None)),
            relations=(RelationAmendment(OTHER, trust=0.05),),
        ),
    ))

    emotion = hero.personality.state.emotion
    assert (emotion.primary.value, emotion.intensity, emotion.valence) == ("anger", 0.7, -0.6)
    assert await _stored_text(container, hero, "e-1") == "我面上不动声色，心里已有了计较"
    assert await _stored_text(container, hero, "e-2") is None
    relation = await hero.relation_system.load_existing(OTHER)
    assert relation.trust_objective == 0.05


@pytest.mark.asyncio
@pytest.mark.parametrize("amendments, reason", [
    (ActorAmendments(memories=(MemoryAmendment("not-listed", "改写"),)), "memory_not_listed"),
    (ActorAmendments(relations=(RelationAmendment("agent-9", trust=0.5),)), "relation_not_listed"),
    (ActorAmendments(emotion=EmotionAmendment(emotion="自得", intensity=0.5, valence=0.2)), "unknown_emotion"),
])
async def test_only_what_the_turn_listed_can_be_amended(container, amendments, reason) -> None:
    runtime, _ms, agents = _setup(container, f"world-actor-feedback-{reason}")
    await _felt(agents[0], "e-1", "我心里一阵发冷", step=1)
    task = asyncio.create_task(runtime.run_step(agents))
    await _next_turn(runtime, task)

    with pytest.raises(ActorAnswerRejected) as refused:
        runtime.actor.submit(ActorDecision(wait=ActorWait(), amendments=amendments))

    assert refused.value.reason == reason
    runtime.actor.submit(_WAIT)
    await task


def test_amendments_are_well_formed() -> None:
    with pytest.raises(ValueError):
        MemoryAmendment("e-1", "  ")
    with pytest.raises(ValueError):
        RelationAmendment(HERO, trust=1.5)
