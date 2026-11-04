"""The character's mind as the actor follows it apart from turns, which the actor amends and
adds insights to. Amendments land at once while a turn is
open, and otherwise at the start of the next step."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from agent.goals import GoalEntity, GoalOrigin, GoalStatus
from agent.memory_types import MemoryStream
from core.interfaces.memory import MemoryKind
from core.interfaces.actor import (
    ActorAmendments,
    ActorAnswerRejected,
    ActorIdleAnswer,
    EmotionAmendment,
    GoalAmendment,
    RelationAmendment,
)
from tests.unit.test_actor_runtime import HERO, OTHER, _next_turn, _setup, _step_answering

_IDLE = ActorIdleAnswer()


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
    short_term_goals=(GoalAmendment(text="今夜先见长孙无忌"),),
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
async def test_goal_details_use_the_world_clock_even_before_the_actor_perceives(container) -> None:
    runtime, _agents, hero = await _hero_with_a_relation(container, "world-actor-goal-details")
    hero.personality.set_short_term_goal_entities([
        GoalEntity(id="appointment", text="赴约", goal_type="short_term", due_step=7,
                   origin=GoalOrigin.RESIDUE, progress_summary="已收到回信"),
        GoalEntity(id="old", text="查明缘由", goal_type="short_term", created_step=0),
        GoalEntity(id="done", text="送信", goal_type="short_term", status=GoalStatus.COMPLETED),
    ])
    for _ in range(6):
        runtime.clock.tick()

    mind = await runtime.actor_mind()
    appointment, old, done = mind.short_term_goals
    assert appointment.origin == "residue"
    assert appointment.due_step == 7
    assert appointment.progress_summary == "已收到回信"
    assert "约定在" in appointment.time_hint
    assert appointment.attention is False
    assert "仍未了结" in old.time_hint and old.attention is True
    assert done.status == "completed" and not done.time_hint and not done.attention

    runtime.clock.tick()
    at_due = (await runtime.actor_mind()).short_term_goals[0]
    assert "就在此刻" in at_due.time_hint and at_due.attention
    runtime.clock.tick()
    late = (await runtime.actor_mind()).short_term_goals[0]
    assert "时刻已过" in late.time_hint and late.attention


@pytest.mark.asyncio
async def test_editing_goal_text_preserves_identity_and_all_other_fields(container) -> None:
    runtime, agents, hero = await _hero_with_a_relation(container, "world-actor-goal-edit")
    original = GoalEntity(
        id="stg-0-1", text="原来的约定", goal_type="short_term", status=GoalStatus.INTERRUPTED,
        created_step=0, due_step=10, last_evaluated_step=0, origin=GoalOrigin.RESIDUE,
        progress_summary="尚未等到答复",
    )
    hero.personality.set_short_term_goal_entities([original])
    task = asyncio.create_task(runtime.run_step(agents))
    await _next_turn(runtime, task)
    try:
        assert await runtime.amend_actor_mind(ActorAmendments(short_term_goals=(
            GoalAmendment(id=original.id, text="修改后的约定"),
            GoalAmendment(text="另一个目标"),
        ))) is True
        edited, added = hero.personality.state.short_term_goal_entities
        assert edited == replace(original, text="修改后的约定")
        assert added.id != original.id and added.origin is GoalOrigin.COGNITIVE
        assert added.due_step is None

        await runtime.amend_actor_mind(ActorAmendments(
            short_term_goals=(), removed_short_term_goal_ids=(edited.id, added.id),
        ))
        dropped = hero.personality.state.short_term_goal_entities
        assert all(g.status is GoalStatus.FAILED for g in dropped)
        assert all(g.progress_summary == "已放下。" for g in dropped)
    finally:
        runtime.answer_actor(_IDLE)
        await task


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
async def test_pending_goal_edits_preserve_feedback_added_after_saving(container) -> None:
    runtime, agents, hero = await _hero_with_a_relation(container, "world-actor-goal-concurrent")
    first = GoalEntity(id="stg-0-1", text="旧目标", goal_type="short_term", due_step=10)
    dropped = GoalEntity(id="stg-0-2", text="不再追求", goal_type="short_term")
    hero.personality.set_short_term_goal_entities([first, dropped])
    await runtime.amend_actor_mind(ActorAmendments(
        short_term_goals=(GoalAmendment(id=first.id, text="新的表达"),),
        removed_short_term_goal_ids=(dropped.id,),
    ))
    await runtime.amend_actor_mind(ActorAmendments(insights=("我愿意继续",)))
    newer = GoalEntity(
        id="stg-0-3", text="反馈刚留下的待办", goal_type="short_term", origin=GoalOrigin.RESIDUE,
    )
    hero.personality.set_short_term_goal_entities([first, dropped, newer])

    await _step_answering(runtime, agents, _IDLE)

    by_id = {g.id: g for g in hero.personality.state.short_term_goal_entities}
    assert by_id[first.id] == replace(first, text="新的表达")
    assert by_id[dropped.id].status is GoalStatus.FAILED
    assert by_id[newer.id] == newer
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
async def test_a_turn_answered_while_an_amendment_lands_resumes_after_it(container) -> None:
    """An amendment spans awaits (the insight is embedded and stored). Answered meanwhile, the
    step must not resume alongside it and write back the goals it planned with."""
    runtime, agents, hero = await _hero_with_a_relation(container, "world-actor-mind-race")
    task = asyncio.create_task(runtime.run_step(agents))
    await _next_turn(runtime, task)
    storing, release = asyncio.Event(), asyncio.Event()
    add_insight = hero.memory_system.add_insight

    async def slow_insight(**kwargs):
        storing.set()
        await release.wait()
        return await add_insight(**kwargs)

    hero.memory_system.add_insight = slow_insight
    amending = asyncio.create_task(runtime.amend_actor_mind(_AMENDED))
    await storing.wait()

    runtime.answer_actor(_IDLE)                     # taken: the turn closes…
    await asyncio.wait({task}, timeout=0.5)
    assert not task.done()                          # …but the step waits for the amendment

    release.set()
    assert await amending is True
    await task
    await _assert_amended(container, hero)


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
    (ActorAmendments(short_term_goals=(GoalAmendment(id="missing", text="无此目标"),)), "goal_not_listed"),
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
        GoalAmendment(text=" ")
    with pytest.raises(ValueError):
        ActorAmendments(removed_short_term_goal_ids=("stg-0-1",))




@pytest.mark.asyncio
async def test_amendments_made_during_a_decision_are_a_mind_beat_of_it(container) -> None:
    from providers.trace.in_memory import InMemoryTraceSink

    sink = InMemoryTraceSink()
    runtime, _ms, agents = _setup(container, "world-actor-mind-beat", trace_sink=sink)
    hero = agents[0]
    await hero.relation_system.apply_interaction(OTHER, trust_delta=-0.1, affection_delta=-0.1, step=1)
    dropped = GoalEntity(id="stg-0-9", text="不再追求", goal_type="short_term")
    hero.personality.set_short_term_goal_entities([dropped])
    task = asyncio.create_task(runtime.run_step(agents))
    turn = await _next_turn(runtime, task)

    await runtime.amend_actor_mind(replace(_AMENDED, removed_short_term_goal_ids=(dropped.id,)))
    runtime.answer_actor(_IDLE)
    await task

    [beat] = [b for b in sink.read_beats("world-actor-mind-beat") if b.beat_kind == "mind"]
    # Landed while the decision waited: the input of that decision.
    assert (beat.step, beat.agent_id, beat.stage, beat.actor) == (turn.step, HERO, "decision", True)
    expected_to = (await hero.relation_system.load_existing(OTHER)).to_name or "某人"
    assert beat.payload == {
        "emotion": {"emotion": "anger", "intensity": 0.7, "valence": -0.6},
        "insights": ["大哥终究容不下我"],
        "relations": [{"to": expected_to, "trust": 0.05, "affection": None}],
        "long_term_goals": ["保全秦王府上下"],
        "short_term_goals": ["今夜先见长孙无忌"],
        "removed_short_term_goals": ["不再追求"],
    }
