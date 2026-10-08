"""Unit tests for the Agent aggregate root."""

from __future__ import annotations

import asyncio

import pytest

from agent.agent import (
    Agent,
    AgentStepPlan,
    _supplement_recent,
    _DECIDE_RECENT_FACTUAL_SUPPLEMENT_K,
    _DECIDE_RECENT_EXPERIENTIAL_SUPPLEMENT_K,
)
from agent.decision import DecisionEngine
from agent.memory import MemorySystem
from agent.memory_types import Memory, MemoryImportance, MemoryStream
from agent.need import NeedEngine, NeedEvaluation, NeedType
from agent.personality import PersonalityLayer, SoulLayer, StateLayer, EmotionState
from agent.relation import RelationSystem
from core.interfaces.action import ActionResult, ActionTarget, ActionType, AgentAction, Ref
from core.interfaces.llm import LLMRouter, LLMScene
from core.interfaces.perception import LocationView, SpatialPerception
from providers.llm.mock import MockLLMProvider
from core.interfaces.urgency import Urgency


def _neutral_emotion() -> EmotionState:
    """Pre-appraised emotion for direct _apply_feedback (pure application) tests."""
    return EmotionState(primary="neutral", intensity=0.3, valence=0.0)


def _make_personality(*, name: str = "Test", agent_id: str = "agent-1", innate_needs=()) -> PersonalityLayer:
    soul = SoulLayer(
        name=name,
        agent_id=agent_id,
        role="advisor",
        core_traits=["pragmatic"],
        core_values=["loyalty"],
        innate_needs=innate_needs,
    )
    state = StateLayer(
        step=1,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
        current_location="palace",
        dominant_need=NeedType.SOCIAL.value,
    )
    return PersonalityLayer(soul=soul, state=state)


def _make_agent(container: object, *, agent_id: str = "agent-1") -> Agent:
    router = LLMRouter({scene: MockLLMProvider() for scene in LLMScene})
    return Agent(
        world_id="world-1",
        agent_id=agent_id,
        personality=_make_personality(agent_id=agent_id),
        decision_engine=DecisionEngine(router),
        llm_router=router,
        memory_system=MemorySystem(
            router,
            container.embedding,  # type: ignore[attr-defined]
            container.vector_store,  # type: ignore[attr-defined]
            world_id="world-1",
            agent_id=agent_id,
        ),
        need_engine=NeedEngine(),
        relation_system=RelationSystem(
            container.agent_store,  # type: ignore[attr-defined]
            world_id="world-1",
            agent_id=agent_id,
        ),
        agent_store=container.agent_store,  # type: ignore[attr-defined]
        is_main_character=False,
    )


def test_build_retrieval_query_is_natural_text_no_ids_or_tags(container: object) -> None:
    """The retrieval query uses names/content (natural text), with no location_id/agent_id or key:
    tag prefixes — embedding treats those as noise, and ids never match memory prose (ids must
    not cross the embedding boundary)."""
    from types import SimpleNamespace
    from core.interfaces.perception import PerceivedPresence, LocationView, PerceivedIdentity, SpatialPerception

    agent = _make_agent(container)
    from agent.goals import GoalEntity

    # personality.state returns a copy → write the internal _state through the goal queue
    agent.personality.set_short_term_goal_entities([GoalEntity(
        id="stg-0", text="查清玄武门门禁部署", goal_type="short_term",
    )])
    spatial = SpatialPerception(
        location_id="xuanwu_gate",   # code-layer id; must not enter the query
        location_view=LocationView(name="玄武门", description="皇城北门"),
        reachable_locations=[],
        visible_agents={   # ids must not enter the query
            "agent-aaa": PerceivedPresence(PerceivedIdentity(name="李世民")),
            "agent-bbb": PerceivedPresence(PerceivedIdentity(name="尉迟恭")),
        },
        ambient_events=[],
        world_time_label="卯时",     # timestamp; must not enter the query
        current_step=1,
    )
    inbox = [SimpleNamespace(content="速控门禁")]  # _build_retrieval_query only reads .content

    q = agent._build_retrieval_query(spatial=spatial, inbox=inbox, broadcasts=[])  # noqa: SLF001

    # names/content present; ids/tags/timestamps absent
    assert "玄武门" in q.spatial_context
    assert "李世民" in q.spatial_context and "尉迟恭" in q.spatial_context
    for leak in ("location:", "visible:", "time:", "xuanwu_gate", "agent-aaa", "agent-bbb", "卯时"):
        assert leak not in q.spatial_context
    assert q.message_context == "速控门禁"          # no "messages:" prefix
    assert "messages:" not in q.message_context
    assert q.as_primary() and q.as_combined()       # structure still joins normally
    # Natural separator ("，") rather than a machine one (" | "): the latter carries no meaning
    # for embedding and never appears in memory prose
    assert " | " not in q.as_primary() and " | " not in q.as_combined()
    # Own short-term goals are intent, not perceived narrative → keep them out of the embedding
    # query (matching them against memory prose is noise)
    assert "查清玄武门门禁部署" not in q.as_combined()


def _make_action_result(
    *,
    succeeded: bool = True,
    target_agent_id: str = "agent-2",
    relation_updates: list | None = None,
) -> ActionResult:
    action = AgentAction(
        action_type=ActionType.TALK,
        action_description="Spoke with agent.",
        agent_id="agent-1",
        target=ActionTarget(acts_on=[Ref.agent(target_agent_id)], claims=[Ref.agent(target_agent_id)]),
        step=1,
    )
    # Explicit relation_updates required: _apply_feedback derives no relation delta of its own.
    if relation_updates is None:
        relation_updates = (
            [(target_agent_id, 0.04, 0.03)] if succeeded
            else [(target_agent_id, -0.05, -0.04)]
        )
    return ActionResult(
        action=action,
        expected_outcome="Build trust.",
        outcome="Conversation went well." if succeeded else "Conversation failed.",
        succeeded=succeeded,
        factual_memory="Talked at the palace.",
        relation_updates=relation_updates,
    )


def _make_spatial() -> SpatialPerception:
    return SpatialPerception(
        location_id="palace",
        location_view=LocationView(name="Palace", description="A grand palace."),
        reachable_locations=[],
        visible_agents={},
        ambient_events=[],
        world_time_label="morning",
        current_step=1,
    )


def _make_need_evaluation() -> NeedEvaluation:
    from agent.need import NeedEvaluation, NeedState
    return NeedEvaluation(
        dominant_need=NeedType.SOCIAL,
        scores={NeedType.SOCIAL: 0.6},
        active_needs=[NeedState(type=NeedType.SOCIAL, label="Connect", intensity=0.6, weight=1.0)],
        short_term_goals=["与X建立信任"],
        long_term_goals=[],
        prompt_context="你眼下最迫切的需求是：维持联系。",
    )


def _make_agent_action() -> AgentAction:
    return AgentAction(
        action_type=ActionType.TALK,
        action_description="与X进行交谈。",
        agent_id="agent-1",
        target=ActionTarget(),
        step=1,
        estimated_steps=1,
        remaining_steps=0,
    )


@pytest.mark.asyncio
async def test_revise_long_term_goals_mirrors_goals_to_personality(container: object) -> None:
    """With recent memories and a changed direction: the need layer is written back and the
    personality mirror matches (Agent owns need→personality)."""
    from agent.memory_types import MemoryStream

    router = LLMRouter(
        {scene: MockLLMProvider(fixed_response='{"goals": ["演化后的新方向"]}') for scene in LLMScene}
    )
    agent = _make_agent(container)
    agent.need_engine = NeedEngine(llm_router=router)
    agent.personality.set_long_term_goals(["旧方向"])  # the source of truth for long-term goals is personality
    assert agent.personality.state.long_term_goals != ["演化后的新方向"]
    # Objective facts are enough to trigger a review — a long-term goal being achieved or void is
    # objective and doesn't need an insight.
    await agent.memory_system.write(
        "皇帝驾崩。", MemoryStream.FACTUAL,
        personality=agent.personality, importance=MemoryImportance.CRITICAL, current_step=55,
    )

    await agent.revise_long_term_goals(step=60)

    assert agent.personality.state.long_term_goals == ["演化后的新方向"]


@pytest.mark.asyncio
async def test_revise_long_term_goals_skips_without_recent_memory(container: object) -> None:
    """No recent memories to go on → skip entirely (experience-driven); goals unchanged."""
    router = LLMRouter(
        {scene: MockLLMProvider(fixed_response='{"goals": ["不该被采用"]}') for scene in LLMScene}
    )
    agent = _make_agent(container)
    agent.need_engine = NeedEngine(llm_router=router)
    agent.personality.set_long_term_goals(["旧方向"])

    await agent.revise_long_term_goals(step=60)  # memory store empty → skip

    assert agent.personality.state.long_term_goals == ["旧方向"]
    assert agent.personality.state.long_term_goals != ["不该被采用"]


@pytest.mark.asyncio
async def test_apply_feedback_on_success_increases_trust(container: object) -> None:
    agent = _make_agent(container)
    target = "agent-2"

    relation_before = await agent.relation_system.get_or_create(target)
    trust_before = relation_before.trust_objective

    result = _make_action_result(succeeded=True, target_agent_id=target)
    await agent._apply_feedback(result=result, dominant_need=NeedType.SOCIAL, emotion=_neutral_emotion())  # noqa: SLF001

    relation_after = await agent.relation_system.get_or_create(target)
    assert relation_after.trust_objective > trust_before


@pytest.mark.asyncio
async def test_apply_feedback_on_failure_decreases_trust_more_than_success_increases(container: object) -> None:
    """Trust loss (×3 multiplier) should exceed trust gain from success."""
    agent = _make_agent(container)
    target = "agent-2"

    relation_base = await agent.relation_system.get_or_create(target)
    base_trust = relation_base.trust_objective

    result_success = _make_action_result(succeeded=True, target_agent_id=target)
    await agent._apply_feedback(result=result_success, dominant_need=NeedType.SOCIAL, emotion=_neutral_emotion())  # noqa: SLF001
    after_success = (await agent.relation_system.get_or_create(target)).trust_objective
    gain = after_success - base_trust

    result_failure = _make_action_result(succeeded=False, target_agent_id=target)
    await agent._apply_feedback(result=result_failure, dominant_need=NeedType.SOCIAL, emotion=_neutral_emotion())  # noqa: SLF001
    after_failure = (await agent.relation_system.get_or_create(target)).trust_objective
    loss = after_success - after_failure

    assert loss > gain


@pytest.mark.asyncio
async def test_apply_feedback_with_none_dominant_need_skips_need_update(container: object) -> None:
    """dominant_need=None must not crash and must not change need intensities."""
    agent = _make_agent(container)
    result = _make_action_result(succeeded=True)

    # Just assert no exception raised — need engine skips update when dominant_need is None
    await agent._apply_feedback(result=result, dominant_need=None, emotion=_neutral_emotion())  # noqa: SLF001


@pytest.mark.asyncio
async def test_finalize_ongoing_action_uses_state_dominant_need(container: object) -> None:
    """finalize_ongoing_action should read dominant_need from personality state, not pass None."""
    agent = _make_agent(container)
    # StateLayer was constructed with dominant_need=NeedType.SOCIAL.value
    assert agent.personality.state.dominant_need == NeedType.SOCIAL.value

    result = _make_action_result(succeeded=True, target_agent_id="agent-2")
    await agent.finalize_ongoing_action(result=result, step=5)


@pytest.mark.asyncio
async def test_completing_goal_writes_no_separate_completed_memory(container: object) -> None:
    """Completing a short-term goal flips its status (the loop's source of truth) but writes
    NO separate goal_completed memory: the action's lived factual_memory (recorded in
    _apply_feedback just before) already covers the event, so a bookkeeping restatement would
    only duplicate it and pollute retrieval."""
    agent = _make_agent(container)

    from agent.goals import GoalEntity, GoalStatus
    from agent.need import NeedType
    spatial = _make_spatial()
    plan = AgentStepPlan(
        agent_id="agent-1",
        step=1,
        spatial=spatial,
        inbox=[],
        broadcasts=[],
        need_evaluation=_make_need_evaluation(),
        action=_make_agent_action(),
    )
    result = _make_action_result(succeeded=True)

    # Born-zero (duration-1): begin marks IN_PROGRESS with 0 remaining, finalize completes same step.
    await agent.begin_ongoing_step(plan=plan, estimated_steps=0)
    # Seed AFTER commit (commit writes the plan's empty goal queue): the short-term goal whose
    # progress the finalize will evaluate. Goals live on personality (single source of truth).
    agent.personality.set_short_term_goal_entities([
        GoalEntity(id="stg-1-0", text="与X建立信任", goal_type="short_term",
                   related_need=NeedType.SOCIAL, created_step=1)
    ])
    from core.interfaces.llm import LLMResponse

    class _Judge:
        async def complete(self, scene, messages, **kwargs):
            return LLMResponse(
                content='{"goals": [{"reason": "已建立", "index": 1, "status": "completed"}]}',
                input_tokens=0, output_tokens=0, model="test",
            )

    agent.need_engine = NeedEngine(llm_router=_Judge())  # type: ignore[arg-type]
    await agent.finalize_ongoing_action(result=result, step=plan.step)

    # Status flipped (loop intact)...
    assert agent.personality.state.short_term_goal_entities[0].status == GoalStatus.COMPLETED
    # ...but no separate goal_completed memory was written.
    completed_memories = [
        m for m in agent.memory_system._entries.values()
        if m.triggered_by == "goal_completed"
    ]
    assert completed_memories == []
    # The lived action memory still exists (the event is not lost).
    assert any(m.triggered_by != "goal_completed" for m in agent.memory_system._entries.values())


_INTERRUPTED_VERDICT = (
    '{"goals": [{"reason": "被卷入他人之事而搁置", "index": 1, "status": "interrupted"}]}'
)


def _make_llm_cognition_agent(container: object, *, goal_eval_response: str) -> Agent:
    """Agent whose goal-evaluation scene returns a fixed verdict. Goal progress is always
    LLM-judged when a router is present; the no-router rule fallback never produces
    INTERRUPTED, so driving an interruption requires the judge LLM."""
    router = LLMRouter({
        scene: MockLLMProvider(fixed_response=goal_eval_response)
        if scene == LLMScene.NEED_GOAL_GENERATION else MockLLMProvider()
        for scene in LLMScene
    })
    return Agent(
        world_id="world-1",
        agent_id="agent-1",
        personality=_make_personality(),
        decision_engine=DecisionEngine(router),
        llm_router=router,
        memory_system=MemorySystem(
            router, container.embedding, container.vector_store,  # type: ignore[attr-defined]
            world_id="world-1", agent_id="agent-1",
        ),
        need_engine=NeedEngine(llm_router=router),
        relation_system=RelationSystem(
            container.agent_store, world_id="world-1", agent_id="agent-1",  # type: ignore[attr-defined]
        ),
        agent_store=container.agent_store,  # type: ignore[attr-defined]
        is_main_character=True,
    )


async def _judge_and_land_goals(agent: Agent, *, step: int, result: ActionResult) -> None:
    """Judge + writeback: the full goal branch of _record_completion."""

    updated = await agent._judge_goal_progress(  # noqa: SLF001
        step=step, result=result,
    )
    await agent._land_goal_progress(updated, step=step)  # noqa: SLF001


@pytest.mark.asyncio
async def test_interrupted_goal_memory_written_once_across_steps(container: object) -> None:
    """An INTERRUPTED goal records its goal_interrupted memory exactly once: only when it
    transitions into INTERRUPTED during this call. The judge never re-judges it (only ACTIVE goals
    are evaluated), but the returned entity table still carries it every time. COMPLETED writes no
    memory."""
    agent = _make_llm_cognition_agent(container, goal_eval_response=_INTERRUPTED_VERDICT)

    from agent.goals import GoalEntity, GoalStatus
    from agent.need import NeedType
    agent.personality._state.short_term_goal_entities = [  # noqa: SLF001
        GoalEntity(id="stg-1-0", text="与X建立信任", goal_type="short_term",
                   related_need=NeedType.SOCIAL, created_step=1)
    ]
    result = _make_action_result(succeeded=True)

    def _interrupted_count() -> int:
        return sum(
            1 for m in agent.memory_system._entries.values()
            if m.triggered_by == "goal_interrupted"
        )

    await _judge_and_land_goals(agent, step=1, result=result)
    assert agent.personality.state.short_term_goal_entities[0].status == GoalStatus.INTERRUPTED
    assert _interrupted_count() == 1  # recorded once, the step it's first interrupted
    # Step 2: the goal is now INTERRUPTED, no longer in the active set → the judge does not
    # re-evaluate it, but the returned entity table still carries it → no duplicate write.
    await _judge_and_land_goals(agent, step=2, result=result)
    assert _interrupted_count() == 1  # still once — it did not transition again


@pytest.mark.asyncio
async def test_interrupted_goal_memory_written_once_within_one_step(container: object) -> None:
    """This can run twice in one step (finishing the interrupted action + the synchronously
    started replacement each trigger feedback), and the same interruption may write only one
    memory. Deduping by step doesn't cover this: the interruption judged first still carries this
    step's last_evaluated_step in the full table returned the second time."""
    agent = _make_llm_cognition_agent(container, goal_eval_response=_INTERRUPTED_VERDICT)

    from agent.goals import GoalEntity, GoalStatus
    from agent.need import NeedType
    agent.personality._state.short_term_goal_entities = [  # noqa: SLF001
        GoalEntity(id="stg-1-0", text="与X建立信任", goal_type="short_term",
                   related_need=NeedType.SOCIAL, created_step=1)
    ]
    result = _make_action_result(succeeded=True)

    await _judge_and_land_goals(agent, step=1, result=result)
    await _judge_and_land_goals(agent, step=1, result=result)

    assert agent.personality.state.short_term_goal_entities[0].status == GoalStatus.INTERRUPTED
    assert sum(
        1 for m in agent.memory_system._entries.values()
        if m.triggered_by == "goal_interrupted"
    ) == 1


_RESIDUE_VERDICT = (
    '{"goals": [{"reason": "仍在推进", "index": 1, "status": "active"}], '
    '"residue": ["我答应了要去查此事"]}'
)


@pytest.mark.asyncio
async def test_action_residue_lands_in_short_term_queue(container: object) -> None:
    """Cross-step continuity: loose ends left by one step's action land in the short-term goal
    queue so the next step's motivation layer can see them.

    This is the end-to-end landing point of the mechanism — without it, loose ends could only be
    squeezed into a memory and hope embedding recall finds them.
    """
    from agent.goals import GoalEntity, GoalOrigin
    from agent.need import NeedType

    agent = _make_llm_cognition_agent(container, goal_eval_response=_RESIDUE_VERDICT)
    agent.personality._state.short_term_goal_entities = [  # noqa: SLF001
        GoalEntity(id="stg-1-0", text="与X建立信任", goal_type="short_term",
                   related_need=NeedType.SOCIAL, created_step=1)
    ]

    await _judge_and_land_goals(agent, step=2, result=_make_action_result(succeeded=True))

    entities = agent.personality.state.short_term_goal_entities
    residue = [g for g in entities if g.origin == GoalOrigin.RESIDUE]
    assert [g.text for g in residue] == ["我答应了要去查此事"]
    # The existing plan isn't pushed out by the loose end — both coexist, with different
    # eviction priorities.
    assert any(g.text == "与X建立信任" for g in entities)


@pytest.mark.asyncio
async def test_adjudication_failure_enqueues_no_residue(container: object) -> None:
    """A failed adjudication is a null step (infra failure, nothing happened in the world) → all
    writeback is skipped and no loose end is left.

    Rule 1 tier-1: a failure must never produce a fake intent that gets persisted and recalled.
    """
    from agent.goals import GoalOrigin

    agent = _make_llm_cognition_agent(container, goal_eval_response=_RESIDUE_VERDICT)
    result = _make_action_result(succeeded=False)
    result.adjudication_failed = True

    await agent._record_completion(result=result, dominant_need=NeedType.SOCIAL, step=1)

    assert not [
        g for g in agent.personality.state.short_term_goal_entities
        if g.origin == GoalOrigin.RESIDUE
    ]


@pytest.mark.asyncio
async def test_not_executed_enqueues_no_residue(container: object) -> None:
    """Foiled (precondition unmet, the action never touched the world) skips goal-eval → no loose
    end. The world didn't change, so nothing was left behind."""
    from agent.goals import GoalOrigin

    agent = _make_llm_cognition_agent(container, goal_eval_response=_RESIDUE_VERDICT)
    result = _make_action_result(succeeded=False)
    result.not_executed = True

    await agent._record_completion(result=result, dominant_need=NeedType.SOCIAL, step=1)

    assert not [
        g for g in agent.personality.state.short_term_goal_entities
        if g.origin == GoalOrigin.RESIDUE
    ]


@pytest.mark.asyncio
async def test_record_completion_isolates_slot_failure(
    container: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 4: appraise→_apply_feedback ‖ _judge_goal_progress run concurrently with
    return_exceptions=True. A failure in one slot must NOT cancel the other (a bare gather
    would cancel the in-flight apply_feedback → half-landed personality) — the failure is
    logged and the step still lands best-effort."""
    agent = _make_agent(container)

    async def _boom(**_kwargs: object) -> None:
        raise RuntimeError("goal eval blew up")

    monkeypatch.setattr(agent, "_judge_goal_progress", _boom)

    result = _make_action_result(succeeded=True)
    # Must NOT raise despite the goal-eval slot exploding.
    await agent._record_completion(result=result, dominant_need=NeedType.SOCIAL, step=1)

    # The other slot (apply_feedback) still fully landed → the lived action memory (provisional
    # factual, synchronously inserted) is present, proving it was not cancelled mid-flight.
    assert any(
        m.triggered_by == ActionType.TALK.value
        for m in agent.memory_system._entries.values()
    )


@pytest.mark.asyncio
async def test_record_completion_skips_goal_eval_for_not_executed(
    container: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A not_executed non-event can't advance goals → goal-eval is skipped entirely (saves an LLM
    call, invents no progress). apply_feedback still runs (records the foil transiently, writes no
    persistent memory)."""
    agent = _make_agent(container)

    goal_eval_calls: list[int] = []

    async def _spy_goal_eval(**kwargs: object) -> None:
        goal_eval_calls.append(1)

    monkeypatch.setattr(agent, "_judge_goal_progress", _spy_goal_eval)

    result = _make_action_result(succeeded=False, relation_updates=[])
    result.not_executed = True
    await agent._record_completion(result=result, dominant_need=NeedType.SOCIAL, step=1)

    # goal-eval never ran.
    assert goal_eval_calls == []
    # The foil wrote no persistent memory but landed in the transient buffer for decide.
    assert agent.memory_system._entries == {}
    assert agent.memory_system.recent_foiled_attempts(1)


@pytest.mark.asyncio
async def test_goal_judge_runs_concurrently_with_outcome_appraisal(
    container: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The judge doesn't depend on emotion, so it needn't wait for emotion evaluation: both are
    in flight at once."""
    agent = _make_agent(container)
    both_in_flight = asyncio.Event()
    inflight = 0
    peak = 0

    async def _meet() -> None:
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        if inflight == 2:
            both_in_flight.set()
        try:
            await asyncio.wait_for(both_in_flight.wait(), timeout=2.0)
        except asyncio.TimeoutError:  # pragma: no cover - only hit when it degrades to serial
            pass
        inflight -= 1

    async def _appraise(_result: object, _need: object) -> None:
        await _meet()
        return None

    async def _judge(**_kwargs: object) -> list[object]:
        await _meet()
        return list(agent.personality.state.short_term_goal_entities)

    monkeypatch.setattr(agent, "_appraise_outcome", _appraise)
    monkeypatch.setattr(agent, "_judge_goal_progress", _judge)

    await agent._record_completion(
        result=_make_action_result(succeeded=True), dominant_need=NeedType.SOCIAL, step=1,
    )

    assert peak == 2


@pytest.mark.asyncio
async def test_feedback_slot_sees_pre_action_goals_even_if_judge_returns_first(
    container: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The memory-importance prompt includes goals: even if the judge returns first it may not
    write back before the feedback slot finishes, or that side would see goals sometimes from
    before the action and sometimes after. Writeback happens after both slots join."""
    from agent.goals import GoalEntity

    agent = _make_agent(container)
    before = GoalEntity(id="stg-1-0", text="与X建立信任", goal_type="short_term",
                        related_need=NeedType.SOCIAL, created_step=1)
    judged = GoalEntity(id="stg-1-1", text="我答应了要去查此事", goal_type="short_term",
                        related_need=NeedType.SOCIAL, created_step=1)
    agent.personality._state.short_term_goal_entities = [before]  # noqa: SLF001
    judge_done = asyncio.Event()

    async def _judge(**_kwargs: object) -> list[GoalEntity]:
        judge_done.set()
        return [before, judged]

    seen_in_feedback: list[list[str]] = []
    real_apply = agent._apply_feedback

    async def _apply(**kwargs: object) -> None:
        await judge_done.wait()  # the judge has returned
        seen_in_feedback.append(
            [g.text for g in agent.personality.state.short_term_goal_entities]
        )
        await real_apply(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(agent, "_judge_goal_progress", _judge)
    monkeypatch.setattr(agent, "_apply_feedback", _apply)

    await agent._record_completion(
        result=_make_action_result(succeeded=True), dominant_need=NeedType.SOCIAL, step=1,
    )

    assert seen_in_feedback == [["与X建立信任"]]
    assert [g.text for g in agent.personality.state.short_term_goal_entities] == [
        "与X建立信任", "我答应了要去查此事",
    ]


@pytest.mark.asyncio
async def test_finalize_adjudication_failed_is_null_step(container: object) -> None:
    """An adjudication_failed result (the executor's judge LLM failed) → null step: the
    feedback layer writes NO narrative — no memory, no goal progress — and the agent
    returns to idle to re-plan next step (Rule 1 fallback tiers, tier-1)."""
    agent = _make_agent(container)

    from agent.goals import GoalEntity
    from agent.need import NeedType
    plan = AgentStepPlan(
        agent_id="agent-1", step=1, spatial=_make_spatial(), inbox=[], broadcasts=[],
        need_evaluation=_make_need_evaluation(), action=_make_agent_action(),
    )
    action = AgentAction(
        action_type=ActionType.WORK, action_description="整理账本",
        agent_id="agent-1", target=ActionTarget(), step=1,
    )
    result = ActionResult(
        action=action, expected_outcome="",
        outcome="我着手「整理账本」，一时未能确知是否做成。",
        succeeded=False, adjudication_failed=True,
    )

    # Begin marks IN_PROGRESS; the adjudication-failed finalize must reset to idle (null step).
    await agent.begin_ongoing_step(plan=plan, estimated_steps=0)
    # Seed AFTER commit (commit writes the plan's empty goal queue).
    agent.personality.set_short_term_goal_entities([
        GoalEntity(id="stg-1-0", text="与X建立信任", goal_type="short_term",
                   related_need=NeedType.SOCIAL, created_step=1)
    ])
    goal_status_before = agent.personality.state.short_term_goal_entities[0].status
    mem_before = len(agent.memory_system._entries)
    await agent.finalize_ongoing_action(result=result, step=plan.step)

    # No narrative memory written (covers both the action outcome and any goal memory).
    assert len(agent.memory_system._entries) == mem_before
    # Goal progress not evaluated → goal unchanged.
    assert agent.personality.state.short_term_goal_entities[0].status == goal_status_before
    # The action lifecycle reset to idle so the agent re-plans cleanly next step.
    from agent.personality import AgentActivityStatus
    assert agent.personality.state.activity_status == AgentActivityStatus.IDLE


# ---------------------------------------------------------------------------
# factual_memory=None must not produce "None ..." in recorded memory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_apply_feedback_none_factual_memory_does_not_write_none_string(container: object) -> None:
    agent = _make_agent(container)
    action = AgentAction(
        action_type=ActionType.TALK,
        action_description="spoke",
        agent_id="agent-1",
        target=ActionTarget(),
        step=1,
    )
    result = ActionResult(
        action=action,
        expected_outcome="connect",
        outcome="Conversation ended.",
        succeeded=False,
        factual_memory=None,
    )
    await agent._apply_feedback(result=result, dominant_need=NeedType.SOCIAL, emotion=_neutral_emotion())  # noqa: SLF001

    for m in agent.memory_system._entries.values():
        assert not m.stored_content.startswith("None"), (
            f"Memory starts with literal 'None': {m.stored_content!r}"
        )


@pytest.mark.asyncio
async def test_not_executed_is_not_persisted_but_reaches_decide(container: object) -> None:
    """A not_executed non-event writes no persistent memory (principle 1); it only lands in the
    transient buffer for decide to read (principle 2).

    Deeper consumers read _entries/vectors; empty here means they can't see it (principle 3,
    structural exclusion).
    """
    agent = _make_agent(container)
    action = AgentAction(
        action_type=ActionType.TALK,
        action_description="与父亲相谈",
        agent_id="agent-1",
        target=ActionTarget(),
        step=3,
    )
    result = ActionResult(
        action=action,
        expected_outcome="connect",
        outcome="李世民本想与父亲相谈，对方不在场。",
        succeeded=False,
        factual_memory="想要做：与父亲相谈，但是结果是：对方不在场",
        not_executed=True,
    )
    await agent._apply_feedback(result=result, dominant_need=NeedType.SOCIAL, emotion=_neutral_emotion())  # noqa: SLF001

    # Principles 1 + 3: not a single persistent memory was written.
    assert agent.memory_system._entries == {}
    # Principle 2: decide sees this foil via the transient buffer (rendered the same way as
    # normal memories, with a relative-recency prefix).
    foiled = agent.memory_system.recent_foiled_attempts(3)
    assert len(foiled) == 1
    assert foiled[0].startswith("（")  # relative-recency prefix
    # The body is ONLY factual_memory itself — never append a template tail (that 3rd-person
    # template lives on a 1st-person channel, and the whole world's failures share the same few
    # sentences, diluting retrieval once embedded). The cause goes in structured failure_reason.
    assert foiled[0].endswith("想要做：与父亲相谈，但是结果是：对方不在场")


@pytest.mark.asyncio
async def test_foiled_key_finds_a_physical_target_person(container: object) -> None:
    """For a PHYSICAL foil against a person, the merge key must resolve to that person.

    A blow is done on a person, so that person is in ``acts_on`` and the key must take him. If the
    key can't get the person, repeated attacks on the same person never merge.
    """
    from agent.agent import _foiled_key, _foiled_key_has_object

    physical = AgentAction(
        action_type=ActionType.PHYSICAL, action_description="扑向李元吉",
        agent_id="agent-1", step=3,
        target=ActionTarget(acts_on=[Ref.agent("agent-yuanji")]),
    )
    assert _foiled_key(physical) == ("physical", "agent-yuanji")

    talk = AgentAction(
        action_type=ActionType.TALK, action_description="与他相谈",
        agent_id="agent-1", step=3, target=ActionTarget(acts_on=[Ref.agent("agent-shimin")], claims=[Ref.agent("agent-shimin")]),
    )
    assert _foiled_key(talk) == ("talk", "agent-shimin")

    move = AgentAction(
        action_type=ActionType.MOVE, action_description="前往玄武门",
        agent_id="agent-1", step=3, target=ActionTarget(acts_on=[Ref.place("xuanwu")]),
    )
    assert _foiled_key(move) == ("move", "xuanwu")

    # No target → half a key: the type half is present, the target half empty. Consumers that
    # decide "same thing" back off on this
    work = AgentAction(
        action_type=ActionType.WORK, action_description="独自拟一份诏书",
        agent_id="agent-1", step=3, target=ActionTarget(),
    )
    assert _foiled_key(work) == ("work", "")
    assert _foiled_key_has_object(_foiled_key(work)) is False
    assert _foiled_key_has_object(_foiled_key(talk)) is True


def test_foiled_key_is_stable_across_recipient_ordering() -> None:
    """With multiple recipients the key takes the sorted full set, so it doesn't drift with the
    LLM's output order.

    SEND_MESSAGE can go to many, and ``acts_on`` keeps the LLM's index order as given. If the key
    took only the first, the same recipients written [1,2] this step and [2,1] next step would get
    two different keys, and repeated foils of the same thing would never merge.
    """
    from agent.agent import _foiled_key

    def send(ids: list[str]) -> AgentAction:
        return AgentAction(
            action_type=ActionType.SEND_MESSAGE, action_description="传话",
            agent_id="agent-1", step=3, target=ActionTarget(acts_on=[Ref.agent(a) for a in ids]),
        )

    assert _foiled_key(send(["a", "b"])) == _foiled_key(send(["b", "a"]))
    # But a different recipient set is a different thing and must not merge
    assert _foiled_key(send(["a", "b"])) != _foiled_key(send(["a"]))


def _foiled_action(step: int, *, target: str = "agent-2", desc: str = "拦住他问话") -> AgentAction:
    return AgentAction(
        action_type=ActionType.TALK, action_description=desc,
        agent_id="agent-1", step=step, target=ActionTarget(acts_on=[Ref.agent(target)], claims=[Ref.agent(target)]),
    )


@pytest.mark.asyncio
async def test_foiled_misses_counts_misses_and_clears_once_the_intent_happens(
    container: object,
) -> None:
    """Foils accumulate; once it really executes, clear it all (count + trace) — what matters is
    whether it happened, not whether it succeeded.

    Executed but failed in the world (the talk broke down) is a real narrative event, not
    starvation, and still clears it.
    """
    agent = _make_agent(container)

    for step in (1, 2, 3):
        result = _make_action_result(succeeded=False)
        result.action = _foiled_action(step)
        result.not_executed = True
        await agent._record_completion(
            result=result, dominant_need=NeedType.SOCIAL, step=step
        )
    assert agent._foiled_misses_for(_foiled_action(3), 3) == 3

    landed = _make_action_result(succeeded=False)   # it landed; the talk just broke down
    landed.action = _foiled_action(4)
    await agent._record_completion(
        result=landed, dominant_need=NeedType.SOCIAL, step=4
    )
    assert agent._foiled_misses_for(_foiled_action(4), 4) == 0
    # The trace shown to decide is cleared too (see Agent._clear_foiled for why)
    assert agent.memory_system.recent_foiled_attempts(4) == []


@pytest.mark.asyncio
async def test_a_landed_intent_clears_only_its_own_foiled_trace(container: object) -> None:
    """Only the same-key entry is cleared; other foils are unaffected."""
    agent = _make_agent(container)
    blocked = _foiled_action(1, target="agent-2")
    elsewhere = _foiled_action(1, target="agent-9", desc="去问他讨个说法")
    agent._note_foiled(blocked, step=1, text="我本想拦住他问话,却未能如愿。")
    agent._note_foiled(elsewhere, step=1, text="我本想去问他讨个说法,却未能如愿。")

    agent._clear_foiled(_foiled_action(2, target="agent-2"))

    remaining = agent.memory_system.recent_foiled_attempts(2)
    assert len(remaining) == 1 and "讨个说法" in remaining[0]


@pytest.mark.asyncio
async def test_an_unrelated_action_in_between_does_not_clear_the_misses(
    container: object,
) -> None:
    """Doing something else in between does NOT reset — it's "N times recently", not "N times in
    a row".

    Someone blocked three times may grab a letter and come back to remonstrate; the grab is effort
    toward the same thing. The only proof of resolution is that the thing actually happened (see
    _clear_foiled).
    """
    agent = _make_agent(container)
    blocked = _foiled_action(1)
    agent._note_foiled(blocked, step=1, text="没能如愿")
    agent._note_foiled(blocked, step=2, text="没能如愿")

    other = AgentAction(
        action_type=ActionType.PHYSICAL, action_description="夺过他手中的密信",
        agent_id="agent-1", step=3, target=ActionTarget(acts_on=[Ref.entity("letter", "object")]),
    )
    agent._clear_foiled(other)               # something else succeeded
    assert agent._foiled_misses_for(blocked, 3) == 2

    agent._note_foiled(blocked, step=4, text="没能如愿")
    assert agent._foiled_misses_for(blocked, 4) == 3


@pytest.mark.asyncio
async def test_a_landed_no_object_action_clears_only_the_same_thing(container: object) -> None:
    """Half a key can't say which thing, so compare the gist too — same rule as merging.

    A different WORK succeeding doesn't count; the same WORK succeeding under different wording
    must clear the trace, or it fills the whole window and counts already-resolved attempts as
    "tried N times".
    """
    def work(desc: str, step: int) -> AgentAction:
        return AgentAction(
            action_type=ActionType.WORK, action_description=desc,
            agent_id="agent-1", step=step, target=ActionTarget(),
        )

    agent = _make_agent(container)
    blocked = work("我独自前往秦王府密室，取出无忌那卷帛书细看。", 1)
    agent._note_foiled(blocked, step=1, text="我本想去密室取那卷帛书,却未能如愿。")

    agent._clear_foiled(work("我清点府中甲仗数目。", 2))          # a different WORK
    assert len(agent.memory_system.recent_foiled_attempts(2)) == 1

    agent._clear_foiled(work("我走向密室，取出无忌那卷帛书，摊开细看上面的每一字。", 2))
    assert agent.memory_system.recent_foiled_attempts(2) == []


@pytest.mark.asyncio
async def test_foiled_misses_ages_out(container: object) -> None:
    """Keys untouched for long are pruned on read — two foils far apart aren't the same
    starvation."""
    from agent.agent import _FOILED_MISS_AGE_STEPS

    agent = _make_agent(container)
    action = _foiled_action(1)
    agent._note_foiled(action, step=1, text="没能如愿")

    assert agent._foiled_misses_for(action, 1 + _FOILED_MISS_AGE_STEPS) == 1
    assert agent._foiled_misses_for(action, 2 + _FOILED_MISS_AGE_STEPS) == 0
    assert agent._foiled_misses == {}


@pytest.mark.asyncio
async def test_foiled_misses_follows_the_intent_not_the_agent(container: object) -> None:
    """The count is per action key: switch to another thing and it's 0 — the boost follows the
    blocked intent and doesn't speed up the agent globally."""
    agent = _make_agent(container)
    blocked = _foiled_action(1, target="agent-2")
    for step in (1, 2, 3):
        agent._note_foiled(blocked, step=step, text="没能如愿")

    assert agent._foiled_misses_for(blocked, 3) == 3
    assert agent._foiled_misses_for(_foiled_action(3, target="agent-9"), 3) == 0
    # Target-less actions don't count (two different WORKs adding up to one starvation is wrong)
    work = AgentAction(
        action_type=ActionType.WORK, action_description="独自拟一份诏书",
        agent_id="agent-1", step=3, target=ActionTarget(),
    )
    agent._note_foiled(work, step=3, text="没能如愿")
    assert agent._foiled_misses_for(work, 3) == 0


# ---------------------------------------------------------------------------
# The memory body holds only factual_memory — never a template tail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_apply_feedback_writes_factual_memory_verbatim(container: object) -> None:
    """Memory body == factual_memory, not one character more.

    Never append gap_description ("未能完成预期事务。", "对话自然展开。", …): every success/failure
    memory would end with the same few template sentences and go into the vector store with them —
    redundant (the fact already says it) and diluting retrieval. The cause of failure goes in the
    structured ActionResult.failure_reason, not the first-person memory body.
    """
    agent = _make_agent(container)
    action = AgentAction(
        action_type=ActionType.WORK,
        action_description="write report",
        agent_id="agent-1",
        target=ActionTarget(),
        step=1,
    )
    result = ActionResult(
        action=action,
        expected_outcome="finish",
        outcome="Report done.",
        succeeded=True,
        factual_memory="送出了报告",
    )
    await agent._apply_feedback(result=result, dominant_need=NeedType.SOCIAL, emotion=_neutral_emotion())  # noqa: SLF001

    factual = [
        m for m in agent.memory_system._entries.values() if "送出了报告" in m.stored_content
    ]
    assert factual, "Expected factual memory to be written"
    # Neither the 3rd-person outcome nor any template tail may mix into the 1st-person memory body.
    assert all(m.stored_content.strip() == "送出了报告" for m in factual)
    assert all("Report done." not in m.stored_content for m in factual)


# ---------------------------------------------------------------------------
# _llm_emotion parses LLM output and falls back on failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_emotion_returns_emotion_state_for_main_character(container: object) -> None:
    from agent.personality import EmotionType
    from core.interfaces.llm import LLMRouter, LLMScene

    # Realistic output carries a leading first-person `reason` (grounding-first lite-CoT);
    # parsing must ignore it and read only emotion/intensity/valence/gap.
    mock_provider = MockLLMProvider(fixed_response='{"reason": "这事成了，但我心里并不轻松", "emotion": "joy", "intensity": 0.6, "valence": 0.4, "gap": 0.2}')
    router = LLMRouter({scene: mock_provider for scene in LLMScene})
    agent = Agent(
        world_id="world-1",
        agent_id="agent-mc",
        personality=_make_personality(agent_id="agent-mc"),
        decision_engine=DecisionEngine(router),
        llm_router=router,
        memory_system=MemorySystem(
            router,
            container.embedding,  # type: ignore[attr-defined]
            container.vector_store,  # type: ignore[attr-defined]
            world_id="world-1",
            agent_id="agent-mc",
        ),
        need_engine=NeedEngine(),
        relation_system=RelationSystem(
            container.agent_store,  # type: ignore[attr-defined]
            world_id="world-1",
            agent_id="agent-mc",
        ),
        agent_store=container.agent_store,  # type: ignore[attr-defined]
        is_main_character=True,
    )
    result = _make_action_result(succeeded=True)
    emotion = await agent._llm_emotion(result, NeedType.SOCIAL)  # noqa: SLF001

    assert emotion is not None
    assert emotion.primary == EmotionType.JOY
    assert abs(emotion.intensity - 0.6) < 0.01
    assert abs(emotion.valence - 0.4) < 0.01


@pytest.mark.asyncio
async def test_llm_emotion_injects_situation_header(container: object) -> None:
    """Situation anchoring: the feedback-emotion prompt's first line injects "我此刻在...,
    时间为..." (first-person header). The situation comes from the cached _situation of the last
    perceive_step."""
    from core.interfaces.llm import LLMRouter, LLMScene
    from core.interfaces.perception import LocationView, Situation

    mock_provider = MockLLMProvider(
        fixed_response='{"reason": "...", "emotion": "joy", "intensity": 0.5, "valence": 0.3, "gap": 0.2}'
    )
    router = LLMRouter({scene: mock_provider for scene in LLMScene})
    agent = Agent(
        world_id="world-1",
        agent_id="agent-mc",
        personality=_make_personality(agent_id="agent-mc"),
        decision_engine=DecisionEngine(router),
        llm_router=router,
        memory_system=MemorySystem(
            router, container.embedding, container.vector_store,  # type: ignore[attr-defined]
            world_id="world-1", agent_id="agent-mc",
        ),
        need_engine=NeedEngine(),
        relation_system=RelationSystem(
            container.agent_store, world_id="world-1", agent_id="agent-mc",  # type: ignore[attr-defined]
        ),
        agent_store=container.agent_store,  # type: ignore[attr-defined]
        is_main_character=True,
    )
    # Simulate the situation runtime caches in perceive_step.
    agent._situation = Situation(
        location_view=LocationView(name="朝堂", description="百官议事的正殿"),
        time_label="辰时",
    )
    await agent._llm_emotion(_make_action_result(succeeded=True), NeedType.SOCIAL)  # noqa: SLF001

    # The LLM call sends [system, user]; join both for
    # the substring/order assertions (mirrors test_decision._decision_prompt).
    prompt = "\n".join(m.content for m in mock_provider.call_history[0])
    # first-person situation header, at the very start of the prompt
    assert "我此刻在朝堂" in prompt and "时间为辰时" in prompt
    assert prompt.index("我此刻在朝堂") < prompt.index("【我是谁】")
    # no leak of code-layer location_id / step
    assert "court_hall" not in prompt and "第3步" not in prompt


@pytest.mark.asyncio
async def test_llm_emotion_injects_agent_specific_need_label(container: object) -> None:
    """_llm_emotion injects the agent's own need labels (subjective meaning generated at build
    time), not the generic English Maslow description."""
    from agent.need import NeedState
    from core.interfaces.llm import LLMRouter, LLMScene

    # Realistic output carries a leading first-person `reason` (grounding-first lite-CoT);
    # parsing must ignore it and read only emotion/intensity/valence/gap.
    mock_provider = MockLLMProvider(fixed_response='{"reason": "这事成了，但我心里并不轻松", "emotion": "joy", "intensity": 0.6, "valence": 0.4, "gap": 0.2}')
    router = LLMRouter({scene: mock_provider for scene in LLMScene})
    # The agent's own need labels live in soul.innate_needs; need_label renders from there.
    innate = (NeedState(type=NeedType.SOCIAL, label="渴望重获父亲的认可", intensity=0.6, weight=1.0),)
    agent = Agent(
        world_id="world-1",
        agent_id="agent-mc",
        personality=_make_personality(agent_id="agent-mc", innate_needs=innate),
        decision_engine=DecisionEngine(router),
        llm_router=router,
        memory_system=MemorySystem(
            router, container.embedding, container.vector_store,  # type: ignore[attr-defined]
            world_id="world-1", agent_id="agent-mc",
        ),
        need_engine=NeedEngine(),
        relation_system=RelationSystem(
            container.agent_store, world_id="world-1", agent_id="agent-mc",  # type: ignore[attr-defined]
        ),
        agent_store=container.agent_store,  # type: ignore[attr-defined]
        is_main_character=True,
    )
    await agent._llm_emotion(_make_action_result(succeeded=True), NeedType.SOCIAL)  # noqa: SLF001

    # The LLM call sends [system, user]; join both for
    # the substring/order assertions (mirrors test_decision._decision_prompt).
    prompt = "\n".join(m.content for m in mock_provider.call_history[0])
    assert "渴望重获父亲的认可" in prompt        # agent-specific label injected
    assert "Belonging" not in prompt           # generic English description not rendered


@pytest.mark.asyncio
async def test_llm_emotion_returns_none_on_parse_failure(container: object) -> None:
    from core.interfaces.llm import LLMRouter, LLMScene

    mock_provider = MockLLMProvider(fixed_response="这是一段无法解析的乱码响应")
    router = LLMRouter({scene: mock_provider for scene in LLMScene})
    agent = Agent(
        world_id="world-1",
        agent_id="agent-mc",
        personality=_make_personality(agent_id="agent-mc"),
        decision_engine=DecisionEngine(router),
        llm_router=router,
        memory_system=MemorySystem(
            router,
            container.embedding,  # type: ignore[attr-defined]
            container.vector_store,  # type: ignore[attr-defined]
            world_id="world-1",
            agent_id="agent-mc",
        ),
        need_engine=NeedEngine(),
        relation_system=RelationSystem(
            container.agent_store,  # type: ignore[attr-defined]
            world_id="world-1",
            agent_id="agent-mc",
        ),
        agent_store=container.agent_store,  # type: ignore[attr-defined]
        is_main_character=True,
    )
    result = _make_action_result(succeeded=True)
    emotion = await agent._llm_emotion(result, NeedType.SOCIAL)  # noqa: SLF001

    assert emotion is None


def _make_bg_emotion_agent(container: object) -> "Agent":
    from core.interfaces.llm import LLMRouter, LLMScene

    mock_provider = MockLLMProvider(
        fixed_response='{"emotion": "joy", "intensity": 0.5, "valence": 0.3, "gap": 0.2}'
    )
    router = LLMRouter({scene: mock_provider for scene in LLMScene})
    return Agent(
        world_id="world-1",
        agent_id="agent-bg",
        personality=_make_personality(agent_id="agent-bg"),
        decision_engine=DecisionEngine(router),
        llm_router=router,
        memory_system=MemorySystem(
            router,
            container.embedding,  # type: ignore[attr-defined]
            container.vector_store,  # type: ignore[attr-defined]
            world_id="world-1",
            agent_id="agent-bg",
        ),
        need_engine=NeedEngine(),
        relation_system=RelationSystem(
            container.agent_store,  # type: ignore[attr-defined]
            world_id="world-1",
            agent_id="agent-bg",
        ),
        agent_store=container.agent_store,  # type: ignore[attr-defined]
        is_main_character=False,
    )


@pytest.mark.asyncio
async def test_llm_emotion_runs_for_background_agent(container: object) -> None:
    """Background agents run the LLM emotion path too — there is no cognition-mode gate.

    Feedback emotion is LLM-derived for **every** agent. There is no rule floor beneath it:
    an LLM failure writes no emotion at all (see
    test_appraise_outcome_llm_failure_writes_no_emotion).
    """
    from agent.personality import EmotionType

    agent = _make_bg_emotion_agent(container)
    result = _make_action_result(succeeded=True)
    emotion = await agent._llm_emotion(result, NeedType.SOCIAL)  # noqa: SLF001

    assert emotion is not None
    assert emotion.primary == EmotionType.JOY


@pytest.mark.asyncio
async def test_appraise_outcome_llm_failure_writes_no_emotion(container: object) -> None:
    """Emotion-evaluation LLM fails → write NO emotion (return None), but gap is still returned.

    Emotion is persistent narrative substrate that colors experiential memory; per fallback tier
    1, if we can't get it, keep the current mood and never synthesize one. Don't add a "conservative
    fallback emotion" or a rule chain deriving it from succeeded.

    gap is unaffected: it's a real value from the executor, not a cognitive product.
    """
    agent = _make_agent(container)
    agent.llm_router = _RaisingRouter()  # type: ignore[assignment]

    result = _make_action_result(succeeded=False)
    emotion = await agent._appraise_outcome(result, NeedType.SOCIAL)  # noqa: SLF001

    assert emotion is None                                 # no fabricated emotion


@pytest.mark.asyncio
async def test_apply_feedback_with_none_emotion_keeps_current_mood(container: object) -> None:
    """emotion=None (evaluation failed) lands as: keep the current mood, but the action itself
    is still recorded in action_history.

    A failure shouldn't lose both emotion and fact — the action result is a known objective event;
    only the emotion is the missing cognitive product.
    """
    from agent.personality import EmotionType

    agent = _make_agent(container)
    agent.personality.update_emotion(primary=EmotionType.ANGER, intensity=0.8, valence=-0.6)

    await agent._apply_feedback(  # noqa: SLF001
        result=_make_action_result(succeeded=False),
        dominant_need=NeedType.SOCIAL,
        emotion=None,
    )

    after = agent.personality.state.emotion
    assert after.primary == EmotionType.ANGER          # mood not overwritten, nor reset to default
    assert after.intensity == pytest.approx(0.8)
    assert agent.personality.state.last_action_succeeded is False   # the fact lands as usual


class _RaisingRouter:
    """LLM router stub whose every call raises — drives the cognition failure path."""

    async def complete(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("llm down")

    async def complete_with_retry(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("llm down")


@pytest.mark.asyncio
async def test_perceive_step_writes_urgent_message(container: object) -> None:
    """Urgent messages in inbox must be written as "[消息]" HIGH memories via perceive_step."""
    from unittest.mock import patch
    from core.interfaces.message import Message
    from agent.memory_types import MemoryImportance

    agent = _make_agent(container)
    written: list[tuple[str, dict]] = []
    original_write = agent.memory_system.write

    async def capturing_write(content: str, *args, **kwargs):
        written.append((content, kwargs))
        return await original_write(content, *args, **kwargs)

    spatial = _make_spatial()
    msg = Message(
        id="msg-1",
        world_id="world-1",
        sender_id="agent-2",
        content="紧急消息：立即行动",
        recipients=["agent-1"],
        location_scope=None,
        created_step=5,
        deliver_step=5,
        urgency=Urgency.HIGH,
    )
    with patch.object(agent.memory_system, "write", side_effect=capturing_write):
        await agent.perceive_step(
            spatial=spatial,
            inbox=[msg],
            broadcasts=[],
            step=5,
        )

    msg_writes = [(c, kw) for c, kw in written if c.startswith("[消息]")]
    assert len(msg_writes) == 1, f"Expected 1 message write, got: {[c for c, _ in written]}"
    assert "紧急消息" in msg_writes[0][0]
    assert msg_writes[0][1].get("importance") == MemoryImportance.HIGH


@pytest.mark.asyncio
async def test_perceive_step_background_agent_higher_threshold(container: object) -> None:
    """Background agent does not write ambient events (signal 0.2 < threshold 0.35)."""
    from unittest.mock import patch

    agent = _make_agent(container)  # is_main_character=False by default
    written_contents: list[str] = []
    original_write = agent.memory_system.write

    async def capturing_write(content: str, *args, **kwargs):
        written_contents.append(content)
        return await original_write(content, *args, **kwargs)

    from core.interfaces.perception import AmbientEvent, LocationView, SpatialPerception
    spatial = SpatialPerception(
        location_id="palace",
        location_view=LocationView(name="Palace", description="A grand palace."),
        reachable_locations=[],
        visible_agents={},
        ambient_events=[AmbientEvent(content="有人路过庭院")],
        world_time_label="morning",
        current_step=5,
    )
    with patch.object(agent.memory_system, "write", side_effect=capturing_write):
        await agent.perceive_step(
            spatial=spatial,
            inbox=[],
            broadcasts=[],
            step=5,
        )

    ambient_writes = [c for c in written_contents if c.startswith("[环境感知]")]
    assert len(ambient_writes) == 0, (
        f"Background agent should not write ambient (signal 0.2 < threshold 0.35), "
        f"got: {written_contents}"
    )


# ---------------------------------------------------------------------------
# decide autobiographical continuity: _supplement_recent
# Vector recall keys on the current perception query and candidates pass a hard relevance gate,
# so it misses recent events that are semantically unrelated to the current scene.
# _supplement_recent adds the recent anchors recall missed (deduped by id) into decide — the part
# a recency weight can't recover.
# ---------------------------------------------------------------------------


def _event(mid: str, *, step: int, stream: MemoryStream = MemoryStream.FACTUAL) -> Memory:
    return Memory(id=mid, stream=stream, importance=0.5, created_step=step, kind="event")


def test_supplement_recent_adds_recall_missed_recent_event() -> None:
    """A recent event that recall missed still makes it into decide's memory list."""
    recalled = [_event("r1", step=3)]
    recent = [_event("recent_only", step=5)]  # not in recalled
    result = _supplement_recent(recalled, recent, limit=5)
    ids = [m.id for m in result]
    assert "recent_only" in ids, "召回漏掉的近期事件应被补进 decide"
    assert ids == ["r1", "recent_only"], "recalled 原序在前,补充项追加在后"


def test_supplement_recent_dedups_by_id() -> None:
    """A recent event already covered by recall (same id) isn't injected twice — only the delta
    is added."""
    shared = _event("shared", step=4)
    recalled = [shared]
    recent = [shared, _event("fresh", step=5)]
    result = _supplement_recent(recalled, recent, limit=5)
    ids = [m.id for m in result]
    assert ids.count("shared") == 1, "召回已覆盖的事件不得重复注入"
    assert "fresh" in ids


def test_supplement_recent_caps_to_most_recent_limit() -> None:
    """When supplements exceed limit, keep only the most recent limit (recent is ascending by
    time → take the tail)."""
    recalled: list[Memory] = []
    recent = [_event(f"m{s}", step=s) for s in (1, 2, 3, 4)]  # ascending by time
    result = _supplement_recent(recalled, recent, limit=2)
    ids = [m.id for m in result]
    assert ids == ["m3", "m4"], f"应保留最近的 2 条,got {ids}"


def test_supplement_recent_skips_none_slots() -> None:
    """sample_recent_events' (factual, experiential) pairs may have None slots; skip them without
    crashing."""
    recalled = [_event("r1", step=2)]
    recent = [None, _event("e1", step=3), None]
    result = _supplement_recent(recalled, recent, limit=5)
    assert [m.id for m in result] == ["r1", "e1"]


def test_supplement_recent_empty_recent_is_noop() -> None:
    """With no recent events, the recalled list comes back unchanged (no change, no crash)."""
    recalled = [_event("r1", step=2), _event("r2", step=3)]
    result = _supplement_recent(recalled, [], limit=5)
    assert [m.id for m in result] == ["r1", "r2"]


def test_supplement_caps_are_factual_heavy() -> None:
    """Contract: the factual supplement quota exceeds experiential (the feeling stream tends to
    make decide drift)."""
    assert _DECIDE_RECENT_FACTUAL_SUPPLEMENT_K == 5
    assert _DECIDE_RECENT_EXPERIENTIAL_SUPPLEMENT_K < _DECIDE_RECENT_FACTUAL_SUPPLEMENT_K


def test_memory_importance_override_maps_trivial_and_send_message_urgency() -> None:
    """MOVE/REST → LOW; SEND_MESSAGE reuses the same URGENCY_TO_STRENGTH table; others
    → None (go to LLM)."""
    from core.prompts import URGENCY_TO_STRENGTH

    def _res(
        action_type: ActionType,
        urgency: Urgency = Urgency.NORMAL,
    ) -> ActionResult:
        return ActionResult(
            action=AgentAction(
                agent_id="a", step=1, action_type=action_type,
                action_description="x", urgency=urgency,
            ),
            expected_outcome="",
            outcome="",
        )
    ov = Agent._memory_importance_override
    assert ov(_res(ActionType.MOVE)) == MemoryImportance.LOW
    assert ov(_res(ActionType.REST)) == MemoryImportance.LOW
    # SEND_MESSAGE reuses the canonical table directly (no separate mapping)
    for urg in (Urgency.LOW, Urgency.NORMAL, Urgency.HIGH, Urgency.CRITICAL):
        assert ov(_res(ActionType.SEND_MESSAGE, urg)) == URGENCY_TO_STRENGTH[urg]
    # Non-trivial, non-SEND_MESSAGE → None (LLM importance evaluation as usual)
    assert ov(_res(ActionType.TALK)) is None
    assert ov(_res(ActionType.PHYSICAL)) is None
    # A not_executed non-event never reaches record_event (see
    # test_not_executed_is_not_persisted_but_reaches_decide), so _memory_importance_override
    # doesn't grade it.


@pytest.mark.asyncio
async def test_llm_emotion_injects_standing_condition_as_backdrop(container: object) -> None:
    """Feedback emotion: the same "didn't get it done" tastes different to someone free and to
    someone bound.

    The condition shares the block and label with the space-time anchor (see the first line of
    decision's 【我所处的现实】) — the four cognition paths (decision / interrupt / perception
    emotion / feedback emotion) look the same, and changing the format means changing only
    render_condition.
    """
    from core.interfaces.condition import BodyCondition
    from core.interfaces.llm import LLMRouter, LLMScene
    from core.interfaces.perception import LocationView, Situation

    mock_provider = MockLLMProvider(
        fixed_response='{"reason": "...", "emotion": "anger", "intensity": 0.6, "valence": -0.5, "gap": 0.2}'
    )
    router = LLMRouter({scene: mock_provider for scene in LLMScene})
    agent = Agent(
        world_id="world-1",
        agent_id="agent-mc",
        personality=_make_personality(agent_id="agent-mc"),
        decision_engine=DecisionEngine(router),
        llm_router=router,
        memory_system=MemorySystem(
            router, container.embedding, container.vector_store,  # type: ignore[attr-defined]
            world_id="world-1", agent_id="agent-mc",
        ),
        need_engine=NeedEngine(),
        relation_system=RelationSystem(
            container.agent_store, world_id="world-1", agent_id="agent-mc",  # type: ignore[attr-defined]
        ),
        agent_store=container.agent_store,  # type: ignore[attr-defined]
        is_main_character=True,
    )
    agent._situation = Situation(
        location_view=LocationView(name="朝堂", description="百官议事的正殿"), time_label="辰时",
    )
    agent.personality.set_condition(BodyCondition("双手被反绑", since_step=0))
    result = _make_action_result(succeeded=False)
    result.action.step = 1
    await agent._llm_emotion(result, NeedType.SOCIAL)  # noqa: SLF001

    prompt = "\n".join(m.content for m in mock_provider.call_history[0])
    assert "我此刻的处境：双手被反绑（已持续约1小时）" in prompt
    assert prompt.index("我此刻在朝堂") < prompt.index("我此刻的处境") < prompt.index("【我是谁】")
