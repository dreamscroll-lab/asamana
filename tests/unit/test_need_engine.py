"""Unit tests for the agent need subsystem."""

from __future__ import annotations

import re

import pytest

from agent.motivation import ExternalDriveType, ExternalGoal
from agent.goals import (
    _SHORT_TERM_GOAL_CAP, GoalEntity, GoalOrigin, GoalStatus, enqueue_goals, _evict_over_cap,
    live_goals, parse_residue, text_to_long_term_goal_entity,
)
from agent.need import (
    LLM_RELEVANCE_WEIGHT, NeedEngine, NeedState, NeedType, ensure_maslow_baseline,
)
from agent.personality import EmotionState, PersonalityLayer, SoulLayer, StateLayer
from core.interfaces.urgency import Urgency


# Default innate needs (intensity = weight = w) for run tests that only need one populated agent.
_DEFAULT_WEIGHTS = {
    NeedType.PHYSIOLOGICAL: 0.2,
    NeedType.SAFETY: 0.5,
    NeedType.SOCIAL: 0.4,
    NeedType.ESTEEM: 0.3,
    NeedType.SELF_ACTUALIZATION: 0.2,
}


def _default_needs() -> list[NeedState]:
    return [NeedState(type=nt, label=nt.description, intensity=w, weight=w) for nt, w in _DEFAULT_WEIGHTS.items()]


def _goal(
    text: str,
    *,
    origin: GoalOrigin = GoalOrigin.COGNITIVE,
    created_step: int = 0,
    status: GoalStatus = GoalStatus.ACTIVE,
) -> GoalEntity:
    """A short-term GoalEntity with the fields these tests care about."""
    return GoalEntity(
        id=f"stg-{created_step}-{text}", text=text, goal_type="short_term",
        status=status, created_step=created_step, origin=origin,
    )


def _make_personality(
    *,
    active: list[NeedState] | None = None,
    hidden: list[NeedState] | None = None,
    long_term: list[str] | None = None,
    short_term_entities: list[GoalEntity] | None = None,
    intensities: dict[str, float] | None = None,
    dominant_emotion: str = "neutral",
    valence: float = 0.0,
    life_goal: str = "Achieve harmony.",
    name: str = "Test Agent",
    agent_id: str = "agent-1",
) -> PersonalityLayer:
    """Build a PersonalityLayer carrying the static needs on soul.innate_needs and the dynamic
    need/goal state on StateLayer — the single source of truth the stateless NeedEngine reads.

    innate_needs is taken EXACTLY as given (no Maslow baseline-ensure) so a test can probe a
    precise need set; need_intensities defaults to each active need's seed intensity.
    """
    active = _default_needs() if active is None else list(active)
    hidden = [] if hidden is None else list(hidden)
    innate = tuple(active + hidden)
    if intensities is None:
        intensities = {n.type.value: n.intensity for n in innate if not n.is_hidden}
    lt = list(long_term or [])
    ste = list(short_term_entities or [])
    soul = SoulLayer(name=name, agent_id=agent_id, life_goal=life_goal, innate_needs=innate)
    state = StateLayer(
        agent_id=agent_id,
        emotion=EmotionState(primary=dominant_emotion, intensity=0.3, valence=valence),
        need_intensities=dict(intensities),
        long_term_goals=lt,
        long_term_goal_entities=[text_to_long_term_goal_entity(t) for t in lt],
        short_term_goal_entities=ste,
        short_term_goals=[g.text for g in ste if g.status in (GoalStatus.ACTIVE, GoalStatus.INTERRUPTED)],
    )
    return PersonalityLayer(soul=soul, state=state)


def _seeded_goal(text: str = "稳住眼下的局势", *, step: int = 1) -> GoalEntity:
    """Seed a live goal directly, for cases that test only the queue/judging, not generation.

    Don't rely on run() generating one: that path either needs a router (which changes what the
    no-judge cases test) or produces nothing.
    """
    return GoalEntity(
        id=f"stg-{step}-0", text=text, goal_type="short_term",
        related_need=NeedType.SAFETY, created_step=step,
    )


class _GoalsLLM:
    """Yield batches of short-term goals in order. In production NeedEngine always has a router,
    so generation is tested in that shape."""

    def __init__(self, *batches: list[str]) -> None:
        self._batches = list(batches) or [["稳住眼下的局势"]]
        self._calls = 0

    async def complete(self, scene, messages, temperature=0.7, max_tokens=1000, **kwargs):
        import json as _json

        from core.interfaces.llm import LLMResponse

        batch = self._batches[min(self._calls, len(self._batches) - 1)]
        self._calls += 1
        return LLMResponse(
            content=_json.dumps({"goals": batch}, ensure_ascii=False),
            input_tokens=0, output_tokens=0, model="test",
        )


def _commit(personality: PersonalityLayer, result) -> None:
    """Mirror Agent.begin_ongoing_step: write a run() NeedEvaluation's need/goal state back into personality."""
    personality.update_needs(
        step=personality.state.step,
        active_needs=[n.type.value for n in result.active_needs],
        dominant_need=result.dominant_need.value if result.dominant_need is not None else None,
    )
    personality.set_short_term_goal_entities(result.short_term_goal_entities)


def test_personality_need_label_prefers_agent_label_else_description() -> None:
    """need_label: a matching innate_need with a non-empty label returns that agent's own label;
    no match or an empty label falls back to the generic NeedType.description."""
    personality = _make_personality(active=[
        NeedState(type=NeedType.SOCIAL, label="渴望重获父亲的认可", intensity=0.6, weight=1.0),
        NeedState(type=NeedType.ESTEEM, label="", intensity=0.5, weight=1.0),
    ])
    assert personality.need_label(NeedType.SOCIAL) == "渴望重获父亲的认可"   # agent's own label
    assert personality.need_label(NeedType.ESTEEM) == NeedType.ESTEEM.description  # empty label → fallback
    assert personality.need_label(NeedType.SAFETY) == NeedType.SAFETY.description  # no match → fallback


def test_need_engine_to_prompt_context_renders_dominant_and_competing() -> None:
    personality = _make_personality(active=[
        NeedState(type=NeedType.SAFETY, label="减少不确定性", intensity=0.9, weight=1.0),
        NeedState(type=NeedType.ESTEEM, label="维护地位", intensity=0.5, weight=1.0),
        NeedState(type=NeedType.SOCIAL, label="维系盟友", intensity=0.3, weight=1.0),
    ])
    engine = NeedEngine()
    visible = [
        NeedState(type=NeedType.SAFETY, label="减少不确定性", intensity=0.9, weight=1.0),
        NeedState(type=NeedType.ESTEEM, label="维护地位", intensity=0.5, weight=1.0),
        NeedState(type=NeedType.SOCIAL, label="维系盟友", intensity=0.3, weight=1.0),
    ]
    text = engine.to_prompt_context(
        personality,
        NeedType.SAFETY,
        visible,
        [_goal("先稳住局面")],
        now_step=0,
    )
    assert "眼下最迫切的需求：safety（减少不确定性）" in text
    assert "同时也在意（仍在牵扯注意力）：esteem（维护地位）" in text
    assert "social" not in text  # intensity 0.3 ≤ 0.4 threshold
    # Short-term goals are an ordered queue: rendered numbered, with the ordering stated to the
    # decision.
    assert "我的短期目标（定了时候的排在最前；其余按先后，越靠后越新；自行斟酌先推进哪条）：\n  1. 先稳住局面" in text


def test_to_prompt_context_competing_keeps_only_strongest_rival() -> None:
    """With several competing needs (>0.4), list only the strongest one: a clear dominant vs. a
    single rival, without flattening the hierarchy."""
    personality = _make_personality(active=[
        NeedState(type=NeedType.SAFETY, label="求稳", intensity=0.9, weight=1.0),
        NeedState(type=NeedType.ESTEEM, label="护尊严", intensity=0.5, weight=1.0),
        NeedState(type=NeedType.SOCIAL, label="求归属", intensity=0.7, weight=1.0),
    ])
    visible = [
        NeedState(type=NeedType.SAFETY, label="求稳", intensity=0.9, weight=1.0),
        NeedState(type=NeedType.ESTEEM, label="护尊严", intensity=0.5, weight=1.0),
        NeedState(type=NeedType.SOCIAL, label="求归属", intensity=0.7, weight=1.0),
    ]
    text = NeedEngine().to_prompt_context(personality, NeedType.SAFETY, visible, [], now_step=0)
    # social (0.7) is the strongest rival → listed; esteem (0.5) is >0.4 but not the strongest →
    # not listed.
    assert "同时也在意（仍在牵扯注意力）：social（求归属）" in text
    assert "esteem" not in text
    assert text.count("同时也在意") == 1


@pytest.mark.asyncio
async def test_need_engine_run_returns_dominant_need() -> None:
    engine = NeedEngine()

    result = await engine.run(
        current_step=0,
        personality=_make_personality(),
        pending_messages=1,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
    )

    assert result.dominant_need is not None
    assert isinstance(result.dominant_need, NeedType)


@pytest.mark.asyncio
async def test_run_reads_long_term_goals_from_personality() -> None:
    """personality.state is the source of truth for long-term goals; run() renders straight from
    it; the engine keeps no local copy."""
    engine = NeedEngine()
    personality = _make_personality(long_term=["演化后的新目标"])

    result = await engine.run(current_step=12, personality=personality)

    assert result.long_term_goals == ["演化后的新目标"]


@pytest.mark.asyncio
async def test_need_engine_social_need_boosted_when_agents_visible() -> None:
    engine = NeedEngine()

    result = await engine.run(
        current_step=0,
        personality=_make_personality(),
        pending_messages=0,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
    )

    # Social need should receive a runtime adjustment
    assert result.scores[NeedType.SOCIAL] > result.scores.get(NeedType.PHYSIOLOGICAL, 0)


@pytest.mark.asyncio
async def test_need_engine_safety_need_boosted_in_dangerous_context() -> None:
    engine = NeedEngine()
    personality = _make_personality()

    result = await engine.run(
        current_step=1,
        personality=personality,
    )

    assert result.scores[NeedType.SAFETY] > result.scores[NeedType.PHYSIOLOGICAL]


def _two_need_personality(safety: tuple[float, float], esteem: tuple[float, float]) -> PersonalityLayer:
    return _make_personality(active=[
        NeedState(type=NeedType.SAFETY, label="safe", intensity=safety[0], weight=safety[1]),
        NeedState(type=NeedType.ESTEEM, label="esteem", intensity=esteem[0], weight=esteem[1]),
    ])


_HIGH_BOOST = LLM_RELEVANCE_WEIGHT * (Urgency.HIGH.level - Urgency.LOW.level) / (
    Urgency.CRITICAL.level - Urgency.LOW.level)


@pytest.mark.asyncio
async def test_run_external_pressure_is_additive_and_dominant_is_argmax() -> None:
    """External pressure is an additive boost to related_need (urgency → magnitude), merged into
    scores, then dominant = argmax. esteem sits slightly below safety; one HIGH external pressure
    (boost≈1.33) lifts esteem to the top → dominant=esteem, the goals follow that dominant, and
    external_goals stays on the evaluation (feeding goal generation/prompt)."""
    base = await NeedEngine().run(
        current_step=0, personality=_two_need_personality((0.6, 1.0), (0.5, 1.0)),
        visible_agents=[], pending_messages=0)
    goal = ExternalGoal(text="奉命即刻执行", source_id="s1", urgency=Urgency.HIGH,
                        drive_type=ExternalDriveType.AUTHORITY, related_need=NeedType.ESTEEM)
    res = await NeedEngine().run(
        current_step=0, personality=_two_need_personality((0.6, 1.0), (0.5, 1.0)),
        visible_agents=[], pending_messages=0, external_goals=[goal])

    # Additive: esteem's score is exactly boost(HIGH) above its no-pressure score.
    assert abs(res.scores[NeedType.ESTEEM] - (base.scores[NeedType.ESTEEM] + _HIGH_BOOST)) < 1e-9
    # dominant agrees with the reported scores (argmax) and is lifted to esteem by the boost.
    assert res.dominant_need == max(res.scores, key=res.scores.get)
    assert res.dominant_need == NeedType.ESTEEM
    assert res.external_goals == [goal]
    assert all(g.related_need == NeedType.ESTEEM for g in res.short_term_goal_entities)


@pytest.mark.asyncio
async def test_run_external_pressure_does_not_blindly_override_overwhelming_score() -> None:
    """A hard override would force dominant to related_need regardless of scores (so dom would
    contradict scores). Additively, one HIGH external pressure (boost≈1.33) can't beat an
    overwhelming safety (I×W=2.0) → dominant stays safety, and == argmax."""
    goal = ExternalGoal(text="奉召入宫", source_id="s1", urgency=Urgency.HIGH,
                        drive_type=ExternalDriveType.AUTHORITY, related_need=NeedType.ESTEEM)
    res = await NeedEngine().run(
        current_step=0, personality=_two_need_personality((1.0, 2.0), (0.3, 1.0)),
        visible_agents=[], pending_messages=0, external_goals=[goal])

    assert res.dominant_need == NeedType.SAFETY                      # not blindly overturned
    assert res.dominant_need == max(res.scores, key=res.scores.get)  # consistent with scores



def test_evolve_intensities_reduces_dominant_need_on_success() -> None:
    engine = NeedEngine()
    current = {NeedType.SOCIAL.value: 0.6, NeedType.SAFETY.value: 0.5}
    updated = engine.evolve_intensities(current, dominant_need=NeedType.SOCIAL, succeeded=True)
    assert updated[NeedType.SOCIAL.value] < current[NeedType.SOCIAL.value]


def test_evolve_intensities_increases_dominant_need_on_failure() -> None:
    engine = NeedEngine()
    current = {NeedType.SAFETY.value: 0.5}
    updated = engine.evolve_intensities(current, dominant_need=NeedType.SAFETY, succeeded=False)
    assert updated[NeedType.SAFETY.value] > current[NeedType.SAFETY.value]


def test_evolve_intensities_no_dominant_still_relaxes() -> None:
    """No dominant need → no outcome shock, but homeostasis still relaxes every need toward B."""
    engine = NeedEngine()
    current = {nt.value: 0.50 for nt in NeedType}
    updated = engine.evolve_intensities(current, dominant_need=None, succeeded=True)
    # 0.50 + 0.25×(0.30 − 0.50) = 0.45 for every need (no shock)
    for v in updated.values():
        assert abs(v - 0.45) < 1e-9


def test_evolve_intensities_is_pure_does_not_mutate_input() -> None:
    """evolve_intensities returns a new dict; the caller (feedback layer) owns the write-back."""
    engine = NeedEngine()
    current = {NeedType.SOCIAL.value: 0.6, NeedType.SAFETY.value: 0.5}
    snapshot = dict(current)
    engine.evolve_intensities(current, dominant_need=NeedType.SOCIAL, succeeded=True)
    assert current == snapshot


def test_score_needs_selects_highest_scoring_visible_need() -> None:
    engine = NeedEngine()
    needs = [
        NeedState(type=NeedType.SAFETY, label="Stay safe", intensity=0.7, weight=1.0),
        NeedState(type=NeedType.ESTEEM, label="Gain respect", intensity=0.75, weight=1.0),
    ]
    # No context keywords — pure intensity × weight comparison
    dominant = max(engine._score_needs(needs), key=lambda x: x[0])[1]  # noqa: SLF001

    assert dominant is not None
    assert dominant.type == NeedType.ESTEEM


def test_score_needs_relevance_override_flips_winner() -> None:
    """An LLM need_relevance entry replaces keyword relevance for that need and can flip
    the ranking — here a strong safety activation beats a higher-intensity ESTEEM."""
    engine = NeedEngine()
    needs = [
        NeedState(type=NeedType.SAFETY, label="Stay safe", intensity=0.7, weight=1.0),
        NeedState(type=NeedType.ESTEEM, label="Gain respect", intensity=0.75, weight=1.0),
    ]
    # No keyword context; without relevance ESTEEM (0.75) > SAFETY (0.7).
    baseline = engine._score_needs(needs)  # noqa: SLF001
    assert max(baseline, key=lambda x: x[0])[1].type == NeedType.ESTEEM
    # additive: safety = 0.7 + 2.0×0.9 = 2.5 ; esteem(omitted→0) = 0.75 → safety wins
    scored = engine._score_needs(  # noqa: SLF001
        needs, need_relevance={NeedType.SAFETY: 0.9}
    )
    assert max(scored, key=lambda x: x[0])[1].type == NeedType.SAFETY


def test_score_needs_nonempty_relevance_zeros_omitted_needs() -> None:
    """Non-empty need_relevance is authoritative: a need it omits scores at the neutral
    baseline (I×W), NOT boosted — the LLM deliberately judged it un-activated."""
    engine = NeedEngine()
    needs = [
        NeedState(type=NeedType.SAFETY, label="safe", intensity=0.6, weight=1.0),
        NeedState(type=NeedType.SOCIAL, label="belong", intensity=0.6, weight=1.0),
    ]
    scored = {s.type: v for v, s in engine._score_needs(  # noqa: SLF001
        needs, need_relevance={NeedType.SOCIAL: 0.5}
    )}
    assert abs(scored[NeedType.SAFETY] - 0.6) < 1e-9          # omitted → baseline I×W
    assert abs(scored[NeedType.SOCIAL] - (0.6 + 2.0 * 0.5)) < 1e-9  # listed → I×W + WEIGHT×0.5


def test_score_needs_none_and_empty_relevance_are_equivalent() -> None:
    """Omitting need_relevance and passing {} both take the rule path
    (disposition I×W + runtime adjustments), producing identical scores."""
    engine = NeedEngine()
    needs = [
        NeedState(type=NeedType.SAFETY, label="Stay safe", intensity=0.6, weight=1.0),
        NeedState(type=NeedType.SOCIAL, label="Belong", intensity=0.6, weight=1.0),
    ]
    none_relevance = {(s.type, round(v, 6)) for v, s in engine._score_needs(needs)}  # noqa: SLF001
    empty_relevance = {  # noqa: SLF001
        (s.type, round(v, 6)) for v, s in engine._score_needs(needs, need_relevance={})
    }
    assert none_relevance == empty_relevance


def test_score_needs_hidden_need_loses_to_visible_under_quarter_weight() -> None:
    """Hidden needs participate at 25% effective weight — a moderately strong
    visible need should still beat a much louder hidden one.
    """
    engine = NeedEngine()
    needs = [
        NeedState(type=NeedType.SOCIAL, label="Hidden longing", intensity=0.99, weight=2.0, is_hidden=True),
        NeedState(type=NeedType.SAFETY, label="Stay safe", intensity=0.5, weight=1.0, is_hidden=False),
    ]
    dominant = max(engine._score_needs(needs), key=lambda x: x[0])[1]  # noqa: SLF001

    # 0.99 * 2.0 * 0.25 = 0.495 < 0.5 * 1.0 * 1.0 = 0.5
    assert dominant is not None
    assert dominant.type == NeedType.SAFETY


def test_score_needs_hidden_need_can_surface_when_visible_needs_are_weak() -> None:
    """Hidden needs aren't excluded — they bias decisions at 25% weight.

    With no visible competition, a strong hidden pressure becomes the dominant
    motive even though the agent does not consciously voice it.
    """
    engine = NeedEngine()
    needs = [
        NeedState(type=NeedType.SOCIAL, label="Hidden longing", intensity=0.9, weight=1.0, is_hidden=True),
    ]
    dominant = max(engine._score_needs(needs), key=lambda x: x[0])[1]  # noqa: SLF001

    assert dominant is not None
    assert dominant.type == NeedType.SOCIAL
    assert dominant.is_hidden is True


@pytest.mark.asyncio
async def test_run_force_goal_update_regenerates_goals() -> None:
    """force_goal_update=True must regenerate goals even when personality already has goals."""
    engine = NeedEngine(llm_router=_GoalsLLM(["改道先探虚实"]))  # type: ignore[arg-type]
    personality = _make_personality()
    # Pre-seed short_term_goals so _should_update_goals would normally return False
    personality.apply_need_state(
        step=0,
        active_needs=[NeedType.SAFETY.value],
        dominant_need=NeedType.SAFETY.value,
        long_term_goals=[],
        short_term_goal_entities=[_goal("existing goal")],
    )

    result = await engine.run(
        current_step=1,
        personality=personality,
        pending_messages=0,
        force_goal_update=True,
    )

    assert result.short_term_goals
    assert result.short_term_goals != ["existing goal"]


@pytest.mark.asyncio
async def test_short_term_generation_parses_thought_first_response() -> None:
    """Think-first lite-CoT: the thought field only anchors reasoning and isn't parsed; only goals
    is read, and the thought text never leaks into the goals."""
    from core.interfaces.llm import LLMResponse

    class _ThoughtLLM:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ):
            return LLMResponse(
                content='{"thought": "我此刻最怕局势失控，刚听到风声", "goals": ["稳住局势", "联络旧部"]}',
                input_tokens=0, output_tokens=0, model="test",
            )

    personality = _make_personality(active=[
        NeedState(type=NeedType.SAFETY, label="减少不确定性", intensity=0.9, weight=1.0),
    ])
    engine = NeedEngine(llm_router=_ThoughtLLM())  # type: ignore[arg-type]
    result = await engine.run(
        current_step=1, personality=personality,
        visible_agents=[], pending_messages=0,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
    )

    assert result.short_term_goals == ["稳住局势", "联络旧部"]
    assert all("我此刻" not in g for g in result.short_term_goals)


@pytest.mark.asyncio
async def test_a_deliberate_empty_answer_does_not_get_a_template_goal_instead() -> None:
    """When the LLM says "no new direction this step", leave the queue empty; never stuff a rule
    goal in instead.

    An empty queue always passes the low-water gate, so empty after generation means no new
    direction, not a failure. A template goal would put personality-free text into the persistent
    FIFO (snapshots, every decision prompt, the goal judge, embedded memory if INTERRUPTED). Even
    when the LLM raises, _generate_short_term_goals returns empty.
    """
    from core.interfaces.llm import LLMResponse

    class _EmptyGoalsLLM:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000, **kwargs):
            return LLMResponse(
                content='{"thought": "手头几件事都还在走,眼下不缺新方向", "goals": []}',
                input_tokens=0, output_tokens=0, model="test",
            )

    personality = _make_personality(active=[
        NeedState(type=NeedType.SAFETY, label="减少不确定性", intensity=0.9, weight=1.0),
    ])
    engine = NeedEngine(llm_router=_EmptyGoalsLLM())  # type: ignore[arg-type]
    result = await engine.run(
        current_step=1, personality=personality,
        visible_agents=[], pending_messages=0,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
    )

    assert result.short_term_goals == []
    assert result.short_term_goal_entities == []


@pytest.mark.asyncio
async def test_short_term_generation_injects_recent_foiled_as_distinct_block() -> None:
    """not_executed (foiled) attempts are injected into the short-term goal prompt as their own
    block, not mixed into "failed goals", and carry the memory-order hint.

    This is what stops "hitting the same wall over and over" at the source: motivation can rewrite
    an intent that isn't working (add a prerequisite step, change direction), which decide's
    single-beat avoidance can't do."""
    from core.interfaces.llm import LLMResponse
    from core.prompts import MEMORY_ORDER_HINT

    captured: dict = {}

    class _CaptureLLM:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000, **kwargs):
            captured["user"] = messages[-1].content
            return LLMResponse(
                content='{"thought": "我", "goals": ["先去找到父亲"]}',
                input_tokens=0, output_tokens=0, model="test",
            )

    personality = _make_personality(active=[
        NeedState(type=NeedType.SOCIAL, label="被接纳", intensity=0.9, weight=1.0),
    ])
    engine = NeedEngine(llm_router=_CaptureLLM())  # type: ignore[arg-type]
    await engine.run(
        current_step=3, personality=personality,
        visible_agents=[], pending_messages=0,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
        recent_foiled_texts=["想要做：与父亲相谈，但是结果是：对方不在场"],
    )

    user = captured["user"]
    assert "想要做：与父亲相谈，但是结果是：对方不在场" in user
    assert "没走通的尝试" in user
    assert MEMORY_ORDER_HINT in user


@pytest.mark.asyncio
async def test_run_annotates_old_and_new_dominant_need_on_goal_call() -> None:
    """The goal-generation LLM call must carry the old/new dominant_need annotation (via
    annotate_call → LLMCallTrace.extra)."""
    from core.context import get_call_annotations
    from core.interfaces.llm import LLMResponse

    captured: dict = {}

    class _CapturingLLM:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ):
            captured.update(get_call_annotations())  # the call happens inside the annotate_call scope
            return LLMResponse(content='{"goals": ["稳住局势"]}', input_tokens=0, output_tokens=0, model="test")

    engine = NeedEngine(llm_router=_CapturingLLM())  # type: ignore[arg-type]
    # innate is safety only → this step's argmax=safety; the old state.dominant_need is set to
    # social.
    personality = _make_personality(active=[
        NeedState(type=NeedType.SAFETY, label="减少不确定性", intensity=0.9, weight=1.0),
    ])
    personality.update_needs(
        step=0, active_needs=[NeedType.SOCIAL.value], dominant_need=NeedType.SOCIAL.value,
    )

    await engine.run(
        current_step=1, personality=personality,
        visible_agents=[], pending_messages=0,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
    )

    assert captured.get("dominant_need_old") == "social"   # old value before this step's recompute
    # Audit contract fields, read from LLMCallTrace.extra rather than parsed from the prompt:
    # dominant_need is this step's argmax after recompute (no separate dominant_need_new).
    assert captured.get("dominant_need") == "safety"       # this step's argmax (innate is safety only)
    assert isinstance(captured.get("prior_goals"), list)
    # The annotation lives only inside the call scope; nothing remains after run() ends.
    from core.context import get_call_annotations as _gca
    assert _gca() == {}


@pytest.mark.asyncio
async def test_run_stores_only_admitted_goals_after_dedup() -> None:
    """The engine stores the deduplicated new goals that were actually enqueued in the motivation
    call's extra.short_term_goals_new (post-call). Near-duplicate LLM goals removed by
    enqueue_goals' literal dedup aren't in it, so audit doesn't mistake engine-deduped repeats for
    new goals."""
    from types import SimpleNamespace

    from agent.goals import GoalEntity
    from agent.need import NeedType
    from core.context import set_active_call
    from core.interfaces.llm import LLMResponse

    fake_call = SimpleNamespace(extra={})  # stands in for LLMCallTrace; annotate_active_call only needs .extra
    set_active_call(fake_call)
    try:
        class _DupGoalLLM:
            async def complete(self, scene, messages, temperature=0.7, max_tokens=1000, **kwargs):
                # Literally very close to an existing queued goal (ratio≈0.9 > 0.6 threshold) →
                # should be deduped, not enqueued.
                return LLMResponse(content='{"goals": ["挺枪逼近御座，以武力威慑逼李渊交出兵权"]}',
                                   input_tokens=0, output_tokens=0, model="test")

        engine = NeedEngine(llm_router=_DupGoalLLM())  # type: ignore[arg-type]
        personality = _make_personality(
            active=[NeedState(type=NeedType.SAFETY, label="减少不确定性", intensity=0.9, weight=1.0)],
            short_term_entities=[GoalEntity(
                id="stg-0", text="持械逼近御座，以武力威慑逼李渊交出兵权",
                goal_type="short_term", related_need=NeedType.SAFETY, created_step=1)])
        # Old dominant=social, this step's argmax=safety → triggers goal recompute.
        personality.update_needs(step=0, active_needs=[NeedType.SOCIAL.value], dominant_need=NeedType.SOCIAL.value)
        await engine.run(current_step=1, personality=personality, visible_agents=[], pending_messages=0,
                         emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0))
        # Near-duplicate deduped → nothing actually enqueued (rather than treating "挺枪…" as a
        # new goal).
        assert fake_call.extra.get("short_term_goals_new") == []
    finally:
        set_active_call(None)


@pytest.mark.asyncio
async def test_evaluate_goal_progress_annotates_goal_texts_for_audit() -> None:
    """Audit contract: the goal-progress call puts an index → goal-text map into
    extra.goal_texts (string keys, 1-based, with duration hint) so audit can map the output's
    index back to goal text. The annotation is captured here inside the real call scope, proving
    the producer's key name matches the audit consumer (tuning/audit_reconstruct.py) exactly."""
    from agent.goals import GoalEntity
    from agent.need import NeedType
    from core.context import get_call_annotations
    from core.interfaces.llm import LLMResponse

    captured: dict = {}

    class _CapturingJudge:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000, **kwargs):
            captured.update(get_call_annotations())  # the call happens inside the annotate_call scope
            return LLMResponse(
                content='{"goals": [{"index": 1, "status": "active"}, {"index": 2, "status": "active"}]}',
                input_tokens=0, output_tokens=0, model="test")

    engine = NeedEngine(llm_router=_CapturingJudge())  # type: ignore[arg-type]
    personality = _make_personality(short_term_entities=[
        GoalEntity(id="stg-1-0", text="目标甲", goal_type="short_term",
                   related_need=NeedType.SOCIAL, created_step=1),
        GoalEntity(id="stg-1-1", text="目标乙", goal_type="short_term",
                   related_need=NeedType.SAFETY, created_step=1),
    ])
    await engine.evaluate_goal_progress(
        step=3, action_description="做了一件事", action_result="结果如此",
        personality=personality,
    )
    texts = captured.get("goal_texts")
    assert isinstance(texts, dict)
    # String keys, 1-based, aligned indices; text starts with the goal text and may carry a
    # trailing "（已历约N…）" duration hint (the contract requires keeping it).
    assert texts.get("1", "").startswith("目标甲") and texts.get("2", "").startswith("目标乙")


@pytest.mark.asyncio
async def test_run_preserves_existing_goals_when_dominant_need_unchanged() -> None:
    """Goals are preserved when the dominant need hasn't changed.

    Between the two run() calls the first evaluation is written into personality, as
    Agent.begin_ongoing_step does; otherwise the second run() sees the init state and regenerates.
    """
    engine = NeedEngine()
    personality = _make_personality()
    # Run once to establish dominant need
    first = await engine.run(
        current_step=1,
        personality=personality,
    )
    # Simulate commit: write first evaluation into personality
    personality.apply_need_state(
        step=1,
        active_needs=[n.type.value for n in first.active_needs],
        dominant_need=first.dominant_need.value if first.dominant_need else None,
        long_term_goals=first.long_term_goals,
        short_term_goal_entities=first.short_term_goal_entities,
    )
    existing_goals = list(first.short_term_goals)

    # Run again with same conditions — dominant need should be unchanged, goals preserved
    second = await engine.run(
        current_step=2,
        personality=personality,
    )

    assert second.dominant_need == first.dominant_need
    assert second.short_term_goals == existing_goals


def _personality_with_long_term(*goals: str) -> PersonalityLayer:
    return _make_personality(long_term=list(goals))


@pytest.mark.asyncio
async def test_revise_long_term_goals_none_without_llm_router() -> None:
    """Without an LLM router → return None (no update)."""
    engine = NeedEngine()
    personality = _personality_with_long_term("Achieve harmony.", "Protect the realm.")

    result = await engine.revise_long_term_goals(
        personality=personality, recent_memory_texts=["something happened"],
        is_main_character=True,
    )

    assert result is None
    assert personality.state.long_term_goals == ["Achieve harmony.", "Protect the realm."]


@pytest.mark.asyncio
async def test_revise_long_term_goals_applies_revised_json() -> None:
    """Valid JSON goals → returned as the revised list. The engine does not write; the agent
    commits via personality.set_long_term_goals."""
    from core.interfaces.llm import LLMResponse

    class _JsonLLM:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ):
            return LLMResponse(
                content='{"goals": ["守护新的同盟", "为枉死者讨回公道"]}',
                input_tokens=0, output_tokens=0, model="test",
            )

    engine = NeedEngine(llm_router=_JsonLLM())  # type: ignore[arg-type]
    personality = _personality_with_long_term("旧志向")
    result = await engine.revise_long_term_goals(
        personality=personality, recent_memory_texts=["背叛让我看清了一切"],
        is_main_character=False,
    )

    assert result == ["守护新的同盟", "为枉死者讨回公道"]
    personality.set_long_term_goals(result)  # committing via personality
    assert personality.state.long_term_goals == ["守护新的同盟", "为枉死者讨回公道"]


@pytest.mark.asyncio
async def test_revise_long_term_goals_none_on_empty_array() -> None:
    """Model outputs [] = direction unchanged → returns None."""
    from core.interfaces.llm import LLMResponse

    class _NoChangeLLM:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ):
            return LLMResponse(content='{"goals": []}', input_tokens=0, output_tokens=0, model="test")

    engine = NeedEngine(llm_router=_NoChangeLLM())  # type: ignore[arg-type]
    result = await engine.revise_long_term_goals(
        personality=_personality_with_long_term("守护家国"), recent_memory_texts=["寻常的一天"],
        is_main_character=False,
    )

    assert result is None


@pytest.mark.asyncio
async def test_revise_long_term_goals_none_when_identical_to_current() -> None:
    """Model echoes back exactly the original list → treated as no update, returns None."""
    from core.interfaces.llm import LLMResponse

    class _SameLLM:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ):
            return LLMResponse(content='{"goals": ["守护家国"]}', input_tokens=0, output_tokens=0, model="test")

    engine = NeedEngine(llm_router=_SameLLM())  # type: ignore[arg-type]
    result = await engine.revise_long_term_goals(
        personality=_personality_with_long_term("守护家国"), recent_memory_texts=["寻常的一天"],
        is_main_character=False,
    )

    assert result is None


@pytest.mark.asyncio
async def test_revise_long_term_goals_main_character_uses_retry() -> None:
    """Main-character path routes through complete_with_retry (CLAUDE.md Rule 1 exception)."""
    from core.interfaces.llm import LLMResponse

    class _RetryLLM:
        def __init__(self) -> None:
            self.retry_called = False

        async def complete_with_retry(self, scene, messages, temperature=0.7, max_tokens=1000, *, retry_delay=1.0, **kwargs):
            self.retry_called = True
            return LLMResponse(content='{"goals": ["新方向"]}', input_tokens=0, output_tokens=0, model="test")

    router = _RetryLLM()
    engine = NeedEngine(llm_router=router)  # type: ignore[arg-type]
    result = await engine.revise_long_term_goals(
        personality=_personality_with_long_term("旧志向"), recent_memory_texts=["重大转折"],
        is_main_character=True,
    )

    assert router.retry_called is True
    assert result == ["新方向"]


@pytest.mark.asyncio
async def test_revise_long_term_goals_none_on_malformed_json() -> None:
    """Malformed / fieldless output → None (no update), no crash."""
    from core.interfaces.llm import LLMResponse

    class _GarbageLLM:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ):
            return LLMResponse(content="维持原方向吧", input_tokens=0, output_tokens=0, model="test")

    engine = NeedEngine(llm_router=_GarbageLLM())  # type: ignore[arg-type]
    result = await engine.revise_long_term_goals(
        personality=_personality_with_long_term("守护家国"), recent_memory_texts=["平静的一天"],
        is_main_character=False,
    )

    assert result is None


@pytest.mark.asyncio
async def test_revise_long_term_goals_none_on_exception() -> None:
    """An LLM exception is caught (Rule 1): None, no raise."""
    class _BoomLLM:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ):
            raise RuntimeError("provider down")

    engine = NeedEngine(llm_router=_BoomLLM())  # type: ignore[arg-type]
    result = await engine.revise_long_term_goals(
        personality=_personality_with_long_term("一统天下"), recent_memory_texts=["噩耗传来"],
        is_main_character=False,
    )

    assert result is None


@pytest.mark.asyncio
async def test_set_long_term_goals_preserves_unchanged_goal_entity_status() -> None:
    """personality.set_long_term_goals: re-writing one goal verbatim preserves its entity
    (status/related_need). The engine returns revised text; personality reconciles entities."""
    from core.interfaces.llm import LLMResponse

    personality = _personality_with_long_term("守护家国", "重整朝纲")
    personality._state.long_term_goal_entities[0].status = GoalStatus.COMPLETED  # noqa: SLF001

    class _JsonLLM:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ):
            # Keep the first goal verbatim, replace the second.
            return LLMResponse(
                content='{"goals": ["守护家国", "肃清奸佞"]}',
                input_tokens=0, output_tokens=0, model="test",
            )

    engine = NeedEngine(llm_router=_JsonLLM())  # type: ignore[arg-type]
    result = await engine.revise_long_term_goals(
        personality=personality, recent_memory_texts=["新仇"],
        is_main_character=False,
    )

    assert result == ["守护家国", "肃清奸佞"]
    personality.set_long_term_goals(result)
    kept = personality.state.long_term_goal_entities[0]
    assert kept.text == "守护家国"
    assert kept.status == GoalStatus.COMPLETED  # preserved by the setter


# ─── GoalEntity lifecycle tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_goal_entity_created_with_active_status_on_first_run() -> None:
    engine = NeedEngine(llm_router=_GoalsLLM(["稳住眼下的局势"]))  # type: ignore[arg-type]
    result = await engine.run(
        current_step=1, personality=_make_personality(),
        visible_agents=[], pending_messages=0,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
    )

    assert len(result.short_term_goal_entities) > 0
    for entity in result.short_term_goal_entities:
        assert entity.status.value == "active"
        assert entity.goal_type == "short_term"
        assert entity.created_step == 1


@pytest.mark.asyncio
async def test_active_goal_entities_preserved_across_runs() -> None:
    engine = NeedEngine()
    personality = _make_personality()

    first = await engine.run(
        current_step=1, personality=personality,
        visible_agents=[], pending_messages=0,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
    )
    _commit(personality, first)
    first_texts = [g.text for g in first.short_term_goal_entities if g.status.value == "active"]

    second = await engine.run(
        current_step=2, personality=personality,
        visible_agents=[], pending_messages=0,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
    )

    assert first_texts == second.short_term_goals


@pytest.mark.asyncio
async def test_evaluate_goal_progress_without_router_rules_nothing() -> None:
    """No router is handled like a judge failure: no ruling, no progress written, goals stay in the
    live queue as they are."""
    engine = NeedEngine()
    personality = _make_personality(short_term_entities=[_seeded_goal()])
    updated = await engine.evaluate_goal_progress(
        step=1,
        action_description="Spoke with ally.",
        action_result="Ally agreed to cooperate.",
        personality=personality,
    )
    assert [g.status for g in updated] == [GoalStatus.ACTIVE]
    assert updated[0].progress_summary == _seeded_goal().progress_summary


@pytest.mark.asyncio
async def test_evaluate_goal_progress_without_router_still_runs_backstop() -> None:
    """The time-limit fallback doesn't depend on the judge: with no router, goals at the threshold
    are still shelved."""
    from agent.goals import _SHORT_TERM_GOAL_STALL_STEPS

    engine = NeedEngine()
    personality = _make_personality(short_term_entities=[
        GoalEntity(id="stg-stall", text="安抚某人情绪", goal_type="short_term",
                   related_need=NeedType.SOCIAL, created_step=1),
    ])
    updated = await engine.evaluate_goal_progress(
        step=1 + _SHORT_TERM_GOAL_STALL_STEPS,
        action_description="又劝了一次", action_result="对方仍焦躁",
        personality=personality,
    )
    assert updated[0].status is GoalStatus.FAILED


@pytest.mark.asyncio
async def test_completed_goals_trigger_regeneration_on_next_run() -> None:
    # Two batches with different text: the second generation must produce a new entity (identical
    # text would be blocked by enqueue_goals' literal dedup).
    engine = NeedEngine(llm_router=_GoalsLLM(["稳住眼下的局势"], ["改道先探虚实"]))  # type: ignore[arg-type]
    personality = _make_personality()
    first = await engine.run(
        current_step=1, personality=personality,
        visible_agents=[], pending_messages=0,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
    )
    _commit(personality, first)
    first_ids = {g.id for g in personality.state.short_term_goal_entities}

    updated = await engine.evaluate_goal_progress(
        step=1,
        action_description="Action.",
        action_result="Success.",
        personality=personality,
    )
    personality.set_short_term_goal_entities(updated)

    second = await engine.run(
        current_step=2, personality=personality,
        visible_agents=[], pending_messages=0,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
    )
    second_ids = {g.id for g in second.short_term_goal_entities}

    assert second_ids != first_ids


def test_goal_need_bias_boosts_related_need() -> None:
    engine = NeedEngine()
    long_term = [
        GoalEntity(id="ltg-0", text="称霸天下", goal_type="long_term",
                   related_need=NeedType.SELF_ACTUALIZATION, created_step=0)
    ]

    biases = engine._goal_need_bias(long_term)  # noqa: SLF001

    assert NeedType.SELF_ACTUALIZATION in biases
    assert biases[NeedType.SELF_ACTUALIZATION] > 0.0


def test_goal_need_bias_capped_at_015() -> None:
    engine = NeedEngine()
    long_term = [
        GoalEntity(id=f"ltg-{i}", text=f"目标{i}", goal_type="long_term",
                   related_need=NeedType.SAFETY, created_step=0)
        for i in range(5)
    ]

    biases = engine._goal_need_bias(long_term)  # noqa: SLF001

    assert biases[NeedType.SAFETY] <= 0.15


def test_text_to_long_term_goal_has_no_inferred_need() -> None:
    """A long-term goal arriving as bare text gets related_need=None (wildcard); no keyword
    guessing (Rule 7)."""
    from agent.goals import text_to_long_term_goal_entity

    entity = text_to_long_term_goal_entity("保护太子，防御敌人威胁")

    assert entity.related_need is None
    assert entity.goal_type == "long_term"


def test_set_long_term_goals_produces_entities() -> None:
    # set_long_term_goals syncs entities (active) from text.
    personality = _make_personality()
    personality.set_long_term_goals(["称霸天下，完成使命"])

    entities = personality.state.long_term_goal_entities
    assert len(entities) == 1
    assert entities[0].text == "称霸天下，完成使命"
    assert entities[0].goal_type == "long_term"
    assert entities[0].status.value == "active"
    assert personality.state.long_term_goals == ["称霸天下，完成使命"]


@pytest.mark.asyncio
async def test_need_evaluation_carries_goal_entities() -> None:
    engine = NeedEngine(llm_router=_GoalsLLM(["稳住眼下的局势"]))  # type: ignore[arg-type]

    result = await engine.run(
        current_step=1,
        personality=_make_personality(),
        pending_messages=0,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
    )

    assert isinstance(result.short_term_goal_entities, list)
    assert len(result.short_term_goal_entities) > 0
    for entity in result.short_term_goal_entities:
        assert entity.status.value == "active"


def test_should_update_goals_when_dominant_need_not_in_active_goal_needs() -> None:
    """Goals generated for SOCIAL must be discarded when SAFETY becomes dominant.

    Seed >= _SHORT_TERM_GOAL_LOW_WATER active goals so the low-water branch does not
    short-circuit — this isolates the need-mismatch trigger.
    """
    engine = NeedEngine()
    entities = [
        GoalEntity(
            id=f"stg-1-{i}", text=text, goal_type="short_term",
            related_need=NeedType.SOCIAL, created_step=1,
        )
        for i, text in enumerate(["试探X的态度", "拉近与Y的关系"])
    ]

    # SAFETY becomes dominant — not in active goal needs → should update
    assert engine._should_update_goals(NeedType.SAFETY, entities) is True  # noqa: SLF001


def test_should_update_goals_preserves_goals_when_dominant_need_matches() -> None:
    """Goals generated for SAFETY must be preserved while SAFETY stays dominant.

    Seed >= _SHORT_TERM_GOAL_LOW_WATER active goals so the low-water branch does not
    short-circuit — this isolates the need-match (no-regen) path.
    """
    engine = NeedEngine()
    entities = [
        GoalEntity(
            id=f"stg-1-{i}", text=text, goal_type="short_term",
            related_need=NeedType.SAFETY, created_step=1,
        )
        for i, text in enumerate(["先观察局势，确认安全", "守在隐蔽处避开锋芒"])
    ]

    assert engine._should_update_goals(NeedType.SAFETY, entities) is False  # noqa: SLF001


# ---------------------------------------------------------------------------
# Time-based need adjustment
# ---------------------------------------------------------------------------


def test_runtime_adjustment_night_boosts_physiological() -> None:
    from agent.need import NeedType
    from agent.personality import EmotionState

    engine = NeedEngine()
    adj = engine._runtime_adjustment(  # noqa: SLF001
        need_type=NeedType.PHYSIOLOGICAL,
        visible_agents=0,
        pending_messages=0,
        emotion=EmotionState(),
        world_time_hour=23,  # late night
    )
    assert adj >= 0.12


def test_runtime_adjustment_night_suppresses_social() -> None:
    from agent.need import NeedType
    from agent.personality import EmotionState

    engine = NeedEngine()
    day_adj = engine._runtime_adjustment(  # noqa: SLF001
        need_type=NeedType.SOCIAL,
        visible_agents=0,
        pending_messages=0,
        emotion=EmotionState(),
        world_time_hour=-1,  # unknown → no time bias
    )
    night_adj = engine._runtime_adjustment(  # noqa: SLF001
        need_type=NeedType.SOCIAL,
        visible_agents=0,
        pending_messages=0,
        emotion=EmotionState(),
        world_time_hour=2,  # deep night
    )
    assert night_adj < day_adj


def test_runtime_adjustment_dawn_gives_small_physiological_boost() -> None:
    from agent.need import NeedType
    from agent.personality import EmotionState

    engine = NeedEngine()
    adj = engine._runtime_adjustment(  # noqa: SLF001
        need_type=NeedType.PHYSIOLOGICAL,
        visible_agents=0,
        pending_messages=0,
        emotion=EmotionState(),
        world_time_hour=6,  # dawn
    )
    assert 0.04 <= adj <= 0.10


def test_runtime_adjustment_daytime_has_no_time_bias() -> None:
    from agent.need import NeedType
    from agent.personality import EmotionState

    engine = NeedEngine()
    adj = engine._runtime_adjustment(  # noqa: SLF001
        need_type=NeedType.PHYSIOLOGICAL,
        visible_agents=0,
        pending_messages=0,
        emotion=EmotionState(),
        world_time_hour=14,  # afternoon
    )
    assert adj == 0.0


def test_classify_time_period_by_hour() -> None:
    from agent.need import _classify_time_period

    assert _classify_time_period(23) == "night"
    assert _classify_time_period(2) == "night"
    assert _classify_time_period(4) == "night"
    assert _classify_time_period(5) == "dawn"
    assert _classify_time_period(6) == "dawn"
    assert _classify_time_period(7) == ""
    assert _classify_time_period(14) == ""
    assert _classify_time_period(-1) == ""   # unknown
    assert _classify_time_period(99) == ""   # out of range


# ---------------------------------------------------------------------------
# Goal generation uses perceived_signal_texts, not structural context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_goal_generation_uses_perceived_signal_texts_not_structural_context() -> None:
    """perceived_signal_texts must reach the goal prompt, not structural location/time tokens."""
    from core.interfaces.llm import LLMMessage, LLMResponse, LLMScene

    captured: list[str] = []

    class _CaptureLLM:
        async def complete(
            self,
            scene: LLMScene,
            messages: list[LLMMessage],
            temperature: float = 0.7,
            max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
            for msg in messages:
                captured.append(msg.content)
            return LLMResponse(
                content='{"goals": ["观察局势，等待时机"]}',
                input_tokens=0, output_tokens=0, model="test",
            )

    engine = NeedEngine(llm_router=_CaptureLLM())  # type: ignore[arg-type]
    personality = _make_personality()

    await engine.run(
        current_step=1,
        personality=personality,
        pending_messages=0,
        perceived_signal_texts=["一场风暴突然袭来，宫墙震动"],
    )

    combined = " ".join(captured)
    assert "一场风暴突然袭来" in combined, "narrative event must appear in goal prompt"
    assert "location:hall" not in combined, "structural context must not leak into goal prompt"


@pytest.mark.asyncio
async def test_short_term_goals_parsed_from_json(monkeypatch) -> None:
    """_generate_short_term_goals consumes the JSON {"goals": [...]} contract."""
    from core.interfaces.llm import LLMResponse

    class _JsonLLM:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ):
            # Fenced JSON with extra prose around it — extract_json must still cope.
            return LLMResponse(
                content='```json\n{"goals": ["稳住阵脚", "联络旧部", ""]}\n```',
                input_tokens=0, output_tokens=0, model="test",
            )

    engine = NeedEngine(llm_router=_JsonLLM())  # type: ignore[arg-type]
    goals = await engine._generate_short_term_goals(  # noqa: SLF001
        dominant_need=NeedType.SAFETY,
        personality=_make_personality(),
    )

    # Empty entries dropped; capped at 2.
    # Bare strings are still accepted: the model sometimes ignores the object schema and emits only
    # text; that should degrade to "this goal has no deadline", not drop the goal (Rule 2
    # field-level silent coercion).
    assert goals == [("稳住阵脚", None), ("联络旧部", None)]


@pytest.mark.asyncio
async def test_unparseable_llm_output_yields_no_goals_rather_than_a_template() -> None:
    """Garbage doesn't crash, and gets no template goal either: returns empty.

    Goals are persisted (FIFO / snapshot / every decision prompt / the judge), so on failure write
    nothing (Rule 1 fallback tiers); an empty queue lets generation through again next step. The
    garbage line itself is unparsed raw text and must never become a goal.
    """
    from core.interfaces.llm import LLMResponse

    class _GarbageLLM:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ):
            return LLMResponse(content="观察局势，等待时机", input_tokens=0, output_tokens=0, model="test")

    engine = NeedEngine(llm_router=_GarbageLLM())  # type: ignore[arg-type]
    goals = await engine._generate_short_term_goals(  # noqa: SLF001
        dominant_need=NeedType.SOCIAL,
        personality=_make_personality(),
    )

    assert goals == []


# ---------------------------------------------------------------------------
# Need intensity persisted and restored across sessions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_scores_track_persisted_drive_single_source_of_truth() -> None:
    """personality.state.need_intensities is the single source of truth for drive: a brand-new
    NeedEngine (= restart, no engine-local intensities) must read exactly the persisted drive, and
    changing the persisted drive must change the next scoring."""
    personality = _make_personality()
    nt = NeedType.SAFETY

    personality.update_need_intensities({n.value: 0.30 for n in NeedType})
    low_engine = NeedEngine()  # fresh ("restart"): drive comes only from persisted state
    low = await low_engine.run(
        current_step=1, personality=personality,
        visible_agents=[], pending_messages=0,
    )

    elevated = {n.value: 0.30 for n in NeedType}
    elevated[nt.value] = 0.90
    personality.update_need_intensities(elevated)
    high_engine = NeedEngine()  # another fresh engine: reflects the new persisted drive directly
    high = await high_engine.run(
        current_step=2, personality=personality,
        visible_agents=[], pending_messages=0,
    )

    # Raising the persisted drive raises that need's score (run reads drive from persisted state,
    # with no engine-local restore needed).
    assert high.scores[nt] > low.scores[nt]


def test_update_need_intensities_stores_drive_in_state() -> None:
    """update_need_intensities (the feedback layer's only write point for drive evolution) must
    land in StateLayer.need_intensities."""
    personality = _make_personality()
    drive = {NeedType.SAFETY.value: 0.88, NeedType.SOCIAL.value: 0.42}

    personality.update_need_intensities(drive)

    state = personality.state
    assert abs(state.need_intensities[NeedType.SAFETY.value] - 0.88) < 1e-9
    assert abs(state.need_intensities[NeedType.SOCIAL.value] - 0.42) < 1e-9


# ---------------------------------------------------------------------------
# Gap fillers: exact deltas, wildcard, cross-step accumulation
# ---------------------------------------------------------------------------


def test_evolve_intensities_success_dominant_relax_plus_shock() -> None:
    """Success: dominant = homeostatic relaxation + relief shock. @0.5 → 0.45 (relax) − 0.12 (k_s)
    = 0.33."""
    engine = NeedEngine()
    current = {nt.value: 0.50 for nt in NeedType}  # away from the clamp bounds
    updated = engine.evolve_intensities(current, dominant_need=NeedType.SAFETY, succeeded=True)
    assert abs(updated[NeedType.SAFETY.value] - 0.33) < 1e-9


def test_evolve_intensities_failure_dominant_relax_plus_shock() -> None:
    """Failure: dominant = relaxation + stress shock. @0.5 → 0.45 (relax) + 0.08 (k_f) = 0.53
    (still a net rise)."""
    engine = NeedEngine()
    current = {nt.value: 0.50 for nt in NeedType}
    updated = engine.evolve_intensities(current, dominant_need=NeedType.SAFETY, succeeded=False)
    assert abs(updated[NeedType.SAFETY.value] - 0.53) < 1e-9


def test_evolve_intensities_non_dominant_relaxes_toward_baseline() -> None:
    """Non-dominant needs relax toward the resting baseline, so they can't drift upward to
    saturation. @0.5 → 0.45."""
    engine = NeedEngine()
    current = {nt.value: 0.50 for nt in NeedType}
    updated = engine.evolve_intensities(current, dominant_need=NeedType.SAFETY, succeeded=True)
    for key, v in updated.items():
        if key == NeedType.SAFETY.value:
            continue
        assert abs(v - 0.45) < 1e-9, f"{key} should relax to 0.45 (toward B), got {v}"


def test_evolve_intensities_intensity_clamped_at_lower_bound_005() -> None:
    """After dominant success, intensity is clamped at a floor of 0.05 (so the need can't drop to 0
    and vanish)."""
    engine = NeedEngine()
    current = {NeedType.SAFETY.value: 0.10}
    updated = engine.evolve_intensities(current, dominant_need=NeedType.SAFETY, succeeded=True)
    assert updated[NeedType.SAFETY.value] == 0.05


def test_evolve_intensities_failure_at_high_intensity_capped_by_homeostasis() -> None:
    """At high levels homeostasis outweighs the failure shock, so intensity doesn't run away to
    saturation (otherwise repeated failures would climb to 1.0).
    @0.95 failure → 0.95 + 0.25×(0.30−0.95) + 0.08 = 0.8675 (net drop, away from 1.0)."""
    engine = NeedEngine()
    current = {NeedType.SAFETY.value: 0.95}
    updated = engine.evolve_intensities(current, dominant_need=NeedType.SAFETY, succeeded=False)
    safety = updated[NeedType.SAFETY.value]
    assert abs(safety - 0.8675) < 1e-9 and safety < 0.95  # homeostasis caps; no runaway to 1.0


def test_should_update_goals_when_no_active_goal() -> None:
    """No active goal at all (0 < low-water) → triggers an update (a new goal must be generated)."""
    engine = NeedEngine()
    entities = [
        GoalEntity(
            id="stg-1-0", text="已完成的目标", goal_type="short_term",
            status=GoalStatus.COMPLETED, related_need=NeedType.SOCIAL, created_step=1,
        ),
    ]
    assert engine._should_update_goals(NeedType.SOCIAL, entities) is True  # noqa: SLF001


def test_should_update_goals_low_water_triggers_topup_even_when_need_matches() -> None:
    """Active goals below low-water (here 1 < 2) → replenish even if the need matches; don't wait
    for the queue to empty."""
    from agent.need import _SHORT_TERM_GOAL_LOW_WATER
    engine = NeedEngine()
    assert _SHORT_TERM_GOAL_LOW_WATER == 2
    entities = [
        GoalEntity(
            id="stg-1-0", text="一个匹配当前需求的 active goal", goal_type="short_term",
            related_need=NeedType.SAFETY, created_step=1,
        ),
    ]
    # One active goal whose related_need matches dominant, but below the water mark → still
    # replenishes.
    assert engine._should_update_goals(NeedType.SAFETY, entities) is True  # noqa: SLF001


def test_should_update_goals_related_need_none_is_wildcard() -> None:
    """goal.related_need=None is a wildcard: switching dominant_need doesn't trigger an update.

    Scenario: a goal restored from a snapshot with no need tag must not be force-dropped just
    because dominant changed. Seed >= low-water goals to avoid the low-water short-circuit and
    isolate the wildcard check.
    """
    engine = NeedEngine()
    entities = [
        GoalEntity(
            id=f"stg-restored-{i}", text=text, goal_type="short_term",
            related_need=None, created_step=0,
        )
        for i, text in enumerate(["从快照恢复的旧 goal", "另一条无 need 标签的旧 goal"])
    ]
    # dominant switches to SAFETY, but the goal's related_need=None → wildcard
    assert engine._should_update_goals(NeedType.SAFETY, entities) is False  # noqa: SLF001


@pytest.mark.asyncio
async def test_evaluate_goal_progress_llm_path_parses_json_by_index() -> None:
    """LLM path: the functional judge returns JSON with 1-based index → per-goal status,
    and a goal reaching COMPLETED sets the refresh flag."""
    from agent.goals import GoalEntity, GoalStatus
    from agent.need import NeedType
    from core.interfaces.llm import LLMResponse

    class _JsonJudge:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ):
            return LLMResponse(
                content='{"goals": [{"reason": "已达成", "index": 1, "status": "completed"}, '
                        '{"reason": "仍在推进", "index": 2, "status": "active"}]}',
                input_tokens=0, output_tokens=0, model="test",
            )

    engine = NeedEngine(llm_router=_JsonJudge())  # type: ignore[arg-type]
    personality = _make_personality(short_term_entities=[
        GoalEntity(id="stg-1-0", text="目标甲", goal_type="short_term",
                   related_need=NeedType.SOCIAL, created_step=1),
        GoalEntity(id="stg-1-1", text="目标乙", goal_type="short_term",
                   related_need=NeedType.SAFETY, created_step=1),
    ])
    updated = await engine.evaluate_goal_progress(
        step=3,
        action_description="做了一件事",
        action_result="结果如此",
        personality=personality,
    )
    by_id = {g.id: g for g in updated}
    assert by_id["stg-1-0"].status == GoalStatus.COMPLETED
    assert by_id["stg-1-0"].progress_summary == "已达成"
    assert by_id["stg-1-1"].status == GoalStatus.ACTIVE


@pytest.mark.asyncio
async def test_evaluate_goal_progress_injects_recent_factuals_for_cumulative_completion() -> None:
    """Cumulative completion: this round's action is unremarkable on its own, but recent factuals
    contain evidence of completion → the judge rules COMPLETED from the trajectory. Verifies
    recent_factuals reach the prompt through the trajectory-evidence channel."""
    from agent.goals import GoalEntity, GoalStatus
    from agent.need import NeedType
    from core.interfaces.llm import LLMResponse

    seen: dict[str, str] = {}

    class _TrajectoryJudge:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ):
            # messages=[system, user] (prefix-cache split); join both for the assertions.
            seen["text"] = "\n".join(m.content for m in messages)
            # Simulate a judge relying on the trajectory: rule completed only when recent
            # experience contains completion evidence.
            status = "completed" if "找到了密信" in seen["text"] else "active"
            return LLMResponse(
                content=f'{{"goals": [{{"reason": "据近期经历判断", "index": 1, "status": "{status}"}}]}}',
                input_tokens=0, output_tokens=0, model="test",
            )

    engine = NeedEngine(llm_router=_TrajectoryJudge())  # type: ignore[arg-type]
    personality = _make_personality(short_term_entities=[
        GoalEntity(id="stg-1-0", text="查清密信下落", goal_type="short_term",
                   related_need=NeedType.SAFETY, created_step=1),
    ])
    updated = await engine.evaluate_goal_progress(
        step=5,
        action_description="回到书房",
        action_result="抵达书房",  # unremarkable this round; this step alone doesn't show the goal is done
        personality=personality,
        recent_factuals=["我向侍卫打听到线索", "我在暗格里找到了密信"],
    )
    assert updated[0].status == GoalStatus.COMPLETED  # cumulative completion ruled from trajectory evidence
    assert "近期的客观经历" in seen["text"]
    assert "找到了密信" in seen["text"]


def test_goal_age_hint_only_for_aged_goals() -> None:
    """Elapsed-time hint: attached only when elapsed>0 (membrane: natural duration only, no step
    leak); not attached to a goal set this step."""
    from agent.goals import GoalEntity, goal_age_hint

    fresh = GoalEntity(id="g", text="x", goal_type="short_term", created_step=5)
    aged = GoalEntity(id="g", text="x", goal_type="short_term", created_step=1)
    assert goal_age_hint(fresh, step=5, seconds_per_step=3600) == ""          # just set, no hint
    hint = goal_age_hint(aged, step=5, seconds_per_step=3600)                 # 4 steps = 4 hours
    assert hint.startswith("（已历约") and "仍未了结" in hint  # "约N小时", not a step count
    assert "step" not in hint  # digits in a natural duration are fine; step is not

    # A goal with a set time goes on the appointment axis and gets no age hint: saying "elapsed…
    # still unresolved" about an appointment that isn't due yet is false, and the judge's shelving
    # criterion keys off that marker.
    dated = GoalEntity(id="g", text="x", goal_type="short_term", created_step=1, due_step=99)
    assert goal_age_hint(dated, step=50, seconds_per_step=3600) == ""


@pytest.mark.asyncio
async def test_evaluate_goal_progress_injects_goal_age_hint() -> None:
    """Every active goal in the judge prompt carries its elapsed time, so the judge can see how
    long it has been hanging."""
    from agent.goals import GoalEntity
    from agent.need import NeedType
    from core.interfaces.llm import LLMResponse

    seen: dict[str, str] = {}

    class _Judge:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ):
            # messages=[system, user] (prefix-cache split); join both for the assertions.
            seen["text"] = "\n".join(m.content for m in messages)
            return LLMResponse(
                content='{"goals": [{"reason": "在推进", "index": 1, "status": "active"}]}',
                input_tokens=0, output_tokens=0, model="test",
            )

    engine = NeedEngine(llm_router=_Judge(), seconds_per_step=3600)  # type: ignore[arg-type]
    personality = _make_personality(short_term_entities=[
        GoalEntity(id="stg-aged", text="安抚某人情绪", goal_type="short_term",
                   related_need=NeedType.SOCIAL, created_step=1),
    ])
    await engine.evaluate_goal_progress(
        step=4, action_description="又劝了一次", action_result="对方仍焦躁",
        personality=personality,
    )
    assert "已历约" in seen["text"] and "仍未了结" in seen["text"]


@pytest.mark.asyncio
async def test_evaluate_goal_progress_backstop_shelves_stalled_active_goal() -> None:
    """Hard fallback: the LLM judge keeps ruling active, but the goal has been active for the
    threshold number of steps → code force-shelves it (FAILED) so cognition doesn't lock up."""
    from agent.goals import GoalEntity, GoalStatus
    from agent.goals import _SHORT_TERM_GOAL_STALL_STEPS
    from agent.need import NeedType
    from core.interfaces.llm import LLMResponse

    class _StubbornJudge:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ):
            return LLMResponse(
                content='{"goals": [{"reason": "还在推进", "index": 1, "status": "active"}]}',
                input_tokens=0, output_tokens=0, model="test",
            )

    engine = NeedEngine(llm_router=_StubbornJudge())  # type: ignore[arg-type]
    created = 1
    personality = _make_personality(short_term_entities=[
        GoalEntity(id="stg-stall", text="安抚某人情绪", goal_type="short_term",
                   related_need=NeedType.SOCIAL, created_step=created),
    ])
    updated = await engine.evaluate_goal_progress(
        step=created + _SHORT_TERM_GOAL_STALL_STEPS,  # exactly at the threshold
        action_description="又劝了一次", action_result="对方仍焦躁,未达预期",
        personality=personality,
    )
    assert updated[0].status is GoalStatus.FAILED           # LLM rules active; fallback force-shelves
    assert "搁置" in updated[0].progress_summary


@pytest.mark.asyncio
async def test_evaluate_goal_progress_backstop_spares_not_yet_stalled_goal() -> None:
    """Boundary: active goals below the threshold aren't killed by the fallback; the LLM's active
    ruling stands."""
    from agent.goals import GoalEntity, GoalStatus
    from agent.goals import _SHORT_TERM_GOAL_STALL_STEPS
    from agent.need import NeedType
    from core.interfaces.llm import LLMResponse

    class _StubbornJudge:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ):
            return LLMResponse(
                content='{"goals": [{"reason": "还在推进", "index": 1, "status": "active"}]}',
                input_tokens=0, output_tokens=0, model="test",
            )

    engine = NeedEngine(llm_router=_StubbornJudge())  # type: ignore[arg-type]
    created = 1
    personality = _make_personality(short_term_entities=[
        GoalEntity(id="stg-young", text="安抚某人情绪", goal_type="short_term",
                   related_need=NeedType.SOCIAL, created_step=created),
    ])
    updated = await engine.evaluate_goal_progress(
        step=created + _SHORT_TERM_GOAL_STALL_STEPS - 1,  # one step short of the threshold
        action_description="劝了一次", action_result="对方仍焦躁",
        personality=personality,
    )
    assert updated[0].status is GoalStatus.ACTIVE           # below threshold, stays active


@pytest.mark.asyncio
async def test_evaluate_goal_progress_backstop_shelves_stalled_interrupted_goal() -> None:
    """The age fallback covers INTERRUPTED too: the judge only evaluates active goals and never
    reclaims INTERRUPTED ones, so a suspended goal that's never evicted by over-capacity would be a
    permanent zombie. Once its age hits the threshold it must be force-shelved (FAILED)."""
    from agent.goals import GoalEntity, GoalStatus
    from agent.goals import _SHORT_TERM_GOAL_STALL_STEPS
    from agent.need import NeedType

    class _NoopJudge:
        """Only INTERRUPTED goals, no active → the judge is never called (the assert below proves
        the fallback works on its own)."""
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ):
            raise AssertionError("judge 不应被调用：无 active 目标可评估")

    engine = NeedEngine(llm_router=_NoopJudge())  # type: ignore[arg-type]
    created = 1
    personality = _make_personality(short_term_entities=[
        GoalEntity(
            id="stg-interrupted-stall", text="安抚某人情绪", goal_type="short_term",
            status=GoalStatus.INTERRUPTED, related_need=NeedType.SOCIAL, created_step=created,
        ),
    ])
    updated = await engine.evaluate_goal_progress(
        step=created + _SHORT_TERM_GOAL_STALL_STEPS,  # exactly at the threshold
        action_description="又劝了一次", action_result="对方仍焦躁,未达预期",
        personality=personality,
    )
    # Non-active (INTERRUPTED) with age at the threshold → the age fallback force-shelves it as
    # FAILED, regardless of the "must be ACTIVE" restriction.
    assert updated[0].status is GoalStatus.FAILED
    assert "搁置" in updated[0].progress_summary


# A goal with a set time is on the appointment axis, not the age axis. The three cases below are
# the complement of is_past_allowance: before the time (even if past the age threshold) it isn't
# reaped / at the due step it isn't reaped / after the due time it's reaped on the first evaluation.
def _stubborn_judge():
    """The judge keeps ruling active; whether a goal is reaped is decided only by the code's
    time-limit fallback."""
    from core.interfaces.llm import LLMResponse

    class _Judge:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000, **kwargs):
            return LLMResponse(
                content='{"goals": [{"reason": "还在推进", "index": 1, "status": "active"}]}',
                input_tokens=0, output_tokens=0, model="test",
            )

    return _Judge()


async def _judge_dated_goal(*, created: int, due: int, step: int):
    from agent.goals import GoalEntity
    from agent.need import NeedType

    engine = NeedEngine(llm_router=_stubborn_judge())  # type: ignore[arg-type]
    personality = _make_personality(short_term_entities=[
        GoalEntity(id="stg-dated", text="明早五更去玄武门见大哥", goal_type="short_term",
                   related_need=NeedType.SOCIAL, created_step=created, due_step=due),
    ])
    return await engine.evaluate_goal_progress(
        step=step, action_description="又等了一阵", action_result="时候还没到",
        personality=personality,
    )


@pytest.mark.asyncio
async def test_backstop_spares_pending_appointment_older_than_the_age_threshold() -> None:
    """Not due yet isn't procrastination: a goal with a distant appointment must not be reaped,
    even if its age is long past the threshold."""
    from agent.goals import _SHORT_TERM_GOAL_STALL_STEPS, GoalStatus

    created = 1
    step = created + _SHORT_TERM_GOAL_STALL_STEPS * 2  # age far past the threshold
    updated = await _judge_dated_goal(created=created, due=step + 1, step=step)
    assert updated[0].status is GoalStatus.ACTIVE


@pytest.mark.asyncio
async def test_backstop_spares_overdue_appointment_within_its_allowance() -> None:
    """Past due is not void: after missing the appointment there is a whole allowance to make up
    for it, and the goal must not be reaped during it."""
    from agent.goals import _SHORT_TERM_GOAL_STALL_STEPS, GoalStatus

    due = 10
    updated = await _judge_dated_goal(
        created=1, due=due, step=due + _SHORT_TERM_GOAL_STALL_STEPS - 1,  # one step short of the allowance
    )
    assert updated[0].status is GoalStatus.ACTIVE


@pytest.mark.asyncio
async def test_backstop_shelves_appointment_after_its_allowance_runs_out() -> None:
    """Make-up allowance also used up → reaped, even if the judge keeps ruling active; the closing
    note is written on the appointment axis."""
    from agent.goals import _SHORT_TERM_GOAL_STALL_STEPS, GoalStatus

    due = 10
    updated = await _judge_dated_goal(
        created=1, due=due, step=due + _SHORT_TERM_GOAL_STALL_STEPS,
    )
    assert updated[0].status is GoalStatus.FAILED
    assert "错过了" in updated[0].progress_summary
    assert "反复推进" not in updated[0].progress_summary  # it may never have been attempted


@pytest.mark.asyncio
async def test_dated_goal_outlives_an_undated_one_created_at_the_same_step() -> None:
    """A goal with a set time always lives strictly longer than an ordinary goal set at the same
    moment: immune until due, and the clock only starts after."""
    from agent.goals import _SHORT_TERM_GOAL_STALL_STEPS, GoalEntity, is_past_allowance

    created, due = 1, 20
    plain = GoalEntity(id="a", text="x", goal_type="short_term", created_step=created)
    dated = GoalEntity(id="b", text="y", goal_type="short_term", created_step=created,
                       due_step=due)
    plain_dies = created + _SHORT_TERM_GOAL_STALL_STEPS
    assert is_past_allowance(plain, plain_dies)
    assert not is_past_allowance(dated, plain_dies)            # the appointment wasn't due yet then
    assert not is_past_allowance(dated, due)                   # still there at the due step
    assert is_past_allowance(dated, due + _SHORT_TERM_GOAL_STALL_STEPS)


@pytest.mark.asyncio
async def test_evaluate_goal_progress_interrupted_not_yet_stale_left_untouched() -> None:
    """The judge only sees ACTIVE goals: an INTERRUPTED goal below the threshold neither enters
    the judge's candidate set nor gets reclaimed, and stays in the queue as is (no resume, no
    fail). Verifies (1) the prompt lists only active goals; (2) the interrupted goal's status is
    unchanged."""
    from agent.goals import GoalEntity, GoalStatus
    from agent.need import NeedType
    from core.interfaces.llm import LLMResponse

    seen: dict[str, str] = {}

    class _Judge:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ):
            seen["text"] = "\n".join(m.content for m in messages)
            return LLMResponse(
                content='{"goals": [{"reason": "本步已达成", "index": 1, "status": "completed"}]}',
                input_tokens=0, output_tokens=0, model="test",
            )

    engine = NeedEngine(llm_router=_Judge())  # type: ignore[arg-type]
    personality = _make_personality(short_term_entities=[
        # Interrupted goal: below threshold (young), not judged, not hit by the fallback.
        GoalEntity(
            id="stg-interrupted", text="送信给盟友",
            goal_type="short_term", status=GoalStatus.INTERRUPTED,
            related_need=NeedType.SOCIAL, created_step=2,
        ),
        # Regular active goal: the only one the judge sees and evaluates.
        GoalEntity(
            id="stg-active", text="打探敌情",
            goal_type="short_term", status=GoalStatus.ACTIVE,
            related_need=NeedType.SAFETY, created_step=2,
        ),
    ])
    updated = await engine.evaluate_goal_progress(
        step=3, action_description="做事", action_result="有进展",  # age=1 < threshold
        personality=personality,
    )
    # (1) The judge's candidate set contains only active goals; the interrupted goal isn't in the
    # prompt.
    assert "打探敌情" in seen["text"]
    assert "送信给盟友" not in seen["text"]
    # (2) The interrupted goal is kept as is (no resume, no fail below threshold); the active goal
    # is judged complete normally.
    by_id = {g.id: g for g in updated}
    assert by_id["stg-interrupted"].status is GoalStatus.INTERRUPTED
    assert by_id["stg-active"].status is GoalStatus.COMPLETED


@pytest.mark.asyncio
async def test_evaluate_goal_progress_no_recent_factuals_omits_block() -> None:
    """Without recent_factuals (default empty), the prompt has no recent-experience block."""
    from agent.goals import GoalEntity
    from agent.need import NeedType
    from core.interfaces.llm import LLMResponse

    seen: dict[str, str] = {}

    class _Judge:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ):
            # messages=[system, user] (prefix-cache split); join both for the assertions.
            seen["text"] = "\n".join(m.content for m in messages)
            return LLMResponse(
                content='{"goals": [{"reason": "r", "index": 1, "status": "active"}]}',
                input_tokens=0, output_tokens=0, model="test",
            )

    engine = NeedEngine(llm_router=_Judge())  # type: ignore[arg-type]
    personality = _make_personality(short_term_entities=[
        GoalEntity(id="stg-1-0", text="某目标", goal_type="short_term",
                   related_need=NeedType.SAFETY, created_step=1),
    ])
    await engine.evaluate_goal_progress(
        step=2, action_description="做事", action_result="结果",
        personality=personality,
    )
    assert "近期的客观经历" not in seen["text"]


def test_ensure_maslow_baseline_fills_missing_physiological() -> None:
    """If the persona omits physiological, ensure_maslow_baseline adds it as an active baseline;
    existing needs are left alone."""
    active = [
        NeedState(type=NeedType.SAFETY, label="安稳", intensity=0.9, weight=1.2),
        NeedState(type=NeedType.ESTEEM, label="尊严", intensity=0.7, weight=1.1),
    ]
    hidden = [NeedState(type=NeedType.SOCIAL, label="归属", intensity=0.4, weight=0.8, is_hidden=True)]
    ensure_maslow_baseline(active, hidden)
    present = {n.type for n in active} | {n.type for n in hidden}
    assert present == set(NeedType)
    phys = next(n for n in active if n.type == NeedType.PHYSIOLOGICAL)
    assert not phys.is_hidden and 0 < phys.intensity < 0.9   # active baseline, dormant
    assert next(n for n in active if n.type == NeedType.SAFETY).intensity == 0.9


def test_ensure_maslow_baseline_is_idempotent() -> None:
    active = [NeedState(type=t, label=t.value, intensity=0.5, weight=1.0) for t in NeedType]
    hidden: list[NeedState] = []
    ensure_maslow_baseline(active, hidden)
    assert len(active) == len(NeedType)


# ---------------------------------------------------------------------------
# Maslow prepotency: an acute survival-floor crisis takes precedence
# ---------------------------------------------------------------------------


def _five_need_personality() -> PersonalityLayer:
    """A Li Shimin-style profile: high safety endowment (I×W=1.08), physiological only at its
    resting baseline (0.3×1.0)."""
    return _make_personality(active=[
        NeedState(type=NeedType.PHYSIOLOGICAL, label="休整", intensity=0.3, weight=1.0),
        NeedState(type=NeedType.SAFETY, label="求安", intensity=0.9, weight=1.2),
        NeedState(type=NeedType.ESTEEM, label="尊严", intensity=0.7, weight=1.1),
    ])


@pytest.mark.asyncio
async def test_prepotency_physiological_crisis_preempts_high_safety() -> None:
    """Physiological activation ≥0.85 (collapse from exhaustion) → overrides the
    higher-endowed safety as dominant, and dominant==argmax."""
    res = await NeedEngine().run(
        current_step=1, personality=_five_need_personality(),
        visible_agents=[], pending_messages=0,
        emotion=EmotionState(primary="frustration", intensity=0.6, valence=-0.7),
        need_relevance={NeedType.PHYSIOLOGICAL: 0.9, NeedType.SAFETY: 0.7},
    )
    assert res.dominant_need == NeedType.PHYSIOLOGICAL
    assert res.dominant_need == max(res.scores, key=res.scores.get)  # invariant


@pytest.mark.asyncio
async def test_prepotency_below_threshold_does_not_fire() -> None:
    """Physiological activation 0.8 (<0.85: dominating thoughts but not a crisis) → no
    trigger; dominant stays the additive argmax (safety)."""
    res = await NeedEngine().run(
        current_step=1, personality=_five_need_personality(),
        visible_agents=[], pending_messages=0,
        need_relevance={NeedType.PHYSIOLOGICAL: 0.8, NeedType.SAFETY: 0.7},
    )
    assert res.dominant_need == NeedType.SAFETY
    assert res.dominant_need == max(res.scores, key=res.scores.get)


@pytest.mark.asyncio
async def test_prepotency_safety_beats_physiological_when_both_in_crisis() -> None:
    """Physiological and safety both in crisis (both ≥0.85) → safety first (get out of danger
    first; don't rest with a killer closing in)."""
    res = await NeedEngine().run(
        current_step=1, personality=_five_need_personality(),
        visible_agents=[], pending_messages=0,
        need_relevance={NeedType.PHYSIOLOGICAL: 0.9, NeedType.SAFETY: 0.95},
    )
    assert res.dominant_need == NeedType.SAFETY


@pytest.mark.asyncio
async def test_prepotency_non_survival_need_never_preempts() -> None:
    """esteem gets no prepotency even at activation=1.0 (no biological override) → normal
    additive competition."""
    res = await NeedEngine().run(
        current_step=1, personality=_five_need_personality(),
        visible_agents=[], pending_messages=0,
        need_relevance={NeedType.ESTEEM: 1.0, NeedType.SAFETY: 0.2},
    )
    # esteem wins naturally through high additive activation (2.0×1.0 + I×W), not a prepotency
    # boost; dominant==argmax still holds.
    assert res.dominant_need == max(res.scores, key=res.scores.get)


# ---------------------------------------------------------------------------
# Situational anchoring: each of the three need-engine prompts (short-term goals / long-term
# goals / goal-progress judging) gets a header
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_short_term_goal_prompt_includes_situation_header() -> None:
    """Situational anchoring: the short-term goal prompt's first line carries the first-person
    header "我此刻在...,时间为...". location_view comes from spatial (existing path);
    world_time_label is passed through run()."""
    from core.interfaces.llm import LLMRouter, LLMScene
    from core.interfaces.perception import LocationView, Situation
    from providers.llm.mock import MockLLMProvider

    provider = MockLLMProvider(fixed_response='{"goals": ["先去看父亲"]}')
    router = LLMRouter({scene: provider for scene in LLMScene})
    engine = NeedEngine(llm_router=router)
    personality = _make_personality()

    await engine.run(
        current_step=1,
        personality=personality,
        pending_messages=0,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
        situation=Situation(location_view=LocationView(name="紫宸殿", description="天子日常理政之所"), time_label="申时"),
        force_goal_update=True,  # force short-term goal generation → calls the LLM
    )
    assert provider.call_history, "短期目标 prompt 应被调用"
    # The call sends [system, user] (prefix-cache split); join both for the assertions.
    prompt = "\n".join(m.content for m in provider.call_history[0])
    assert "我此刻在紫宸殿" in prompt and "时间为申时" in prompt
    assert "step=" not in prompt


@pytest.mark.asyncio
async def test_short_term_goal_prompt_no_flat_context_blob_and_separate_memory_section() -> None:
    """Projection separation: with no perception signals, the short-term goal prompt
    must not get the flat embedding/keyword query (context). That's a machine retrieval format
    (machine separators + own goals + code tags) and is unreadable as LLM context. The agent's
    own recent memories go in a separate labeled section."""
    from core.interfaces.llm import LLMRouter, LLMScene
    from providers.llm.mock import MockLLMProvider

    provider = MockLLMProvider(fixed_response='{"goals": ["静观其变"]}')
    router = LLMRouter({scene: provider for scene in LLMScene})
    engine = NeedEngine(llm_router=router)

    flat_query_blob = "太极宫，密奏副本，谋先发制人"  # projection A flat query (never in the prompt)
    await engine.run(
        current_step=1,
        personality=_make_personality(),
        pending_messages=0,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
        perceived_signal_texts=[],                       # no perception signal
        recent_memory_texts=["我昨夜辗转难眠，总想起父亲的话。"],
        force_goal_update=True,
    )
    assert provider.call_history
    # The call sends [system, user] (prefix-cache split); join both for the assertions.
    prompt = "\n".join(m.content for m in provider.call_history[0])
    # 1. Under "我刚刚感知到" is "无特殊事件", not the context blob.
    assert "- 无特殊事件" in prompt
    assert flat_query_blob not in prompt             # the flat query is never LLM context
    # 2. Own recent memories are in their own labeled section (two streams: objective course +
    # interpretation), not under "我刚刚感知到".
    assert "我近来经历的事（客观经过；括号内是我当时的主观理解与感受" in prompt
    assert "我昨夜辗转难眠" in prompt
    assert prompt.index("我刚刚感知到") < prompt.index("我近来经历的事")
    # 3. No code tags / machine separators leak.
    assert "最近记忆:" not in prompt and "recent_memory" not in prompt


@pytest.mark.asyncio
async def test_short_term_goal_empty_llm_result_is_noop_not_rule_fallback() -> None:
    """LLM deliberately returns empty goals ("no new short-term goal this step") → return []
    as a no-op; never fabricate a placeholder goal."""
    from core.interfaces.llm import LLMRouter, LLMScene
    from providers.llm.mock import MockLLMProvider

    provider = MockLLMProvider(fixed_response='{"thought": "短期目标已覆盖，无需新增", "goals": []}')
    router = LLMRouter({scene: provider for scene in LLMScene})
    engine = NeedEngine(llm_router=router)

    goals = await engine._generate_short_term_goals(
        dominant_need=NeedType.SAFETY,
        personality=_make_personality(),
    )
    assert goals == []
    # A template SAFETY goal must never appear when the LLM succeeds but returns empty.
    assert "先观察局势，确认安全后再行动。" not in goals


@pytest.mark.asyncio
async def test_short_term_goal_prompt_injects_step_duration() -> None:
    """The short-term goal prompt includes the per-step duration, steering the LLM toward
    directions that span more than one step rather than a single one-step action."""
    from core.interfaces.llm import LLMRouter, LLMScene
    from providers.llm.mock import MockLLMProvider

    provider = MockLLMProvider(fixed_response='{"goals": ["静观其变"]}')
    router = LLMRouter({scene: provider for scene in LLMScene})
    engine = NeedEngine(llm_router=router, seconds_per_step=3600)

    await engine._generate_short_term_goals(
        dominant_need=NeedType.SAFETY,
        personality=_make_personality(),
    )
    # The call sends [system, user] (prefix-cache split); join both for the assertions.
    prompt = "\n".join(m.content for m in provider.call_history[0])
    assert "约1小时" in prompt  # describe_duration(1, 3600)
    assert "不止一步" in prompt


@pytest.mark.asyncio
async def test_short_term_goal_thought_audits_a_long_neglected_direction() -> None:
    """The check for an absent long-term direction must be in the thought checklist: goals are
    derived from thought, so listing it only in user doesn't count.

    It's a check, not an instruction: the same passage must keep "don't force a goal", or it
    collides with "don't let long-term goals override the present".
    """
    from core.interfaces.llm import LLMRouter, LLMScene
    from providers.llm.mock import MockLLMProvider

    provider = MockLLMProvider(fixed_response='{"thought": "我此刻……", "goals": []}')
    router = LLMRouter({scene: provider for scene in LLMScene})
    engine = NeedEngine(llm_router=router, seconds_per_step=3600)

    await engine._generate_short_term_goals(
        dominant_need=NeedType.SAFETY,
        personality=_make_personality(),
    )
    system = provider.call_history[0][0].content
    thought_spec = system[system.index("先在 thought 里"):]
    assert "哪个长期方向我一直没沾" in thought_spec
    assert "别硬凑" in system
    assert "更不能用它盖过眼前" in system


@pytest.mark.asyncio
async def test_long_term_goal_update_prompt_includes_situation_header() -> None:
    """Situational anchoring: the revise_long_term_goals prompt's first line carries the
    first-person header "我此刻在...,时间为...". The caller (Agent) passes location /
    world_time_label through from the spatial snapshot."""
    from core.interfaces.llm import LLMRouter, LLMScene
    from core.interfaces.perception import LocationView, Situation
    from providers.llm.mock import MockLLMProvider

    provider = MockLLMProvider(fixed_response='{"thought": "...", "goals": []}')
    router = LLMRouter({scene: provider for scene in LLMScene})
    engine = NeedEngine(llm_router=router)
    personality = _make_personality()

    await engine.revise_long_term_goals(
        personality=personality,
        recent_memory_texts=["发生了一些事。"],
        is_main_character=True,
        situation=Situation(location_view=LocationView(name="书房", description="僻静的内书房"), time_label="子时"),
    )
    assert provider.call_history, "长期目标 prompt 应被调用"
    # The call sends [system, user] (prefix-cache split); join both for the assertions.
    prompt = "\n".join(m.content for m in provider.call_history[0])
    assert "我此刻在书房" in prompt and "时间为子时" in prompt
    # The header is at the very top of the prompt (before the persona block).
    assert prompt.index("我此刻在书房") < prompt.index("【我是谁】")


@pytest.mark.asyncio
async def test_goal_evaluation_prompt_includes_situation_header_third_person() -> None:
    """Situational anchoring: evaluate_goal_progress is a functional third-person ruling → uses a
    voice='third' header, shaped like "当前时间：...；当前地点：..."."""
    from agent.goals import GoalEntity, GoalStatus
    from core.interfaces.llm import LLMRouter, LLMScene
    from core.interfaces.perception import LocationView, Situation
    from providers.llm.mock import MockLLMProvider

    provider = MockLLMProvider(
        fixed_response='{"goals": [{"reason": "ok", "index": 1, "status": "active"}]}'
    )
    router = LLMRouter({scene: provider for scene in LLMScene})
    engine = NeedEngine(llm_router=router)
    personality = _make_personality(short_term_entities=[
        GoalEntity(
            id="g1", text="去看父亲", goal_type="short_term",
            related_need=NeedType.SOCIAL, status=GoalStatus.ACTIVE, created_step=1,
        ),
    ])

    await engine.evaluate_goal_progress(
        step=5,
        action_description="入殿",
        action_result="见到父亲",
        personality=personality,
        situation=Situation(location_view=LocationView(name="太极宫", description="正殿"), time_label="巳时"),
    )
    assert provider.call_history, "目标进度判定 prompt 应被调用"
    # The call sends [system, user] (prefix-cache split); join both for the assertions.
    prompt = "\n".join(m.content for m in provider.call_history[0])
    # functional third-person header (colon + semicolon format)
    assert "当前时间：巳时" in prompt and "当前地点：太极宫" in prompt
    assert "我此刻在" not in prompt
    assert "第5步" not in prompt


# ---------------------------------------------------------------------------
# Short-term goal FIFO queue: state changes that preserve existing goals
# ---------------------------------------------------------------------------

def _short_term_entity(text: str, *, need: NeedType | None, step: int = 0):
    from agent.goals import GoalEntity, GoalStatus
    return GoalEntity(
        id=f"stg-{step}-{abs(hash(text)) % 10000}",
        text=text,
        goal_type="short_term",
        status=GoalStatus.ACTIVE,
        related_need=need,
        created_step=step,
    )


@pytest.mark.asyncio
async def test_run_appends_new_goals_without_dropping_existing_active() -> None:
    """Triggering generation doesn't overwrite the whole list: existing unfinished goals must stay
    and new goals are appended."""
    engine = NeedEngine()
    # Put a sentinel goal that generation would never produce straight into the live queue
    # (related_need=None is a wildcard, so it won't be judged misaligned).
    personality = _make_personality(short_term_entities=[
        _short_term_entity("SENTINEL_未完成的旧意图", need=None, step=0)
    ])

    result = await engine.run(
        current_step=1,
        personality=personality,
        pending_messages=0,
        force_goal_update=True,  # force generation; unfinished intents must stay queued
    )

    assert "SENTINEL_未完成的旧意图" in result.short_term_goals
    survivor = next(
        g for g in result.short_term_goal_entities if g.text == "SENTINEL_未完成的旧意图"
    )
    assert survivor.status == GoalStatus.ACTIVE


def test_enqueue_dedups_against_live_queue() -> None:
    """Generation dedup: text identical to a goal already in the live queue isn't enqueued again
    and doesn't push out valid old goals."""
    ents = enqueue_goals([], ["A", "B"], dominant_need=NeedType.SAFETY, current_step=1)
    ents = enqueue_goals(ents, ["A", "C"], dominant_need=NeedType.SAFETY, current_step=2)
    live = [g.text for g in live_goals(ents)]
    assert live == ["A", "B", "C"]


def test_enqueue_dedups_near_duplicate_by_similarity() -> None:
    """Near-verbatim duplicates (SequenceMatcher.ratio >= threshold) aren't enqueued; broader
    than exact matching."""
    ents = enqueue_goals([], ["去和李将军谈判结盟"], dominant_need=NeedType.SOCIAL, current_step=1)
    # Light rewording with high literal overlap → treated as a duplicate, not enqueued.
    ents = enqueue_goals(ents, ["和李将军谈判结盟"], dominant_need=NeedType.SOCIAL, current_step=2)
    live = [g.text for g in live_goals(ents)]
    assert live == ["去和李将军谈判结盟"]


def test_enqueue_keeps_lexically_distinct_goal() -> None:
    """Distinct intents that read alike but mean different things aren't merged; a false merge
    means a lost intent."""
    ents = enqueue_goals([], ["去和李将军谈判结盟"], dominant_need=NeedType.SOCIAL, current_step=1)
    ents = enqueue_goals(ents, ["去找张将军喝酒叙旧"], dominant_need=NeedType.SOCIAL, current_step=2)
    live = [g.text for g in live_goals(ents)]
    assert live == ["去和李将军谈判结盟", "去找张将军喝酒叙旧"]


def test_enqueue_over_cap_evicts_oldest_non_silently() -> None:
    """Over capacity, the oldest goals are evicted from the head and marked FAILED as history;
    they never silently vanish.

    Goal names differ a lot literally, so similarity dedup can't interfere with the eviction test.
    """
    from agent.goals import _SHORT_TERM_GOAL_CAP

    pool = [
        "守住玄武门", "拉拢秦王旧部", "打探东宫动向", "筹措粮草军械", "安抚后宫诸妃",
        "联络城外驻军", "起草讨逆檄文", "修缮宫墙箭楼", "清点府库存银", "遣散闲杂人等",
    ]
    assert len(pool) >= _SHORT_TERM_GOAL_CAP + 2  # need two over capacity to trigger eviction
    texts = pool[:_SHORT_TERM_GOAL_CAP + 2]
    oldest_two = texts[:2]
    ents = enqueue_goals([], texts, dominant_need=NeedType.SAFETY, current_step=1)

    live = [g.text for g in live_goals(ents)]
    assert len(live) == _SHORT_TERM_GOAL_CAP
    # The two oldest are evicted from the live queue,
    assert all(t not in live for t in oldest_two)
    # but they don't silently vanish: still in entities and marked FAILED (into the review channel).
    evicted = [g for g in ents if g.text in oldest_two]
    assert evicted and all(g.status == GoalStatus.FAILED for g in evicted)
    assert all(g.progress_summary.startswith("搁置") for g in evicted)  # narrative only, no code-layer counts


@pytest.mark.asyncio
async def test_completing_one_goal_preserves_other_active_goals() -> None:
    """Completing one goal leaves the other still-ACTIVE goals in the live queue (partial
    completion doesn't affect the rest)."""
    from core.interfaces.llm import LLMResponse

    class _Judge:
        async def complete(self, scene, messages, **kwargs):
            return LLMResponse(
                content='{"goals": [{"reason": "宫门已控", "index": 1, "status": "completed"}, '
                        '{"reason": "未涉及", "index": 2, "status": "active"}]}',
                input_tokens=0, output_tokens=0, model="test",
            )

    engine = NeedEngine(llm_router=_Judge())  # type: ignore[arg-type]
    ents = enqueue_goals([], ["守住玄武门"], dominant_need=NeedType.SAFETY, current_step=1)
    ents = enqueue_goals(ents, ["拉拢盟友"], dominant_need=NeedType.SOCIAL, current_step=1)
    personality = _make_personality(short_term_entities=ents)

    updated = await engine.evaluate_goal_progress(
        step=2,
        action_description="封锁宫门",
        action_result="玄武门已在掌控",
        personality=personality,
    )

    live = [g.text for g in live_goals(updated)]
    assert "拉拢盟友" in live
    assert "守住玄武门" not in live
    completed = [g for g in updated if g.text == "守住玄武门"]
    assert completed and completed[0].status == GoalStatus.COMPLETED  # kept as history, not silent


# ── residue: cross-step continuity ─────────────────────────────────────────────────────


def test_evict_over_cap_drops_cognitive_before_residue() -> None:
    """Over capacity, the oldest plan is evicted first; unresolved matters stay in the queue.
    That is the whole point of residue.

    Otherwise two newly generated routine goals could push "something I promised" out of the cap-6
    queue.
    """
    residue = _goal("我答应了要去查此事", origin=GoalOrigin.RESIDUE, created_step=0)
    entities = [residue] + [
        _goal(f"计划{i}", created_step=i) for i in range(1, _SHORT_TERM_GOAL_CAP + 1)
    ]
    _evict_over_cap(entities, current_step=99)

    assert residue.status == GoalStatus.ACTIVE, "没有了结的事项不该被新计划挤下去"
    evicted = [g for g in entities if g.status == GoalStatus.FAILED]
    assert [g.text for g in evicted] == ["计划1"], "该逐出的是最旧的那条计划"


def test_evict_over_cap_falls_back_to_residue_when_no_cognitive() -> None:
    """When the live queue is all unresolved matters, the oldest is still evicted: residue is
    bounded and can't hold slots forever."""
    entities = [
        _goal(f"没了结{i}", origin=GoalOrigin.RESIDUE, created_step=i)
        for i in range(_SHORT_TERM_GOAL_CAP + 1)
    ]
    _evict_over_cap(entities, current_step=99)

    evicted = [g for g in entities if g.status == GoalStatus.FAILED]
    assert [g.text for g in evicted] == ["没了结0"]
    assert len(live_goals(entities)) == _SHORT_TERM_GOAL_CAP


def test_enqueue_goals_defaults_to_cognitive_origin() -> None:
    """Goals produced by need-generation LLM are plans, not unresolved matters (the default origin
    must not drift)."""
    entities = enqueue_goals([], ["我要去打探消息"], dominant_need=NeedType.SAFETY, current_step=1)
    assert entities[0].origin == GoalOrigin.COGNITIVE


def test_residue_dedups_against_existing_goals() -> None:
    """If the judge reports something already in the queue as a new unresolved matter, it isn't
    enqueued again (reuses the existing literal dedup)."""
    existing = enqueue_goals(
        [], ["我要去查探敌情"], dominant_need=NeedType.SAFETY, current_step=1,
    )
    after = enqueue_goals(
        existing, ["我要去查探敌情"], dominant_need=None, current_step=2,
        origin=GoalOrigin.RESIDUE,
    )
    assert len(live_goals(after)) == 1


@pytest.mark.parametrize(
    "raw",
    [
        '{"goals": []}',                                  # field missing
        '{"goals": [], "residue": "我答应了要去查此事"}',    # not a list
        '{"goals": [], "residue": [null, 123, {}]}',       # element not a string
        '{"goals": [], "residue": ["", "   "]}',           # empty string / whitespace only
        'not json at all',                                 # unparseable as a whole
        '["not", "a", "dict"]',                            # not a dict
    ],
)
def test_parse_residue_degrades_to_empty(raw: str) -> None:
    """Every anomaly falls back to empty: better to leave no matter this step than fabricate a
    pseudo-intent that gets persisted and recalled."""
    items, _reason = parse_residue(raw)
    assert items == []


class _ResidueJudge:
    """Judge stub returning a fixed verdict; records the prompts it was handed."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.messages: list = []

    async def complete(self, scene, messages, temperature=0.7, max_tokens=1000, **kwargs):
        from core.interfaces.llm import LLMResponse

        self.messages = messages
        return LLMResponse(content=self._content, input_tokens=0, output_tokens=0, model="test")


@pytest.mark.asyncio
async def test_evaluate_goal_progress_enqueues_residue() -> None:
    """Unresolved matters reported by the judge are enqueued as RESIDUE short-term goals; this is
    where cross-step continuity lands."""
    judge = _ResidueJudge('{"goals": [], "residue": ["我答应了要去查此事"]}')
    engine = NeedEngine(llm_router=judge)  # type: ignore[arg-type]
    personality = _make_personality(short_term_entities=[_goal("我要稳住局面", created_step=1)])

    updated = await engine.evaluate_goal_progress(
        step=2, action_description="与人交谈", action_result="谈定了",
        personality=personality,
    )

    residue = [g for g in updated if g.origin == GoalOrigin.RESIDUE]
    assert [g.text for g in residue] == ["我答应了要去查此事"]
    assert residue[0].status == GoalStatus.ACTIVE
    assert residue[0].created_step == 2


@pytest.mark.asyncio
async def test_evaluate_goal_progress_runs_with_empty_active_queue() -> None:
    """The judge still runs when the live queue is empty: a promise made by an agent that just
    cleared its goals is exactly the kind that must not be lost."""
    judge = _ResidueJudge('{"goals": [], "residue": ["我答应了明日去回话"]}')
    engine = NeedEngine(llm_router=judge)  # type: ignore[arg-type]

    updated = await engine.evaluate_goal_progress(
        step=2, action_description="与人交谈", action_result="谈定了",
        personality=_make_personality(short_term_entities=[]),
    )

    assert judge.messages, "空队列时也必须真的调用判官"
    assert [g.text for g in updated] == ["我答应了明日去回话"]


@pytest.mark.asyncio
async def test_evaluate_goal_progress_passes_expected_outcome_to_judge() -> None:
    """For actions like MOVE, the purpose ("what was I going to do once there") lives only in
    expected_outcome and must be fed to the judge."""
    judge = _ResidueJudge('{"goals": [], "residue": []}')
    engine = NeedEngine(llm_router=judge)  # type: ignore[arg-type]

    await engine.evaluate_goal_progress(
        step=2, action_description="前往太极殿", action_result="已抵达",
        personality=_make_personality(short_term_entities=[]),
        expected_outcome="面见皇帝，问明那桩事",
    )

    user_prompt = judge.messages[-1].content
    assert "面见皇帝，问明那桩事" in user_prompt


@pytest.mark.asyncio
async def test_residue_judge_prompt_marks_owed_goals() -> None:
    """Judging a residue resolved depends on whether it was actually fulfilled, so the candidate
    list must mark which items were left unresolved earlier."""
    judge = _ResidueJudge('{"goals": [{"index": 1, "status": "active"}], "residue": []}')
    engine = NeedEngine(llm_router=judge)  # type: ignore[arg-type]

    await engine.evaluate_goal_progress(
        step=2, action_description="做了一件事", action_result="结果如此",
        personality=_make_personality(short_term_entities=[
            _goal("我答应了要去查此事", origin=GoalOrigin.RESIDUE, created_step=1),
        ]),
    )

    assert "是否真的兑现" in judge.messages[-1].content


@pytest.mark.asyncio
async def test_evaluate_goal_progress_llm_failure_enqueues_no_residue() -> None:
    """Judge LLM failure → no unresolved matters written (Rule 1 tier-1: better nothing than a
    fabrication)."""
    class _Boom:
        async def complete(self, *a, **kw):
            raise RuntimeError("boom")

    engine = NeedEngine(llm_router=_Boom())  # type: ignore[arg-type]
    updated = await engine.evaluate_goal_progress(
        step=2, action_description="与人交谈", action_result="谈定了",
        personality=_make_personality(short_term_entities=[_goal("我要稳住局面", created_step=1)]),
    )
    assert not [g for g in updated if g.origin == GoalOrigin.RESIDUE]


def test_to_prompt_context_separates_plans_from_owed() -> None:
    """Plans and unresolved matters render as two blocks: flattening them into one list loses the
    distinction of "already on my plate"."""
    text = NeedEngine().to_prompt_context(
        _make_personality(),
        NeedType.SAFETY,
        [],
        [_goal("我要稳住局面"), _goal("我答应了要去查此事", origin=GoalOrigin.RESIDUE)],
        now_step=0,
    )
    assert "我的短期目标" in text and "我要稳住局面" in text
    assert "我还没有了结的事项" in text and "我答应了要去查此事" in text
    # Code-layer words never enter narrative-layer text.
    assert "residue" not in text and "origin" not in text and "cognitive" not in text


def test_parse_residue_reason_is_captured_but_never_gates_items() -> None:
    """residue_reason is the think-first anchor, captured into the trace for audit; parsing never
    depends on it (§5 boundary).

    Both directions: a given reason is captured, and a missing one doesn't drop the entry (a wording
    slip must not lose a real outstanding item).
    """
    items, reason = parse_residue(
        '{"residue_reason": "他在对话末尾应下了明日回话", "residue": ["我答应了明日去回话"]}'
    )
    assert items == [("我答应了明日去回话", None)]
    assert reason == "他在对话末尾应下了明日回话"

    # No reason: the entry is enqueued as usual.
    items2, reason2 = parse_residue('{"residue": ["我答应了明日去回话"]}')
    assert items2 == [("我答应了明日去回话", None)] and reason2 == ""

    # Reason given but the judge finds nothing left: an empty array is a valid answer, and the
    # reason is still captured.
    items3, reason3 = parse_residue('{"residue_reason": "只是独自赶路，未与人交涉", "residue": []}')
    assert items3 == [] and reason3 == "只是独自赶路，未与人交涉"


@pytest.mark.asyncio
async def test_short_term_goal_prompt_injects_standing_condition() -> None:
    """Where I can push is first limited by what I can still do right now.

    Otherwise someone tied up sets "go scout the Eastern Palace"; short-term goals are reread by
    every later decision, so a goal that spins idle keeps spinning.
    """
    from core.interfaces.condition import BodyCondition
    from core.interfaces.llm import LLMRouter, LLMScene
    from core.interfaces.perception import LocationView, Situation
    from providers.llm.mock import MockLLMProvider

    provider = MockLLMProvider(fixed_response='{"thought": "我先想法子挣开", "goals": ["设法脱身"]}')
    router = LLMRouter({scene: provider for scene in LLMScene})
    engine = NeedEngine(llm_router=router, seconds_per_step=3600)

    personality = _make_personality()
    personality.set_condition(BodyCondition("双手被反绑", since_step=0))
    await engine._generate_short_term_goals(  # noqa: SLF001
        dominant_need=NeedType.SAFETY,
        personality=personality,
        situation=Situation(
            location_view=LocationView(name="玄武门", description="宫城北门"), time_label="寅时",
        ),
        now_step=24,
    )
    prompt = "\n".join(m.content for m in provider.call_history[0])
    # Same label and same with-duration form as decision / interrupt / both emotion paths.
    assert "我此刻的处境：双手被反绑（已持续约1天）" in prompt
    # Right after the time/place anchor, before the persona.
    assert prompt.index("我此刻在玄武门") < prompt.index("我此刻的处境") < prompt.index("【我是谁】")
    # Narrative layer: no step counts.
    assert "24" not in prompt


@pytest.mark.asyncio
async def test_short_term_goal_prompt_omits_condition_when_unencumbered() -> None:
    """No predicament is the norm: the whole line is omitted rather than adding noise for everyone
    every beat."""
    from core.interfaces.llm import LLMRouter, LLMScene
    from providers.llm.mock import MockLLMProvider

    provider = MockLLMProvider(fixed_response='{"thought": "…", "goals": []}')
    router = LLMRouter({scene: provider for scene in LLMScene})
    engine = NeedEngine(llm_router=router, seconds_per_step=3600)

    await engine._generate_short_term_goals(  # noqa: SLF001
        dominant_need=NeedType.SAFETY,
        personality=_make_personality(),
    )
    prompt = "\n".join(m.content for m in provider.call_history[0])
    assert "我此刻的处境" not in prompt


# ---------------------------------------------------------------------------
# A goal's appointed time: letting the agent see that "the time has come"
# ---------------------------------------------------------------------------


def _dated_goal(text: str, *, due: int | None, created: int = 0, gid: str = "g") -> GoalEntity:
    return GoalEntity(
        id=gid, text=text, goal_type="short_term", created_step=created, due_step=due,
    )


def test_goal_due_hint_renders_the_four_positions_relative_to_now() -> None:
    """No deadline means no parentheses; the other three tiers each read differently."""
    from agent.goals import goal_due_hint

    sps = 3600
    assert goal_due_hint(_dated_goal("x", due=None), 10, sps) == ""
    assert goal_due_hint(_dated_goal("x", due=14), 10, sps) == "（约定在约4小时之后）"
    assert goal_due_hint(_dated_goal("x", due=10), 10, sps) == "（约定就在此刻）"
    assert goal_due_hint(_dated_goal("x", due=4), 10, sps) == "（约定时刻已过约6小时）"


def test_goal_due_hint_never_leaks_a_step_count_or_doubles_the_approx_prefix() -> None:
    """step is a code-layer coordinate and stays out of prompt text; describe_duration already
    includes "约", so don't add another."""
    from agent.goals import goal_due_hint

    text = goal_due_hint(_dated_goal("x", due=47), 5, 3600)
    assert "步" not in text and "47" not in text
    assert "约约" not in text


def test_dated_goals_sort_ahead_of_undated_ones_by_how_soon() -> None:
    """Goals with a set time go first (most imminent first); undated ones follow in their original
    queue order.

    A due goal mixed into a queue ordered by creation would be read as the oldest under "later is
    newer"; the ordering would bury the thing most needing doing.
    """
    from agent.goals import order_goals_for_prompt

    a = _dated_goal("无期限甲", due=None, gid="a")
    b = _dated_goal("三日后", due=30, gid="b")
    c = _dated_goal("无期限乙", due=None, gid="c")
    d = _dated_goal("就在此刻", due=10, gid="d")

    assert [g.id for g in order_goals_for_prompt([a, b, c, d])] == ["d", "b", "a", "c"]


def test_parse_goal_item_reads_hours_and_treats_anything_unusable_as_no_deadline() -> None:
    """The upward unit is hours, not steps: step is an engine coordinate, and the narrative layer
    has only points in time and durations.

    No deadline is the normal answer, not a parse failure: never drop the goal because of it.
    """
    from agent.goals import parse_goal_item

    assert parse_goal_item({"text": "去东宫", "due_in_hours": 10}) == ("去东宫", 10)
    assert parse_goal_item({"text": "去东宫", "due_in_hours": 9.5}) == ("去东宫", 9)
    for bad in (None, "十小时", 0, -2):
        assert parse_goal_item({"text": "去东宫", "due_in_hours": bad}) == ("去东宫", None)
    assert parse_goal_item({"text": "去东宫"}) == ("去东宫", None)
    assert parse_goal_item("去东宫") == ("去东宫", None)          # a bare string is accepted too
    assert parse_goal_item({"text": "  "}) is None
    assert parse_goal_item({"due_in_hours": 10}) is None


def test_enqueued_goal_turns_the_relative_due_into_an_absolute_step() -> None:
    """The LLM only outputs "how many steps remain" (a controlled upward channel); code computes
    the absolute time."""
    from agent.goals import enqueue_goals

    out = enqueue_goals([], [("明晨面谈", 5), ("随手打听", None)],
                         dominant_need=None, current_step=12)

    assert [(g.text, g.due_step) for g in out] == [("明晨面谈", 17), ("随手打听", None)]


def test_decision_goal_list_shows_the_due_hint_and_leads_with_it() -> None:
    """In the decision list, being due must be an already-rendered fact, not something the agent
    has to infer."""
    engine = NeedEngine(seconds_per_step=3600)
    goals = [
        _dated_goal("从太子口中问出实情", due=None, gid="s1"),
        _dated_goal("到玄武门依约行事", due=20, gid="s2"),
    ]

    text = engine.to_prompt_context(
        _make_personality(), NeedType.SAFETY, [], goals, now_step=20,
    )

    # Goals whose time has come are pulled into their own block before the list and no longer
    # appear in it. Buried in the list it's just a parenthetical at the end of the first line,
    # looking the same as "（已历约3小时仍未了结）".
    assert "截止时间已经到的事" in text
    assert "  - 到玄武门依约行事（约定就在此刻）" in text
    assert "1. 从太子口中问出实情" in text
    assert text.index("截止时间已经到的事") < text.index("我的短期目标")
    assert text.count("到玄武门依约行事") == 1      # must not appear in both places


def test_goal_due_step_survives_a_persist_restore_round_trip() -> None:
    """Without it, every appointment loses its time after restore and the agent is time-blind
    again."""
    from world.initializer import _goal_entity_from_dict

    stored = {
        "id": "stg-12-1", "text": "明晨面谈", "goal_type": "short_term",
        "status": "active", "related_need": None,
        "created_step": 12, "due_step": 17,
        "last_evaluated_step": None, "progress_summary": "", "origin": "cognitive",
    }

    assert _goal_entity_from_dict(stored, fallback_id="x").due_step == 17
    assert _goal_entity_from_dict({**stored, "due_step": None}, fallback_id="x").due_step is None
    assert _goal_entity_from_dict({k: v for k, v in stored.items() if k != "due_step"},
                                  fallback_id="x").due_step is None


def test_both_goal_prompts_show_a_deadline_and_a_null_side_by_side() -> None:
    """The schema example must show both kinds of value: showing only one anchors the answer to
    that kind.

    With a null-only example the model almost always answers null and misses real appointments.
    """
    from agent.need import _GOAL_EVALUATION_SYSTEM, _SHORT_TERM_GOAL_SYSTEM

    for prompt in (_SHORT_TERM_GOAL_SYSTEM, _GOAL_EVALUATION_SYSTEM):
        assert '"due_in_hours": null' in prompt
        assert re.search(r'"due_in_hours":\s*\d', prompt)
        # Hours, not steps: engine coordinates don't go upward.
        assert "due_in_steps" not in prompt


def test_the_judge_never_has_to_know_how_long_a_step_is() -> None:
    """The LLM answers in hours, so the judge prompt has no step-length slot.

    Don't add one: an unread slot invites someone to put engine units into the prompt.
    """
    from agent.need import _GOAL_EVALUATION_SYSTEM

    assert "step_duration" not in _GOAL_EVALUATION_SYSTEM
    assert "每一步" not in _GOAL_EVALUATION_SYSTEM


def test_a_goal_without_a_deadline_renders_no_parenthetical_at_all() -> None:
    """No deadline means writing nothing, not an empty shell like "(TBD)".

    Most goals have no deadline; hanging an empty marker on each just piles noise into the list and
    drowns the one that is actually due. 0 / null / invalid values / missing field all take this
    path.
    """
    from agent.goals import enqueue_goals, goal_due_hint, parse_goal_item

    for raw in ({"text": "甲", "due_in_hours": 0}, {"text": "甲", "due_in_hours": None},
                {"text": "甲", "due_in_hours": "十小时"}, {"text": "甲", "due_in_hours": -3},
                {"text": "甲"}, "甲"):
        goal = enqueue_goals([], [parse_goal_item(raw)],
                              dominant_need=None, current_step=10)[0]
        assert goal.due_step is None
        assert goal_due_hint(goal, 10, 3600) == ""

    engine = NeedEngine(seconds_per_step=3600)
    text = engine.to_prompt_context(
        _make_personality(), NeedType.SAFETY, [],
        [_dated_goal("联络上房玄龄", due=None, gid="s1")], now_step=10,
    )
    assert "联络上房玄龄" in text
    assert "约定" not in text                      # the whole line carries no due wording


def test_hours_convert_to_steps_and_round_early_rather_than_late() -> None:
    """Converting hours → steps belongs to code (the inverse of describe_duration).

    The rounding direction is deliberate: better early than late. Late makes it look like there's
    slack, which is exactly what this field must prevent. So round down; don't use round(), which
    rounds .5 to even, so 3.5→4 lands on the late side. Under one step still counts as one step:
    an appointment half a step away is "due next step", not "due now".
    """
    from agent.goals import _due_step_from_hours

    assert _due_step_from_hours(10, 10, 3600) == 20        # one hour per step
    assert _due_step_from_hours(10, 10, 7200) == 15        # two hours per step
    assert _due_step_from_hours(10, 10, 4 * 3600) == 12    # 2.5 steps → the early side
    assert _due_step_from_hours(10, 14, 4 * 3600) == 13    # 3.5 steps → round() would go late
    assert _due_step_from_hours(10, 1, 4 * 3600) == 11     # under one step still counts as one step
    for none_ish in (None, 0, -5):
        assert _due_step_from_hours(10, none_ish, 3600) is None


@pytest.mark.asyncio
async def test_short_term_generation_carries_vitality_when_it_is_failing() -> None:
    """Someone near collapse should set "catch my breath", not "rush to the Eastern Palace
    overnight"; goals are enqueued and read by every later beat."""
    from core.interfaces.llm import LLMResponse

    captured: dict = {}

    class _CaptureLLM:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000, **kwargs):
            captured["user"] = messages[-1].content
            return LLMResponse(
                content='{"thought": "我", "goals": ["找个地方歇口气"]}',
                input_tokens=0, output_tokens=0, model="test",
            )

    personality = _make_personality(active=[
        NeedState(type=NeedType.SAFETY, label="减少不确定性", intensity=0.9, weight=1.0),
    ])
    personality.apply_vitality_damage(0.8)
    engine = NeedEngine(llm_router=_CaptureLLM())  # type: ignore[arg-type]
    await engine.run(
        current_step=3, personality=personality,
        visible_agents=[], pending_messages=0,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
    )

    assert "我此刻的体力：将竭" in captured["user"]


def test_the_goal_text_field_itself_asks_for_a_concrete_date() -> None:
    """A block of time rules isn't enough: when the model writes text, only the field description
    is in front of it.

    Goal text is reread every step for many steps, so a bare "明日" enqueued there gets faithfully
    copied downstream; the field description must mention it too.
    """
    from agent.need import _GOAL_EVALUATION_SYSTEM, _SHORT_TERM_GOAL_SYSTEM

    for prompt in (_SHORT_TERM_GOAL_SYSTEM, _GOAL_EVALUATION_SYSTEM):
        flat = re.sub(r"\s+", "", prompt)
        assert "写具体时间点，不写「明日/今夜」" in flat
    # And the sentence must sit right on the text field itself, not in a separate general paragraph.
    flat = re.sub(r"\s+", "", _SHORT_TERM_GOAL_SYSTEM)
    assert "每条不超过32字；里头若提到时间" in flat


def test_completed_goals_carry_when_they_were_finished() -> None:
    """Resolved goals are things that happened in the past, so they get a time label from the
    memory system's relative-recency words.

    This section takes every completed item in the lookback window (trimmed by count, not by
    duration), so one can stay there for many steps; without a time label, something finished a
    day ago would be called "刚刚".
    """
    from core.prompts import MEMORY_ORDER_HINT

    # 1 hour per step, starting at 8am. now=30 (2pm the next day): the one completed at step 2
    # crossed a day boundary; the one at step 28 was only two hours ago.
    engine = NeedEngine(seconds_per_step=3600, world_start_second_of_day=8 * 3600)
    yesterday = _goal("请动裴寂出面", created_step=0, status=GoalStatus.COMPLETED)
    yesterday.last_evaluated_step = 2
    recent = _goal("备下六月初四的马", created_step=20, status=GoalStatus.COMPLETED)
    recent.last_evaluated_step = 28

    text = engine.to_prompt_context(
        _make_personality(), NeedType.SAFETY, [], [],
        recently_completed=[recent, yesterday], now_step=30,
    )

    assert f"我近期已经完成的事项（{MEMORY_ORDER_HINT}）：" in text
    assert "刚刚完成" not in text
    # Later is newer: the one across the day boundary comes first, then the two-hours-ago one.
    assert text.index("（昨日）请动裴寂出面") < text.index("（几小时前）备下六月初四的马")


def test_a_finished_goal_and_a_memory_of_the_same_moment_read_the_same() -> None:
    """Something that happened at one moment must be phrased the same in the goal section and the
    memory section; otherwise the LLM can't place them on one timeline."""
    from core.prompts import render_memory

    class _Mem:
        kind = "event"
        stored_content = "内容"
        created_step = 4

    engine = NeedEngine(seconds_per_step=3600, world_start_second_of_day=8 * 3600)
    goal = _goal("同一刻了结的事", created_step=0, status=GoalStatus.COMPLETED)
    goal.last_evaluated_step = 4

    text = engine.to_prompt_context(
        _make_personality(), NeedType.SAFETY, [], [],
        recently_completed=[goal], now_step=30,
    )
    prefix = render_memory(
        _Mem(), now_step=30, seconds_per_step=3600, world_start_second_of_day=8 * 3600,
    ).removesuffix("内容")

    assert f"  - {prefix}同一刻了结的事" in text


async def test_the_recent_history_block_says_when_each_thing_ended() -> None:
    """Saying "I've already done it (no need to repeat)" weighs differently for something done
    yesterday vs. just now; without a time label they can't be told apart.

    Time labels use the same vocabulary as the decision side and memory; with more than five, keep
    the five most recent and list them in order.
    """
    from core.interfaces.llm import LLMResponse

    captured: dict = {}

    class _CaptureLLM:
        async def complete(self, scene, messages, temperature=0.7, max_tokens=1000, **kwargs):
            captured["user"] = messages[-1].content
            return LLMResponse(content='{"thought": "我", "goals": []}',
                               input_tokens=0, output_tokens=0, model="test")

    done = _goal("请动裴寂出面", created_step=0, status=GoalStatus.COMPLETED)
    done.last_evaluated_step = 2
    flopped = _goal("取走那封密信", created_step=10, status=GoalStatus.FAILED)
    flopped.last_evaluated_step = 28
    personality = _make_personality(
        active=[NeedState(type=NeedType.SAFETY, label="减少不确定性", intensity=0.9, weight=1.0)],
        short_term_entities=[done, flopped],
    )
    # 1 hour per step, starting at 8am: the one resolved at step 2 crossed a day boundary; the one
    # at step 28 was only two hours ago.
    engine = NeedEngine(llm_router=_CaptureLLM(), seconds_per_step=3600,  # type: ignore[arg-type]
                        world_start_second_of_day=8 * 3600)
    await engine.run(
        current_step=30, personality=personality,
        visible_agents=[], pending_messages=0,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
    )

    assert "  - （昨日）请动裴寂出面" in captured["user"]
    assert "  - （几小时前）取走那封密信" in captured["user"]
