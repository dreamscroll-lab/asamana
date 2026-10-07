"""Unit tests for external drive evaluation and motivation blending."""

from __future__ import annotations



from agent.motivation import ExternalDriveType, ExternalGoal, MotivationBlender
from agent.need import NeedEvaluation, NeedState, NeedType, LLM_RELEVANCE_WEIGHT
from core.interfaces.urgency import Urgency


def _goal(urgency: Urgency, need: NeedType | None) -> ExternalGoal:
    return ExternalGoal(text="x", source_id="s", urgency=urgency,
                        drive_type=ExternalDriveType.THREAT, related_need=need)


# urgency ordinal mapped linearly to [0,1] x WEIGHT (anchored to the same additive scale as appraisal)
def _expected(urgency: Urgency) -> float:
    span = Urgency.CRITICAL.level - Urgency.LOW.level
    return LLM_RELEVANCE_WEIGHT * (urgency.level - Urgency.LOW.level) / span


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bare_need_evaluation(dominant: NeedType | None = NeedType.SOCIAL) -> NeedEvaluation:
    return NeedEvaluation(
        dominant_need=dominant,
        scores={dominant: 0.6} if dominant else {},
        active_needs=[NeedState(type=dominant, label="social", intensity=0.6)] if dominant else [],
        short_term_goals=[],
        long_term_goals=[],
        prompt_context="",
    )


# ---------------------------------------------------------------------------
# MotivationBlender.external_pressure_boost tests
#
# The boost RULE is tested here in isolation (urgency→additive magnitude, per related_need,
# summed). Its wiring through NeedEngine.run (boost folded into scores → dominant=argmax) lives
# in test_need_engine.py, the sole caller.
# ---------------------------------------------------------------------------

def test_boost_empty_external_goals() -> None:
    assert MotivationBlender().external_pressure_boost([]) == {}


def test_boost_scales_with_urgency_anchored_to_weight() -> None:
    """urgency ordinal mapped linearly to [0,1] x WEIGHT: LOW=0 / NORMAL~0.67 / HIGH~1.33 / CRITICAL=2.0."""
    b = MotivationBlender()
    assert b.external_pressure_boost([_goal(Urgency.LOW, NeedType.SAFETY)]) == {NeedType.SAFETY: 0.0}
    assert b.external_pressure_boost([_goal(Urgency.CRITICAL, NeedType.SAFETY)]) == {
        NeedType.SAFETY: LLM_RELEVANCE_WEIGHT}  # CRITICAL → full WEIGHT (2.0)
    assert abs(b.external_pressure_boost([_goal(Urgency.HIGH, NeedType.ESTEEM)])[NeedType.ESTEEM]
               - _expected(Urgency.HIGH)) < 1e-9


def test_boost_skips_goals_without_related_need() -> None:
    assert MotivationBlender().external_pressure_boost([_goal(Urgency.HIGH, None)]) == {}


def test_boost_sums_per_related_need_across_goals() -> None:
    b = MotivationBlender()
    boosts = b.external_pressure_boost([
        _goal(Urgency.HIGH, NeedType.SAFETY),
        _goal(Urgency.NORMAL, NeedType.SAFETY),   # same need → summed
        _goal(Urgency.CRITICAL, NeedType.ESTEEM),
    ])
    assert abs(boosts[NeedType.SAFETY] - (_expected(Urgency.HIGH) + _expected(Urgency.NORMAL))) < 1e-9
    assert abs(boosts[NeedType.ESTEEM] - _expected(Urgency.CRITICAL)) < 1e-9


# ---------------------------------------------------------------------------
# NeedEvaluation field presence
# ---------------------------------------------------------------------------

def test_need_evaluation_has_external_goals_field() -> None:
    ne = _bare_need_evaluation()
    assert hasattr(ne, "external_goals")
    assert ne.external_goals == []
