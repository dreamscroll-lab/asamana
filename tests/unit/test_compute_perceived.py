"""Unit tests for the perceived-relation distortion formula.

`_compute_perceived` bends an objective trust/affection baseline by the agent's current
emotion and recent-memory bias:

    perceived = clamp(objective + memory_bias·K_MEMORY_PULL + intensity·valence·K_EMOTION_PULL)

These tests pin two things:
  1. **Structural properties** the formula must satisfy for any reasonable constants
     (rest identity, continuity, monotonicity, clamping, sign correctness).
  2. **Scale-anchored behavior bands** — concrete (objective, emotion, memory) inputs whose
     expected perceived value is fixed by the trust/affection scale semantics and the design
     intent that emotion *bends but cannot flip* a settled baseline. The constants are fit
     against this table; the table is the contract.

Pure-function calls — no store, no fixtures.
"""

from __future__ import annotations

from agent.personality import EmotionState, EmotionType
from agent.relation import (
    AFFECTION_RANGE,
    K_EMOTION_PULL,
    K_MEMORY_PULL,
    TRUST_RANGE,
    _compute_perceived,
)


def _emotion(intensity: float, valence: float) -> EmotionState:
    return EmotionState(primary=EmotionType.NEUTRAL, intensity=intensity, valence=valence)


def _trust(objective: float, *, intensity: float = 0.2, valence: float = 0.0, memory: float = 0.0) -> float:
    return _compute_perceived(
        objective=objective,
        recent_memory_bias=memory,
        emotion=_emotion(intensity, valence),
        value_range=TRUST_RANGE,
    )


def _affection(objective: float, *, intensity: float = 0.2, valence: float = 0.0, memory: float = 0.0) -> float:
    return _compute_perceived(
        objective=objective,
        recent_memory_bias=memory,
        emotion=_emotion(intensity, valence),
        value_range=AFFECTION_RANGE,
    )


# ---------------------------------------------------------------------------
# Structural properties (must hold for any sane constants)
# ---------------------------------------------------------------------------


def test_rest_identity_neutral_emotion_no_memory_leaves_baseline_untouched() -> None:
    """A truly neutral emotion (valence 0) with no memory bias must return the baseline exactly.
    """
    for objective in (0.0, 0.3, 0.5, 0.7, 1.0):
        assert _trust(objective, intensity=0.2, valence=0.0, memory=0.0) == objective
    for objective in (-1.0, -0.3, 0.0, 0.3, 1.0):
        assert _affection(objective, intensity=0.2, valence=0.0, memory=0.0) == objective


def test_negative_emotion_lowers_positive_emotion_raises() -> None:
    base = 0.5
    assert _trust(base, intensity=0.8, valence=-0.7) < base
    assert _trust(base, intensity=0.8, valence=0.7) > base


def test_memory_bias_sign_is_respected() -> None:
    base = 0.5
    assert _trust(base, memory=-0.8) < base
    assert _trust(base, memory=0.8) > base


def test_monotonic_in_emotion_intensity() -> None:
    base = 0.6
    mild = _trust(base, intensity=0.3, valence=-0.6)
    strong = _trust(base, intensity=0.9, valence=-0.6)
    assert strong < mild < base  # stronger negative emotion pulls trust further down


def test_valence_magnitude_matters_not_just_sign() -> None:
    """Mild displeasure must distort less than rage (a sign-only formula could not express this)."""
    base = 0.7
    mild = _trust(base, intensity=0.6, valence=-0.2)
    harsh = _trust(base, intensity=0.6, valence=-0.9)
    assert harsh < mild < base


def test_continuous_through_zero_valence() -> None:
    """No jump as valence crosses 0 (a sign-flip term would be discontinuous at 0)."""
    base = 0.5
    just_below = _trust(base, intensity=0.6, valence=-0.001)
    at_zero = _trust(base, intensity=0.6, valence=0.0)
    just_above = _trust(base, intensity=0.6, valence=0.001)
    assert abs(just_below - at_zero) < 0.01
    assert abs(just_above - at_zero) < 0.01


def test_clamped_to_range_at_both_bounds() -> None:
    # Trust cannot exceed 1.0 even under maximal positive pull.
    assert _trust(0.95, intensity=1.0, valence=1.0, memory=1.0) == 1.0
    # Trust cannot drop below 0.0.
    assert _trust(0.05, intensity=1.0, valence=-1.0, memory=-1.0) == 0.0
    # Affection clamps to [-1, 1].
    assert _affection(0.95, intensity=1.0, valence=1.0, memory=1.0) == 1.0
    assert _affection(-0.95, intensity=1.0, valence=-1.0, memory=-1.0) == -1.0


def test_emotion_pull_bounded_to_one_trust_band() -> None:
    """Design invariant: a passing mood bends but cannot flip a settled bond across a full band.

    Max emotion displacement (intensity=1, |valence|=1) must not exceed one trust band (0.20):
    a "信任" (0.7) baseline under maximal anger stays at or above "中立" (0.5).
    """
    assert K_EMOTION_PULL <= 0.20 + 1e-9
    assert _trust(0.70, intensity=1.0, valence=-1.0) >= 0.50 - 1e-9


# ---------------------------------------------------------------------------
# Scale-anchored behavior bands (the constants were fit to these)
# ---------------------------------------------------------------------------


def test_strong_anger_on_trusted_ally_drops_toward_neutral_but_not_below() -> None:
    # "信任" (0.70) ally, strong anger → bends toward "警惕" but stays ≥ "中立".
    perceived = _trust(0.70, intensity=0.8, valence=-0.7)
    assert 0.50 <= perceived <= 0.605


def test_mild_irritation_barely_moves_a_trusted_bond() -> None:
    perceived = _trust(0.70, intensity=0.3, valence=-0.3)
    assert 0.64 <= perceived <= 0.695


def test_warm_joy_gives_a_modest_lift() -> None:
    perceived = _trust(0.50, intensity=0.6, valence=0.5)
    assert 0.54 <= perceived <= 0.62


def test_recent_betrayal_memory_pulls_trust_down_while_calm() -> None:
    # Calm now (neutral emotion) but recent painful memory of this person.
    perceived = _trust(0.60, intensity=0.2, valence=0.0, memory=-0.8)
    assert 0.30 <= perceived <= 0.42


def test_affection_hatred_flash_stays_within_range() -> None:
    perceived = _affection(0.30, intensity=0.9, valence=-0.8)
    assert -0.10 <= perceived <= 0.20


def test_constants_match_fitted_values() -> None:
    """Guard the derived constants — changing them is a deliberate re-derivation, not a tweak."""
    assert K_EMOTION_PULL == 0.20
    assert K_MEMORY_PULL == 0.30
