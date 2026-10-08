"""Multi-step characterization + property locks for need intensity homeostasis.

Pins the four dynamics constants (`_NEED_REST_BASELINE` B, `_NEED_HOMEOSTASIS_RATE` λ,
`_NEED_SUCCESS_RELIEF` k_s, `_NEED_FAILURE_PRESSURE` k_f) to interpretable design targets,
derived globally over a multi-step simulation — NOT fit to any scenario. Per action-feedback tick:
    I_next = clamp(I + shock + λ·(B − I), 0.05, 1.0)   (shock = −k_s success / +k_f fail / 0 idle)

Need intensities live in personality.state.need_intensities; ``NeedEngine.evolve_intensities`` is
the **pure** evolution function (dict → dict), so these property locks thread an intensities dict
through the steps.

Targets / properties:
- Idle: converges to B and stays.
- A saturated need de-saturates within ~T_desaturate steps.
- A fail shock still raises a need that has room (I < sustained-fail equilibrium).
- Sustained failure keeps the need clearly elevated (doesn't relax to B).
- No monotonic upward drift.
- Acting successfully satisfies fast (T_satisfy).
"""

from __future__ import annotations

from agent.need import (
    NeedEngine, NeedType,
    _NEED_FAILURE_PRESSURE, _NEED_HOMEOSTASIS_RATE, _NEED_REST_BASELINE, _NEED_SUCCESS_RELIEF,
)

B = _NEED_REST_BASELINE
SAFETY = NeedType.SAFETY.value
SOCIAL = NeedType.SOCIAL.value


def _drive(safety_intensity: float) -> dict[str, float]:
    return {SAFETY: safety_intensity, SOCIAL: B}


def test_constants_are_principled() -> None:
    assert (B, _NEED_HOMEOSTASIS_RATE, _NEED_SUCCESS_RELIEF, _NEED_FAILURE_PRESSURE) == (0.3, 0.25, 0.12, 0.08)


def test_h1_idle_converges_to_baseline() -> None:
    """No shock (dominant_need=None) → all needs relax toward B and settle there."""
    engine = NeedEngine()
    drive = _drive(1.0)
    for _ in range(20):
        drive = engine.evolve_intensities(drive, dominant_need=None, succeeded=True)
    assert abs(drive[SAFETY] - B) < 0.05
    # fixed point: a need already at B stays at B
    drive2 = engine.evolve_intensities(_drive(B), dominant_need=None, succeeded=True)
    assert abs(drive2[SAFETY] - B) < 1e-9


def test_h2_saturated_need_desaturates_within_window() -> None:
    """A need pinned at 1.0, never acted on, relaxes to ~B within T_desaturate ≈ 8–12 steps."""
    engine = NeedEngine()
    drive = _drive(1.0)
    n = None
    for step in range(1, 21):
        drive = engine.evolve_intensities(drive, dominant_need=NeedType.SOCIAL, succeeded=True)  # SAFETY non-dominant → idle relax
        if abs(drive[SAFETY] - B) <= 0.05:
            n = step
            break
    assert n is not None and 8 <= n <= 12, f"de-saturated in {n} steps (want 8–12)"


def test_h3_failure_raises_when_room() -> None:
    """One failure shock net-raises a need that still has room (below the sustained-fail equilibrium)."""
    engine = NeedEngine()
    drive = engine.evolve_intensities(_drive(0.5), dominant_need=NeedType.SAFETY, succeeded=False)
    assert drive[SAFETY] > 0.5


def test_h4_sustained_failure_stays_elevated() -> None:
    """Repeated failure keeps the need clearly elevated (≈ B + k_f/λ ≈ 0.62), not relaxed to B."""
    engine = NeedEngine()
    drive = _drive(B)
    for _ in range(30):
        drive = engine.evolve_intensities(drive, dominant_need=NeedType.SAFETY, succeeded=False)
    assert abs(drive[SAFETY] - (B + _NEED_FAILURE_PRESSURE / _NEED_HOMEOSTASIS_RATE)) < 0.02
    assert drive[SAFETY] > 0.55  # still "clearly care"


def test_t_satisfy_success_relieves_fast() -> None:
    """Acting on a saturated need successfully brings it to ~B within a few steps."""
    engine = NeedEngine()
    drive = _drive(1.0)
    n = None
    for step in range(1, 11):
        drive = engine.evolve_intensities(drive, dominant_need=NeedType.SAFETY, succeeded=True)
        if drive[SAFETY] <= B + 0.05:
            n = step
            break
    assert n is not None and n <= 6


def test_h5_no_monotonic_drift_for_ignored_need() -> None:
    """Ignored needs settle at B and never drift up."""
    engine = NeedEngine()
    drive = _drive(B)  # SAFETY ignored (SOCIAL dominant) — must not creep up
    intensities = []
    for _ in range(15):
        drive = engine.evolve_intensities(drive, dominant_need=NeedType.SOCIAL, succeeded=False)
        intensities.append(drive[SAFETY])
    assert max(intensities) <= B + 1e-9  # never rises above baseline
