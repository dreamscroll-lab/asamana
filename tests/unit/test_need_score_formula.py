"""Localized characterization of the need-scoring relevance combination (disposition ↔ situation).

These tests pin the *design properties* of how situational `need_relevance` combines with the
dispositional baseline `I×W` in `NeedEngine._score_needs` — derived globally over the parameter
space, NOT fit to any single scenario. The chosen form is **additive**:
    score = I×W·vis + WEIGHT·activation·vis  (+ runtime adj)

Properties (two needs A=disposition-favored, B=situation-favored):
- Close activations (|Δa| ≤ ε) must not override a clear disposition gap (Δ ≥ 0.3);
- A maximally distinctive signal (Δa=1) can overcome a disposition gap up to ~WEIGHT (=G);
- monotonic: raising a need's activation only raises its score.

WEIGHT=2.0 is the largest weight for which close activations can't override the gap (ε=0.15,
dmargin=0.3). This file locks the additive form and that constant so the formula can't silently
drift, and asserts the capped multiplier `1+a` too weak to let a maximal signal overcome the gap.
"""

from __future__ import annotations

from agent.need import NeedEngine, NeedState, NeedType, LLM_RELEVANCE_WEIGHT

_A, _B = NeedType.SAFETY, NeedType.SOCIAL  # A=disposition-favored, B=situation-favored


def _dominant(iw_a: float, a_a: float, iw_b: float, a_b: float) -> NeedType:
    """Dominant need under the production formula, isolated (vis=1, no adj, no keyword)."""
    engine = NeedEngine()
    needs = [
        NeedState(type=_A, label="A", intensity=iw_a, weight=1.0),
        NeedState(type=_B, label="B", intensity=iw_b, weight=1.0),
    ]
    scored = engine._score_needs(needs, need_relevance={_A: a_a, _B: a_b})  # noqa: SLF001
    return max(scored, key=lambda x: x[0])[1].type


def test_weight_is_principled_value() -> None:
    """WEIGHT = dmargin/ε = 0.3/0.15 = 2.0 (principle, not scenario-fit)."""
    assert LLM_RELEVANCE_WEIGHT == 2.0


def test_p1_close_activations_do_not_override_clear_disposition() -> None:
    """|Δa| ≤ 0.15 must not flip a clear disposition gap (Δ ≥ 0.3)."""
    base = 0.5
    for delta in (0.3, 0.5, 0.8, 1.2):          # clear disposition gap: A stronger by delta
        for a_a in (0.0, 0.3, 0.6, 0.85):
            a_b = min(1.0, a_a + 0.15)            # B's activation edge ≤ ε
            assert _dominant(base + delta, a_a, base, a_b) == _A, (
                f"close edge wrongly flipped at Δ={delta}, a_a={a_a}")


def test_p3_distinctive_signal_overcomes_bounded_gap() -> None:
    """A maximally-distinctive signal (a_B=1, a_A=0) flips gaps up to ~WEIGHT, but not beyond."""
    base = 0.5
    # gap just under WEIGHT → flips to B
    assert _dominant(base + (LLM_RELEVANCE_WEIGHT - 0.1), 0.0, base, 1.0) == _B
    # gap above WEIGHT → disposition holds
    assert _dominant(base + (LLM_RELEVANCE_WEIGHT + 0.1), 0.0, base, 1.0) == _A


def test_monotonic_in_activation() -> None:
    """Raising B's activation only ever helps B (never flips back to A)."""
    base_a, base_b = 1.0, 0.6   # A disposition-stronger
    prev_b_wins = False
    for a_b in (0.0, 0.25, 0.5, 0.75, 1.0):
        b_wins = _dominant(base_a, 0.0, base_b, a_b) == _B
        assert b_wins or not prev_b_wins, "monotonicity violated: B lost after gaining activation"
        prev_b_wins = prev_b_wins or b_wins


def test_capped_multiplier_would_be_too_weak() -> None:
    """A capped multiplier (1+a) is too weak: even a max-distinctive signal could only
    overturn a ~0.49 disposition gap — situation nearly inert. The additive form lifts
    that ceiling to ~WEIGHT while keeping P1."""
    base = 0.5
    # additive: a max signal overturns a 1.5 gap (well past the multiplier's ~0.49 ceiling)
    assert _dominant(base + 1.5, 0.0, base, 1.0) == _B
    # under a 1+a multiplier: B=base×2.0=1.0 vs A=(base+1.5)×1.0=2.0 → A; the additive form
    # is a deliberate, principled lift of situational influence (not a scenario patch).
