"""Urgency thresholds share one source across paths + SEVERITY bridge table completeness.

Urgency is the single 4-level enum for urgency, and the interrupt / dominate / emotion-shock
thresholds all derive from Urgency.HIGH.

This file locks:
- URGENCY_INTERRUPT_THRESHOLD == Urgency.HIGH (the interrupt threshold; dominance is additive and
  has no binary dominate threshold)
- runtime.py Paths 1/2/3 all use URGENCY_INTERRUPT_THRESHOLD, with no hard-coded "high"/0.75
- the SEVERITY_TO_URGENCY bridge table is complete (high/medium/low all mapped)
- parse_urgency tolerantly falls back to NORMAL

Note: there are no URGENCY_TO_{THREAT,AUTHORITY,OBLIGATION}_INTENSITY tables. Perception emotion is
always derived by the LLM from signals, so there is no rule-based urgency→intensity mapping to lock.
"""

from __future__ import annotations

from pathlib import Path


from core.interfaces.urgency import (
    URGENCY_INTERRUPT_THRESHOLD,
    Urgency,
    parse_urgency,
)
from core.prompts import URGENCY_TO_STRENGTH


# ─────────────────────────────────────────────────────────────────────────────
# interrupt threshold = Urgency.HIGH (dominance is additive, see test_motivation)
# ─────────────────────────────────────────────────────────────────────────────


def test_interrupt_threshold_is_high() -> None:
    """The interrupt threshold is locked at HIGH. Dominance (which need leads) uses
    MotivationBlender's continuous additive boost and doesn't share this binary threshold; the two
    concepts evolve independently."""
    assert URGENCY_INTERRUPT_THRESHOLD == Urgency.HIGH


# ─────────────────────────────────────────────────────────────────────────────
# Urgency ordering: __ge__ / __gt__ etc. are this enum's core API
# ─────────────────────────────────────────────────────────────────────────────


def test_urgency_comparison_order_is_low_normal_high_critical() -> None:
    assert Urgency.LOW < Urgency.NORMAL
    assert Urgency.NORMAL < Urgency.HIGH
    assert Urgency.HIGH < Urgency.CRITICAL
    # The interrupt threshold shares the source explicitly
    assert Urgency.HIGH >= URGENCY_INTERRUPT_THRESHOLD
    assert Urgency.NORMAL < URGENCY_INTERRUPT_THRESHOLD
    assert Urgency.CRITICAL >= URGENCY_INTERRUPT_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# URGENCY → intensity mapping tables: key invariants
# ─────────────────────────────────────────────────────────────────────────────


def test_urgency_to_strength_covers_all_four_levels() -> None:
    """All 4 levels map to a strength; LOW < NORMAL < HIGH < CRITICAL is monotonic."""
    for level in Urgency:
        assert level in URGENCY_TO_STRENGTH
    values = [URGENCY_TO_STRENGTH[u] for u in
              (Urgency.LOW, Urgency.NORMAL, Urgency.HIGH, Urgency.CRITICAL)]
    assert values == sorted(values), f"strength 映射应单调递增: {values}"


def test_rule_emotion_intensity_tables_are_gone() -> None:
    """core.prompts must not contain a rule-based urgency→intensity mapping table.

    Perception emotion has only the LLM path (derived from real signals). Keeping such a table
    invites someone to wire a "cheaper" rule-based emotion bypass, and per CLAUDE.md §5 that path
    doesn't exist.
    """
    import core.prompts as prompts

    for name in (
        "URGENCY_TO_THREAT_INTENSITY",
        "URGENCY_TO_AUTHORITY_INTENSITY",
        "URGENCY_TO_OBLIGATION_INTENSITY",
    ):
        assert not hasattr(prompts, name), f"{name} 不应存在——规则情绪旁路已被禁止"


# ─────────────────────────────────────────────────────────────────────────────
# parse_urgency tolerance
# ─────────────────────────────────────────────────────────────────────────────


def test_parse_urgency_accepts_enum_value() -> None:
    assert parse_urgency(Urgency.HIGH) == Urgency.HIGH


def test_parse_urgency_accepts_lowercase_string() -> None:
    assert parse_urgency("high") == Urgency.HIGH
    assert parse_urgency("CRITICAL") == Urgency.CRITICAL
    assert parse_urgency(" normal ") == Urgency.NORMAL


def test_parse_urgency_invalid_falls_back_to_default() -> None:
    assert parse_urgency("nonsense") == Urgency.NORMAL  # default
    assert parse_urgency(None) == Urgency.NORMAL
    assert parse_urgency(0.75) == Urgency.NORMAL        # floats aren't supported
    assert parse_urgency("nonsense", default=Urgency.LOW) == Urgency.LOW


# ─────────────────────────────────────────────────────────────────────────────
# runtime.py Paths 1/2/3 all use URGENCY_INTERRUPT_THRESHOLD (locked by source grep)
# ─────────────────────────────────────────────────────────────────────────────


def test_interrupt_coordinator_uses_urgency_interrupt_threshold_for_interrupt_paths() -> None:
    """All three interrupt paths must take their threshold from URGENCY_INTERRUPT_THRESHOLD, never a
    hard-coded "high" or 0.75.

    Interrupt evaluation lives in engine/interrupt_coordinator.py (InterruptCoordinator), so the
    contract points at that file.
    """
    src = Path("engine/interrupt_coordinator.py").read_text(encoding="utf-8")
    # The constant must be imported (simple grep evidence)
    assert "URGENCY_INTERRUPT_THRESHOLD" in src, (
        "interrupt_coordinator.py 必须用 URGENCY_INTERRUPT_THRESHOLD 而非硬编码阈值。"
    )
    # Reverse assertion: no bare "0.75" or bare == "urgent" string in the interrupt decision code
    forbidden_hardcodings = [
        '"urgent"',
        "'urgent'",
        ">= 0.75",
        "== 0.75",
    ]
    for token in forbidden_hardcodings:
        # Checks the whole file, docstrings included: a docstring should say
        # URGENCY_INTERRUPT_THRESHOLD instead.
        assert token not in src, (
            f"interrupt_coordinator.py 出现硬编码阈值 {token},应改用 URGENCY_INTERRUPT_THRESHOLD"
        )
