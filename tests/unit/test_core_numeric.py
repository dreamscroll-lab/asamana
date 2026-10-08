"""Unit tests for the shared numeric primitive."""

from __future__ import annotations

from core.numeric import clamp


def test_clamp_within_range_passes_through() -> None:
    assert clamp(0.5, 0.0, 1.0) == 0.5


def test_clamp_below_low_returns_low() -> None:
    assert clamp(-2.0, 0.0, 1.0) == 0.0
    assert clamp(-2.0, -1.0, 1.0) == -1.0


def test_clamp_above_high_returns_high() -> None:
    assert clamp(9.0, 0.0, 1.0) == 1.0


def test_clamp_degenerate_range() -> None:
    assert clamp(5.0, 0.3, 0.3) == 0.3
