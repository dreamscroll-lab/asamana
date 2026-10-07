"""Unit tests for the shared duration-rendering primitive."""

from __future__ import annotations

from core.duration import describe_duration, describe_seconds


def test_describe_duration_unit_thresholds() -> None:
    assert describe_duration(1, 30) == "约30秒"
    assert describe_duration(1, 300) == "约5分钟"
    assert describe_duration(1, 7200) == "约2小时"
    assert describe_duration(1, 172800) == "约2天"


def test_describe_duration_scales_with_steps() -> None:
    assert describe_duration(3, 3600) == "约3小时"


def test_describe_duration_clamps_negatives_to_zero() -> None:
    assert describe_duration(-5, 3600) == "约0秒"
    assert describe_duration(5, -3600) == "约0秒"


def test_describe_seconds_keeps_the_minutes_of_larger_units() -> None:
    # A 1h59m walk must not read as "约1小时": the remainder down to the minute is kept.
    assert describe_seconds(5616) == "约1小时33分钟"
    assert describe_seconds(7199) == "约1小时59分钟"
    assert describe_seconds(100800) == "约1天4小时"
    assert describe_seconds(90061) == "约1天1小时1分钟"
