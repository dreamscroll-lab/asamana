"""Strict validation: a value inside its rule passes untouched, anything else raises naming the
field, NaN included."""

from __future__ import annotations

import math

import pytest

from core.validate import require_at_least, require_in_range, require_member, require_positive, require_text


def test_values_inside_their_rule_pass() -> None:
    require_in_range(0.0, (0.0, 1.0), "trust")
    require_in_range(1.0, (0.0, 1.0), "trust")
    require_at_least(1, 1, "steps")
    require_positive(0.5, "cooldown")
    require_text(" 有话 ", "goal")
    require_member("file", ("stdout", "file"), "output")


@pytest.mark.parametrize("check", [
    lambda: require_in_range(1.5, (0.0, 1.0), "trust"),
    lambda: require_in_range(math.nan, (0.0, 1.0), "trust"),
    lambda: require_at_least(0, 1, "trust"),
    lambda: require_at_least(math.nan, 0, "trust"),
    lambda: require_positive(0, "trust"),
    lambda: require_positive(math.nan, "trust"),
    lambda: require_text("  ", "trust"),
    lambda: require_member("syslog", ("stdout", "file"), "trust"),
])
def test_anything_else_raises_naming_the_field(check) -> None:
    with pytest.raises(ValueError, match="trust"):
        check()
