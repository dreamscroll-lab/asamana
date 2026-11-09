"""Spans become whole steps rounded half up, at least one."""

import pytest

from core.action_duration import travel_duration_steps


@pytest.mark.parametrize(("seconds", "steps"), [(10800, 2), (18000, 3), (9000, 1), (600, 1)])
def test_a_walk_takes_its_steps_rounded_half_up(seconds: int, steps: int) -> None:
    # Two-hour steps: 2.5 steps is three, 1.5 is two (round() would give two for both).
    assert travel_duration_steps(seconds, 7200) == steps
