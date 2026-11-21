"""Action scheduling durations shared by execution and actor summaries."""

import math

from core.interfaces.action import ActionType

_SECONDS_PER_HOUR = 3600


def action_duration_steps(action_type: ActionType, estimated_steps: int, *, live_talk: bool = False) -> int:
    if action_type is ActionType.MOVE:
        raise ValueError("Movement duration must be derived from travel time")
    if action_type in (ActionType.PHYSICAL, ActionType.SEND_MESSAGE, ActionType.ERRAND):
        return 1
    if action_type is ActionType.TALK:
        return 1 if live_talk else max(estimated_steps, 1)
    return estimated_steps


def _whole_steps(seconds: float, seconds_per_step: int) -> int:
    """A span as whole steps, rounded half up and at least one. Not round(): its banker's rounding
    makes 2.5 steps two and 1.5 steps two."""
    return max(1, math.floor(seconds / seconds_per_step + 0.5))


def travel_duration_steps(travel_seconds: int, seconds_per_step: int) -> int:
    return _whole_steps(travel_seconds, seconds_per_step)


def estimated_hours_to_steps(hours: object, seconds_per_step: int) -> int:
    """The steps a model's estimated duration in hours fills, rounded half up and at least one: a step is
    the simulation's grain, so a shorter act still takes a whole one. The model estimates the
    duration and code divides, since the division is where models go wrong. A missing or
    non-positive estimate is one step."""
    if isinstance(hours, bool) or not isinstance(hours, int | float) or not math.isfinite(hours) or hours <= 0:
        return 1
    return _whole_steps(hours * _SECONDS_PER_HOUR, seconds_per_step)
