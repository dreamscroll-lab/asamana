"""Action scheduling durations shared by execution and actor summaries."""

from core.interfaces.action import ActionType


def action_duration_steps(action_type: ActionType, estimated_steps: int, *, live_talk: bool = False) -> int:
    if action_type is ActionType.MOVE:
        raise ValueError("Movement duration must be derived from travel time")
    if action_type in (ActionType.PHYSICAL, ActionType.SEND_MESSAGE, ActionType.ERRAND):
        return 1
    if action_type is ActionType.TALK:
        return 1 if live_talk else max(estimated_steps, 1)
    return estimated_steps


def travel_duration_steps(travel_seconds: int, seconds_per_step: int) -> int:
    return max(1, round(travel_seconds / seconds_per_step))
