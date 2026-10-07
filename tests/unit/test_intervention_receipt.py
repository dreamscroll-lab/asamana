"""Intervention receipt: who an intervention actually moved.

The most important assertion is the negative one: delivered, but nobody acted on it. Without it,
"the pressure evaluator decided this wasn't a push" and "the feature is broken" look the same from
outside, and the director can't learn how to use the tool.

The receipt is a pure function of who was reached plus three signals from this step. It isn't part
of the director channel: the channel finishes at the start of the step, and the three signals only
exist at the end.
"""

from __future__ import annotations


from agent.motivation import ExternalDriveType, ExternalGoal
from agent.personality import (
    EmotionState,
    EmotionType,
    PersonalityLayer,
    SoulLayer,
    StateLayer,
)
from core.interfaces.urgency import Urgency
from engine.directory import LiveWorldDirectory
from engine.environment import EnvironmentSystem
from engine.intervention_receipt import build_receipt


class _Agent:
    """The receipt reads only pending_external_goals and the name; no full Agent needed."""

    def __init__(self, agent_id: str, name: str, goals: list[ExternalGoal] | None = None) -> None:
        self.agent_id = agent_id
        self.is_active = True
        self.pending_external_goals = goals or []
        self.personality = PersonalityLayer(
            soul=SoulLayer(name=name, agent_id=agent_id, role="臣"),
            state=StateLayer(
                agent_id=agent_id, step=1,
                emotion=EmotionState(primary=EmotionType.NEUTRAL, intensity=0.3, valence=0.0),
                current_location="palace", vitality=1.0,
            ),
        )


def _goal(urgency: Urgency) -> ExternalGoal:
    return ExternalGoal(
        text="立刻离开", source_id="world", urgency=urgency,
        drive_type=ExternalDriveType.THREAT,
    )


def _directory(agents: dict[str, _Agent]) -> LiveWorldDirectory:
    return LiveWorldDirectory.from_agents(agents, EnvironmentSystem())


def test_receipt_reports_who_it_reached_and_who_it_actually_moved() -> None:
    agents = {"a2": _Agent("a2", "李建成", [_goal(Urgency.HIGH)])}

    receipt = build_receipt(
        ["a2"], agents=agents, directory=_directory(agents),
        decided_ids={"a2"}, interrupted_ids=set(),
    )

    assert receipt["delivered_to"] == [{"agent_id": "a2", "name": "李建成"}]
    assert receipt["pressure"] == [{"agent_id": "a2", "name": "李建成", "urgency": "紧急"}]
    assert receipt["decided"] == [{"agent_id": "a2", "name": "李建成"}]
    assert receipt["interrupted"] == []


def test_receipt_shows_an_intervention_that_moved_nobody() -> None:
    """Delivered, but the pressure evaluator decided it wasn't a push for this person, so nothing
    followed.

    This is why the receipt exists: to let the UI tell "had no effect" apart from "not their turn yet".
    """
    agents = {"a2": _Agent("a2", "李建成", goals=[])}

    receipt = build_receipt(
        ["a2"], agents=agents, directory=_directory(agents),
        decided_ids=set(), interrupted_ids=set(),
    )

    assert receipt["delivered_to"]       # it really was delivered
    assert receipt["pressure"] == []     # but nobody was moved by it
    assert receipt["decided"] == []


def test_receipt_reports_the_heaviest_pressure_not_every_stirring() -> None:
    """One intervention can raise several goals. The receipt reports how strongly it hit the person,
    not a per-goal list."""
    agents = {"a2": _Agent("a2", "李建成", [_goal(Urgency.LOW), _goal(Urgency.CRITICAL)])}

    receipt = build_receipt(
        ["a2"], agents=agents, directory=_directory(agents),
        decided_ids=set(), interrupted_ids=set(),
    )

    assert receipt["pressure"] == [{"agent_id": "a2", "name": "李建成", "urgency": "危急"}]


def test_other_columns_stay_inside_the_set_this_intervention_touched() -> None:
    """"Who did this move" is not "who acted anywhere this step".

    Other agents may be admitted to decide, or interrupted, in the same step for unrelated reasons.
    Crediting them to this intervention would make the receipt overstate its influence.
    """
    agents = {
        "a1": _Agent("a1", "李世民", [_goal(Urgency.HIGH)]),
        "a2": _Agent("a2", "李建成", [_goal(Urgency.HIGH)]),
    }

    receipt = build_receipt(
        ["a2"], agents=agents, directory=_directory(agents),
        decided_ids={"a1", "a2"}, interrupted_ids={"a1"},
    )

    assert [d["agent_id"] for d in receipt["decided"]] == ["a2"]
    assert receipt["interrupted"] == []      # a1 was interrupted by something else
    assert [p["agent_id"] for p in receipt["pressure"]] == ["a2"]


def test_an_unknown_target_is_skipped_rather_than_faked() -> None:
    """An agent who was reached but is no longer in agents (e.g. died this step and was removed) is
    skipped rather than given a made-up row."""
    agents: dict[str, _Agent] = {}
    receipt = build_receipt(
        ["ghost"], agents=agents, directory=_directory(agents),
        decided_ids=set(), interrupted_ids=set(),
    )
    assert receipt["pressure"] == []
    # Delivery is still recorded: the message was sent, the recipient is just gone.
    assert receipt["delivered_to"] == [{"agent_id": "ghost", "name": "某人"}]
