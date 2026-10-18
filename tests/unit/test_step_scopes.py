"""A step is told by scope: each event sits where it happened, once, and each place that only saw
it gets what it saw.

Movement is the case that spans scopes. Within one step a body can leave a room, pass a waypoint
and arrive; across steps it can stand on a waypoint or be between two. None of it needs a special
event: the engine records where the doer was and what each place observed, and the read model
files them.
"""

from __future__ import annotations

from engine.environment import IN_TRANSIT
from interaction.models import ActionSummary, ObservationSummary, StepView
from tests.unit.step_view import TRANSIT, agent, events, snapshot_of, step_of_records, the_action


def _record(**fields: object) -> dict[str, object]:
    return {"seq": 5, "execution_id": "x1", "initiator_id": "a", "agent_id": "a",
            "participant_ids": ["a"], "phase": "settled", "observations": [], **fields}


def _seen(where: str, text: str) -> dict[str, object]:
    return {"location_id": where, "text": text, "strength": None}


def test_a_joint_act_is_one_event_told_by_its_initiator() -> None:
    talk = {"action_type": "talk", "location_id": "loc-hall", "participant_ids": ["a", "b"],
            "observations": [_seen("loc-hall", "甲与乙低声交谈")], "overheard_by": ["c"]}
    step = StepView.from_snapshot(snapshot_of(
        agents={i: {"agent_id": i, "location_id": "loc-hall"} for i in ("a", "b", "c")},
        records=[
            # The conscript's record arrives first; it only restates the intent back at him.
            _record(**talk, agent_id="b", outcome="乙被甲叫住", inner_monologue=""),
            _record(**talk, outcome="甲对乙说了实话", inner_monologue="我得说出来",
                    acted_upon=["d"], dialogue=[{"speaker_id": "a", "speaker": "甲", "line": "是我"}]),
        ],
    ))

    [(where, act)] = events(step, "action")
    assert isinstance(act, ActionSummary)
    assert where == "loc-hall"
    assert act.participant_ids == ["a", "b"]
    assert act.public is not None and act.public.text == "甲与乙低声交谈"
    assert act.private.outcome == "甲对乙说了实话"
    # Inside it: who spent a turn, who it landed on, who overheard. Nobody else.
    assert act.private.audience == ["a", "b", "d", "c"]
    # The thought is its owner's, filed under the act it led to.
    [thought] = agent(step, "a").private.thoughts
    assert (thought.seq, thought.text) == (5, "我得说出来")
    assert agent(step, "b").private.thoughts == []


def test_a_move_through_a_waypoint_in_one_step_is_told_at_each_place() -> None:
    step = step_of_records(_record(
        action_type="move", location_id="loc-hall", outcome="甲从玄武门经内院到了大殿",
        observations=[_seen("loc-gate", "甲离开了"), _seen("loc-yard", "甲穿过院子"),
                      _seen("loc-hall", "甲走了进来")],
    ))

    by_place = {where: e for where, e in events(step)}
    assert isinstance(by_place["loc-hall"], ActionSummary)
    assert by_place["loc-hall"].public is not None
    assert by_place["loc-hall"].public.text == "甲走了进来"
    assert isinstance(by_place["loc-gate"], ObservationSummary)
    assert by_place["loc-gate"].text == "甲离开了"
    assert by_place["loc-yard"].text == "甲穿过院子"
    # One act, three places: one seq.
    assert {e.seq for e in by_place.values()} == {5}


def test_a_journey_over_steps_opens_at_its_origin_and_runs_on_the_way() -> None:
    begin = step_of_records(_record(
        action_type="move", phase="begin", location_id=IN_TRANSIT, outcome="甲动身前往大殿",
        observations=[_seen("loc-gate", "甲离开了"), _seen(IN_TRANSIT, "途中")],
    ))
    [(where, act)] = events(begin, "action")
    assert where == TRANSIT
    # Nobody watches from the way between two waypoints.
    assert act.public is None
    [(origin, seen)] = events(begin, "observation")
    assert (origin, seen.text) == ("loc-gate", "甲离开了")

    tick = step_of_records(_record(
        action_type="move", phase="ongoing_tick", location_id="loc-yard", outcome="甲在赶路",
        observations=[_seen("loc-yard", "甲穿过院子")], elapsed_steps=1, total_steps=3,
    ))
    [(where, act)] = events(tick, "action")
    assert where == "loc-yard"
    assert (act.private.elapsed_steps, act.private.total_steps) == (1, 3)


def test_an_act_nobody_saw_has_no_public_side() -> None:
    act = step_of_records(_record(action_type="covert", location_id="loc-hall", outcome="甲翻看了密信"))
    [(_, covert)] = events(act, "action")
    assert covert.public is None
    assert covert.private.outcome == "甲翻看了密信"


def test_beats_of_one_execution_stay_apart_from_a_new_one() -> None:
    """An agent who finishes one act and starts the next in a step has two executions."""
    step = step_of_records(
        _record(execution_id="x1", phase="ongoing_complete", location_id="loc-hall", seq=3),
        _record(execution_id="x2", phase="begin", location_id="loc-hall", seq=4),
    )
    assert [(e.execution_id, e.phase) for _, e in events(step, "action")] == [
        ("x1", "ongoing_complete"), ("x2", "begin"),
    ]


def test_injections_stay_with_the_observer() -> None:
    step = StepView.from_snapshot(snapshot_of(
        events=[{"kind": "injection", "id": "ev1", "authored_by": "director",
                 "narrative_desc": "城中起火", "directive_text": "让城里着火", "seq": 2}],
    ))
    assert [e.narrative for e in step.observer.injections] == ["城中起火"]
    assert events(step) == []


def test_what_an_injection_changed_is_seen_where_it_landed() -> None:
    """The record stays with the observer; what onlookers perceived of its change is told in each
    place, under the injection's seq, done by nobody in the world."""
    step = StepView.from_snapshot(snapshot_of(
        events=[{"kind": "injection", "id": "ev1", "authored_by": "system",
                 "narrative_desc": "诏书加印", "seq": 7,
                 "observations": [_seen("loc-hall", "封皮上多出一道朱印。")]}],
    ))
    assert [e.narrative for e in step.observer.injections] == ["诏书加印"]
    assert events(step) == [("loc-hall", ObservationSummary(
        seq=7, initiator_id="", participant_ids=[], text="封皮上多出一道朱印。", strength=None,
    ))]


def test_short_term_goals_are_one_typed_queue() -> None:
    step = StepView.from_snapshot(snapshot_of(
        agents={"a": {"agent_id": "a", "location_id": "loc-hall",
                            "short_term_goals": ["查账"],
                            "short_term_goal_entities": [
                                {"text": "查账", "status": "active", "origin": "cognitive"},
                                {"text": "还钱", "status": "completed", "origin": "residue"},
                            ]}},
    ))
    goals = agent(step, "a").private.short_term_goals
    assert [(g.text, g.status, g.origin) for g in goals] == [
        ("查账", "active", "cognitive"), ("还钱", "completed", "residue"),
    ]


def test_what_shows_of_an_agent_is_not_what_he_is_doing() -> None:
    step = StepView.from_snapshot(snapshot_of(
        agents={"a": {"agent_id": "a", "location_id": "loc-hall", "activity_status": "covert",
                            "gender": "男", "age": 30}},
    ))
    seen = agent(step, "a")
    # Visible at a glance.
    assert (seen.gender, seen.age) == ("男", 30)
    # Someone acting covertly doesn't look like it; what he is doing is his own.
    assert seen.public.activity_status == ""
    assert seen.private.activity_status == "covert"


def test_an_act_tells_onlookers_what_they_saw_and_keeps_the_intent() -> None:
    act = the_action(_record(
        action_type="covert", phase="begin", location_id="loc-hall", outcome="甲开始盯梢",
        action_description="假装路过，暗中盯着乙", duration_label="约2小时", elapsed_steps=0,
        total_steps=2, observations=[_seen("loc-hall", "甲在院中闲逛")],
    ))
    assert act.public is not None and act.public.text == "甲在院中闲逛"
    assert act.private.action_description == "假装路过，暗中盯着乙"
    assert (act.private.duration_label, act.private.total_steps) == ("约2小时", 2)
