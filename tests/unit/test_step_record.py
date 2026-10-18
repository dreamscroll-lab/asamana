"""The step record tells each event once.

The runtime keeps one action record per participant (each writes its own memory); the snapshot
folds them into one event per beat, with every participant's own part inside. It is the only
fold: readers of a snapshot never fold again.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from core.interfaces.message import Message
from core.interfaces.perception import Broadcast, BroadcastType
from core.interfaces.snapshot import EngineState, WorldSnapshot, WorldState
from engine.event_seq import EventSequencer
from engine.message_system import MessageDelivery
from engine.step_record import action_events, message_events
from providers.snapshot.file import FileSnapshotProvider


def _record(agent_id: str, **fields: object) -> dict[str, object]:
    return {"agent_id": agent_id, "initiator_id": "a", "execution_id": "x1", "phase": "settled",
            "participant_ids": ["a", "b"], "location_id": "hall", **fields}


def test_a_joint_beat_is_one_event_with_every_participants_part() -> None:
    seen = {"location_id": "hall", "text": "甲与乙交谈", "strength": None}
    [event] = action_events([
        # The conscript's record may arrive first.
        _record("b", action_description="被甲叫住", outcome="乙被甲叫住", observations=[seen],
                acted_upon=[], overheard_by=["c"]),
        _record("a", action_description="找乙谈", outcome="甲对乙说了实话", inner_monologue="我得说",
                observations=[seen], acted_upon=["d"], overheard_by=["c"]),
    ], routes={})

    assert event["initiator_id"] == "a"
    assert event["participant_ids"] == ["a", "b"]
    # The initiator's part first; each keeps its own words and thought.
    assert [(p["agent_id"], p["outcome"], p["inner_monologue"]) for p in event["parts"]] == [
        ("a", "甲对乙说了实话", "我得说"), ("b", "乙被甲叫住", ""),
    ]
    # What places saw is told once, not once per participant.
    assert event["observations"] == [seen]
    assert (event["acted_upon"], event["overheard_by"]) == (["d"], ["c"])


def test_two_executions_of_one_agent_stay_two_events() -> None:
    events = action_events([
        _record("a", execution_id="x1", phase="ongoing_complete", participant_ids=["a"]),
        _record("a", execution_id="x2", phase="begin", participant_ids=["a"]),
    ], routes={})
    assert [(e["execution_id"], e["phase"]) for e in events] == [("x1", "ongoing_complete"), ("x2", "begin")]


def test_the_route_walked_rides_the_beat_the_move_ended_on() -> None:
    route = {"path": ["gate", "hall"], "elapsed_steps": 1, "total_steps": 1}
    closing, opening = action_events([
        _record("a", action_type="move", phase="settled", participant_ids=["a"]),
        _record("a", action_type="move", phase="begin", execution_id="x2", participant_ids=["a"]),
    ], routes={"a": route})
    assert closing["route"] == route
    assert opening["route"] is None


def test_a_letter_records_where_each_reader_stood() -> None:
    letter = Message(id="m1", world_id="w", sender_id="s", content="来", recipients=["b", "c"],
                     location_scope=None, created_step=1, deliver_step=2)
    delivery = MessageDelivery(step=2, delivered_messages=[letter], inboxes={"b": [letter], "c": [letter]})

    [event] = message_events(delivery, where={"b": "hall", "c": "gate"})

    assert event["message_id"] == "m1"
    assert event["receivers"] == [
        {"agent_id": "b", "location_id": "hall"}, {"agent_id": "c", "location_id": "gate"},
    ]


def test_events_are_ordered_and_numbered_once_per_step() -> None:
    sequencer = EventSequencer(start_seq=10)
    ordered = sequencer.order_events(
        messages=[{"kind": "message"}], broadcasts=[{"kind": "broadcast"}],
        actions=[{"kind": "action"}, {"kind": "action"}], injections=[{"kind": "injection"}],
    )
    assert [(e["kind"], e["seq"]) for e in ordered] == [
        ("message", 10), ("broadcast", 11), ("action", 12), ("action", 13), ("injection", 14),
    ]
    assert sequencer.next_seq == 15


@pytest.mark.asyncio
async def test_a_snapshot_survives_the_file_round_trip(tmp_path) -> None:
    provider = FileSnapshotProvider(path=str(tmp_path))
    in_flight = Message(id="m9", world_id="w", sender_id="s", content="后到", recipients=["a"],
                        location_scope=None, created_step=1, deliver_step=9)
    notice = Broadcast(content="讣闻", source="system", broadcast_type=BroadcastType.WORLD_EVENT,
                       deliver_step=4)
    snapshot = WorldSnapshot(
        world_id="w", step=3, timestamp=datetime(2026, 1, 1),
        world_time={"label": "辰时", "hour": 8, "minute": 0},
        state=WorldState(agents={"a": {"agent_name": "甲", "location_id": "hall"}}),
        events=[{"kind": "action", "seq": 7, "initiator_id": "a", "parts": []}],
        engine=EngineState(pending_messages=[in_flight], pending_broadcasts=[notice], event_seq=8,
                           environment={"transit_origins": {}}),
    )

    await provider.save("w", 3, snapshot)
    loaded = await provider.load("w", 3)

    assert loaded == snapshot
