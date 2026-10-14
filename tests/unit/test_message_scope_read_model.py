"""A delivered message's ADDRESSING decides the scope it lands in, and place ids must not leak.

1. MessageSystem resolves receivers from (recipients × location_scope): a named list, a place, or
   the whole world. Showing only the resolved inbox list makes all three identical and silently
   turns one proclamation into N private letters.

2. ``location_scope`` is an id. A narrative surface gets the place's NAME from the snapshot's own
   place table (the environment's), never by looking ids up itself: that would be a second
   id→name path beside WorldDirectory (see the WorldDirectory usage rule in CLAUDE.md).
"""

from __future__ import annotations

from typing import Any

import pytest

from core.interfaces.perception import Broadcast, BroadcastType
from core.interfaces.place import Place
from core.interfaces.severity import Severity
from engine.environment import IN_TRANSIT
from interaction.models import StepView
from tests.unit.bus_tap import tap
from tests.unit.step_view import TRANSIT, WORLD, events, snapshot_of
from tests.unit.test_runtime_smoke import _build_agent, _build_runtime, _set_work_decision

PLACES = {"loc-gate": "玄武门", "loc-yard": "内院"}


def _message(**fields: Any) -> dict[str, Any]:
    """A delivered message as the step record writes it: who got it and where each stood."""
    return {
        "kind": "message", "message_id": "m1", "sender_id": "agent-a", "sender_name": "甲",
        "recipients": ["agent-b", "agent-c"], "location_scope": None, "content": "来",
        "metadata": {},
        "receivers": [
            {"agent_id": "agent-b", "location_id": "loc-gate"},
            {"agent_id": "agent-c", "location_id": "loc-yard"},
        ],
        **fields,
    }


def _step(message: dict[str, Any]) -> StepView:
    return StepView.from_snapshot(snapshot_of(places=PLACES, events=[message]))


def test_named_recipients_are_a_private_letter_where_each_of_them_stands() -> None:
    delivered = events(_step(_message(seq=4)), "message")

    assert [(where, msg.receiver_ids) for where, msg in delivered] == [
        ("loc-gate", ["agent-b"]), ("loc-yard", ["agent-c"]),
    ]
    # One message, so one seq; nobody outside it may read it.
    assert {msg.seq for _, msg in delivered} == {4}
    assert all(msg.scope == "direct" for _, msg in delivered)
    assert all(msg.audience == ["agent-a", "agent-b", "agent-c"] for _, msg in delivered)


def test_a_letter_reaching_a_traveler_lands_on_the_way() -> None:
    [(where, _)] = events(_step(_message(
        recipients=["agent-b"], receivers=[{"agent_id": "agent-b", "location_id": IN_TRANSIT}],
    )), "message")
    assert where == TRANSIT


def test_no_recipients_plus_a_location_is_a_public_announcement_there() -> None:
    [(where, msg)] = events(_step(_message(
        recipients=None, location_scope="loc-gate", content="肃静",
    )), "message")

    assert (where, msg.scope, msg.audience) == ("loc-gate", "place", None)


def test_no_recipients_and_no_location_is_a_world_proclamation() -> None:
    [(where, msg)] = events(_step(_message(recipients=None, content="昭告")), "message")

    assert (where, msg.scope, msg.audience) == (WORLD, "world", None)


@pytest.mark.asyncio
async def test_the_producer_names_places_once_and_leaks_no_enum(container) -> None:
    """The broadcast is routed by the place's id, and the snapshot carries the environment's own
    name for it in its place table, so no consumer resolves ids itself. ``broadcast_type`` gets
    the same treatment as ``action_type``: the VALUE, never the live enum, which would reach live
    observers as "BroadcastType.WORLD_EVENT".
    """
    published = tap(container.event_bus)
    world_id = "world-broadcast-place"
    runtime, environment, _ = _build_runtime(container, world_id=world_id)
    # A location that HAS a name, so a real resolution is distinguishable from the directory's
    # miss-fallback (「某地」).
    environment.space.register_place(Place(place_id="palace", name="太极宫"))
    environment.place_agent(agent_id="agent-1", location_id="palace")
    agents = [_build_agent(container, world_id=world_id, agent_id="agent-1", name="Li Shimin",
                           is_main_character=True)]
    _set_work_decision(agents[0])
    runtime._broadcast_channel.publish(
        Broadcast(
            content="门外有异动。",
            source="system",
            broadcast_type=BroadcastType.WORLD_EVENT,
            deliver_step=1,
            location_scope="palace",  # an ID — routed by it, but never rendered as one
            severity=Severity.HIGH,
        )
    )

    await runtime.run_step(agents)
    snapshot = published()[0]["snapshot"]
    (record,) = snapshot.events_of("broadcast")

    assert record["location_scope"] == "palace"          # the id routes it…
    assert snapshot.state.places["palace"] == "太极宫"    # …and the world names it once
    assert isinstance(record["broadcast_type"], str)
    assert "BroadcastType" not in record["broadcast_type"]
    [(where, _)] = events(StepView.from_snapshot(snapshot), "broadcast")
    assert where == "palace"


def test_a_broadcast_lands_in_the_scope_that_perceived_it() -> None:
    """A senderless broadcast is place-scoped or world-wide — and that is WHO PERCEIVED IT.

    BroadcastChannel.for_location() hands a place-scoped broadcast only to the agents standing
    there, and a world one to everybody.
    """
    step = StepView.from_snapshot(snapshot_of(
        places={"loc-gate": "玄武门"},
        events=[
            {"kind": "broadcast", "seq": 1, "content": "门外有异动。", "location_scope": "loc-gate",
             "severity": "low"},
            {"kind": "broadcast", "seq": 2, "content": "有人死了。", "location_scope": None,
             "severity": "high"},
        ],
    ))
    local, world = events(step, "broadcast")

    assert local[0] == "loc-gate"
    assert step.locations["loc-gate"].name == "玄武门"
    assert world[0] == WORLD  # no place at all — that IS "everyone heard it"
    assert world[1].severity == "high"
