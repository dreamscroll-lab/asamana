"""A delivered message's ADDRESSING must survive into the read model, and place ids must not.

1. MessageSystem resolves receivers from (recipients × location_scope): a named list, a place, or
   the whole world. Showing only the resolved inbox list makes all three identical and silently
   turns one proclamation into N private letters.

2. ``location_scope`` is an id. A narrative surface gets the NAME, baked by the PRODUCER (the
   runtime, which holds the WorldDirectory), as ``sender_name`` sits beside ``sender_id``. A read
   model looking ids up itself would be a second id→name path beside WorldDirectory (see the
   WorldDirectory usage rule in CLAUDE.md).
"""

from __future__ import annotations

import pytest

from core.interfaces.perception import Broadcast, BroadcastType
from core.interfaces.severity import Severity
from core.interfaces.snapshot import WorldSnapshot
from interaction.models import StepEvent, _messages_from_snapshot
from tests.unit.test_runtime_smoke import _build_agent, _build_runtime, _set_work_decision
from core.interfaces.place import Place
from tests.unit.bus_tap import tap


def _snapshot(*delivered: dict) -> WorldSnapshot:
    return WorldSnapshot(
        world_id="w",
        step=7,
        timestamp="2026-07-14 00:00:00",
        metadata={
            "messages": {
                "delivered": list(delivered),
                # Everyone who got it — the routing already happened upstream.
                "inboxes": {"agent-b": ["m1"], "agent-c": ["m1"]},
            }
        },
    )


def test_named_recipients_are_a_direct_message() -> None:
    (msg,) = _messages_from_snapshot(
        _snapshot({"id": "m1", "sender_id": "agent-a", "sender_name": "甲",
                   "recipients": ["agent-b", "agent-c"], "location_scope": None,
                   "location_name": "", "content": "来"})
    )

    assert msg.scope == "direct"
    assert msg.place == ""
    assert msg.receiver_ids == ["agent-b", "agent-c"]


def test_no_recipients_plus_a_location_is_a_place_announcement() -> None:
    (msg,) = _messages_from_snapshot(
        _snapshot({"id": "m1", "sender_id": "agent-a", "sender_name": "甲",
                   "recipients": None, "location_scope": "loc-gate",
                   "location_name": "玄武门", "content": "肃静"})
    )

    assert msg.scope == "place"
    # The NAME the producer baked in — this string goes straight into the narrative feed.
    assert msg.place == "玄武门"
    assert "loc-" not in msg.place


def test_no_recipients_and_no_location_is_a_world_proclamation() -> None:
    (msg,) = _messages_from_snapshot(
        _snapshot({"id": "m1", "sender_id": "agent-a", "sender_name": "甲",
                   "recipients": None, "location_scope": None,
                   "location_name": "", "content": "昭告"})
    )

    assert msg.scope == "world"
    assert msg.place == ""


@pytest.mark.asyncio
async def test_the_producer_bakes_the_place_name_and_leaks_no_enum(container) -> None:
    """The runtime — which holds the WorldDirectory — resolves the name at write time.

    This is the invariant the read model relies on: it does NOT look ids up itself. And a
    place-scoped broadcast must carry a NAME, not the id it is routed by, because the feed
    prints it. ``broadcast_type`` gets the same treatment as ``action_type``: the VALUE, never
    the live enum, which would reach live observers as "BroadcastType.WORLD_EVENT".
    """
    published = tap(container.event_bus)
    world_id = "world-broadcast-place"
    runtime, environment, _ = _build_runtime(container, world_id=world_id)
    # A location that HAS a name, so a real resolution is distinguishable from the directory's
    # miss-fallback (「某地」).
    environment.space.register_place(
        Place(place_id="palace", name="太极宫")
    )
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
    (record,) = published()[0]["broadcasts"]

    # The environment's own name for "palace" — resolved by the producer, not by a consumer.
    assert record["location_name"] == "太极宫"       # the NAME…
    assert record["location_scope"] == "palace"      # …while the id still routes it
    assert isinstance(record["broadcast_type"], str)
    assert "BroadcastType" not in record["broadcast_type"]


def test_broadcast_scope_and_its_baked_place_survive_to_the_observer() -> None:
    """A senderless broadcast is place-scoped or world-wide — and that is WHO PERCEIVED IT.

    BroadcastChannel.for_location() hands a place-scoped broadcast only to the agents standing
    there, and a world one to everybody. The observer must be able to tell them apart.
    """
    snapshot = WorldSnapshot(
        world_id="w",
        step=7,
        timestamp="2026-07-14 00:00:00",
        metadata={
            "broadcasts": [
                {"content": "门外有异动。", "location_scope": "loc-gate",
                 "location_name": "玄武门", "severity": "low"},
                {"content": "有人死了。", "location_scope": None,
                 "location_name": "", "severity": "high"},
            ]
        },
    )

    local, world = StepEvent.from_snapshot(snapshot).broadcasts

    assert local.location_name == "玄武门"
    assert local.location_scope == "loc-gate"
    assert world.location_name == ""  # no place at all — that IS "everyone heard it"
    assert world.severity == "high"
