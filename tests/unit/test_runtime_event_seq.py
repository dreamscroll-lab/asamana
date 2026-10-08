"""Runtime event-seq contract.

`seq` is a per-world monotonic ordinal stamped on every per-step narrative event (action record,
delivered message, broadcast snapshot dict, world event) in emission order: an engine-level
property, not a render concept. Within a step the order is messages → broadcasts → actions →
world events (arrivals-before-actions: deliveries and broadcasts were emitted last step).
Consumers order events by it; the renderer sorts intra-step events by seq, so a death broadcast
can't show up before the victim's last action, and the replayer preserves order.

Contract these tests pin:
- Strictly increasing across every event within one step.
- Continues monotonically across steps (never resets).
- Survives snapshot save/restore: the counter's next-to-assign value is
  persisted in ``metadata["event_seq"]`` and the runtime resumes from it via
  ``start_event_seq`` — no collision, no rewind across restart.
- Flows through read-models (``ActionSummary`` / ``MessageSummary`` /
  ``WorldEventSummary``) built via ``StepEvent.from_runtime_payload`` and
  ``StepEvent.from_snapshot``.
"""
from __future__ import annotations

import pytest

from engine.scheduler import DEFAULT_MAIN_MAX_IDLE_STEPS

from core.interfaces.message import Message
from engine.application import NarrativeApplication
from interaction.models import StepEvent
from tests.unit.test_runtime_smoke import (
    _build_agent,
    _build_runtime,
    _set_work_decision,
)
from tests.unit.bus_tap import tap


def _collect_seqs(records) -> list[int]:
    """Extract seq values from a list of dicts, preserving order."""
    return [r["seq"] for r in records]


def _is_strictly_increasing(seqs: list[int]) -> bool:
    return all(a < b for a, b in zip(seqs, seqs[1:]))


@pytest.mark.asyncio
async def test_step_assigns_seq_across_all_channels_in_emission_order(container) -> None:
    """One step's action records + delivered messages + broadcast snapshots
    + world events all carry a strictly increasing ``seq`` — messages first,
    then broadcasts, then actions, then events (arrivals-before-actions)."""
    published = tap(container.event_bus)
    world_id = "world-seq-order"
    runtime, environment, message_system = _build_runtime(container, world_id=world_id)
    environment.place_agent(agent_id="agent-1", location_id="palace")
    environment.place_agent(agent_id="agent-2", location_id="palace")
    agents = [
        _build_agent(container, world_id=world_id, agent_id="agent-1", name="Li Shimin", is_main_character=True),
        _build_agent(container, world_id=world_id, agent_id="agent-2", name="Li Jiancheng", is_main_character=False),
    ]
    _set_work_decision(container)

    # Publish a message that will be delivered at step 2 so this step has
    # deliveries alongside the action records.
    await message_system.publish(
        Message(
            id="msg-1",
            world_id=world_id,
            sender_id="agent-2",
            content="At dawn.",
            recipients=["agent-1"],
            location_scope=None,
            created_step=1,
            deliver_step=2,
        )
    )
    # Advance one step so message becomes deliverable next step.
    await runtime.run_step(agents)
    # Consume its event to keep bus tidy.
    published()

    # Manually enqueue a broadcast + world event into the second step's payload
    # by driving another run_step where deliveries + a broadcast fire together.
    runtime._broadcast_channel.publish  # sanity import
    from core.interfaces.perception import Broadcast, BroadcastType
    from core.interfaces.severity import Severity
    runtime._broadcast_channel.publish(
        Broadcast(
            content="A distant bell tolls.",
            source="system",
            broadcast_type=BroadcastType.WORLD_EVENT,
            deliver_step=2,
            location_scope=None,
            severity=Severity.LOW,
        )
    )

    await runtime.run_step(agents)
    step_event = published()[0]

    actions = step_event["actions"]
    delivered = step_event["messages"]["delivered"]
    broadcasts = step_event["broadcasts"]
    events = step_event["events"]

    # Fixture: this step really did produce all four channels.
    assert actions, "expected action records this step"
    assert delivered, "expected at least one delivered message this step"
    assert broadcasts, "expected at least one broadcast this step"

    # Each individual channel is strictly increasing (holds trivially for one item
    # and pins the emission-order property when there are several).
    assert _is_strictly_increasing(_collect_seqs(actions))
    assert _is_strictly_increasing(_collect_seqs(delivered))
    assert _is_strictly_increasing(_collect_seqs(broadcasts))
    assert _is_strictly_increasing(_collect_seqs(events))

    # Cross-channel ordering: messages < broadcasts < actions < events.
    max_message_seq = max(r["seq"] for r in delivered)
    min_broadcast_seq = min(r["seq"] for r in broadcasts)
    max_broadcast_seq = max(r["seq"] for r in broadcasts)
    min_action_seq = min(r["seq"] for r in actions)
    max_action_seq = max(r["seq"] for r in actions)

    assert max_message_seq < min_broadcast_seq
    assert max_broadcast_seq < min_action_seq
    if events:
        min_event_seq = min(r["seq"] for r in events)
        assert max_action_seq < min_event_seq

    # Global step-scope monotonicity: every seq across every channel is distinct.
    all_seqs = (
        _collect_seqs(actions)
        + _collect_seqs(delivered)
        + _collect_seqs(broadcasts)
        + _collect_seqs(events)
    )
    assert len(set(all_seqs)) == len(all_seqs), "seq collision within a step"


@pytest.mark.asyncio
async def test_seq_continues_monotonically_across_steps(container) -> None:
    """The counter is per-world and never resets — step N+1's minimum seq >
    step N's maximum seq."""
    published = tap(container.event_bus)
    world_id = "world-seq-cross-step"
    runtime, environment, _ = _build_runtime(container, world_id=world_id)
    environment.place_agent(agent_id="agent-1", location_id="palace")
    environment.place_agent(agent_id="agent-2", location_id="palace")
    agents = [
        _build_agent(container, world_id=world_id, agent_id="agent-1", name="A", is_main_character=True),
        _build_agent(container, world_id=world_id, agent_id="agent-2", name="B", is_main_character=False),
    ]
    _set_work_decision(container)

    await runtime.run_step(agents)
    step1 = published()[0]
    await runtime.run_step(agents)
    step2 = published()[0]

    step1_max = max(r["seq"] for r in step1["actions"])
    step2_min = min(r["seq"] for r in step2["actions"])
    assert step2_min > step1_max, "seq must not reset or overlap across steps"


@pytest.mark.asyncio
async def test_seq_persists_in_snapshot_and_resumes_without_collision(
    mock_build_container, test_config
) -> None:
    """The counter's next-to-assign value is persisted in the snapshot's
    ``metadata["event_seq"]``. A restored runtime resumes from it so a new step
    after restart keeps a strictly increasing seq — no collision, no rewind."""
    container = mock_build_container
    published = tap(container.event_bus)
    application = NarrativeApplication(container, test_config)
    world = await application.build_world("玄武门之变", template="changan_iso")
    await application.run_world(world.world_id, steps=2)

    snapshot = await container.snapshot.load(world.world_id, 2)
    assert snapshot is not None
    persisted_seq = snapshot.metadata.get("event_seq")
    assert isinstance(persisted_seq, int)
    # The persisted value equals: max seq assigned across steps 1..2, + 1.
    seqs_step1 = [r["seq"] for r in (await container.snapshot.load(world.world_id, 1)).actions_this_step]
    seqs_step2 = [r["seq"] for r in snapshot.actions_this_step]
    assert persisted_seq >= max(seqs_step1 + seqs_step2) + 1

    # Restore the session and run one more step; the new step's action seqs must
    # be >= the persisted counter (strictly increasing, no restart-from-zero).
    application._sessions.clear()  # noqa: SLF001 — force a fresh restore path
    await application.restore_session(world.world_id)
    published()  # clear leftovers from build/run
    # Run until someone acts again: the scheduler's cadence gate (scheduler.DEFAULT_MAIN_MAX_IDLE_STEPS)
    # needn't wake anyone the next step, and what's under test is only that seq doesn't go
    # backwards across a restart.
    await application.run_world(world.world_id, steps=DEFAULT_MAIN_MAX_IDLE_STEPS + 1)
    resumed_seqs = [
        r["seq"] for event in published() for r in event.get("actions") or []
    ]
    assert resumed_seqs, "restart 后若干步内应至少有一次行动"
    assert min(resumed_seqs) >= persisted_seq, (
        f"resumed seq {min(resumed_seqs)} must be >= persisted next-to-assign {persisted_seq}"
    )


@pytest.mark.asyncio
async def test_step_event_read_models_carry_seq(container) -> None:
    """``StepEvent.from_runtime_payload`` and ``from_snapshot`` propagate seq
    into ``ActionSummary`` / ``MessageSummary`` / ``WorldEventSummary`` and
    keep raw ``broadcasts`` dicts that already carry seq."""
    published = tap(container.event_bus)
    world_id = "world-seq-readmodels"
    runtime, environment, message_system = _build_runtime(container, world_id=world_id)
    environment.place_agent(agent_id="agent-1", location_id="palace")
    environment.place_agent(agent_id="agent-2", location_id="palace")
    agents = [
        _build_agent(container, world_id=world_id, agent_id="agent-1", name="A", is_main_character=True),
        _build_agent(container, world_id=world_id, agent_id="agent-2", name="B", is_main_character=False),
    ]
    _set_work_decision(container)
    await message_system.publish(
        Message(
            id="msg-1",
            world_id=world_id,
            sender_id="agent-2",
            content="Meet me.",
            recipients=["agent-1"],
            location_scope=None,
            created_step=1,
            deliver_step=2,
        )
    )
    await runtime.run_step(agents)
    published()

    from core.interfaces.perception import Broadcast, BroadcastType
    from core.interfaces.severity import Severity
    runtime._broadcast_channel.publish(  # noqa: SLF001
        Broadcast(
            content="A bell tolls.",
            source="system",
            broadcast_type=BroadcastType.WORLD_EVENT,
            deliver_step=2,
            location_scope=None,
            severity=Severity.LOW,
        )
    )
    await runtime.run_step(agents)
    payload = published()[0]

    # Live path (from_runtime_payload) — same data the WS observer feeds the FE.
    live_event = StepEvent.from_runtime_payload(payload)
    assert live_event.actions and all(action.seq > 0 for action in live_event.actions)
    assert live_event.messages and all(msg.seq > 0 for msg in live_event.messages)
    assert live_event.broadcasts and all(b.seq > 0 for b in live_event.broadcasts)

    # Cross-channel ordering also visible through read-model dataclasses:
    # messages < broadcasts < actions (arrivals-before-actions).
    assert max(m.seq for m in live_event.messages) < min(b.seq for b in live_event.broadcasts)
    assert max(b.seq for b in live_event.broadcasts) < min(a.seq for a in live_event.actions)

    # Snapshot path (from_snapshot) — same seqs on replay.
    snapshot = await container.snapshot.load(world_id, 2)
    assert snapshot is not None
    replay_event = StepEvent.from_snapshot(snapshot)
    assert {a.seq for a in replay_event.actions} == {a.seq for a in live_event.actions}
    assert {m.seq for m in replay_event.messages} == {m.seq for m in live_event.messages}
    assert {b.seq for b in replay_event.broadcasts} == {b.seq for b in live_event.broadcasts}
