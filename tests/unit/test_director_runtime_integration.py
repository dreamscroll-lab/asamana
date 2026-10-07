"""End-to-end contract for a director intervention passing through the whole step loop.

Unit tests cover each piece: parse rejection, channel dispatch, applying mutations, assembling the
receipt. These tests check the properties that only hold once they run together in a real step:

- Injection happens at ``poll_event``, so a broadcast is perceived in the same step, not the next.
- Both authors' registers merge into one event stream at step end, so snapshots, replay and the
  frontend see a single card type.
- The receipt is assembled at step end, and its four signals (delivery / pressure / admission /
  interrupt) all come from values already computed this step.
- phenomenon is stored on the broadcast record in the snapshot, so replay can reproduce the effect.
"""

from __future__ import annotations

import json

import pytest

from core.interfaces.llm import LLMScene
from agent.personality import ActionStatus
from engine.broadcast import BroadcastChannel
from engine.clock import GlobalClock, WorldTimeConfig
from engine.director import NpcOnMenu
from engine.directory import LiveWorldDirectory
from engine.environment import EnvironmentSystem
from engine.event import EventSettings
from engine.executors import build_default_registry
from engine.message_system import MessageSystem
from engine.runtime import NarrativeRuntime
from core.interfaces.message import Message
from core.interfaces.urgency import Urgency
from engine.scheduler import AgentScheduler
from tests.unit.test_runtime_smoke import _SignalPressure, _build_agent, _set_work_decision
from worlds.tiled import TiledWorldConfig

WORLD_ID = "world-director"


def _directive_json(**overrides) -> str:
    payload = {
        "reason": "导演要在场上放一把火",
        "feasible": True,
        "refusal": "",
        "broadcast": None,
        "message": None,
        "mutations": [],
        "narrative_desc": "一场大火",
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def _npc_menu(environment):
    """Mirrors ``_director_inputs`` in production: put each NPC and its current location in the menu."""
    return [
        NpcOnMenu(npc=npc, location_id=environment.get_body_location(npc.npc_id))
        for npc in environment.all_npcs()
    ]


async def _submit(runtime, environment, agents, text: str):
    """Mirrors the orchestration layer in production: the channel doesn't own EnvironmentSystem, so
    the menus are passed in."""
    return await runtime.director.submit(
        text,
        all_agents={a.agent_id: a for a in agents},
        npcs=_npc_menu(environment),
        locations=environment.space.all_places(),
        entities=environment.all_live_entities(),
        world_time=runtime.clock.current,
    )


def _runtime(container):
    """A runtime with a human author and a cast the directory knows.

    Build the cast before the directory: a directory built from an empty map answers 「某人」 for
    everyone, so the receipt assertions would pass without testing name resolution.
    """
    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    agents = [
        _build_agent(container, world_id=WORLD_ID, agent_id="a1", name="李世民", is_main_character=True),
        _build_agent(container, world_id=WORLD_ID, agent_id="a2", name="李建成", is_main_character=False),
    ]
    location = next(iter(environment.snapshot_state()["location_names"]))
    for agent in agents:
        environment.place_agent(agent_id=agent.agent_id, location_id=location)
        agent.personality.update_location(step=0, location=location)

    message_system = MessageSystem(container.message_provider, world_id=WORLD_ID)
    broadcast_channel = BroadcastChannel()
    directory = LiveWorldDirectory.from_agents({a.agent_id: a for a in agents}, environment)
    runtime = NarrativeRuntime(
        world_id=WORLD_ID,
        clock=GlobalClock(WorldTimeConfig(start_hour=6, seconds_per_step=60)),
        scheduler=AgentScheduler(),
        environment=environment,
        message_system=message_system,
        event_settings=EventSettings(check_interval=2, max_events_per_window=1),
        snapshot_provider=container.snapshot,
        agent_store=container.agent_store,
        event_bus=container.event_bus,
        broadcast_channel=broadcast_channel,
        directory=directory,
        executor_registry=build_default_registry(container.llm_router, directory),
        pressure_evaluator=_SignalPressure(),
        llm_router=container.llm_router,
    )
    return runtime, environment, agents, location


@pytest.mark.asyncio
async def test_directive_lands_in_the_same_step_it_drains(container) -> None:
    """Injection happens at ``poll_event``, before broadcasts are collected, so it is perceived in
    the same step, not the next. This ordering is why the director sees results after one step."""
    _set_work_decision(container)
    runtime, environment, agents, _loc = _runtime(container)
    # Fire is a located phenomenon (see core/interfaces/phenomenon.py): without a location the
    # Broadcast boundary downgrades it to none and it can't reach the snapshot.
    container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json(
        narrative_desc="东市火起",
        broadcast={"content": "东市火起,浓烟蔽日。", "severity": "high",
                   "location_scope": 1, "phenomenon": "fire"},
    )
    result = await _submit(runtime, environment, agents, "东市起火")
    assert result.accepted is True

    await runtime.run_step(agents)

    snapshot = await container.snapshot.load(WORLD_ID, 1)
    broadcasts = snapshot.metadata["broadcasts"]
    assert [b["content"] for b in broadcasts] == ["东市火起,浓烟蔽日。"]
    assert broadcasts[0]["phenomenon"] == "fire"    # the renderer plays fire from this, and so does replay
    assert runtime.director.pending_count() == 0


@pytest.mark.asyncio
async def test_both_authors_share_one_event_stream_distinguished_by_hand(container) -> None:
    """For an observer, "what was injected into the world this step" is one question: one stream,
    one card type, with authored_by saying who wrote it rather than two schemas."""
    _set_work_decision(container)
    runtime, environment, agents, _loc = _runtime(container)
    container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json(
        broadcast={"content": "钟声大作。", "severity": "medium", "location_scope": None},
    )
    await _submit(runtime, environment, agents, "敲钟")

    await runtime.run_step(agents)

    snapshot = await container.snapshot.load(WORLD_ID, 1)
    assert [e["authored_by"] for e in snapshot.events_this_step] == ["director"]
    # Events carry a seq and share one emission-ordered timeline with actions, messages and broadcasts.
    assert isinstance(snapshot.events_this_step[0]["seq"], int)


@pytest.mark.asyncio
async def test_receipt_is_attached_at_step_end_with_real_signals(container) -> None:
    """All four receipt signals come from values computed this step. This checks that the receipt is
    assembled and accurate: the recipient got the message and, since _SignalPressure puts pressure on
    recipients, was actually pushed."""
    _set_work_decision(container)
    runtime, environment, agents, _loc = _runtime(container)
    container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json(
        narrative_desc="密信送到",
        message={"recipients": [2], "content": "速离此地。", "urgency": "high"},
    )
    await _submit(runtime, environment, agents, "给李建成送封急信")

    await runtime.run_step(agents)

    snapshot = await container.snapshot.load(WORLD_ID, 1)
    receipt = snapshot.events_this_step[0]["receipt"]
    assert receipt is not None
    assert receipt["delivered_to"] == [{"agent_id": "a2", "name": "李建成"}]
    # Delivered, and the pressure evaluator did put him under pressure. That is what separates an
    # effective intervention from one that went nowhere.
    assert [p["agent_id"] for p in receipt["pressure"]] == ["a2"]


@pytest.mark.asyncio
async def test_directive_killing_someone_is_reaped_by_the_normal_death_path(container) -> None:
    """A death caused by the director is handled by the normal death path in the same step, with no
    special case.

    This relies on injection happening after ``pre_step_active`` is sampled: death handling compares
    "alive at step start" with "dead at step end".
    """
    _set_work_decision(container)
    runtime, environment, agents, location = _runtime(container)
    container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json(
        narrative_desc="李建成暴毙",
        mutations=[{"kind": "vitality", "person": 2, "effect": "kill",
                    "observation": "李建成猝然倒地,气息全无。"}],
    )
    await _submit(runtime, environment, agents, "让李建成死")

    await runtime.run_step(agents)

    victim = agents[1]
    assert victim.is_active is False
    assert victim.death_cause == "李建成猝然倒地,气息全无。"
    # World-side death handling ran: he was removed from the environment and others can't see him.
    assert environment.get_body_location("a2") == "unknown"
    assert "a2" not in environment.bodies_at(location)


@pytest.mark.asyncio
async def test_someone_the_director_kills_mid_action_does_not_act_on_that_step(container) -> None:
    """The kill lands before anything acts, so the victim's ongoing work must not tick or finish
    on the step that killed him."""
    work = '{"selected_index": 3, "action_description": "批阅文书", "estimated_steps": 3}'
    container.llm_router.get(LLMScene.AGENT_DECISION_MAIN).fixed_response = work
    runtime, environment, agents, location = _runtime(container)
    await runtime.run_step(agents)
    victim = agents[1]
    assert victim.personality.state.action_status == ActionStatus.IN_PROGRESS
    container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json(
        narrative_desc="李建成暴毙",
        mutations=[{"kind": "vitality", "person": 2, "effect": "kill",
                    "observation": "李建成猝然倒地,气息全无。"}],
    )
    await _submit(runtime, environment, agents, "让李建成死")

    await runtime.run_step(agents)

    snap = await container.snapshot.load("world-director", 2)
    assert [a["phase"] for a in snap.actions_this_step
            if a["agent_id"] == "a2" and str(a.get("phase", "")).startswith("ongoing")] == []


@pytest.mark.asyncio
async def test_the_sentence_that_caused_an_intervention_is_kept_with_it(container) -> None:
    """An intervention record must keep the director's exact words, not just the engine's version.

    Without them, an intervention that moved nobody can't be diagnosed: was the directive badly
    worded, or does this world not respond to it? That is the director's only way to learn.
    """
    _set_work_decision(container)
    runtime, environment, agents, _loc = _runtime(container)
    container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json(
        narrative_desc="密信送到",
        message={"recipients": [2], "content": "速离此地。", "urgency": "high"},
    )
    await _submit(runtime, environment, agents, "让个不认识的人给李建成捎句话，叫他快走")

    await runtime.run_step(agents)

    snapshot = await container.snapshot.load(WORLD_ID, 1)
    assert snapshot.events_this_step[0]["directive_text"] == (
        "让个不认识的人给李建成捎句话，叫他快走"
    )


@pytest.mark.asyncio
async def test_a_message_from_the_director_makes_him_nobody_to_know(container) -> None:
    """A director's message has no one behind it, so the recipient forms no relation.

    The sender declares whether it can be related to (``sender_is_agent``). Don't let the recipient
    guess from the id: the director would get a relation, a message-list slot and a pressure line.
    """
    _set_work_decision(container)
    runtime, environment, agents, _loc = _runtime(container)
    container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json(
        narrative_desc="密信送到",
        message={"recipients": [2], "content": "速离此地。", "urgency": "high"},
    )
    await _submit(runtime, environment, agents, "给李建成送封急信")

    await runtime.run_step(agents)

    assert await container.agent_store.load_relation(WORLD_ID, "a2", "director") is None


@pytest.mark.asyncio
async def test_the_record_of_my_interventions_is_readable_without_watching_them(
    container,
) -> None:
    """The intervention history is an index read from snapshots, not a slice of the narrative feed
    (which holds only steps watched live), so it works even for a world never opened.
    """
    from interaction.replayer import Replayer

    _set_work_decision(container)
    runtime, environment, agents, _loc = _runtime(container)
    container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json(
        narrative_desc="密信送到",
        message={"recipients": [2], "content": "速离此地。", "urgency": "high"},
    )
    await _submit(runtime, environment, agents, "给李建成送封急信")
    await runtime.run_step(agents)

    records = await Replayer(WORLD_ID, container.snapshot).list_interventions()

    assert len(records) == 1
    # Each row pairs what was said with what the engine did, readable without going back to that
    # step. The receipt (who was moved) needs that step's context, so it stays on the feed card.
    assert records[0].directive_text == "给李建成送封急信"
    assert records[0].narrative == "密信送到"
    # Times use the world's own calendar, not iso_label: step numbers are code-layer coordinates and
    # don't belong in human-readable fields.
    assert records[0].time_label and "step=" not in records[0].time_label
    # Incremental read: steps already scanned aren't returned again, or every poll from the director
    # panel would rescan everything.
    assert await Replayer(WORLD_ID, container.snapshot).list_interventions(after=1) == []


@pytest.mark.asyncio
async def test_a_torn_down_action_and_a_fresh_decision_land_in_one_step(container) -> None:
    """A second way for one step to hold two deeds: the director tears down an in-flight execution
    (``_director.drain`` runs before the ``in_progress_at_step_start`` snapshot, so the body is
    already free), and the step both settles the torn-down action and runs a new decision. The
    review groups by step, so both appear in the same block, listed separately, neither overwriting
    the other.
    """
    from providers.trace.in_memory import InMemoryTraceSink
    from core.interfaces.trace import Stage
    from tuning.audit_reconstruct import StageCall, summarise

    sink = InMemoryTraceSink()
    container.llm_router._trace_sink = sink  # noqa: SLF001 — assertions inspect real calls
    runtime, environment, agents, location = _runtime(container)
    runtime._trace_sink = sink  # noqa: SLF001
    message_system = runtime._message_system  # noqa: SLF001
    mover = agents[0]

    container.llm_router.get(LLMScene.AGENT_DECISION_MAIN).fixed_response = (
        '{"selected_index": 3, "action_description": "闭门批阅文书", "estimated_steps": 6}'
    )
    await runtime.run_step(agents)

    places = environment.space.all_places()
    elsewhere = next(i for i, p in enumerate(places, 1) if p.place_id != location)
    container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = _directive_json(
        mutations=[{"kind": "relocate", "person": 1, "location": elsewhere,
                    "observation": "李世民忽然出现在别处。"}],
    )
    assert (await _submit(runtime, environment, agents, "把李世民挪到别处")).accepted is True
    # A freed body only makes a new decision possible; it still has to pass the cadence gate. A
    # message arriving this step wakes him through the same pressure path as production (see
    # _SignalPressure).
    await message_system.publish(Message(
        id="m-1", world_id=WORLD_ID, sender_id=agents[1].agent_id, content="速来议事",
        recipients=[mover.agent_id], location_scope=None,
        created_step=1, deliver_step=2, urgency=Urgency.CRITICAL,
    ))
    await runtime.run_step(agents)
    assert mover.personality.state.last_decision_step == 2, "这一步他确实跑了一轮新的认知循环"

    step2 = [c for c in sink.llm_calls if c.step == 2 and c.agent_id == mover.agent_id]
    s = summarise([StageCall(stage=c.stage, scene=c.scene, prompt=c.prompt_messages,
                             output={}, parse_ok=c.parse_ok, ok=c.ok, adopted=c.adopted,
                             reject_reason=c.reject_reason, extra=c.extra or {})
                   for c in step2], mover.personality.soul.name)
    assert s.get("decision"), "这一步他自己的新决策要在"
    dec = next(c for c in step2 if c.stage == Stage.DECISION.value)
    assert dec.extra["verdict"] == "executed"   # the verdict comes from arbitration, not inferred indirectly


