"""Tests for NarrativeRuntime._evaluate_interrupts (collect → decide → apply).

Covers signal aggregation into a single per-agent decision, Path 3 edge-triggering
of persistent external THREAT goals, single-interrupt semantics for a shared
multi-participant action, and concurrent execution of the decide pass.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from core.interfaces.llm import LLMScene

from agent.agent import Agent
from agent.decision import DecisionEngine
from agent.memory import MemorySystem
from agent.motivation import ExternalDriveType, ExternalGoal
from agent.need import NeedEngine
from agent.personality import PersonalityLayer, SoulLayer, activity_status_for
from agent.relation import RelationSystem
from core.interfaces.action import ActionTarget, ActionType, Ref
from core.interfaces.message import Message
from core.interfaces.perception import Broadcast, BroadcastType
from engine.broadcast import BroadcastChannel
from engine.clock import GlobalClock, WorldTimeConfig
from engine.directory import LiveWorldDirectory
from engine.environment import EnvironmentSystem
from engine.event import EventSettings
from engine.executors import build_default_registry
from engine.executors.base import ActionExecutionState
from engine.message_system import MessageDelivery, MessageSystem
from engine.runtime import NarrativeRuntime
from engine.scheduler import AgentScheduler
from core.interfaces.urgency import Urgency
from worlds.tiled import TiledWorldConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _always_interrupt(container) -> None:
    """Make the interrupt-decision LLM always rule "interrupt".

    Every agent weighs interrupts via the LLM, and the mock's default non-JSON reply counts as a
    failure → "don't interrupt", so without this the test silently tests nothing.
    """
    container.llm_router.get(LLMScene.AGENT_INTERRUPT_DECISION).fixed_response = (
        '{"thought": "此事更急，我得先放下手里的活", "interrupt": true}'
    )


def _build_agent(container, *, world_id: str, agent_id: str, is_main: bool) -> Agent:
    return Agent(
        world_id=world_id,
        agent_id=agent_id,
        personality=PersonalityLayer(
            soul=SoulLayer(
                name=agent_id,
                role="court_official",
                agent_id=agent_id,
                core_traits=["careful"],
                core_values=["order"],
                hard_constraints=[],
            )
        ),
        decision_engine=DecisionEngine(container.llm_router),
        llm_router=container.llm_router,
        memory_system=MemorySystem(
            container.llm_router,
            container.embedding,
            container.vector_store,
            world_id=world_id,
            agent_id=agent_id,
        ),
        need_engine=NeedEngine(),
        relation_system=RelationSystem(container.agent_store, world_id=world_id, agent_id=agent_id),
        agent_store=container.agent_store,
        is_main_character=is_main,
    )


def _build_runtime(container, world_id: str):
    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    directory = LiveWorldDirectory.from_agents({}, environment)
    registry = build_default_registry(container.llm_router, directory)
    message_system = MessageSystem(
        container.message_provider, world_id=world_id
    )
    broadcast_channel = BroadcastChannel()
    runtime = NarrativeRuntime(
        world_id=world_id,
        clock=GlobalClock(WorldTimeConfig(start_hour=6, seconds_per_step=60)),
        scheduler=AgentScheduler(),
        environment=environment,
        message_system=message_system,
        event_settings=EventSettings(check_interval=2, max_events_per_window=0),
        snapshot_provider=container.snapshot,
        agent_store=container.agent_store,
        event_bus=container.event_bus,
        broadcast_channel=broadcast_channel,
        directory=directory,
        executor_registry=registry,
        llm_router=container.llm_router,
    )
    return runtime, registry


def _start_rest(agents_list, registry, *, step: int = 1, estimated_steps: int = 8):
    """Register one REST execution state shared by all given agents and mark them
    IN_PROGRESS — mirrors a multi-step action in flight."""
    initiator = agents_list[0]
    state = ActionExecutionState.create(
        target=ActionTarget(),
        action_type=ActionType.REST,
        initiator_id=initiator.agent_id,
        participant_ids=[a.agent_id for a in agents_list],
        purpose="rest",
        started_step=step,
        opening_outcome="开始",
        estimated_steps=estimated_steps,
    )
    registry.add_active(state)
    for agent in agents_list:
        agent.personality.begin_action(
            step=step,
            description="rest",
            activity_status=activity_status_for(ActionType.REST),
            estimated_steps=estimated_steps,
        )
    return state


def _urgent_message(receiver_id: str, content: str) -> Message:
    """A targeted high-urgency message (recipients=[receiver_id])."""
    return Message(
        id=f"msg-{receiver_id}-{content}",
        world_id="world",
        sender_id="sender",
        content=content,
        recipients=[receiver_id],
        location_scope=None,
        created_step=1,
        deliver_step=1,
        urgency=Urgency.HIGH,
    )


def _delivery(step: int, *messages: Message) -> MessageDelivery:
    """Route messages into per-agent inboxes by recipients and build a MessageDelivery.

    Interrupt detection Path 1 reads each agent's real inbox_for() (receivers already resolved by
    the delivery layer), not all of delivered_messages, so tests must also put messages in the
    matching inbox or no interrupt fires.
    """
    inboxes: dict[str, list[Message]] = {}
    for message in messages:
        for receiver_id in (message.recipients or []):
            inboxes.setdefault(receiver_id, []).append(message)
    return MessageDelivery(step=step, delivered_messages=list(messages), inboxes=inboxes)


class _DecisionSpy:
    """Stand-in for Agent.evaluate_interrupt: records reasons, returns a fixed
    decision, and optionally drives a shared concurrency tracker."""

    def __init__(self, result: tuple[bool, str], tracker: "_ConcurrencyTracker | None" = None):
        self.result = result
        self.reasons: list[str] = []
        self._tracker = tracker

    async def __call__(self, *, step: int, reason: str, current_action_desc: str,
                        intent: str, progress_hint: str) -> tuple[bool, str]:
        self.reasons.append(reason)
        if self._tracker is not None:
            await self._tracker.enter()
        return self.result


class _ConcurrencyTracker:
    """Records the peak number of overlapping decide-pass coroutines."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def enter(self) -> None:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1


# ---------------------------------------------------------------------------
# Signal aggregation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_signals_aggregate_into_single_decision(container) -> None:
    runtime, registry = _build_runtime(container, "world-agg")
    agent = _build_agent(container, world_id="world-agg", agent_id="mc", is_main=True)
    _start_rest([agent], registry)
    spy = _DecisionSpy((False, ""))
    agent.evaluate_interrupt = spy  # type: ignore[method-assign]

    deliveries = _delivery(2, _urgent_message("mc", "城门失守"), _urgent_message("mc", "粮仓起火"))
    agent.pending_external_goals = [
        ExternalGoal(text="刺客逼近", source_id="scout", urgency=Urgency.CRITICAL,
                     drive_type=ExternalDriveType.THREAT),
    ]

    await runtime._interrupts.evaluate_interrupts(deliveries, [], {"mc": agent}, step=2)

    # Two urgent messages + one THREAT goal → exactly one interrupt decision.
    assert len(spy.reasons) == 1
    reason = spy.reasons[0]
    assert "城门失守" in reason
    assert "粮仓起火" in reason
    assert "刺客逼近" in reason
    # Channel labels: the interrupt judgment must see each signal's source, not bare content.
    assert "急讯" in reason          # message channel
    assert "外部压力" in reason       # external THREAT driver channel
    # No id leaks: neither sender_id="sender" nor source_id="scout" may appear in the reason fed to
    # the LLM.
    assert "sender" not in reason and "scout" not in reason


@pytest.mark.asyncio
async def test_passive_participant_action_desc_is_joiner_pov(container) -> None:
    """A conscripted TALK participant's interrupt prompt must NOT show the
    initiator's target-naming purpose ("与李世民交谈") — that reads as Li Shimin talking to
    himself. exec_state is shared (one, owned by the initiator); the passive participant
    gets a joiner-POV description built from the initiator's name."""
    runtime, registry = _build_runtime(container, "world-talk-vp")
    initiator = _build_agent(container, world_id="world-talk-vp", agent_id="a", is_main=False)
    joiner = _build_agent(container, world_id="world-talk-vp", agent_id="b", is_main=True)
    # Seed the directory so initiator's name resolves (soul.name == agent_id here).
    runtime._directory = LiveWorldDirectory.from_agents({"a": initiator, "b": joiner}, runtime._environment)

    exec_state = ActionExecutionState.create(
        target=ActionTarget(),
        action_type=ActionType.TALK,
        initiator_id="a",
        participant_ids=["a", "b"],
        purpose="与b当面交谈，探明局势",   # initiator a's view: names target b
        started_step=1,
        estimated_steps=4,
        opening_outcome="开始",
    )

    from engine.executors.base import participant_action_desc

    # Initiator sees their own framing verbatim.
    assert participant_action_desc(runtime._directory, "a", exec_state) == "与b当面交谈，探明局势"
    # Passive participant b must NOT get the bare self-referential purpose as "my action";
    # it's reframed as the initiator's action so b reads it as a's intent, not self-talk.
    joiner_desc = participant_action_desc(runtime._directory, "b", exec_state)
    assert joiner_desc != exec_state.purpose               # not the self-referential bare purpose
    assert joiner_desc.startswith("参与a发起的行动")        # framed as "a's action", with b as participant


@pytest.mark.asyncio
async def test_participant_intent_belongs_to_the_initiator_only(container) -> None:
    """``expected_outcome`` is what the initiator hoped for. The conscripted never wrote it, and
    treating it as his intention fabricates his motive, so his side returns an empty string and the
    caller omits the whole section."""
    exec_state = ActionExecutionState.create(
        target=ActionTarget(),
        action_type=ActionType.TALK,
        initiator_id="a",
        participant_ids=["a", "b"],
        purpose="与b当面交谈，探明局势",
        started_step=1,
        estimated_steps=4,
        opening_outcome="开始",
        expected_outcome="问出他昨夜的去处",
    )

    from engine.executors.base import participant_intent

    assert participant_intent("a", exec_state) == "我打算「问出他昨夜的去处」。"
    assert participant_intent("b", exec_state) == ""


@pytest.mark.asyncio
async def test_undelivered_broadcast_message_does_not_interrupt(container) -> None:
    """A high-urgency broadcast that resolved to nobody must NOT trigger an interrupt.

    With recipients=None and location_scope limited to the sender's location, the recipient set is
    empty: the message sits in delivered_messages but in no agent's inbox. Path 1 must read inboxes;
    reading delivered_messages and expanding recipients=None to all agents would bypass
    location_scope and interrupt an out-of-scope agent mid-action."""
    runtime, registry = _build_runtime(container, "world-undeliv")
    agent = _build_agent(container, world_id="world-undeliv", agent_id="mc", is_main=True)
    _start_rest([agent], registry)  # mid multi-step action → interruptible
    spy = _DecisionSpy((False, ""))
    agent.evaluate_interrupt = spy  # type: ignore[method-assign]

    # Broadcast (recipients=None) scoped elsewhere → delivery layer put it in NO inbox.
    broadcast_msg = Message(
        id="msg-undelivered",
        world_id="world",
        sender_id="sender",
        content="着尔即刻入太极宫偏殿见朕",
        recipients=None,
        location_scope="taiji_palace",
        created_step=1,
        deliver_step=1,
        urgency=Urgency.HIGH,
    )
    # inboxes empty for mc — exactly what deliver_for_agents produces for an out-of-scope broadcast.
    deliveries = MessageDelivery(step=2, delivered_messages=[broadcast_msg], inboxes={"mc": []})
    await runtime._interrupts.evaluate_interrupts(deliveries, [], {"mc": agent}, step=2)

    assert spy.reasons == []             # no interrupt decision was even asked
    assert registry.is_agent_active("mc")  # the ongoing action was not torn down


@pytest.mark.asyncio
async def test_apply_passes_interrupted_participant_not_trigger_source(container) -> None:
    """The executor must receive the interrupted participant id (the agent
    that perceived the signal and broke off), not the external trigger source. Feeding the
    trigger source would make social.interrupt's is_triggered ~always False (the sender/threat
    is never a participant), so the agent that actually broke off would lose the interrupt reason."""
    runtime, registry = _build_runtime(container, "world-trig")
    _always_interrupt(container)   # every agent weighs interrupts via LLM
    a = _build_agent(container, world_id="world-trig", agent_id="a", is_main=False)
    b = _build_agent(container, world_id="world-trig", agent_id="b", is_main=False)
    _start_rest([a, b], registry)  # one shared action, a + b
    agents = {"a": a, "b": b}

    captured: dict = {}
    executor = registry.get_executor(ActionType.REST)
    orig = executor.interrupt

    async def spy(*args, **kwargs):
        captured["interrupted_agent_id"] = kwargs.get("interrupted_agent_id")
        return await orig(*args, **kwargs)

    executor.interrupt = spy  # type: ignore[method-assign]

    # Urgent message to "a" from external sender "sender" (NOT a participant).
    deliveries = _delivery(2, _urgent_message("a", "城门失守"))
    await runtime._interrupts.evaluate_interrupts(deliveries, [], agents, step=2)

    # The receiver who broke off — never the external sender.
    assert captured["interrupted_agent_id"] == "a"


@pytest.mark.asyncio
async def test_broadcast_reason_is_labeled_as_environmental(container) -> None:
    """Path 2 broadcast (source=system, no sender) → reason labeled as a stir nearby, with no name
    and no leaked source."""
    runtime, registry = _build_runtime(container, "world-bc-label")
    agent = _build_agent(container, world_id="world-bc-label", agent_id="mc", is_main=True)
    _start_rest([agent], registry)
    spy = _DecisionSpy((False, ""))
    agent.evaluate_interrupt = spy  # type: ignore[method-assign]

    bc = Broadcast(content="宫殿坍塌", source="system", broadcast_type=BroadcastType.WORLD_EVENT,
                   deliver_step=2, location_scope=None, severity="high")
    await runtime._interrupts.evaluate_interrupts(MessageDelivery(step=2), [bc], {"mc": agent}, step=2)

    assert len(spy.reasons) == 1
    reason = spy.reasons[0]
    assert "宫殿坍塌" in reason and "广播" in reason
    assert "system" not in reason  # the code-layer source token never enters the narrative reason


# ---------------------------------------------------------------------------
# Path 3 edge-triggering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_path3_threat_is_edge_triggered(container) -> None:
    runtime, registry = _build_runtime(container, "world-edge")
    agent = _build_agent(container, world_id="world-edge", agent_id="mc", is_main=True)
    _start_rest([agent], registry)
    spy = _DecisionSpy((False, ""))  # decline so the action stays IN_PROGRESS
    agent.evaluate_interrupt = spy  # type: ignore[method-assign]
    agents = {"mc": agent}
    threat = ExternalGoal(text="叛军压境", source_id="rebels", urgency=Urgency.HIGH,
                          drive_type=ExternalDriveType.THREAT)

    # Step 1: newly appearing threat → evaluated.
    agent.pending_external_goals = [threat]
    await runtime._interrupts.evaluate_interrupts(MessageDelivery(step=1), [], agents, step=1)
    assert len(spy.reasons) == 1

    # Step 2: same threat persists → NOT re-evaluated.
    agent.pending_external_goals = [threat]
    await runtime._interrupts.evaluate_interrupts(MessageDelivery(step=2), [], agents, step=2)
    assert len(spy.reasons) == 1

    # Step 3: threat gone.
    agent.pending_external_goals = []
    await runtime._interrupts.evaluate_interrupts(MessageDelivery(step=3), [], agents, step=3)
    assert len(spy.reasons) == 1

    # Step 4: threat reappears → evaluated again.
    agent.pending_external_goals = [threat]
    await runtime._interrupts.evaluate_interrupts(MessageDelivery(step=4), [], agents, step=4)
    assert len(spy.reasons) == 2


# ---------------------------------------------------------------------------
# Shared multi-participant action — interrupted once
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_shared_action_interrupted_once(container) -> None:
    runtime, registry = _build_runtime(container, "world-shared")
    _always_interrupt(container)   # every agent weighs interrupts via LLM
    a = _build_agent(container, world_id="world-shared", agent_id="a", is_main=False)
    b = _build_agent(container, world_id="world-shared", agent_id="b", is_main=False)
    state = _start_rest([a, b], registry)
    agents = {"a": a, "b": b}

    executor = registry.get_executor(ActionType.REST)
    original_interrupt = executor.interrupt
    interrupt_calls: list[str] = []

    async def counting_interrupt(*args, **kwargs):
        interrupt_calls.append(args[0].execution_id)
        return await original_interrupt(*args, **kwargs)

    executor.interrupt = counting_interrupt  # type: ignore[method-assign]

    bc = Broadcast(
        content="宫殿坍塌",
        source="system",
        broadcast_type=BroadcastType.WORLD_EVENT,
        deliver_step=3,
        location_scope=None,
        severity="high",
    )
    await runtime._interrupts.evaluate_interrupts(MessageDelivery(step=3), [bc], agents, step=3)

    # Both participants are candidates, but the shared state is torn down only once.
    assert interrupt_calls == [state.execution_id]
    assert not registry.is_agent_active("a")
    assert not registry.is_agent_active("b")


@pytest.mark.asyncio
async def test_interrupt_apply_executor_failure_is_non_destructive(container) -> None:
    """executor.interrupt raising → doesn't kill the step: exec_state isn't torn down, no writeback,
    the action stays active for re-evaluation next step (Rule 1/5, same direction as decision
    failure → no interrupt)."""
    runtime, registry = _build_runtime(container, "world-int-fail")
    agent = _build_agent(container, world_id="world-int-fail", agent_id="mc", is_main=True)
    _start_rest([agent], registry)
    agent.evaluate_interrupt = _DecisionSpy((True, ""))  # type: ignore[method-assign]

    executor = registry.get_executor(ActionType.REST)

    async def boom(*args, **kwargs):
        raise RuntimeError("interrupt executor down")

    executor.interrupt = boom  # type: ignore[method-assign]

    # Must not raise — the step survives an executor.interrupt failure.
    await runtime._interrupts.evaluate_interrupts(
        _delivery(2, _urgent_message("mc", "城门失守")),
        [], {"mc": agent}, step=2,
    )

    # Conservatively NOT interrupted: the ongoing action stays active (no remove_active).
    assert registry.is_agent_active("mc")


# ---------------------------------------------------------------------------
# Observation/replay display record (phase="interrupt")
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_interrupt_emits_display_record(container) -> None:
    """A decided interrupt surfaces one terminal record (phase="interrupt") into the
    observer/replay action stream.

    It carries who cut it (structured ``interrupted_by``), why (quoted and attributed in
    ``outcome``, the god-view record that states why every action ended; not ``inner_monologue``,
    which is the thought behind the abandoned action), and ``execution_id`` so a cut-short JOINT
    action folds into ONE card.

    It must NOT carry ``reason``: that's the interrupter's raw perception-signal prompt input
    ("收到X的急讯：…"). Spliced into the outcome it puts a first-person cognition signal in a
    third-person channel and persists it into memory."""
    runtime, registry = _build_runtime(container, "world-int-rec")
    agent = _build_agent(container, world_id="world-int-rec", agent_id="mc", is_main=True)
    runtime._directory = LiveWorldDirectory.from_agents({"mc": agent}, runtime._environment)
    _start_rest([agent], registry)
    agent.evaluate_interrupt = _DecisionSpy((True, "我必须立刻动身"))  # type: ignore[method-assign]

    records = await runtime._interrupts.evaluate_interrupts(
        _delivery(2, _urgent_message("mc", "城门失守")),
        [], {"mc": agent}, step=2,
    )

    assert len(records) == 1
    rec = records[0]
    assert rec["phase"] == "interrupt"
    assert rec["agent_id"] == "mc"
    assert rec["succeeded"] is False
    # Who interrupted: a structured field, not guessed from prose.
    assert rec["interrupted_by"] == "mc"
    # Why: quoted and attributed, in the 3p authoritative outcome (like every outcome, it says why
    # this was the result).
    assert "我必须立刻动身" in rec["outcome"]
    # Never via inner_monologue: that's the thought behind doing the thing, not behind dropping it.
    assert not rec.get("inner_monologue")
    # An interrupted joint action is still one execution → the observer folds it into one card.
    assert rec["execution_id"]
    # The triggering signal (prompt input) never enters the narrative channel.
    assert "城门失守" not in rec["outcome"]
    assert "打断" in rec["outcome"]
    # Interrupts produce no bystander text (see ActionExecutor.interrupt): the scene is told by what
    # cut it short.
    assert rec["observations"] == []
    # The action is actually torn down (the state transition happened).
    assert not registry.is_agent_active("mc")

    # The record maps to a valid ActionSummary → it really reaches step_event / the snapshot display
    # layer.
    from interaction.models import _action_from_record

    summary = _action_from_record(rec)
    assert summary.phase == "interrupt"
    assert summary.agent_id == "mc"
    assert summary.succeeded is False
    assert summary.interrupted_by == "mc"
    assert "我必须立刻动身" in summary.outcome
    # The gist reaches the read model without the thought.
    assert "打断" in summary.gist and "我必须立刻动身" not in summary.gist
    assert "城门失守" not in summary.outcome  # trigger stays out of narrative


@pytest.mark.asyncio
async def test_no_interrupt_emits_no_records(container) -> None:
    """Declined decision and the no-candidate path both yield an empty record list —
    nothing spurious enters the action stream."""
    runtime, registry = _build_runtime(container, "world-int-none")
    agent = _build_agent(container, world_id="world-int-none", agent_id="mc", is_main=True)
    _start_rest([agent], registry)
    agent.evaluate_interrupt = _DecisionSpy((False, ""))  # decline

    declined = await runtime._interrupts.evaluate_interrupts(
        _delivery(2, _urgent_message("mc", "城门失守")),
        [], {"mc": agent}, step=2,
    )
    assert declined == []
    assert registry.is_agent_active("mc")  # nothing torn down

    # No qualifying signal at all → no candidates → empty.
    none = await runtime._interrupts.evaluate_interrupts(MessageDelivery(step=3), [], {"mc": agent}, step=3)
    assert none == []


@pytest.mark.asyncio
async def test_shared_interrupt_emits_record_per_participant(container) -> None:
    """A shared multi-participant action is torn down once, but each participant gets its
    own terminal record (one per executor.interrupt result) — so the timeline shows every
    party's break-off, not just the initiator's."""
    runtime, registry = _build_runtime(container, "world-int-shared")
    a = _build_agent(container, world_id="world-int-shared", agent_id="a", is_main=False)
    b = _build_agent(container, world_id="world-int-shared", agent_id="b", is_main=False)
    runtime._directory = LiveWorldDirectory.from_agents({"a": a, "b": b}, runtime._environment)
    _start_rest([a, b], registry)
    a.evaluate_interrupt = _DecisionSpy((True, ""))  # type: ignore[method-assign]
    b.evaluate_interrupt = _DecisionSpy((True, ""))  # type: ignore[method-assign]

    bc = Broadcast(content="宫殿坍塌", source="system", broadcast_type=BroadcastType.WORLD_EVENT,
                   deliver_step=3, location_scope=None, severity="high")
    records = await runtime._interrupts.evaluate_interrupts(
        MessageDelivery(step=3), [bc], {"a": a, "b": b}, step=3
    )

    assert len(records) == 2
    assert {r["agent_id"] for r in records} == {"a", "b"}
    assert all(r["phase"] == "interrupt" for r in records)
    assert not registry.is_agent_active("a")
    assert not registry.is_agent_active("b")


# ---------------------------------------------------------------------------
# Concurrent decide pass
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_decide_pass_runs_concurrently(container) -> None:
    runtime, registry = _build_runtime(container, "world-conc")
    tracker = _ConcurrencyTracker()
    agents: dict[str, Agent] = {}
    for i in range(3):
        agent_id = f"mc{i}"
        agent = _build_agent(container, world_id="world-conc", agent_id=agent_id, is_main=True)
        _start_rest([agent], registry)
        agent.evaluate_interrupt = _DecisionSpy((False, ""), tracker=tracker)  # type: ignore[method-assign]
        agents[agent_id] = agent

    deliveries = _delivery(5, *[_urgent_message(agent_id, f"信号-{agent_id}") for agent_id in agents])
    await runtime._interrupts.evaluate_interrupts(deliveries, [], agents, step=5)

    # All three per-agent decisions overlap — serial execution would peak at 1.
    assert tracker.peak == 3


# ---------------------------------------------------------------------------
# Two conscription modes: invitation vs. compulsion
# ---------------------------------------------------------------------------

def _register_rooms(env, node_ids: list[str]) -> None:
    """Actually register a chain of connected locations in the spatial graph.

    place_agent alone isn't enough: it only writes the ``_locations`` map with no graph nodes, so
    the feasibility check fails early with "no path", and the test passes before any admission
    criterion is touched.
    """
    from core.interfaces.place import Place

    for i, nid in enumerate(node_ids):
        conn = {}
        if i > 0:
            conn[node_ids[i - 1]] = 1
        if i < len(node_ids) - 1:
            conn[node_ids[i + 1]] = 1
        env.space.register_place(Place(
            place_id=nid, name=nid, description="",
            connections=conn, is_public=True, capacity=50,
        ))


def _plan_for(agent, env, *, step: int, action):
    from agent.agent import AgentStepPlan
    from agent.need import NeedEvaluation, NeedType

    return AgentStepPlan(
        agent_id=agent.agent_id, step=step,
        spatial=env.spatial_for(agent_id=agent.agent_id, step=step),
        inbox=[], broadcasts=[],
        need_evaluation=NeedEvaluation(
            dominant_need=NeedType.SAFETY, scores={}, active_needs=[],
            short_term_goals=[], long_term_goals=[], prompt_context="",
        ),
        action=action,
    )


def _two_agents_in(container, env, *, main: str, other: str, at: str = "n0"):
    agents = {
        aid: _build_agent(container, world_id="world", agent_id=aid, is_main=(aid == main))
        for aid in (main, other)
    }
    for aid in agents:
        env.place_agent(agent_id=aid, location_id=at)
    return agents


@pytest.mark.asyncio
async def test_invite_gives_up_when_the_body_is_already_committed(container) -> None:
    """Invitation: the target is busy with an earlier action → the whole action is rejected, and
    the initiator is foiled this step.

    "He's busy with something else" is a valid world fact, not a failure; it's how TALK always
    behaves, and exactly what COMPEL overrides.
    """
    from core.interfaces.action import AgentAction

    runtime, registry = _build_runtime(container, "world")
    env = runtime._environment  # noqa: SLF001
    _register_rooms(env, ["n0", "n1"])
    agents = _two_agents_in(container, env, main="talker", other="busy")
    _start_rest([agents["busy"]], registry, step=1, estimated_steps=8)   # the earlier commitment

    talk = AgentAction(
        agent_id="talker", step=2, action_type=ActionType.TALK,
        action_description="找他谈", estimated_steps=3,
        target=ActionTarget(acts_on=[Ref.agent("busy")], claims=[Ref.agent("busy")]),
    )
    plans = [
        ("main", agents["talker"], _plan_for(agents["talker"], env, step=2, action=talk)),
        ("background", agents["busy"], _plan_for(agents["busy"], env, step=2, action=None)),
    ]

    arb, seized = await runtime._arbiter.arbitrate(  # noqa: SLF001
        plans, agents, 2, {"busy"},
    )

    assert seized == [], "邀请制不拆任何人的行动"
    stub = registry.get_active_for_agent("talker")
    assert stub is not None and stub.extra.get("completed_result") is not None
    assert "正在歇息" in stub.extra["completed_result"].failure_reason   # what he visibly is doing
    assert registry.get_active_for_agent("busy").action_type == ActionType.REST, "他手上的事没被动"


@pytest.mark.asyncio
async def test_compel_takes_the_body_out_of_whatever_it_was_doing(container) -> None:
    """Compulsion: what he was doing is forcibly cut short, then he joins as usual; he has no chance
    to refuse.

    On this side "he's busy" is no reason to refuse; that's the only point where the two modes
    differ. The one seized remembers it (teardown_with_writeback) and leaves a display record,
    or his action would vanish from the stream.
    """
    from core.interfaces.action import AgentAction

    runtime, registry = _build_runtime(container, "world")
    env = runtime._environment  # noqa: SLF001
    _register_rooms(env, ["n0", "n1"])
    agents = _two_agents_in(container, env, main="carrier", other="busy")
    _start_rest([agents["busy"]], registry, step=1, estimated_steps=8)

    carry = AgentAction(
        agent_id="carrier", step=2, action_type=ActionType.MOVE,
        action_description="拽着他走", estimated_steps=1,
        target=ActionTarget(acts_on=[Ref.place("n1")], claims=[Ref.agent("busy")]),
    )
    plans = [
        ("main", agents["carrier"], _plan_for(agents["carrier"], env, step=2, action=carry)),
        ("background", agents["busy"], _plan_for(agents["busy"], env, step=2, action=None)),
    ]

    arb, seized = await runtime._arbiter.arbitrate(  # noqa: SLF001
        plans, agents, 2, {"busy"},
    )

    move = registry.get_active_for_agent("carrier")
    assert move is not None and move.action_type == ActionType.MOVE
    assert "busy" in move.participant_ids, "他被收编进这趟行程"
    assert arb["busy"].is_passive_join
    # His earlier action is gone, and he got the writeback himself (the record is only produced on
    # the writeback branch).
    assert [r["agent_id"] for r in seized] == ["busy"]
    assert seized[0]["phase"] == "interrupt"


@pytest.mark.asyncio
async def test_compel_also_dissolves_a_third_partys_execution(container) -> None:
    """If the one seized is caught up in a third party's execution, that execution is torn down too:
    drag one side out of a conversation and it ends."""
    from core.interfaces.action import AgentAction

    runtime, registry = _build_runtime(container, "world")
    env = runtime._environment  # noqa: SLF001
    _register_rooms(env, ["n0", "n1"])
    agents = {
        aid: _build_agent(container, world_id="world", agent_id=aid, is_main=(aid == "carrier"))
        for aid in ("carrier", "victim", "partner")
    }
    for aid in agents:
        env.place_agent(agent_id=aid, location_id="n0")
    # victim and partner share a multi-step execution.
    _start_rest([agents["victim"], agents["partner"]], registry, step=1, estimated_steps=8)

    carry = AgentAction(
        agent_id="carrier", step=2, action_type=ActionType.MOVE,
        action_description="拽着他走", estimated_steps=1,
        target=ActionTarget(acts_on=[Ref.place("n1")], claims=[Ref.agent("victim")]),
    )
    plans = [
        ("main", agents["carrier"], _plan_for(agents["carrier"], env, step=2, action=carry)),
        ("background", agents["victim"], _plan_for(agents["victim"], env, step=2, action=None)),
        ("background", agents["partner"], _plan_for(agents["partner"], env, step=2, action=None)),
    ]

    _arb, seized = await runtime._arbiter.arbitrate(  # noqa: SLF001
        plans, agents, 2, {"victim", "partner"},
    )

    assert "victim" in registry.get_active_for_agent("carrier").participant_ids
    assert registry.get_active_for_agent("partner") is None, "留下的那一方也脱离了那个执行体"
    # One display record per participant, not just the one seized. The one left behind has their
    # action end here too; recording only the seized one leaves the collaboration in the observer
    # stream with an opening marker and no ending.
    assert {r["agent_id"] for r in seized} == {"victim", "partner"}
    assert all(r["phase"] == "interrupt" for r in seized)


def _same_step_compel_world(container, *, listener_action=None, with_x=False):
    """A (talker) starts a conversation with B (listener) this step; the seizer (carrier) comes
    later and drags one of them off.

    Within a tier COMPEL goes before INVITE, so the test puts the seizer in background: only a lower
    tier makes it act after the conversation.
    """
    runtime, registry = _build_runtime(container, "world")
    env = runtime._environment  # noqa: SLF001
    _register_rooms(env, ["n0", "n1"])
    ids = ["talker", "listener", "carrier"] + (["x"] if with_x else [])
    agents = {
        aid: _build_agent(container, world_id="world", agent_id=aid, is_main=True) for aid in ids
    }
    for aid in agents:
        env.place_agent(agent_id=aid, location_id="n0")
    return runtime, registry, env, agents


def _talk(actor: str, other: str):
    from core.interfaces.action import AgentAction

    return AgentAction(
        agent_id=actor, step=2, action_type=ActionType.TALK, action_description=f"{actor}找{other}议事",
        estimated_steps=3, target=ActionTarget(acts_on=[Ref.agent(other)], claims=[Ref.agent(other)]),
    )


def _carry(seized: str):
    from core.interfaces.action import AgentAction

    return AgentAction(
        agent_id="carrier", step=2, action_type=ActionType.MOVE, action_description="拽着他走",
        estimated_steps=1, target=ActionTarget(acts_on=[Ref.place("n1")], claims=[Ref.agent(seized)]),
    )


def _no_talk_that_never_happened(agent) -> None:
    """A conversation where nothing was said must not leave "interrupted" in anyone's memory."""
    for m in agent.memory_system._entries.values():  # noqa: SLF001
        assert "中断" not in m.stored_content and "中止" not in m.stored_content, m.stored_content


@pytest.mark.asyncio
async def test_same_step_compel_leaves_the_talker_a_refusal_not_a_dangling_talk(container, caplog) -> None:
    """The invitee was dragged away: the initiator acted and didn't succeed; record a failed
    attempt, not something hanging on a dismantled conversation."""
    from agent.personality import ActionStatus

    runtime, registry, env, agents = _same_step_compel_world(container)
    plans = [
        ("main", agents["talker"], _plan_for(agents["talker"], env, step=2, action=_talk("talker", "listener"))),
        ("main", agents["listener"], _plan_for(agents["listener"], env, step=2, action=None)),
        ("background", agents["carrier"], _plan_for(agents["carrier"], env, step=2, action=_carry("listener"))),
    ]
    arb, seized = await runtime._arbiter.arbitrate(plans, agents, 2, set())  # noqa: SLF001
    await runtime._commit_execution(plans, arb, agents)  # noqa: SLF001

    assert "listener" in registry.get_active_for_agent("carrier").participant_ids
    refusal = registry.get_active_for_agent("talker")
    assert refusal is not None and refusal.extra["completed_result"].not_executed
    assert "强行拉走未能如愿" in refusal.extra["completed_result"].outcome   # empty roster: names are "某人"
    assert seized == []
    _no_talk_that_never_happened(agents["talker"])
    assert not any(r.message == "stranded_participant_released" for r in caplog.records)
    assert agents["talker"].personality.state.action_status == ActionStatus.IN_PROGRESS  # settled now


@pytest.mark.parametrize("own", [None, "work"])
@pytest.mark.asyncio
async def test_same_step_compel_frees_a_listener_whose_turn_has_not_come(container, caplog, own) -> None:
    """The initiator was dragged off before the invitee's turn: he's free and acts on his own
    decision when his turn comes (idle if none)."""
    from agent.personality import ActionStatus
    from core.interfaces.action import AgentAction

    runtime, registry, env, agents = _same_step_compel_world(container)
    work = AgentAction(agent_id="listener", step=2, action_type=ActionType.WORK,
                       action_description="理账", estimated_steps=2) if own else None
    plans = [
        ("main", agents["talker"], _plan_for(agents["talker"], env, step=2, action=_talk("talker", "listener"))),
        ("background", agents["carrier"], _plan_for(agents["carrier"], env, step=2, action=_carry("talker"))),
        ("background", agents["listener"], _plan_for(agents["listener"], env, step=2, action=work)),
    ]
    caplog.set_level(logging.INFO, logger="engine.arbiter")
    arb, _ = await runtime._arbiter.arbitrate(plans, agents, 2, set())  # noqa: SLF001
    assert any(r.message == "admission_revoked" for r in caplog.records)
    await runtime._commit_execution(plans, arb, agents)  # noqa: SLF001

    assert "talker" in registry.get_active_for_agent("carrier").participant_ids
    mine = registry.get_active_for_agent("listener")
    if own:
        assert mine is not None and mine.action_type == ActionType.WORK
        assert not arb["listener"].is_passive_join
    else:
        assert mine is None and "listener" not in arb
        assert agents["listener"].personality.state.action_status == ActionStatus.IDLE
    _no_talk_that_never_happened(agents["listener"])
    assert not any(r.message == "stranded_participant_released" for r in caplog.records)


@pytest.mark.asyncio
async def test_same_step_compel_gives_a_listener_whose_turn_has_passed_his_own_turn_back(container, caplog) -> None:
    """The invitee already had his turn and was recorded as joining; then the conversation was
    retracted. Nobody had acted, so his step wasn't spent: back of the queue, acting on his own
    decision, with no join recorded and no intent lost."""
    runtime, registry, env, agents = _same_step_compel_world(container, with_x=True)
    kept: list = []
    agents["listener"].defer_decided_intent = lambda action, step: kept.append(action)  # type: ignore[method-assign]
    plans = [
        ("main", agents["talker"], _plan_for(agents["talker"], env, step=2, action=_talk("talker", "listener"))),
        ("main", agents["listener"], _plan_for(agents["listener"], env, step=2, action=_talk("listener", "x"))),
        ("main", agents["x"], _plan_for(agents["x"], env, step=2, action=None)),
        ("background", agents["carrier"], _plan_for(agents["carrier"], env, step=2, action=_carry("talker"))),
    ]
    caplog.set_level(logging.INFO, logger="engine.arbiter")
    arb, _ = await runtime._arbiter.arbitrate(plans, agents, 2, set())  # noqa: SLF001
    assert any(r.message == "admission_revoked" for r in caplog.records)
    await runtime._commit_execution(plans, arb, agents)  # noqa: SLF001

    mine = registry.get_active_for_agent("listener")
    assert mine is not None and mine.action_type == ActionType.TALK
    assert set(mine.participant_ids) == {"listener", "x"}
    assert not arb["listener"].is_passive_join and arb["x"].is_passive_join
    assert kept == []
    _no_talk_that_never_happened(agents["listener"])
    assert not any(r.message == "stranded_participant_released" for r in caplog.records)


def _move_carrying(actor: str, carried: str, to: str = "n1"):
    from core.interfaces.action import AgentAction

    return AgentAction(
        agent_id=actor, step=2, action_type=ActionType.MOVE, action_description="拽着他走",
        estimated_steps=1, target=ActionTarget(acts_on=[Ref.place(to)], claims=[Ref.agent(carried)]),
    )


@pytest.mark.asyncio
async def test_same_step_compel_revokes_a_carry_that_never_set_out(container, caplog) -> None:
    """A means to drag off B this step, then C drags off A: A's attempt never got under way; B stays put,
    free, nobody hanging on a dead execution."""
    from agent.personality import ActionStatus

    runtime, registry = _build_runtime(container, "world")
    env = runtime._environment  # noqa: SLF001
    _register_rooms(env, ["n0", "n1"])
    agents = {aid: _build_agent(container, world_id="world", agent_id=aid, is_main=True) for aid in ("a", "b", "c")}
    for aid in agents:
        env.place_agent(agent_id=aid, location_id="n0")
    plans = [
        ("main", agents["a"], _plan_for(agents["a"], env, step=2, action=_move_carrying("a", "b"))),
        ("main", agents["b"], _plan_for(agents["b"], env, step=2, action=None)),
        ("main", agents["c"], _plan_for(agents["c"], env, step=2, action=_move_carrying("c", "a"))),
    ]
    arb, seized = await runtime._arbiter.arbitrate(plans, agents, 2, set())  # noqa: SLF001
    await runtime._commit_execution(plans, arb, agents)  # noqa: SLF001

    assert set(registry.get_active_for_agent("c").participant_ids) == {"c", "a"}
    assert arb["a"].is_passive_join
    assert registry.get_active_for_agent("b") is None and "b" not in arb
    assert env.get_body_location("b") == "n0"
    assert agents["b"].personality.state.action_status == ActionStatus.IDLE
    assert seized == []
    assert not any(r.message == "stranded_participant_released" for r in caplog.records)


@pytest.mark.asyncio
async def test_a_revoked_seizure_leaves_the_old_action_it_meant_to_break_untouched(container) -> None:
    """C means to drag off B, who is resting with D, then E drags off C: C's attempt never happened,
    so B's action isn't torn down and continues."""
    runtime, registry = _build_runtime(container, "world")
    env = runtime._environment  # noqa: SLF001
    _register_rooms(env, ["n0", "n1"])
    ids = ("b", "d", "c", "e")
    agents = {aid: _build_agent(container, world_id="world", agent_id=aid, is_main=True) for aid in ids}
    for aid in agents:
        env.place_agent(agent_id=aid, location_id="n0")
    _start_rest([agents["b"], agents["d"]], registry, step=1, estimated_steps=8)
    old = registry.get_active_for_agent("b")
    plans = [
        ("main", agents["c"], _plan_for(agents["c"], env, step=2, action=_move_carrying("c", "b"))),
        ("main", agents["e"], _plan_for(agents["e"], env, step=2, action=_move_carrying("e", "c"))),
        ("main", agents["b"], _plan_for(agents["b"], env, step=2, action=None)),
        ("main", agents["d"], _plan_for(agents["d"], env, step=2, action=None)),
    ]
    arb, seized = await runtime._arbiter.arbitrate(plans, agents, 2, {"b", "d"})  # noqa: SLF001

    assert seized == []
    assert registry.get_active_for_agent("b") is old and registry.get_active_for_agent("d") is old
    assert set(registry.get_active_for_agent("e").participant_ids) == {"e", "c"}
    assert "b" not in arb and "d" not in arb


@pytest.mark.parametrize("seized", ["listener", "talker"])
@pytest.mark.asyncio
async def test_a_body_freed_by_a_revoked_admission_can_still_be_enrolled(container, caplog, seized) -> None:
    """A failed conscription holds nobody: whoever's left after the retraction (the rejected
    initiator or the released invitee) can still be engaged by those who come later."""
    runtime, registry, env, agents = _same_step_compel_world(container, with_x=True)
    left = "talker" if seized == "listener" else "listener"
    plans = [
        ("main", agents["talker"], _plan_for(agents["talker"], env, step=2, action=_talk("talker", "listener"))),
        ("background", agents["carrier"], _plan_for(agents["carrier"], env, step=2, action=_carry(seized))),
        ("background", agents["x"], _plan_for(agents["x"], env, step=2, action=_talk("x", left))),
        ("background", agents["listener"], _plan_for(agents["listener"], env, step=2, action=None)),
    ]
    caplog.set_level(logging.INFO, logger="engine.arbiter")
    arb, _ = await runtime._arbiter.arbitrate(plans, agents, 2, set())  # noqa: SLF001
    assert any(r.message == "admission_revoked" for r in caplog.records)

    talk = registry.get_active_for_agent("x")
    assert talk is not None and set(talk.participant_ids) == {"x", left}
    assert arb[left].is_passive_join and arb[left].ongoing_execution_id == talk.execution_id
    assert "未能如愿" not in arb[left].action_result.outcome


@pytest.mark.parametrize("seized", ["talker", "listener"])
@pytest.mark.asyncio
async def test_a_refusal_never_cites_an_admission_that_was_revoked(container, seized) -> None:
    """x is rejected because "the invitee is talking with the initiator", then that conversation is
    retracted: it never happened, so the reason must not remain (it would enter x's first-person
    memory and get embedded). The released invitee can still be engaged by x; for one dragged away,
    x's reason is what he is really doing now."""
    runtime, registry, env, agents = _same_step_compel_world(container, with_x=True)
    plans = [
        ("main", agents["talker"], _plan_for(agents["talker"], env, step=2, action=_talk("talker", "listener"))),
        ("main", agents["x"], _plan_for(agents["x"], env, step=2, action=_talk("x", "listener"))),
        ("background", agents["carrier"], _plan_for(agents["carrier"], env, step=2, action=_carry(seized))),
        ("background", agents["listener"], _plan_for(agents["listener"], env, step=2, action=None)),
    ]
    arb, _ = await runtime._arbiter.arbitrate(plans, agents, 2, set())  # noqa: SLF001

    mine = registry.get_active_for_agent("x")
    if seized == "talker":
        assert set(mine.participant_ids) == {"x", "listener"}
        assert arb["listener"].is_passive_join
    else:
        refusal = mine.extra["completed_result"]
        assert refusal.not_executed
        assert "交谈" not in refusal.outcome and "交谈" not in refusal.factual_memory


@pytest.mark.asyncio
async def test_commit_releases_a_body_whose_execution_is_already_gone(container, caplog) -> None:
    """Invariant guard: arbitration gave him an execution, but registry has no such execution at
    commit → don't attach him; reset and log an error."""
    from agent.personality import ActionStatus
    from engine.arbiter import ArbitratedAction
    from core.interfaces.action import ActionResult

    runtime, registry, env, agents = _same_step_compel_world(container)
    plan = _plan_for(agents["talker"], env, step=2, action=_talk("talker", "listener"))
    dangling = ArbitratedAction(
        agent_id="talker", location_id="n0", ongoing_execution_id="talk_gone",
        action_result=ActionResult(action=plan.action, expected_outcome="", outcome="x", succeeded=True),
    )
    await runtime._commit_execution([("main", agents["talker"], plan)], {"talker": dangling}, agents)  # noqa: SLF001

    assert agents["talker"].personality.state.action_status == ActionStatus.IDLE
    assert any(r.message == "stranded_participant_released" for r in caplog.records)


@pytest.mark.asyncio
async def test_a_body_in_an_ongoing_action_still_gets_a_plan_entry(container) -> None:
    """Bodies mid-action need plan entries too, or compulsory conscription can't find them.

    Arbitration judges anyone without an entry "no response", so COMPEL ("take him even if he's
    busy") would only work on the idle.
    """
    runtime, registry = _build_runtime(container, "world")
    env = runtime._environment  # noqa: SLF001
    _register_rooms(env, ["n0"])
    agents = _two_agents_in(container, env, main="a", other="busy")
    _start_rest([agents["busy"]], registry, step=1, estimated_steps=8)

    from engine.scheduler import StepExecutionPlan

    planned = await runtime._plan_execution(  # noqa: SLF001
        plan=StepExecutionPlan(step=2),          # nobody scheduled → all idle-body entries
        agents=list(agents.values()),
        in_progress_at_step_start={"busy"},
        deliveries=MessageDelivery({}, []),
        step=2, world_time_label="辰时", broadcasts=[],
        agent_spatials={aid: env.spatial_for(agent_id=aid, step=2) for aid in agents},
    )

    assert "busy" in {plan.agent_id for _phase, _agent, plan in planned}
