"""Unit tests for the engine event system (a pure narrative channel adapter).

Contract (see the engine/event.py module docstring):
- Scope: only pacing control + channel adaptation; never mutates agents / world
- No agent mutation (no personality.* / need_engine.* / memory_system.* calls)
- Data-driven: no EventCategory / AgentEventType / GlobalEventType enums
- The LLM picks the channels (broadcast / message fields in its JSON output)
- Multiple channels may be used together
- Private effects: world mutations go through engine/world_mutation.py; EventSystem may only touch unheld
  entities on the ground
- Agents react autonomously
- Communication goes only through BroadcastChannel / MessageSystem
"""

from __future__ import annotations

import asyncio
import gc
import json
from collections.abc import Sequence

import pytest

from agent.agent import Agent
from agent.decision import DecisionEngine
from agent.memory import MemorySystem
from agent.need import NeedEngine
from agent.personality import PersonalityLayer, SoulLayer
from agent.relation import RelationSystem
from core.interfaces.llm import LLMMessage, LLMResponse, LLMScene
from core.interfaces.perception import Broadcast, BroadcastType
from core.prompts import PHENOMENON_DEFINITION
from engine.broadcast import BroadcastChannel
from engine.clock import GlobalClock, WorldTime, WorldTimeConfig
from engine.directory import LiveWorldDirectory
from engine.environment import EnvironmentSystem
from engine.event import EventSettings, EventSystem
from engine.execution_processor import ExecutionProcessor
from engine.executors.registry import ActionExecutorRegistry
from engine.injection import Author, InjectionDispatcher, WorldEvent, serialize_world_event
from engine.message_system import MessageSystem
from engine.world_mutation import WorldMutationChannel
from providers.message.in_memory import InMemoryMessageProvider
from core.interfaces.urgency import Urgency


def _gate(inject: bool = True, reason: str = "节奏需要") -> str:
    """Stub response for the reason-first pacing gate."""
    return json.dumps({"reason": reason, "inject": inject})


def _joined(messages) -> str:
    """Join all messages of one LLM call into one string, for assertions that don't care whether a
    phrase sits in the system or the user part of the prefix-cache split.
    """
    return "\n".join(getattr(m, "content", "") for m in messages)


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────


class _EventRouter:
    """LLM router stub. Returns responses in call order; once exhausted, returns a "don't inject" gate
    so nothing raises."""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[LLMScene, list[LLMMessage]]] = []

    async def complete(
        self,
        scene: LLMScene,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        self.calls.append((scene, list(messages)))
        return LLMResponse(
            content=next(self._responses, _gate(False)),
            input_tokens=0,
            output_tokens=0,
            model="event-test",
        )


def _world_time(step: int = 1) -> WorldTime:
    return WorldTime.from_step(step, WorldTimeConfig(start_hour=6, seconds_per_step=60))


def _build_agent(
    container, *, world_id: str, agent_id: str, name: str, background: str = "", life_goal: str = "",
) -> Agent:
    return Agent(
        world_id=world_id,
        agent_id=agent_id,
        personality=PersonalityLayer(
            soul=SoulLayer(
                name=name,
                role="court_official",
                agent_id=agent_id,
                core_traits=["careful"],
                core_values=["order"],
                background=background,
                life_goal=life_goal,
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
    )


def _event_system(
    container,
    *,
    router: _EventRouter | None = None,
    world_id: str = "world-1",
    broadcast_channel: BroadcastChannel | None = None,
    directory: LiveWorldDirectory | None = None,
    agents: dict | None = None,
    environment: EnvironmentSystem | None = None,
    **kwargs,
) -> EventSystem:
    """``environment`` is the world mutations land in; tests that check only dispatch use an empty one."""
    environment = environment or EnvironmentSystem()
    message_system = MessageSystem(container.message_provider, world_id=world_id)
    if directory is None:
        directory = LiveWorldDirectory.from_agents(agents or {}, environment)
    return EventSystem(
        llm_router=router or _EventRouter([]),
        snapshot_provider=container.snapshot,
        dispatcher=InjectionDispatcher(
            broadcast_channel=broadcast_channel or BroadcastChannel(),
            message_system=message_system,
            mutation_channel=_mutation_channel(environment),
            directory=directory,
        ),
        directory=directory,
        settings=EventSettings(
            core_tension=kwargs.get("core_tension", "皇位之争,手足相残"),
            narrative_theme=kwargs.get("narrative_theme", "权力与亲情的撕扯"),
            enabled=kwargs.get("enabled", True),
            max_events_per_window=kwargs.get("max_events_per_window", 2),
            quota_window=kwargs.get("quota_window", 20),
            check_interval=kwargs.get("check_interval", 6),
        ),
    )


def _messages(system: EventSystem) -> MessageSystem:
    """Where the editor's directed messages land: the dispatcher's MessageSystem."""
    return system._dispatcher._message_system


def _mutation_channel(environment: EnvironmentSystem | None = None) -> WorldMutationChannel:
    environment = environment or EnvironmentSystem()
    return WorldMutationChannel(
        environment=environment,
        seconds_per_step=3600,
        processor=ExecutionProcessor(
            executor_registry=ActionExecutorRegistry(),
            environment=environment,
            message_system=MessageSystem(InMemoryMessageProvider(), world_id="world-1"),
            directory=LiveWorldDirectory.from_agents({}, environment),
        ),
    )


def _plan_json(
    *,
    narrative_desc: str = "测试事件",
    is_positive: bool | None = None,
    broadcast: dict | None = None,
    message: dict | None = None,
    **things: dict | None,
) -> str:
    return json.dumps(
        {
            "narrative_desc": narrative_desc,
            "is_positive": is_positive,
            **things,
            "broadcast": broadcast,
            "message": message,
        }
    )


async def _run_check(
    system: EventSystem,
    *,
    current_step: int,
    world_time: WorldTime,
    world_id: str,
    all_agents: dict,
    clock: object | None = None,  # unused
    locations: Sequence = (),
    entities: Sequence = (),
) -> dict | None:
    """Drive one full event check synchronously (pacing gate -> gate + plan -> commit).

    Production exposes only the fire-and-forget poll_event; these tests cover dispatch / parsing /
    fallback, not background scheduling, so this drives the internals directly.
    """
    if not system._passes_rule_check(current_step):
        return None
    plan = await system._generate_plan(current_step, world_time, world_id, all_agents, locations, entities)
    if plan is None:
        return None
    committed = await system._commit_plan(plan, current_step, all_agents)
    return None if committed is None else serialize_world_event(committed.event)


# ─────────────────────────────────────────────────────────────────────────────
# Constructor / rule gates
# ─────────────────────────────────────────────────────────────────────────────


def test_constructor_rejects_invalid_config(container) -> None:
    with pytest.raises(ValueError, match="check_interval"):
        _event_system(container, check_interval=0)
    with pytest.raises(ValueError, match="max_events_per_window"):
        _event_system(container, max_events_per_window=-1)
    with pytest.raises(ValueError, match="quota_window"):
        _event_system(container, quota_window=0)


def test_rule_check_respects_check_interval(container) -> None:
    system = _event_system(container, check_interval=6)
    assert system._passes_rule_check(current_step=7) is False
    assert system._passes_rule_check(current_step=12) is True


def test_rule_check_respects_window_quota(container) -> None:
    """Window filled -> gate closes; window slides past -> reopens (the quota is not a lifetime total)."""
    system = _event_system(container, max_events_per_window=2, quota_window=20,
                           check_interval=1)
    for event in (
        WorldEvent(id="e1", triggered_step=1, narrative_desc="x", is_positive=None,
                   dispatched_to=["broadcast"]),
        WorldEvent(id="e2", triggered_step=2, narrative_desc="y", is_positive=None,
                   dispatched_to=["message"]),
    ):
        system._ledger.record(event)
    assert system._passes_rule_check(current_step=20) is False   # both within window (0,20]
    assert system._passes_rule_check(current_step=21) is True    # e1 slides out of the window -> frees one slot
    assert system._passes_rule_check(current_step=50) is True    # window now empty


def test_disabled_event_system_never_passes_rule_check(container) -> None:
    """enabled=False master switch: even with every other gate passing, rule check short-circuits to False."""
    system = _event_system(container, enabled=False, check_interval=1)
    assert system._passes_rule_check(current_step=10) is False


@pytest.mark.asyncio
async def test_disabled_event_system_skips_injection_and_llm(container) -> None:
    """enabled=False: the event check short-circuits: no injection, no LLM calls."""
    world_id = "world-1"
    router = _EventRouter([_gate(), _plan_json(
        narrative_desc="不该出现",
        broadcast={"content": "...", "severity": "low", "location_scope": None},
    )])
    system = _event_system(container, router=router, world_id=world_id,
                            enabled=False, check_interval=1)
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="X")}

    fired = await _run_check(system, 
        current_step=2, world_time=_world_time(2), world_id=world_id,
        all_agents=agents, clock=GlobalClock(WorldTimeConfig()),
    )

    assert fired is None
    assert system.list_events() == []
    assert router.calls == []  # neither gate nor plan was called


# ─────────────────────────────────────────────────────────────────────────────
# fire-and-forget poll_event (the only runtime entry: use a ready event, else generate in background)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_poll_event_empty_returns_none_and_schedules(container) -> None:
    """First poll: nothing ready -> returns None and starts one background generation (non-blocking)."""
    world_id = "world-1"
    router = _EventRouter([_gate(True), _plan_json(
        broadcast={"content": "x", "severity": "low", "location_scope": None})])
    system = _event_system(container, router=router, world_id=world_id,
                            check_interval=1)
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="李世民")}

    fired = await system.poll_event(current_step=1, world_time=_world_time(1),
                                    world_id=world_id, all_agents=agents,
                                    locations=[], entities=[])
    assert fired is None
    assert system._inflight is not None          # background generation started
    await system.aclose()


@pytest.mark.asyncio
async def test_poll_event_roundtrip_generates_then_commits(container) -> None:
    """The poll on step N starts generation and returns None; once it finishes, a later poll commits:
    dispatch + return the event. deliver_step = commit step, so it is perceivable that same step
    (consume runs before collect)."""
    world_id = "world-1"
    bc = BroadcastChannel()
    router = _EventRouter([_gate(True), _plan_json(
        narrative_desc="暴雨袭击",
        is_positive=False,
        broadcast={"content": "天降暴雨,街道泥泞", "severity": "medium", "location_scope": None},
    )])
    system = _event_system(container, router=router, world_id=world_id,
                            broadcast_channel=bc, check_interval=1)
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="李世民")}

    # Step 1: nothing ready -> start in background, non-blocking.
    assert await system.poll_event(current_step=1, world_time=_world_time(1),
                                   world_id=world_id, all_agents=agents,
                                    locations=[], entities=[]) is None
    await system._inflight  # let background generation finish (awaited explicitly in the test)

    # Step 2: ready -> commit. deliver_step=2, perceivable this step.
    fired = await system.poll_event(current_step=2, world_time=_world_time(2),
                                    world_id=world_id, all_agents=agents,
                                    locations=[], entities=[])
    assert fired is not None
    assert fired.event.dispatched_to == ["broadcast"]
    assert fired.event.triggered_step == 2       # commit step, not trigger step
    assert len(bc.peek_pending()) == 1
    assert bc.peek_pending()[0].deliver_step == 2
    assert len(system.list_events()) == 1        # quota is counted at commit
    assert system._inflight is None              # ready was consumed; no new generation this step


@pytest.mark.asyncio
async def test_poll_event_single_in_flight(container) -> None:
    """One in flight: polling again while a task is pending doesn't start a second one."""
    world_id = "world-1"
    router = _EventRouter([_gate(True), _plan_json(
        broadcast={"content": "x", "severity": "low", "location_scope": None})])
    system = _event_system(container, router=router, world_id=world_id,
                            check_interval=1)
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="李世民")}

    await system.poll_event(current_step=1, world_time=_world_time(1),
                            world_id=world_id, all_agents=agents,
                                    locations=[], entities=[])
    first = system._inflight
    await system.poll_event(current_step=1, world_time=_world_time(1),
                            world_id=world_id, all_agents=agents,
                                    locations=[], entities=[])
    assert system._inflight is first  # not replaced
    await system.aclose()


@pytest.mark.asyncio
async def test_poll_event_gate_no_fires_nothing(container) -> None:
    """Gate says no -> background task ends with None; poll emits nothing and counts nothing."""
    world_id = "world-1"
    bc = BroadcastChannel()
    router = _EventRouter([_gate(False)])  # gate says don't inject, so plan is never called
    system = _event_system(container, router=router, world_id=world_id,
                            broadcast_channel=bc, check_interval=1)
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="李世民")}

    await system.poll_event(current_step=1, world_time=_world_time(1),
                            world_id=world_id, all_agents=agents,
                                    locations=[], entities=[])
    await system._inflight
    fired = await system.poll_event(current_step=2, world_time=_world_time(2),
                                    world_id=world_id, all_agents=agents,
                                    locations=[], entities=[])
    assert fired is None
    assert system.list_events() == []            # not counted
    assert bc.peek_pending() == []
    await system.aclose()


@pytest.mark.asyncio
async def test_aclose_cancels_in_flight_generation(container) -> None:
    """aclose cancels the in-flight task and leaves nothing pending."""
    world_id = "world-1"
    router = _EventRouter([_gate(True), _plan_json(
        broadcast={"content": "x", "severity": "low", "location_scope": None})])
    system = _event_system(container, router=router, world_id=world_id,
                            check_interval=1)
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="李世民")}

    await system.poll_event(current_step=1, world_time=_world_time(1),
                            world_id=world_id, all_agents=agents,
                                    locations=[], entities=[])
    assert system._inflight is not None
    await system.aclose()
    assert system._inflight is None


@pytest.mark.asyncio
async def test_aclose_retrieves_the_exception_of_an_already_finished_generation() -> None:
    """A task that already finished with an exception must still have its result retrieved, or the
    exception surfaces only as a warning at GC. Skipping everything when ``done()`` misses this.

    Asserts the real symptom (the loop's exception handler on collection). Don't assert on
    ``task.exception()``: asking counts as retrieving, so the test would pass regardless.
    """
    system = EventSystem.__new__(EventSystem)   # only aclose is under test; skip the full dependency set

    loop = asyncio.get_running_loop()
    reported: list[dict] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: reported.append(context))
    try:
        async def boom() -> None:
            raise RuntimeError("generation blew up after the last poll")

        task = asyncio.create_task(boom())
        await asyncio.sleep(0)                  # let it finish with nobody reading the exception
        assert task.done()
        system._inflight = task

        await system.aclose()
        assert system._inflight is None

        del task                                # drop the last reference
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous)

    assert not reported, f"aclose 没取回异常,GC 时被报了出来: {reported}"


# ─────────────────────────────────────────────────────────────────────────────
# restore_state — throttle recovery on session restore
# ─────────────────────────────────────────────────────────────────────────────


def test_restore_state_rehydrates_window_quota(container) -> None:
    """A restored EventSystem keeps counting prior injections inside its window."""
    system = _event_system(container, max_events_per_window=2, quota_window=40,
                           check_interval=6)
    # Two events already fired in the prior session — serialized as snapshots stored them.
    fired = [
        serialize_world_event(
            WorldEvent(id="e1", triggered_step=6, narrative_desc="alpha",
                       is_positive=True, dispatched_to=["broadcast"], metadata={"k": "v"})
        ),
        serialize_world_event(
            WorldEvent(id="e2", triggered_step=30, narrative_desc="beta",
                       is_positive=None, dispatched_to=["message"], metadata={})
        ),
    ]

    system.restore_state(fired)

    # Both prior events sit inside the 40-step window → quota full, gate closed.
    assert len(system.list_events()) == 2
    assert system._passes_rule_check(current_step=42) is False
    # Window slid past both → quota free again (a restore must not freeze the world).
    assert system._passes_rule_check(current_step=72) is True
    # Rehydrated events round-trip through the public serializer.
    assert serialize_world_event(system.list_events()[0])["narrative_desc"] == "alpha"


def test_restore_state_counts_director_free_history(container) -> None:
    """Without restore the window would start empty; restore re-arms it."""
    system = _event_system(container, max_events_per_window=1, quota_window=24,
                           check_interval=6)
    fired = [
        serialize_world_event(
            WorldEvent(id="e1", triggered_step=30, narrative_desc="last",
                       is_positive=None, dispatched_to=["broadcast"], metadata={})
        ),
    ]

    system.restore_state(fired)

    # Next check_interval-aligned step is still inside the 24-step window → blocked.
    assert system._passes_rule_check(current_step=36) is False
    assert system._passes_rule_check(current_step=54) is True


def test_restore_state_ignores_empty_and_malformed_events(container) -> None:
    system = _event_system(container, max_events_per_window=2)
    system.restore_state([])
    assert system.list_events() == []
    # Entries without a usable step are skipped, not crashed on.
    system.restore_state([{"id": "x"}, "not-a-mapping", {"step": "bad"}])
    assert system.list_events() == []


# ─────────────────────────────────────────────────────────────────────────────
# Core channel contracts
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pure_broadcast_plan_dispatches_only_broadcast(container) -> None:
    """Broadcast-only plan -> published only to BroadcastChannel, not the message queue."""
    world_id = "world-1"
    bc = BroadcastChannel()
    router = _EventRouter([_gate(), _plan_json(
        narrative_desc="暴雨袭击",
        is_positive=False,
        broadcast={"content": "天降暴雨,街道泥泞", "severity": "medium", "location_scope": None},
    )])
    system = _event_system(container, router=router, world_id=world_id,
                            broadcast_channel=bc, check_interval=1)
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="李世民")}

    fired = await _run_check(system, 
        current_step=1, world_time=_world_time(1), world_id=world_id,
        all_agents=agents, clock=GlobalClock(WorldTimeConfig()),
    )

    assert fired is not None
    assert fired["dispatched_to"] == ["broadcast"]
    assert fired["narrative_desc"] == "暴雨袭击"
    assert fired["is_positive"] is False
    assert len(bc.peek_pending()) == 1
    assert bc.peek_pending()[0].content == "天降暴雨,街道泥泞"
    assert bc.peek_pending()[0].severity == "medium"
    assert await _messages(system).peek_pending() == []


@pytest.mark.asyncio
async def test_pure_message_plan_dispatches_only_message(container) -> None:
    """Message-only plan -> published only to MessageSystem, not broadcast."""
    world_id = "world-1"
    bc = BroadcastChannel()
    router = _EventRouter([_gate(), _plan_json(
        narrative_desc="李世民心头不安",
        is_positive=False,
        message={"content": "你心头突然涌起一阵莫名的不安", "recipients": [1], "urgency": "high"},
    )])
    system = _event_system(container, router=router, world_id=world_id,
                            broadcast_channel=bc, check_interval=1)
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="李世民")}

    fired = await _run_check(system, 
        current_step=1, world_time=_world_time(1), world_id=world_id,
        all_agents=agents, clock=GlobalClock(WorldTimeConfig()),
    )

    assert fired is not None
    assert fired["dispatched_to"] == ["message"]
    assert bc.peek_pending() == []
    # message delivered correctly: sent by narrator, recipients=[a1], metadata.narrative=True
    pending = await _messages(system).peek_pending()
    assert len(pending) == 1
    msg = pending[0]
    assert msg.sender_id == "narrator"
    # The narrator's sender_name is a narrative referent ("不知来源"), never a system concept:
    # people in the story don't know the narrative engine exists.
    assert msg.sender_name == "不知来源"
    assert msg.recipients == ["a1"]
    assert msg.urgency == Urgency.HIGH
    # The narrator isn't a person one can form a relation with, and the sender declares that (left
    # to each recipient, they'd each decide differently): undeclared, the recipient would take on a
    # relation with "不知来源" and list it as someone they can message.
    assert msg.sender_is_agent is False
    assert msg.metadata.get("narrative") is True
    assert msg.metadata.get("source") == "system"


@pytest.mark.asyncio
async def test_dual_channel_plan_dispatches_both(container) -> None:
    """Broadcast + message both set -> each channel publishes once; dispatched_to order is fixed."""
    world_id = "world-1"
    bc = BroadcastChannel()
    router = _EventRouter([_gate(), _plan_json(
        narrative_desc="皇帝驾崩",
        is_positive=False,
        broadcast={"content": "宫中传来噩耗,皇帝昨夜驾崩", "severity": "high", "location_scope": None},
        message={"content": "父皇驾崩,你心如刀绞", "recipients": [1, 2], "urgency": "high"},
    )])
    system = _event_system(container, router=router, world_id=world_id,
                            broadcast_channel=bc, check_interval=1)
    agents = {
        "a1": _build_agent(container, world_id=world_id, agent_id="a1", name="李世民"),
        "a2": _build_agent(container, world_id=world_id, agent_id="a2", name="李建成"),
    }

    fired = await _run_check(system, 
        current_step=1, world_time=_world_time(1), world_id=world_id,
        all_agents=agents, clock=GlobalClock(WorldTimeConfig()),
    )

    assert fired is not None
    assert fired["dispatched_to"] == ["broadcast", "message"]  # fixed order
    assert len(bc.peek_pending()) == 1
    pending = await _messages(system).peek_pending()
    assert len(pending) == 1
    assert pending[0].recipients == ["a1", "a2"]


# ─────────────────────────────────────────────────────────────────────────────
# Failure semantics (failures don't consume quota)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invalid_json_does_not_consume_event_quota(container) -> None:
    """LLM returns invalid JSON -> return None; the ledger doesn't grow (no window quota consumed)."""
    world_id = "world-1"
    router = _EventRouter([_gate(), "this is not valid json {{{"])
    system = _event_system(container, router=router, world_id=world_id,
                            check_interval=1)
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="X")}

    fired = await _run_check(system, 
        current_step=5, world_time=_world_time(5), world_id=world_id,
        all_agents=agents, clock=GlobalClock(WorldTimeConfig()),
    )

    assert fired is None
    assert system.list_events() == []


@pytest.mark.asyncio
async def test_both_channels_null_drops_event(container) -> None:
    """LLM plan with both channels null -> dropped, no quota consumed."""
    world_id = "world-1"
    router = _EventRouter([_gate(), _plan_json(narrative_desc="空事件")])
    system = _event_system(container, router=router, world_id=world_id,
                            check_interval=1)
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="X")}

    fired = await _run_check(system, 
        current_step=3, world_time=_world_time(3), world_id=world_id,
        all_agents=agents, clock=GlobalClock(WorldTimeConfig()),
    )

    assert fired is None
    assert system.list_events() == []


@pytest.mark.asyncio
async def test_invalid_recipients_degrades_message_channel(container) -> None:
    """All recipients invalid -> the message part falls away; if broadcast is null too, the whole
    event is dropped."""
    world_id = "world-1"
    router = _EventRouter([_gate(), _plan_json(
        narrative_desc="无效定向",
        message={"content": "...", "recipients": [99, "x", -1], "urgency": "normal"},
    )])
    system = _event_system(container, router=router, world_id=world_id,
                            check_interval=1)
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="X")}

    fired = await _run_check(system, 
        current_step=2, world_time=_world_time(2), world_id=world_id,
        all_agents=agents, clock=GlobalClock(WorldTimeConfig()),
    )
    # broadcast also null -> whole event dropped
    assert fired is None
    assert system.list_events() == []


@pytest.mark.asyncio
async def test_invalid_recipients_keeps_broadcast(container) -> None:
    """All recipients invalid but broadcast valid -> the event is kept and only broadcast is dispatched."""
    world_id = "world-1"
    bc = BroadcastChannel()
    router = _EventRouter([_gate(), _plan_json(
        narrative_desc="只剩广播",
        broadcast={"content": "城内有动静", "severity": "low", "location_scope": None},
        message={"content": "...", "recipients": [99], "urgency": "normal"},
    )])
    system = _event_system(container, router=router, world_id=world_id,
                            broadcast_channel=bc, check_interval=1)
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="X")}

    fired = await _run_check(system, 
        current_step=1, world_time=_world_time(1), world_id=world_id,
        all_agents=agents, clock=GlobalClock(WorldTimeConfig()),
    )

    assert fired is not None
    assert fired["dispatched_to"] == ["broadcast"]
    assert len(bc.peek_pending()) == 1
    assert await _messages(system).peek_pending() == []


# ─────────────────────────────────────────────────────────────────────────────
# IndexedRef + normalization
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_indexed_ref_resolves_recipients_correctly(container) -> None:
    """recipients: [1, 3] in a 3-agent world resolves to the agent_ids of #1 and #3."""
    world_id = "world-1"
    router = _EventRouter([_gate(), _plan_json(
        narrative_desc="点名",
        message={"content": "...", "recipients": [1, 3], "urgency": "normal"},
    )])
    system = _event_system(container, router=router, world_id=world_id,
                            check_interval=1)
    # order matters: #1=a1, #2=a2, #3=a3
    agents = {
        "a1": _build_agent(container, world_id=world_id, agent_id="a1", name="一"),
        "a2": _build_agent(container, world_id=world_id, agent_id="a2", name="二"),
        "a3": _build_agent(container, world_id=world_id, agent_id="a3", name="三"),
    }

    await _run_check(system, 
        current_step=1, world_time=_world_time(1), world_id=world_id,
        all_agents=agents, clock=GlobalClock(WorldTimeConfig()),
    )

    pending = await _messages(system).peek_pending()
    assert len(pending) == 1
    assert pending[0].recipients == ["a1", "a3"]


@pytest.mark.asyncio
async def test_severity_urgency_normalization(container) -> None:
    """Invalid severity / urgency from the LLM -> falls back to 'low' / NORMAL."""
    world_id = "world-1"
    bc = BroadcastChannel()
    router = _EventRouter([_gate(), _plan_json(
        narrative_desc="归一化测试",
        broadcast={"content": "...", "severity": "紧急", "location_scope": None},
        message={"content": "...", "recipients": [1], "urgency": "极高"},  # not an enum literal
    )])
    system = _event_system(container, router=router, world_id=world_id,
                            broadcast_channel=bc, check_interval=1)
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="X")}

    await _run_check(system, 
        current_step=1, world_time=_world_time(1), world_id=world_id,
        all_agents=agents, clock=GlobalClock(WorldTimeConfig()),
    )

    assert bc.peek_pending()[0].severity == "low"  # invalid severity -> low
    pending = await _messages(system).peek_pending()
    assert pending[0].urgency == Urgency.NORMAL   # invalid urgency -> NORMAL


@pytest.mark.asyncio
async def test_is_positive_chinese_aliases(container) -> None:
    """is_positive accepts the string aliases '正面' / '负面' / '中性'."""
    world_id = "world-1"
    router = _EventRouter([_gate(), _plan_json(
        narrative_desc="测试",
        is_positive="正面",
        broadcast={"content": "...", "severity": "low", "location_scope": None},
    )])
    system = _event_system(container, router=router, world_id=world_id,
                            check_interval=1)
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="X")}

    fired = await _run_check(system, 
        current_step=1, world_time=_world_time(1), world_id=world_id,
        all_agents=agents, clock=GlobalClock(WorldTimeConfig()),
    )
    assert fired["is_positive"] is True


@pytest.mark.asyncio
async def test_unknown_location_scope_degrades_to_null(container) -> None:
    """Unknown location_id from the LLM -> location_scope falls back to null; the event still dispatches."""
    world_id = "world-1"
    bc = BroadcastChannel()
    router = _EventRouter([_gate(), _plan_json(
        narrative_desc="未知地点",
        broadcast={"content": "...", "severity": "low", "location_scope": "不存在的地方"},
    )])
    system = _event_system(container, router=router, world_id=world_id,
                            broadcast_channel=bc, check_interval=1)
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="X")}
    agents["a1"].personality.update_location(location="palace")

    fired = await _run_check(system, 
        current_step=1, world_time=_world_time(1), world_id=world_id,
        all_agents=agents, clock=GlobalClock(WorldTimeConfig()),
    )

    assert fired is not None
    assert bc.peek_pending()[0].location_scope is None


@pytest.mark.asyncio
async def test_location_scope_index_resolves_to_location_id(container) -> None:
    """location_scope uses an IndexedRef index: `1` -> the location_id of the first candidate place."""
    from core.interfaces.place import Place

    world_id = "world-1"
    bc = BroadcastChannel()
    env = EnvironmentSystem()
    env.space.register_place(Place(
        place_id="palace_throne", name="太极殿", ))
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="李世民")}
    agents["a1"].personality.update_location(location="palace_throne")
    directory = LiveWorldDirectory.from_agents(agents, env)

    router = _EventRouter([_gate(), _plan_json(
        narrative_desc="殿内异动",
        broadcast={"content": "殿中烛火无风自灭", "severity": "high", "location_scope": 1},
    )])
    system = _event_system(container, router=router, world_id=world_id,
                            broadcast_channel=bc, directory=directory,
                            check_interval=1)

    fired = await _run_check(system, 
        current_step=1, world_time=_world_time(1), world_id=world_id,
        all_agents=agents, locations=env.space.all_places(), clock=GlobalClock(WorldTimeConfig()),
    )

    assert fired is not None
    # index 1 -> the real location_id of the (only) candidate place
    assert bc.peek_pending()[0].location_scope == "palace_throne"


@pytest.mark.asyncio
async def test_location_scope_true_is_not_read_as_the_first_place(container) -> None:
    """``true`` is not an index: it points at no place and must fall back to world-wide.

    ``int(True)`` is 1, so it would pass as "item 1" and silently confine a world-wide broadcast to
    the first place on the menu (``false`` gives 0, out of range, and already falls back).
    """
    from core.interfaces.place import Place

    world_id = "world-1"
    bc = BroadcastChannel()
    env = EnvironmentSystem()
    env.space.register_place(Place(place_id="palace_throne", name="太极殿"))
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="李世民")}
    agents["a1"].personality.update_location(location="palace_throne")
    directory = LiveWorldDirectory.from_agents(agents, env)

    router = _EventRouter([_gate(), _plan_json(
        narrative_desc="殿内异动",
        broadcast={"content": "殿中烛火无风自灭", "severity": "high", "location_scope": True},
    )])
    system = _event_system(container, router=router, world_id=world_id,
                           broadcast_channel=bc, directory=directory, check_interval=1)

    fired = await _run_check(system,
        current_step=1, world_time=_world_time(1), world_id=world_id,
        all_agents=agents, locations=env.space.all_places(), clock=GlobalClock(WorldTimeConfig()),
    )

    assert fired is not None
    assert bc.peek_pending()[0].location_scope is None   # world-wide, not "the first place"


@pytest.mark.asyncio
async def test_reason_first_gate_inject_false_skips(container) -> None:
    """Reason-first gate returns inject=false -> no injection (even if a valid plan would follow)."""
    world_id = "world-1"
    router = _EventRouter([_gate(inject=False), _plan_json(
        narrative_desc="本不该出现",
        broadcast={"content": "...", "severity": "low", "location_scope": None},
    )])
    system = _event_system(container, router=router, world_id=world_id,
                            check_interval=1)
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="X")}

    fired = await _run_check(system, 
        current_step=1, world_time=_world_time(1), world_id=world_id,
        all_agents=agents, clock=GlobalClock(WorldTimeConfig()),
    )

    assert fired is None
    assert system.list_events() == []
    # plan call never happened (gate blocked it); router called once (the gate)
    assert len(router.calls) == 1


@pytest.mark.asyncio
async def test_malformed_gate_response_is_conservative(container) -> None:
    """Gate returns invalid JSON -> conservatively "don't inject", no raise."""
    world_id = "world-1"
    router = _EventRouter(["这不是 JSON {{{"])
    system = _event_system(container, router=router, world_id=world_id,
                            check_interval=1)
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="X")}

    fired = await _run_check(system, 
        current_step=1, world_time=_world_time(1), world_id=world_id,
        all_agents=agents, clock=GlobalClock(WorldTimeConfig()),
    )
    assert fired is None


@pytest.mark.asyncio
async def test_brief_carries_premise_and_dossier_no_step_or_id_leak(container) -> None:
    """The brief includes 【故事前提】 + main-character profiles; gate/plan prompts leak no step or raw
    location id."""
    from core.interfaces.place import Place

    world_id = "world-1"
    env = EnvironmentSystem()
    env.space.register_place(Place(
        place_id="palace_throne", name="太极殿", ))
    # main character with role / core_values / life_goal, placed in a named location
    main = _build_agent(container, world_id=world_id, agent_id="a1", name="李世民")
    main.is_main_character = True
    main.personality.update_location(location="palace_throne")
    agents = {"a1": main}
    directory = LiveWorldDirectory.from_agents(agents, env)

    router = _EventRouter([_gate(), _plan_json(
        narrative_desc="测试",
        broadcast={"content": "...", "severity": "low", "location_scope": None},
    )])
    system = _event_system(container, router=router, world_id=world_id,
                            directory=directory,
                            core_tension="秦王功高震主,太子忌惮",
                            check_interval=1)

    await _run_check(system, 
        current_step=5, world_time=_world_time(5), world_id=world_id,
        all_agents=agents, locations=env.space.all_places(), clock=GlobalClock(WorldTimeConfig()),
    )

    # both gate (calls[0]) and plan (calls[1]) carry the brief
    gate_prompt = _joined(router.calls[0][1])
    plan_prompt = _joined(router.calls[1][1])
    # The rule that a sourced phenomenon needs a place of origin is one shared text for both
    # authors (core.prompts.PHENOMENON_DEFINITION). Both sides' broadcasts go through the same
    # parse_broadcast_spec, so separate wording would give fire to one side and not the other.
    assert PHENOMENON_DEFINITION in plan_prompt
    for prompt in (gate_prompt, plan_prompt):
        assert "【故事前提】" in prompt
        assert "秦王功高震主" in prompt
        assert "李世民" in prompt
        assert "Step " not in prompt
        assert "palace_throne" not in prompt
    # plan candidate places are shown by name
    assert "太极殿" in plan_prompt


@pytest.mark.asyncio
async def test_characters_block_excludes_first_person_state(container) -> None:
    """Main-character profiles hold only externally observable identity, with no emotion / dominant_need
    / short_term_goal or other first-person state."""
    world_id = "world-1"
    main = _build_agent(container, world_id=world_id, agent_id="a1", name="李世民")
    main.is_main_character = True
    # give some first-person state and confirm it does NOT reach the brief
    main.personality.state.dominant_need = "求生欲"
    main.personality.state.short_term_goals = ["夺取兵权"]
    agents = {"a1": main}
    system = _event_system(container, world_id=world_id, agents=agents,
                            core_tension="边境危局", narrative_theme="忠义抉择")

    brief = await system._build_narrative_summary(world_id, 5, agents)

    assert "【主要人物】" in brief
    assert "李世民" in brief
    assert "求生欲" not in brief
    assert "夺取兵权" not in brief
    assert "主导需求" not in brief
    assert "当前目标" not in brief
    assert "此刻" not in brief           # emotion summary wording excluded


@pytest.mark.asyncio
async def test_characters_block_marks_the_dead_and_drops_their_living_attrs(container) -> None:
    """The dead are marked "已死亡" in the brief, with no location/vitality/situation (a leftover
    "生命垂危" would contradict the death notice); dead background characters don't count toward the
    distribution."""
    from core.interfaces.condition import BodyCondition

    world_id = "world-1"
    dead = _build_agent(container, world_id=world_id, agent_id="a1", name="李建成")
    dead.is_main_character = True
    dead.personality.state.current_location = "玄武门"
    dead.personality.set_condition(BodyCondition(description="生命垂危"))
    dead.is_active = False
    alive = _build_agent(container, world_id=world_id, agent_id="a2", name="李世民")
    alive.is_main_character = True
    dead_bg = _build_agent(container, world_id=world_id, agent_id="a3", name="侍卫")
    dead_bg.is_active = False
    agents = {"a1": dead, "a2": alive, "a3": dead_bg}
    system = _event_system(container, world_id=world_id, agents=agents)

    brief = await system._build_narrative_summary(world_id, 5, agents)

    dead_line = next(line for line in brief.splitlines() if "李建成" in line)
    assert "已死亡" in dead_line
    assert "现位于" not in dead_line and "体力" not in dead_line and "生命垂危" not in dead_line
    alive_line = next(line for line in brief.splitlines() if "李世民" in line)
    assert "已死亡" not in alive_line and "现位于" in alive_line
    assert "背景人物" not in brief


@pytest.mark.asyncio
async def test_characters_block_puts_where_and_how_they_are_before_the_profile(container) -> None:
    """Location, vitality and situation follow the identity header, ahead of background and other
    profile fields. Buried after a long background, they'd land in the middle, where attention is
    weakest."""
    from core.interfaces.condition import BodyCondition

    world_id = "world-1"
    bound = _build_agent(container, world_id=world_id, agent_id="a1", name="李建成",
                         background="太子,长居东宫。", life_goal="守住东宫。")
    bound.is_main_character = True
    bound.personality.set_condition(BodyCondition(description="双手被反绑"))
    agents = {"a1": bound}
    system = _event_system(container, world_id=world_id, agents=agents)

    brief = await system._build_narrative_summary(world_id, 5, agents)

    line = next(line for line in brief.splitlines() if "李建成" in line)
    for now in ("现位于", "体力", "处境双手被反绑"):
        assert line.index(now) < line.index("背景"), line
    # When profile fields end with their own full stop (common in build-time prose), the joined
    # text must not contain "。；" or "。。".
    assert "。；" not in line and not line.endswith("。。"), line


@pytest.mark.asyncio
async def test_design_menus_list_only_the_living(container) -> None:
    """Recipient/place menus list only the living: indices number the living, so #2 is the second
    living agent, not a dead one."""
    world_id = "world-1"
    router = _EventRouter([_gate(), _plan_json(
        narrative_desc="点名",
        message={"content": "...", "recipients": [2], "urgency": "normal"},
    )])
    system = _event_system(container, router=router, world_id=world_id, check_interval=1)
    agents = {
        "a1": _build_agent(container, world_id=world_id, agent_id="a1", name="一"),
        "a2": _build_agent(container, world_id=world_id, agent_id="a2", name="死者"),
        "a3": _build_agent(container, world_id=world_id, agent_id="a3", name="三"),
    }
    agents["a2"].personality.state.current_location = "墓地"
    agents["a2"].is_active = False

    await _run_check(system, current_step=1, world_time=_world_time(1), world_id=world_id,
                     all_agents=agents)

    plan_user = router.calls[-1][1][-1].content
    menu = [line for line in plan_user.splitlines() if line.startswith("#")]
    assert not any("死者" in line or "墓地" in line for line in menu)
    pending = await _messages(system).peek_pending()
    assert pending[0].recipients == ["a3"]


def test_relations_block_carries_labels_and_history_desc(container) -> None:
    """【当前关系】 injects labels + history_summary description; no trust/affection numbers."""
    from datetime import datetime
    from core.interfaces.snapshot import WorldSnapshot

    world_id = "world-1"
    a1 = _build_agent(container, world_id=world_id, agent_id="a1", name="李世民")
    a2 = _build_agent(container, world_id=world_id, agent_id="a2", name="李建成")
    a1.is_main_character = True
    agents = {"a1": a1, "a2": a2}
    system = _event_system(container, world_id=world_id, agents=agents)

    snap = WorldSnapshot(
        world_id=world_id, step=4, timestamp=datetime.now(),
        world_time=_world_time(4).clock_payload(),
        agent_relations={
            "a1->a2": {
                "from_id": "a1", "to_id": "a2", "to_name": "李建成",
                "labels": ["兄长", "政敌"], "history_summary": "近日数次当庭争执",
                "trust_objective": 0.2, "affection_objective": -0.6,
            }
        },
    )
    block = system._brief_relations(snap, {"a1"})

    assert "【当前关系】" in block
    assert "李世民 对 李建成" in block
    assert "兄长、政敌" in block
    assert "近日数次当庭争执" in block
    assert "0.2" not in block and "-0.6" not in block  # trust/affection numbers excluded


def test_brief_recent_excludes_not_executed_but_keeps_genuine_failure(container) -> None:
    """【近况】 is a deeds list: not_executed (precondition unmet, never touched the world) is
    excluded, or the editor designs events around something that didn't happen. A real in-world
    failure (not_executed=False) is a deed and stays. This channel reads action records, not memory,
    so it must filter here itself."""
    from datetime import datetime
    from core.interfaces.snapshot import WorldSnapshot

    world_id = "world-1"
    a1 = _build_agent(container, world_id=world_id, agent_id="a1", name="李世民")
    a1.is_main_character = True
    agents = {"a1": a1}
    system = _event_system(container, world_id=world_id, agents=agents)

    snap = WorldSnapshot(
        world_id=world_id, step=4, timestamp=datetime.now(),
        world_time=_world_time(4).clock_payload(),
        actions_this_step=[
            # Real failure: adjudication happened and ruled against -> real deed, kept.
            {"agent_id": "a1", "agent_name": "李世民", "is_main_character": True,
             "phase": "completed", "outcome": "李世民强攻宫门却被禁军击退",
             "succeeded": False, "not_executed": False},
            # Non-event: the target wasn't present and the action never touched the world ->
            # non-deed, excluded.
            {"agent_id": "a1", "agent_name": "李世民", "is_main_character": True,
             "phase": "completed", "outcome": "李世民欲面见父皇却扑了个空",
             "succeeded": False, "not_executed": True},
        ],
    )
    block = system._brief_recent([snap], {"a1"})

    assert "被禁军击退" in block          # real failure is a deed, kept
    assert "扑了个空" not in block         # not_executed is a non-deed, excluded


def _recent_snapshot(world_id: str, step: int, actions: list[dict]):
    """Minimal snapshot shared by the folding cases (only actions_this_step)."""
    from datetime import datetime
    from core.interfaces.snapshot import WorldSnapshot

    return WorldSnapshot(
        world_id=world_id, step=step, timestamp=datetime.now(),
        world_time=_world_time(step).clock_payload(),
        actions_this_step=actions,
    )


def _joint_talk(execution_id: str, initiator: tuple[str, str], other: tuple[str, str],
                *, outcome: str, phase: str = "ongoing_complete") -> list[dict]:
    """Runtime shape of a joint TALK: one record per participant, sharing execution_id and the same dialogue."""
    return [
        {"agent_id": aid, "agent_name": name, "is_main_character": True,
         "phase": phase, "outcome": outcome, "not_executed": False,
         "execution_id": execution_id, "initiator_id": initiator[0]}
        for aid, name in (initiator, other)
    ]


def test_brief_recent_folds_joint_action_into_one_line(container) -> None:
    """A joint action yields one record per participant with the same dialogue as outcome. The
    brief must fold them by execution_id into one (represented by the record where agent_id ==
    initiator_id), the same contract as the frontend feed. Unfolded, the same dialogue appears twice
    in 【近况】 and the gate sees double the event density."""
    world_id = "world-1"
    system = _event_system(container, world_id=world_id, agents={})

    dialogue = "在崇仁坊，李世民与房玄龄的交谈：李世民：明日的事，再推一遍。"
    snap = _recent_snapshot(world_id, 4, _joint_talk(
        "talk_a1_4_abc", ("a1", "李世民"), ("a2", "房玄龄"), outcome=dialogue,
    ))
    block = system._brief_recent([snap], {"a1", "a2"})

    assert block.count(dialogue) == 1          # folded into one
    assert block.count("李世民 在崇仁坊") == 1  # represented by the initiator's record
    assert "房玄龄 在崇仁坊" not in block


def test_brief_recent_keeps_deed_when_initiator_filtered_out(container) -> None:
    """Folding must run after the main-character filter, falling back to the first record when the
    group has no initiator: when a background agent starts a conversation with a main character,
    the initiator's record is already filtered out, and "keep only the initiator" would drop the
    whole deed."""
    world_id = "world-1"
    system = _event_system(container, world_id=world_id, agents={})

    dialogue = "在东宫，魏徵与李建成的交谈：魏徵：殿下，事不宜迟。"
    records = _joint_talk("talk_bg_5_xyz", ("bg1", "魏徵"), ("a1", "李建成"), outcome=dialogue)
    records[0]["is_main_character"] = False     # initiator is a background character -> filtered out first
    block = system._brief_recent([_recent_snapshot(world_id, 5, records)], {"a1"})

    assert block.count(dialogue) == 1           # the deed is still there, as a single entry
    assert "李建成 在东宫，" in block             # represented by the surviving participant's record
    assert "魏徵 在东宫，" not in block           # initiator already removed by the main-character filter


def test_brief_recent_folds_within_step_only(container) -> None:
    """Fold only within a step. A multi-step execution has one entry on its start step and one on its
    end step ("开始聊了起来" / full dialogue); both belong. Folding across steps would swallow the
    closing dialogue. The ongoing_ticks in between are removed by the existing filter."""
    world_id = "world-1"
    system = _event_system(container, world_id=world_id, agents={})

    eid = "talk_a1_4_abc"
    begin = _recent_snapshot(world_id, 4, _joint_talk(
        eid, ("a1", "李世民"), ("a2", "房玄龄"), outcome="李世民与房玄龄开始聊了起来。",
        phase="main",
    ))
    tick = _recent_snapshot(world_id, 5, [
        {"agent_id": "a1", "agent_name": "李世民", "is_main_character": True,
         "phase": "ongoing_tick", "outcome": "交谈仍在继续，已持续约6小时。",
         "not_executed": False, "execution_id": eid, "initiator_id": "a1"},
    ])
    done = _recent_snapshot(world_id, 6, _joint_talk(
        eid, ("a1", "李世民"), ("a2", "房玄龄"), outcome="李世民：明日的事，再推一遍。",
    ))
    block = system._brief_recent([begin, tick, done], {"a1", "a2"})

    assert "开始聊了起来" in block               # the start step
    assert "明日的事，再推一遍" in block          # the end step, not folded away across steps
    assert "已持续约6小时" not in block           # ongoing_tick still excluded


def test_brief_recent_keeps_records_without_execution_id_separate(container) -> None:
    """Records without an execution_id (step-0 placement records etc.) stay separate rows and are
    never merged because they share an empty value."""
    world_id = "world-1"
    system = _event_system(container, world_id=world_id, agents={})

    snap = _recent_snapshot(world_id, 0, [
        {"agent_id": "a1", "agent_name": "李世民", "is_main_character": True,
         "phase": "main", "outcome": "在崇仁坊就位", "not_executed": False},
        {"agent_id": "a2", "agent_name": "房玄龄", "is_main_character": True,
         "phase": "main", "outcome": "在东宫就位", "not_executed": False},
    ])
    block = system._brief_recent([snap], {"a1", "a2"})

    assert "在崇仁坊就位" in block
    assert "在东宫就位" in block


@pytest.mark.asyncio
async def test_prior_events_carry_affected_names_and_location(container) -> None:
    """【已注入事件】 must say who was affected and where; affected_names/location_label survive restore."""
    from core.interfaces.place import Place

    world_id = "world-1"
    env = EnvironmentSystem()
    env.space.register_place(Place(
        place_id="palace_throne", name="太极殿", ))
    a1 = _build_agent(container, world_id=world_id, agent_id="a1", name="李世民")
    a2 = _build_agent(container, world_id=world_id, agent_id="a2", name="李建成")
    for a in (a1, a2):
        a.is_main_character = True
        a.personality.update_location(location="palace_throne")
    agents = {"a1": a1, "a2": a2}
    directory = LiveWorldDirectory.from_agents(agents, env)

    # First injection: broadcast lands in "太极殿" (index 1) + message to a1, a2 (indices 1, 2).
    # Second injection: capture its plan prompt and check 【已注入事件】 carries names + place name.
    router = _EventRouter([
        _gate(), _plan_json(
            narrative_desc="殿中惊变",
            broadcast={"content": "殿内灯烛尽灭", "severity": "high", "location_scope": 1},
            message={"content": "一阵寒意袭来", "recipients": [1, 2], "urgency": "high"},
        ),
        _gate(), _plan_json(narrative_desc="第二事件",
                            broadcast={"content": "...", "severity": "low", "location_scope": None}),
    ])
    system = _event_system(container, router=router, world_id=world_id, directory=directory,
                            max_events_per_window=5, check_interval=1)

    first = await _run_check(system, 
        current_step=1, world_time=_world_time(1), world_id=world_id,
        all_agents=agents, locations=env.space.all_places(), clock=GlobalClock(WorldTimeConfig()),
    )
    assert first is not None
    # the ledger entry holds narrative-layer referents
    ev = system.list_events()[0]
    assert ev.affected_names == ["李世民", "李建成"]
    assert ev.location_label == "太极殿"

    await _run_check(system, 
        current_step=2, world_time=_world_time(2), world_id=world_id,
        all_agents=agents, locations=env.space.all_places(), clock=GlobalClock(WorldTimeConfig()),
    )
    second_plan_prompt = _joined(router.calls[3][1])
    assert "【已注入事件】" in second_plan_prompt
    assert "殿中惊变" in second_plan_prompt
    assert "李世民" in second_plan_prompt and "李建成" in second_plan_prompt
    assert "太极殿" in second_plan_prompt

    # names/place names survive restore (serialize -> restore round trip)
    serialized = [serialize_world_event(ev) for ev in system.list_events() if ev.triggered_step == 1]
    fresh = _event_system(container, world_id=world_id, directory=directory)
    fresh.restore_state(serialized)
    restored = fresh.list_events()[0]
    assert restored.affected_names == ["李世民", "李建成"]
    assert restored.location_label == "太极殿"


# ─────────────────────────────────────────────────────────────────────────────
# Serialize schema (one field, one meaning; no aliases + WorldEvent ledger entry)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_serialized_event_emits_one_key_per_meaning(container) -> None:
    """Serialized shape: one field, one meaning; no aliases and no always-empty dead fields.

    Putting the same string under description / narrative / narrative_desc, plus two always-empty
    affected_agent_ids / location_id, makes every downstream consumer guess which to read.
    This locks the current shape so aliases don't grow back.
    """
    world_id = "world-1"
    router = _EventRouter([_gate(), _plan_json(
        narrative_desc="某事件",
        broadcast={"content": "...", "severity": "low", "location_scope": None},
    )])
    system = _event_system(container, router=router, world_id=world_id,
                            check_interval=1)
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="X")}

    await _run_check(system,
        current_step=1, world_time=_world_time(1), world_id=world_id,
        all_agents=agents, clock=GlobalClock(WorldTimeConfig()),
    )

    serialized = [serialize_world_event(ev) for ev in system.list_events()]
    assert len(serialized) == 1
    s = serialized[0]
    assert set(s) == {
        "id", "step", "narrative_desc", "is_positive", "dispatched_to",
        "affected_names", "location_label", "authored_by", "directive_text", "metadata",
    }
    # Neither seq nor receipt is here: they are step-level fields stamped by Runtime when it
    # assembles the snapshot. At commit time this step's emission order isn't settled and the three
    # phases haven't run.
    assert "seq" not in s and "receipt" not in s
    # directive_text is present (it records the injection's origin and exists at commit), but the
    # editor doesn't grow from a single sentence: it reads the narrative brief, so this is always
    # empty here. Only the director's entries have content.
    assert s["directive_text"] == ""
    assert s["narrative_desc"] == "某事件"
    assert s["step"] == 1
    assert s["dispatched_to"] == ["broadcast"]
    # Written by the LLM event editor, not the director; observers use this to tell "the world
    # did it" from "you caused it".
    assert s["authored_by"] == "system"


def test_world_event_dataclass_minimal_shape() -> None:
    """WorldEvent is a ledger entry, not a behavior carrier."""
    event = WorldEvent(
        id="evt-1", triggered_step=5,
        narrative_desc="X", is_positive=False,
        dispatched_to=["broadcast", "message"],
    )
    assert event.id == "evt-1"
    assert event.triggered_step == 5
    assert event.dispatched_to == ["broadcast", "message"]
    assert event.authored_by is Author.SYSTEM   # default author is the LLM event editor


# ─────────────────────────────────────────────────────────────────────────────
# No agent mutation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_does_not_mutate_agent_state(container) -> None:
    """Even when the LLM describes events like "X is wounded", EventSystem doesn't call
    personality.update_emotion and the like; it only publishes broadcast/message.
    """
    world_id = "world-1"
    bc = BroadcastChannel()
    router = _EventRouter([_gate(), _plan_json(
        narrative_desc="李世民受伤",
        broadcast={"content": "李世民负伤", "severity": "high", "location_scope": None},
        message={"content": "你感到剧痛袭来", "recipients": [1], "urgency": "high"},
    )])
    system = _event_system(container, router=router, world_id=world_id,
                            broadcast_channel=bc, check_interval=1)
    agent = _build_agent(container, world_id=world_id, agent_id="a1", name="李世民")
    agents = {"a1": agent}

    original_emotion = agent.personality.state.emotion
    original_active = agent.is_active
    original_action_status = agent.personality.state.action_status
    original_long_term = list(agent.personality.state.long_term_goals)
    original_short_term = list(agent.personality.state.short_term_goals)

    await _run_check(system, 
        current_step=1, world_time=_world_time(1), world_id=world_id,
        all_agents=agents, clock=GlobalClock(WorldTimeConfig()),
    )

    # agent state was not mutated directly
    assert agent.personality.state.emotion == original_emotion
    assert agent.is_active == original_active
    assert agent.personality.state.action_status == original_action_status
    assert agent.personality.state.long_term_goals == original_long_term
    assert agent.personality.state.short_term_goals == original_short_term
    # but the event did dispatch
    assert len(bc.peek_pending()) == 1
    pending = await _messages(system).peek_pending()
    assert len(pending) == 1


# ─────────────────────────────────────────────────────────────────────────────
# BroadcastChannel: location-scoped filtering (independent of EventSystem)
# ─────────────────────────────────────────────────────────────────────────────


def test_broadcast_channel_global_broadcast_visible_everywhere() -> None:
    channel = BroadcastChannel()
    channel.publish(Broadcast(content="rain begins", source="system",
                              broadcast_type=BroadcastType.WORLD_EVENT, deliver_step=1))
    due = channel.collect(step=1)
    assert len(BroadcastChannel.for_location(due, "tavern")) == 1
    assert len(BroadcastChannel.for_location(due, "market")) == 1


def test_broadcast_channel_scoped_broadcast_visible_only_at_target_location() -> None:
    channel = BroadcastChannel()
    channel.publish(Broadcast(
        content="fire in the tavern", source="system",
        broadcast_type=BroadcastType.WORLD_EVENT, deliver_step=1, location_scope="tavern",
    ))
    due = channel.collect(step=1)
    assert len(BroadcastChannel.for_location(due, "tavern")) == 1
    assert len(BroadcastChannel.for_location(due, "market")) == 0


def test_broadcast_channel_mixed_broadcasts_filtered_correctly() -> None:
    channel = BroadcastChannel()
    channel.publish(Broadcast(content="global news", source="system",
                              broadcast_type=BroadcastType.WORLD_EVENT, deliver_step=1))
    channel.publish(Broadcast(content="local event", source="system",
                              broadcast_type=BroadcastType.WORLD_EVENT, deliver_step=1,
                              location_scope="tavern"))
    due = channel.collect(step=1)
    assert len(BroadcastChannel.for_location(due, "tavern")) == 2
    assert len(BroadcastChannel.for_location(due, "market")) == 1


def test_broadcast_channel_collect_dequeues_due_and_defers_future() -> None:
    """collect takes and dequeues only broadcasts with deliver_step <= step; later ones wait (death
    notices arrive next step)."""
    channel = BroadcastChannel()
    channel.publish(Broadcast(content="event now", source="system",
                              broadcast_type=BroadcastType.WORLD_EVENT, deliver_step=5))
    channel.publish(Broadcast(content="death next step", source="system",
                              broadcast_type=BroadcastType.WORLD_EVENT, deliver_step=6))
    now = channel.collect(step=5)
    assert [b.content for b in now] == ["event now"]          # due ones taken
    assert [b.content for b in channel.peek_pending()] == ["death next step"]  # not yet due, kept
    later = channel.collect(step=6)
    assert [b.content for b in later] == ["death next step"]  # taken on the next step
    assert channel.peek_pending() == []                        # dequeued once consumed


# ─────────────────────────────────────────────────────────────────────────────
# Private effects: entity spawn / alter / destroy, only for unheld things on the ground
# ─────────────────────────────────────────────────────────────────────────────


def _ground_world(container, *, world_id: str = "world-1"):
    """Agent "李世民" stands in "太极殿", with a notice on the floor and a secret letter in his hand;
    in "西市" a knife lies on the ground with nobody there."""
    from world.models import EntityPresence, WorldEntity, WorldEntityType
    from core.interfaces.place import Place

    env = EnvironmentSystem()
    env.space.register_place(Place(place_id="palace", name="太极殿"))
    env.space.register_place(Place(place_id="market", name="西市"))
    env.place_agent(agent_id="a1", location_id="palace")
    env.register_entity(WorldEntity(
        entity_id="notice", name="告示", entity_type=WorldEntityType.LANDMARK,
        description="黄纸黑字", content="明日辰时开城门。",
        presence=EntityPresence.AT_LOCATION, presence_ref="palace",
    ))
    env.register_entity(WorldEntity(
        entity_id="letter", name="密信", entity_type=WorldEntityType.ITEM, is_takeable=True,
        presence=EntityPresence.HELD, presence_ref="a1",
    ))
    env.register_entity(WorldEntity(
        entity_id="knife", name="短刀", entity_type=WorldEntityType.ITEM, is_takeable=True,
        presence=EntityPresence.AT_LOCATION, presence_ref="market",
    ))
    agents = {"a1": _build_agent(container, world_id=world_id, agent_id="a1", name="李世民")}
    agents["a1"].personality.update_location(location="palace")
    return env, agents, LiveWorldDirectory.from_agents(agents, env)


def _ground_index(env: EnvironmentSystem, entity_id: str) -> int:
    """Menu index of a thing on the ground: the menu follows the order of entities, not names
    (every place in ``_ground_world`` makes the menu)."""
    grounded = [e.entity_id for e in env.all_live_entities() if e.location_id is not None]
    return grounded.index(entity_id) + 1


@pytest.mark.asyncio
async def test_entity_menu_lists_unheld_things_here_and_elsewhere(container) -> None:
    env, agents, directory = _ground_world(container)
    router = _EventRouter([_gate(), _plan_json(broadcast={"content": "x", "severity": "low"})])
    system = _event_system(container, router=router, directory=directory, environment=env,
                           check_interval=1)

    await _run_check(system, current_step=1, world_time=_world_time(1), world_id="world-1",
                     all_agents=agents, locations=env.space.all_places(),
                     entities=env.all_live_entities())

    user = router.calls[-1][1][-1].content
    menu = user.split("地上的东西")[1]
    assert "告示" in menu and "上面写着：明日辰时开城门。" in menu
    assert "密信" not in menu      # in someone's hand
    assert "现于西市" in menu.split("短刀")[1].split("\n")[0]      # nobody there, but the editor can reach it


@pytest.mark.asyncio
async def test_place_menu_shows_who_is_where(container) -> None:
    from core.interfaces.place import Place

    env, agents, directory = _ground_world(container)
    env.space.register_place(Place(place_id="gate", name="玄武门"))     # nobody, nothing: still drawn
    router = _EventRouter([_gate(), _plan_json(broadcast={"content": "x", "severity": "low"})])
    system = _event_system(container, router=router, directory=directory, environment=env,
                           check_interval=1)

    await _run_check(system, current_step=1, world_time=_world_time(1), world_id="world-1",
                     all_agents=agents, locations=env.space.all_places(),
                     entities=env.all_live_entities())

    user = router.calls[-1][1][-1].content
    places = user.split("可选地点")[1].split("可选人物")[0]
    assert "太极殿（此刻1人）" in places
    assert "西市（此刻无人）" in places
    assert "玄武门（此刻无人）" in places
    assert "#1 李世民（在太极殿）" in user.split("可选人物")[1]


def test_menu_places_keeps_occupied_caps_stocked_and_draws_bare() -> None:
    import random
    from collections import Counter

    from engine.event import _MAX_BARE_PLACES, _MAX_STOCKED_PLACES, _menu_places
    from core.interfaces.place import Place

    stocked_count = _MAX_STOCKED_PLACES + 2
    places = [Place(place_id=f"p{i}", name=f"地{i}") for i in range(1 + stocked_count + 10)]
    headcount = Counter({"p0": 2})
    # p0 is occupied with nothing on the ground; p1..p7 have i things; the rest are bare.
    things = Counter({f"p{i}": i for i in range(1, 1 + stocked_count)})
    bare = {p.place_id for p in places[1 + stocked_count:]}

    chosen = [p.place_id for p in _menu_places(places, headcount, things, random.Random(0))]

    assert chosen[0] == "p0"                                    # occupied, kept regardless
    assert "p1" not in chosen and "p2" not in chosen           # fewest things, cut first
    assert len([c for c in chosen if c in bare]) == _MAX_BARE_PLACES
    assert len(chosen) == 1 + _MAX_STOCKED_PLACES + _MAX_BARE_PLACES
    assert chosen == sorted(chosen, key=lambda pid: int(pid[1:]))   # input order kept


def test_menu_places_rotates_bare_places_across_steps() -> None:
    """No bare place is shut out for good: over enough steps every one gets drawn."""
    import random
    from collections import Counter

    from engine.event import _menu_places
    from core.interfaces.place import Place

    places = [Place(place_id=f"p{i}", name=f"地{i}") for i in range(12)]
    seen: set[str] = set()
    for step in range(60):
        seen.update(p.place_id for p in _menu_places(
            places, Counter({"p0": 1}), Counter(), random.Random(step)))

    assert seen == {p.place_id for p in places}


@pytest.mark.asyncio
async def test_entity_menu_hides_what_a_takeable_thing_on_the_ground_says(container) -> None:
    """A portable letter on the ground: those present can't read it, and the editor shouldn't see
    its content either (same rule as readable_by)."""
    from world.models import EntityPresence, WorldEntity, WorldEntityType

    env, agents, directory = _ground_world(container)
    env.register_entity(WorldEntity(
        entity_id="dropped", name="遗信", entity_type=WorldEntityType.ITEM, is_takeable=True,
        content="今夜三更动手。", presence=EntityPresence.AT_LOCATION, presence_ref="palace",
    ))
    router = _EventRouter([_gate(), _plan_json(broadcast={"content": "x", "severity": "low"})])
    system = _event_system(container, router=router, directory=directory, environment=env,
                           check_interval=1)

    await _run_check(system, current_step=1, world_time=_world_time(1), world_id="world-1",
                     all_agents=agents, locations=env.space.all_places(),
                     entities=env.all_live_entities())

    menu = router.calls[-1][1][-1].content.split("地上的东西")[1]
    assert "遗信" in menu
    assert "今夜三更动手" not in menu
    assert "明日辰时开城门" in menu      # the notice is fixed in place; those present can read it


@pytest.mark.asyncio
async def test_spawn_alter_destroy_land_in_the_world(container) -> None:
    env, agents, directory = _ground_world(container)
    router = _EventRouter([_gate(), _plan_json(
        narrative_desc="宫门换了告示",
        spawn={"location": [p.place_id for p in env.space.all_places()].index("palace") + 1,
               "entity_type": "item", "name": "血书", "description": "一方白绢",
               "content": "速救东宫", "observation": "殿门口落下一方白绢。"},
        alter={"entity": _ground_index(env, "notice"), "state": "撕破", "observation": "告示被撕去半边。"},
    )])
    system = _event_system(container, router=router, directory=directory, environment=env,
                           check_interval=1)

    fired = await _run_check(system, current_step=1, world_time=_world_time(1), world_id="world-1",
                             all_agents=agents, locations=env.space.all_places(),
                     entities=env.all_live_entities())

    assert fired is not None and fired["dispatched_to"] == ["mutation"]
    assert env.get_entity("notice").state == "撕破"
    [made] = [e for e in env.get_items_at("palace") if e.name == "血书"]
    assert (made.content, made.is_takeable) == ("速救东宫", True)

    router2 = _EventRouter([_gate(), _plan_json(
        destroy={"entity": _ground_index(env, "notice"), "observation": "告示被一把火烧了。"},
    )])
    system2 = _event_system(container, router=router2, directory=directory, environment=env,
                            check_interval=1)
    await _run_check(system2, current_step=2, world_time=_world_time(2), world_id="world-1",
                     all_agents=agents, locations=env.space.all_places(),
                     entities=env.all_live_entities())
    assert env.get_entity("notice").is_destroyed
    assert not made.is_destroyed


@pytest.mark.asyncio
@pytest.mark.parametrize("things", [
    {"destroy": {"entity": 9, "observation": "…"}},                            # out of range
    {"destroy": {"entity": 1}},                                                # missing observation
    {"alter": {"entity": 1, "observation": "告示动了。"}},                       # nothing changed
    {"spawn": {"location": 1, "entity_type": "item", "observation": "…"}},     # no name
], ids=["out-of-range", "no-observation", "no-change", "no-name"])
async def test_malformed_entity_slots_are_dropped(container, things) -> None:
    env, agents, directory = _ground_world(container)
    router = _EventRouter([_gate(), _plan_json(**things)])
    system = _event_system(container, router=router, directory=directory, environment=env,
                           check_interval=1)

    fired = await _run_check(system, current_step=1, world_time=_world_time(1), world_id="world-1",
                             all_agents=agents, locations=env.space.all_places(),
                     entities=env.all_live_entities())

    assert fired is None                           # the only channel came up empty -> not recorded, no quota used
    assert system.list_events() == []
    assert env.get_entity("notice").state == "intact" and not env.get_entity("notice").is_destroyed


@pytest.mark.asyncio
async def test_a_thing_picked_up_before_commit_is_spared_but_the_broadcast_still_goes(container) -> None:
    from core.interfaces.action import EntityStateChange
    from world.models import EntityPresence, WorldEntity, WorldEntityType

    env, agents, directory = _ground_world(container)
    env.register_entity(WorldEntity(
        entity_id="seal", name="兵符", entity_type=WorldEntityType.ITEM, is_takeable=True,
        presence=EntityPresence.AT_LOCATION, presence_ref="palace",
    ))
    bc = BroadcastChannel()
    router = _EventRouter([_gate(), _plan_json(
        destroy={"entity": _ground_index(env, "seal"), "observation": "兵符碎了。"},
        broadcast={"content": "殿中一声脆响", "severity": "medium", "location_scope": 1},
    )])
    system = _event_system(container, router=router, directory=directory, environment=env,
                           broadcast_channel=bc, check_interval=1)

    plan = await system._generate_plan(
        1, _world_time(1), "world-1", agents, env.space.all_places(), env.all_live_entities())
    env.change_entity_state(EntityStateChange(entity_id="seal", owner_id="a1"))
    fired = await system._commit_plan(plan, 1, agents)

    assert fired is not None and fired.event.dispatched_to == ["broadcast"]
    assert not env.get_entity("seal").is_destroyed
    assert len(bc.peek_pending()) == 1
