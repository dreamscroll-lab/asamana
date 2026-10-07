"""End-to-end: a real run_step records stage-tagged LLM traces + a StepTrace.

Verifies the observability seams actually fire inside the runtime loop — that
the LLMRouter chokepoint records calls with the stage set by ``observe_stage``,
that world_id/step land on every call, and that the per-step wall-clock total is
captured. Uses the test container's (mock) LLM providers behind a sink-enabled
router so the whole step funnels through the recording path.
"""

from __future__ import annotations

import pytest

from agent.agent import Agent
from agent.decision import DecisionEngine
from agent.memory import MemorySystem
from agent.need import NeedEngine
from agent.personality import PersonalityLayer, SoulLayer
from agent.relation import RelationSystem
from core.interfaces.llm import LLMRouter, LLMScene
from core.interfaces.trace import Stage
from engine.broadcast import BroadcastChannel
from engine.clock import GlobalClock, WorldTimeConfig
from engine.directory import LiveWorldDirectory
from engine.environment import EnvironmentSystem
from engine.event import EventSettings
from engine.executors import build_default_registry
from engine.message_system import MessageSystem
from engine.runtime import NarrativeRuntime
from engine.scheduler import AgentScheduler
from providers.trace import InMemoryTraceSink
from worlds.tiled import TiledWorldConfig


def _traced_router(container, sink: InMemoryTraceSink) -> LLMRouter:
    providers = {scene: container.llm_router.get(scene) for scene in LLMScene}
    return LLMRouter(providers, trace_sink=sink)


def _agent(container, router, *, world_id, agent_id, name, main) -> Agent:
    return Agent(
        world_id=world_id,
        agent_id=agent_id,
        personality=PersonalityLayer(
            soul=SoulLayer(
                name=name, role="court_official", agent_id=agent_id,
                core_traits=["careful"], core_values=["order"], hard_constraints=[],
            )
        ),
        decision_engine=DecisionEngine(router),
        llm_router=router,
        memory_system=MemorySystem(
            router, container.embedding, container.vector_store,
            world_id=world_id, agent_id=agent_id,
        ),
        need_engine=NeedEngine(),
        relation_system=RelationSystem(container.agent_store, world_id=world_id, agent_id=agent_id),
        agent_store=container.agent_store,
        is_main_character=main,
    )


@pytest.mark.asyncio
async def test_run_step_records_stage_tagged_traces(container) -> None:
    world_id = "obs-world"
    sink = InMemoryTraceSink()
    router = _traced_router(container, sink)

    environment = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    message_system = MessageSystem(container.message_provider, world_id=world_id)
    broadcast_channel = BroadcastChannel()
    event_settings = EventSettings(check_interval=2, max_events_per_window=1)
    runtime = NarrativeRuntime(
        world_id=world_id,
        clock=GlobalClock(WorldTimeConfig(start_hour=6, seconds_per_step=60)),
        scheduler=AgentScheduler(),
        environment=environment,
        message_system=message_system,
        event_settings=event_settings,
        snapshot_provider=container.snapshot,
        agent_store=container.agent_store,
        event_bus=container.event_bus,
        broadcast_channel=broadcast_channel,
        directory=LiveWorldDirectory.from_agents({}, environment),
        executor_registry=build_default_registry(
            router, LiveWorldDirectory.from_agents({}, environment), seconds_per_step=60,
        ),
        llm_router=router,
        trace_sink=sink,
    )
    environment.place_agent(agent_id="a1", location_id="palace")
    environment.place_agent(agent_id="a2", location_id="palace")
    agents = [
        _agent(container, router, world_id=world_id, agent_id="a1", name="Li", main=True),
        _agent(container, router, world_id=world_id, agent_id="a2", name="Jc", main=False),
    ]

    await runtime.run_step(agents)

    # The step funnelled at least one LLM call through the chokepoint.
    assert sink.llm_calls, "expected the run_step to record LLM calls"
    # Decision is the most reliable stage to fire (decision engine has the router).
    stages = {c.stage for c in sink.llm_calls}
    assert Stage.DECISION.value in stages
    assert Stage.UNKNOWN.value not in stages
    assert all(c.world_id == world_id and c.step == 1 for c in sink.llm_calls)
    assert {c.agent_id for c in sink.llm_calls if c.stage == Stage.DECISION.value} <= {"a1", "a2"}

    # The per-step wall-clock total is recorded.
    summaries = sink.read_step_summaries(world_id)
    assert len(summaries) == 1
    assert summaries[0].step == 1
    assert summaries[0].wall_ms > 0

    # Every step-loop phase is timed and captured in the per-phase breakdown.
    phase_ms = summaries[0].phase_ms
    assert set(phase_ms) == {
        "event_check", "message", "pressure", "perceive", "interrupt",
        "executor", "plan", "exec", "cognition", "snapshot",
    }
    assert all(v >= 0 for v in phase_ms.values())
