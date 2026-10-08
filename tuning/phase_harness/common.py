"""Shared by every stage harness: restoring a world with tracing, sandboxing an agent's writes,
the dry-run runtime, and views more than one stage reports."""

from __future__ import annotations

import dataclasses

from config.models import Config
from core.container import Container
from core.interfaces.agent_store import AgentRelation, AgentState, AgentStoreProvider
from core.interfaces.perception import PerceivedIdentity
from engine.broadcast import BroadcastChannel
from engine.clock import GlobalClock
from engine.event import EventSettings
from engine.runtime import NarrativeRuntime
from engine.scheduler import AgentScheduler
from world import World
from world.initializer import WorldInitializer
from providers.vector_store.in_memory import InMemoryVectorStore

from tuning.trace import InMemoryTraceSink, traced_router


async def restore_traced(
    container: Container, config: Config, world_id: str, sink: InMemoryTraceSink
) -> tuple[Container, World, int, str]:
    """Wrap the router for tracing, restore the world, and resolve the dry-run step."""
    router = traced_router(
        container.llm_router, sink, max_concurrent=config.engine.max_concurrent_llm
    )
    traced_container = dataclasses.replace(container, llm_router=router)
    world = await WorldInitializer(traced_container).restore(world_id)
    run_step = world.current_step + 1
    world_time_label = GlobalClock(world.clock_config, start_step=run_step).current.time_label
    return traced_container, world, run_step, world_time_label


def emotion_view(emotion) -> dict | None:
    """JSON-safe view of a perception-emotion EmotionState (None → None)."""
    if emotion is None:
        return None
    primary = emotion.primary
    return {
        "primary": primary.value if hasattr(primary, "value") else str(primary),
        "intensity": round(emotion.intensity, 3),
        "valence": round(emotion.valence, 3),
        "reason": emotion.triggered_by,
    }


def make_action_runtime(container, config, world_id, world, registry, message_system) -> NarrativeRuntime:
    bc = BroadcastChannel()
    return NarrativeRuntime(
        world_id=world_id,
        clock=GlobalClock(world.clock_config),
        scheduler=AgentScheduler(),
        environment=world.environment,
        message_system=message_system,
        event_settings=EventSettings(
            core_tension=world.analysis.core_tension,
            narrative_theme=world.analysis.narrative_theme,
            check_interval=99, max_events_per_window=0,
        ),
        snapshot_provider=container.snapshot,
        agent_store=container.agent_store,
        event_bus=container.event_bus,
        broadcast_channel=bc,
        directory=world.directory,
        executor_registry=registry,
        llm_router=container.llm_router,
    )


class _DryRunAgentStore(AgentStoreProvider):
    """Read-through / write-capture wrapper around the real store.

    Loads delegate to the real store (so the dry-run sees the baseline world's agent
    state + relations); saves go to in-memory shadows (and shadow subsequent loads).
    Nothing the feedback layer writes touches ./data — keeps the baseline world unpolluted.
    """

    def __init__(self, real: AgentStoreProvider) -> None:
        self._real = real
        self._states: dict[tuple[str, str], AgentState] = {}
        self._relations: dict[tuple[str, str, str], AgentRelation] = {}

    async def save_agent_state(self, world_id: str, agent_id: str, state: AgentState) -> None:
        self._states[(world_id, agent_id)] = state

    async def load_agent_state(self, world_id: str, agent_id: str) -> AgentState | None:
        if (world_id, agent_id) in self._states:
            return self._states[(world_id, agent_id)]
        return await self._real.load_agent_state(world_id, agent_id)

    async def save_relation(self, relation: AgentRelation) -> None:
        self._relations[(relation.world_id, relation.from_id, relation.to_id)] = relation

    async def load_relation(self, world_id: str, from_id: str, to_id: str) -> AgentRelation | None:
        key = (world_id, from_id, to_id)
        if key in self._relations:
            return self._relations[key]
        return await self._real.load_relation(world_id, from_id, to_id)

    async def load_all_relations(self, world_id: str, agent_id: str) -> list[AgentRelation]:
        return await self._real.load_all_relations(world_id, agent_id)

    async def save_initial_agent_state(self, world_id: str, agent_id: str, state: AgentState) -> None:
        return None

    async def load_initial_agent_state(self, world_id: str, agent_id: str) -> AgentState | None:
        return await self._real.load_initial_agent_state(world_id, agent_id)

    async def save_initial_relation(self, relation: AgentRelation) -> None:
        return None

    async def load_all_initial_relations(self, world_id: str, agent_id: str) -> list[AgentRelation]:
        return await self._real.load_all_initial_relations(world_id, agent_id)

    async def list_agent_ids(self, world_id: str) -> list[str]:
        return await self._real.list_agent_ids(world_id)

    async def clear_relations(self, world_id: str) -> None:
        # Shadow-only: never mutate the baseline world's persisted relations.
        self._relations = {
            key: rel for key, rel in self._relations.items() if key[0] != world_id
        }

    async def delete_world(self, world_id: str) -> None:
        # Shadow-only: the baseline world's persisted state on ./data is never touched.
        self._states = {key: st for key, st in self._states.items() if key[0] != world_id}
        self._relations = {
            key: rel for key, rel in self._relations.items() if key[0] != world_id
        }


def isolate_agent(agent, name_by_id: dict | None = None) -> None:
    """Swap an agent's persistence to in-memory dry-run shadows (idempotent per harness).

    Also warms the agent's known-name cache from the directory: in production perception
    populates _known_agents each step, so by finalize the actor knows co-cast names;
    the dry-run skips perception, so without warming the experiential rewrite would fall back
    to raw ids and leak them into memory prose.
    """
    dry = _DryRunAgentStore(agent.agent_store)
    agent.agent_store = dry
    agent.relation_system._store = dry  # noqa: SLF001 — test scaffolding
    agent.memory_system._vector_store = InMemoryVectorStore()  # noqa: SLF001
    for aid, nm in (name_by_id or {}).items():
        agent.remember_agent(aid, PerceivedIdentity(name=nm))
