"""Shared by every stage harness: restoring a world with tracing, sandboxing an agent's writes,
the dry-run runtime, and views more than one stage reports."""

from __future__ import annotations

import dataclasses

from config.models import Config
from core.container import Container
from core.interfaces.perception import PerceivedIdentity
from engine.broadcast import BroadcastChannel
from engine.clock import GlobalClock
from engine.narrative_editor import NarrativeEditorSettings
from engine.runtime import NarrativeRuntime
from engine.scheduler import AgentScheduler
from world import World
from world.initializer import WorldInitializer
from providers.vector_store.in_memory import InMemoryVectorStore

from tuning.trace import InMemoryTraceSink, traced_router
from tuning.dry_run import isolate_container


async def restore_traced(
    container: Container, config: Config, world_id: str, sink: InMemoryTraceSink
) -> tuple[Container, World, int, str]:
    """Wrap the router for tracing, restore the world, and resolve the dry-run step."""
    router = traced_router(
        container.llm_router, sink, max_concurrent=config.engine.max_concurrent_llm
    )
    traced_container = dataclasses.replace(await isolate_container(container, world_id), llm_router=router)
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
        narrative_editor_settings=NarrativeEditorSettings(
            core_tension=world.setup.lore.core_tension,
            narrative_theme=world.setup.lore.narrative_theme,
            check_interval=99, max_injections_per_window=0,
        ),
        snapshot_provider=container.snapshot,
        event_bus=container.event_bus,
        broadcast_channel=bc,
        directory=world.directory,
        executor_registry=registry,
        llm_router=container.llm_router,
    )


def isolate_agent(agent, name_by_id: dict | None = None) -> None:
    """Prepare one agent for a single-stage scenario. Storage is already isolated by
    ``restore_traced`` (``isolate_container``).

    Starts the agent's vector memory empty, so the scenario's seeded memories are the only ones
    the stage can retrieve. Also warms the known-name cache from the directory: in production
    perception populates _known_agents each step, so by finalize the doer knows co-cast names;
    the dry-run skips perception, so without warming the experiential rewrite would fall back
    to raw ids and leak them into memory prose.
    """
    agent.memory_system._vector_store = InMemoryVectorStore()  # noqa: SLF001 — test scaffolding
    for aid, nm in (name_by_id or {}).items():
        agent.remember_agent(aid, PerceivedIdentity(name=nm))
