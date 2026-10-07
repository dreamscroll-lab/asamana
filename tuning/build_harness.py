"""Run the world-build pipeline with tracing enabled.

``run_build`` injects a traced ``LLMRouter`` into a copy of the standard
container (via ``dataclasses.replace`` — no container-wiring change) and runs
``WorldBuilder.build`` with an ``on_phase`` hook that records each structured
product. Every LLM call is captured automatically by the traced router.
Output lands in ``{trace_dir}/{world_id}/build.jsonl``.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any
from uuid import uuid4

from config.models import Config
from core.container import Container
from core.interfaces.trace import to_jsonable
from core.logging import get_logger
from world import World, WorldBuilder
from worlds.tiled import TiledWorldConfig, list_templates

from tuning.agent_view import initialized_agent_view
from tuning.trace import InMemoryTraceSink, JsonlTraceSink, PhaseTrace, traced_router

logger = get_logger(__name__)


async def run_build(
    container: Container,
    config: Config,
    theme: str,
    *,
    template: str | None = None,
    sink: InMemoryTraceSink | None = None,
    trace_dir: str = "./data/tuning_traces",
) -> tuple[str, World]:
    """Build a world with tracing on. Returns ``(world_id, world)``.

    ``sink`` defaults to a ``JsonlTraceSink`` rooted at ``trace_dir``. Tests can
    pass an ``InMemoryTraceSink`` instead.

    ``template`` names the map to build on; left out, every installed map is a
    candidate and the build reads the theme to pick one — same as production.
    Name it when a tuning run has to land on a specific map to stay comparable
    with the runs before it.
    """
    trace_sink = sink if sink is not None else JsonlTraceSink(trace_dir)
    world_id = str(uuid4())

    router = traced_router(
        container.llm_router,
        trace_sink,
        max_concurrent=config.engine.max_concurrent_llm,
    )
    traced_container = dataclasses.replace(container, llm_router=router)

    def on_phase(phase: str, product: Any) -> None:
        # Observation only — never let a recording failure interrupt the build.
        try:
            if phase == "world_build.initialization":
                # product is the assembled World; show each agent's post-init view.
                outputs: Any = [
                    initialized_agent_view(agent) for agent in product.agents.values()
                ]
            else:
                outputs = to_jsonable(product)
            trace_sink.record_phase(
                PhaseTrace(
                    world_id=world_id,
                    phase=phase,
                    outputs=outputs,
                    timestamp=datetime.now().isoformat(),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("phase_trace_record_failed", extra={"phase": phase, "error": str(exc)})

    builder = WorldBuilder(traced_container)
    world = await builder.build(
        theme=theme,
        world_id=world_id,
        world_configs=[
            TiledWorldConfig(template=name)
            for name in ([template] if template else list_templates())
        ],
        min_agents=config.world.min_agents,
        max_agents=config.world.max_agents,
        on_phase=on_phase,
    )
    logger.info(
        "tuning_build_complete",
        extra={"world_id": world_id, "world_name": world.analysis.world_name, "theme": theme},
    )
    return world_id, world
