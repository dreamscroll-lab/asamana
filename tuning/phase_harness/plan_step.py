"""Run a single per-step cognition phase (dry-run) with tracing enabled.

``run_plan_step`` restores a built/persisted world, then runs each agent's
``plan_step`` (perception → motivation → decision). plan_step is naturally
non-destructive — it returns an AgentStepPlan without writing memory, relations,
snapshots, or advancing the clock (finalizing the action is what persists). Every LLM call
is captured by the traced router; the structured plan output is recorded as a
PhaseTrace under the ``plan_step`` segment.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

from config.models import Config
from core.container import Container
from core.context import clear_log_context, set_log_context
from core.logging import get_logger
from engine.clock import GlobalClock
from engine.presence import attach_presence
from world.initializer import WorldInitializer

from tuning.plan_view import plan_step_view
from tuning.trace import InMemoryTraceSink, JsonlTraceSink, PhaseTrace, traced_router

logger = get_logger(__name__)


_SEGMENT = "plan_step"


async def run_plan_step(
    container: Container,
    config: Config,
    world_id: str,
    *,
    agent_ids: list[str] | None = None,
    sink: InMemoryTraceSink | None = None,
    trace_dir: str = "./data/tuning_traces",
) -> int:
    """Dry-run plan_step for a restored world. Returns the simulated step.

    Restores ``world_id`` from storage, runs each (active) agent's plan_step
    against the post-restore world state, and records each plan + its LLM calls.
    Nothing is persisted back to the stores.
    """
    trace_sink = sink if sink is not None else JsonlTraceSink(trace_dir, segment=_SEGMENT)
    if isinstance(trace_sink, JsonlTraceSink):
        trace_sink.truncate(world_id)
    router = traced_router(
        container.llm_router, trace_sink, max_concurrent=config.engine.max_concurrent_llm
    )
    traced_container = dataclasses.replace(container, llm_router=router)

    initializer = WorldInitializer(traced_container)
    world = await initializer.restore(world_id)

    run_step = world.current_step + 1
    world_time = GlobalClock(world.clock_config, start_step=run_step).current

    targets = [
        a for aid, a in world.agents.items()
        if a.is_active and (agent_ids is None or aid in agent_ids)
    ]
    for agent in targets:
        spatial = world.environment.spatial_for(
            agent_id=agent.agent_id, step=run_step, world_time=world_time.time_label
        )
        attach_presence(
            spatial, directory=world.directory, agents=world.agents,
            environment=world.environment,
        )
        # Attribute each agent's LLM calls (captured by the traced router) and
        # plan to this step/agent.
        set_log_context(world_id=world_id, agent_id=agent.agent_id, step=str(run_step))
        try:
            plan = await agent.plan_step(step=run_step, spatial=spatial, inbox=[], broadcasts=[])
            trace_sink.record_phase(
                PhaseTrace(
                    world_id=world_id,
                    phase="step.plan",
                    step=run_step,
                    agent_id=agent.agent_id,
                    outputs=plan_step_view(agent, plan),
                    timestamp=datetime.now().isoformat(),
                )
            )
        except Exception as exc:  # noqa: BLE001 — one agent's failure must not abort the run
            logger.warning(
                "plan_step_dryrun_failed",
                extra={"world_id": world_id, "agent_id": agent.agent_id, "error": str(exc)},
            )
    clear_log_context()
    logger.info(
        "tuning_plan_step_dry_run_complete",
        extra={"world_id": world_id, "step": run_step, "agent_count": len(targets)},
    )
    return run_step
