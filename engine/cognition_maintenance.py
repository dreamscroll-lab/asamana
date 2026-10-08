"""Periodic slow cognition for living agents: memory decay and compression, reflection,
long-term goal revision, relation label evolution."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, List

from core.context import observe_stage
from core.interfaces.trace import Stage
from core.logging import get_logger

if TYPE_CHECKING:
    from agent.agent import Agent

logger = get_logger(__name__)


class CognitionMaintenance:
    """Runs the slow-cognition phases that fall due on a step; an interval of 0 turns a phase off."""

    def __init__(
        self,
        *,
        memory_decay_interval: int = 5,
        memory_compression_interval: int = 20,
        reflection_interval: int = 20,
        label_evolution_interval: int = 20,
        long_term_goal_revision_interval: int = 30,
        compression_enabled: bool = True,
        reflection_enabled: bool = True,
        relation_evolution_enabled: bool = True,
        long_term_goal_revision_enabled: bool = True,
    ) -> None:
        self._memory_decay_interval = memory_decay_interval
        self._memory_compression_interval = memory_compression_interval
        self._reflection_interval = reflection_interval
        self._label_evolution_interval = label_evolution_interval
        self._long_term_goal_revision_interval = long_term_goal_revision_interval
        # Ablation switches, ANDed with each interval.
        self._compression_enabled = compression_enabled
        self._reflection_enabled = reflection_enabled
        self._relation_evolution_enabled = relation_evolution_enabled
        self._long_term_goal_revision_enabled = long_term_goal_revision_enabled

    async def run(self, agents: List[Agent], *, step: int) -> None:
        """Run the slow-cognition phases due this step: memory decay, compression, reflection,
        long-term goal revision, relation label evolution.

        The memory phases must run in the order decay → compression → reflection: compression's
        candidate filter reads decay_score, and compression deletes memories that reflection
        must not source. Only compression deletes memories here.

        Compression / reflection / goal / relation failures log a warning and skip that agent
        (Rule 1). Decay absorbs its own I/O failures, so an exception from it is a real bug and
        is not swallowed. Every phase runs for every living agent (§5: no per-tier cognition).
        """
        # Agents who died this step are already inactive; without this filter they would get
        # posthumous insights/memories.
        agents = [agent for agent in agents if agent.is_active]

        # Compression/reflection mutate _entries and the vector store, so drain in-flight
        # background writes first; otherwise a write ends up phantom or an insight sources an
        # unpersisted memory. Decay doesn't need this.
        runs_compression = (
            self._compression_enabled and self._memory_compression_interval > 0
            and step % self._memory_compression_interval == 0
        )
        runs_reflection = (
            self._reflection_enabled and self._reflection_interval > 0
            and step % self._reflection_interval == 0
        )
        if runs_compression or runs_reflection:
            await asyncio.gather(
                *(agent.memory_system.drain_writes() for agent in agents),
                return_exceptions=True,
            )

        if self._memory_decay_interval > 0 and step % self._memory_decay_interval == 0:
            for agent in agents:
                await agent.memory_system.apply_decay(step)

        # Compression acts only on the FACTUAL stream: it merges at the information level. The
        # experiential stream is never compressed; it is distilled by reflection and fades via decay.
        if self._compression_enabled and self._memory_compression_interval > 0 and step % self._memory_compression_interval == 0:
            from agent.memory_types import MemoryStream

            async def _compress_agent(agent: "Agent") -> None:
                try:
                    with observe_stage(Stage.COMPRESS, agent_id=agent.agent_id):
                        await agent.memory_system.compress(MemoryStream.FACTUAL, current_step=step)
                except Exception as exc:
                    logger.warning(
                        "compression_failed",
                        extra={"agent_id": agent.agent_id, "stream": MemoryStream.FACTUAL.value, "error": str(exc)},
                    )

            await asyncio.gather(
                *(_compress_agent(agent) for agent in agents),
                return_exceptions=True,
            )

        if self._reflection_enabled and self._reflection_interval > 0 and step % self._reflection_interval == 0:

            async def _reflect_agent(agent: "Agent") -> None:
                if agent.reflection_engine is None:
                    return
                try:
                    with observe_stage(Stage.REFLECTION, agent_id=agent.agent_id):
                        await agent.reflection_engine.reflect(step)
                except Exception as exc:
                    logger.warning(
                        "reflection_failed",
                        extra={"agent_id": agent.agent_id, "step": step, "error": str(exc)},
                    )

            await asyncio.gather(
                *(_reflect_agent(agent) for agent in agents),
                return_exceptions=True,
            )

        # Long-term goal revision has its own cycle and reads recent factual + experiential
        # memories (see Agent.revise_long_term_goals), not reflection's insights.
        if self._long_term_goal_revision_enabled and self._long_term_goal_revision_interval > 0 and step % self._long_term_goal_revision_interval == 0:

            async def _revise_long_term_goals(agent: "Agent") -> None:
                try:
                    with observe_stage(Stage.LONG_TERM_GOALS, agent_id=agent.agent_id):
                        await agent.revise_long_term_goals(step=step)
                except Exception as exc:
                    logger.warning(
                        "long_term_goal_revision_failed",
                        extra={"agent_id": agent.agent_id, "step": step, "error": str(exc)},
                    )

            await asyncio.gather(
                *(_revise_long_term_goals(agent) for agent in agents),
                return_exceptions=True,
            )

        # Safe to run concurrently: each agent writes only its own outgoing relations.
        if self._relation_evolution_enabled and self._label_evolution_interval > 0 and step % self._label_evolution_interval == 0:

            async def _evolve_agent(agent: "Agent") -> None:
                if agent.relation_evolution is None:
                    return
                try:
                    with observe_stage(Stage.RELATION_EVOLUTION, agent_id=agent.agent_id):
                        await agent.relation_evolution.evaluate(step)
                except Exception as exc:
                    logger.warning(
                        "label_evolution_failed",
                        extra={"agent_id": agent.agent_id, "step": step, "error": str(exc)},
                    )

            await asyncio.gather(
                *(_evolve_agent(agent) for agent in agents),
                return_exceptions=True,
            )
