"""In-memory trace sink — used by tests and tuning harnesses."""

from __future__ import annotations

from typing import Any

from core.context import note_call_adoption
from core.factory import ComponentKind, ProviderFactory
from core.interfaces.trace import (
    LLMCallTrace,
    StepTrace,
    TraceSink,
    stage_sort_key,
)


@ProviderFactory.register("in_memory", kind=ComponentKind.TRACE)
class InMemoryTraceSink(TraceSink):
    """Accumulate records in memory. flush() is a no-op (nothing to persist)."""

    def __init__(self) -> None:
        self.llm_calls: list[LLMCallTrace] = []
        self.steps: list[StepTrace] = []

    def record_llm_call(self, trace: LLMCallTrace) -> None:
        self.llm_calls.append(trace)

    def record_step(self, trace: StepTrace) -> None:
        self.steps.append(trace)

    def _recorded(
        self, world_id: str, step: int, agent_id: str, stage: str
    ) -> list[LLMCallTrace]:
        return [c for c in self.llm_calls
                if (c.world_id, c.step, c.agent_id, c.stage) == (world_id, step, agent_id, stage)]

    def mark_call_unadopted(
        self, world_id: str, *, step: int, agent_id: str, stage: str, reason: str
    ) -> None:
        for call in self._recorded(world_id, step, agent_id, stage):
            note_call_adoption(call, False, reason)

    def annotate_recorded_call(
        self, world_id: str, *, step: int, agent_id: str, stage: str, **fields: Any
    ) -> None:
        for call in self._recorded(world_id, step, agent_id, stage):
            call.extra.update(fields)

    async def flush(self, world_id: str, step: int | None = None) -> None:
        return None

    def delete_world(self, world_id: str) -> None:
        self.llm_calls = [c for c in self.llm_calls if c.world_id != world_id]
        self.steps = [s for s in self.steps if s.world_id != world_id]

    def delete_run_traces(self, world_id: str) -> None:
        # Keep this world's build calls (step is None); drop everything a run produced.
        self.llm_calls = [
            c for c in self.llm_calls if c.world_id != world_id or c.step is None
        ]
        self.steps = [s for s in self.steps if s.world_id != world_id]

    def read_calls(
        self,
        world_id: str,
        *,
        step: int | None = None,
        stage: str | None = None,
        agent_id: str | None = None,
    ) -> list[LLMCallTrace]:
        calls = [c for c in self.llm_calls if c.world_id == world_id]
        if step is not None:
            calls = [c for c in calls if c.step == step]
        if stage is not None:
            calls = [c for c in calls if c.stage == stage]
        if agent_id is not None:
            calls = [c for c in calls if c.agent_id == agent_id]
        return calls

    def read_step_summaries(self, world_id: str) -> list[StepTrace]:
        return sorted((s for s in self.steps if s.world_id == world_id), key=lambda s: s.step)

    def list_steps(self, world_id: str) -> list[int]:
        return sorted({c.step for c in self.llm_calls if c.world_id == world_id and c.step is not None})

    def list_agents(self, world_id: str) -> list[str]:
        return sorted({c.agent_id for c in self.llm_calls if c.world_id == world_id and c.agent_id})

    def list_stages(self, world_id: str) -> list[str]:
        stages = {c.stage for c in self.llm_calls if c.world_id == world_id and c.stage}
        return sorted(stages, key=stage_sort_key)
