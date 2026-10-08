"""No-op trace sink — selected when observability is disabled (zero overhead)."""

from __future__ import annotations

from typing import Any

from core.factory import ComponentKind, ProviderFactory
from core.interfaces.trace import LLMCallTrace, StepTrace, TraceSink


@ProviderFactory.register("null", kind=ComponentKind.TRACE)
class NullTraceSink(TraceSink):
    """Discards every record and reads back empty."""

    def record_llm_call(self, trace: LLMCallTrace) -> None:
        return None

    def record_step(self, trace: StepTrace) -> None:
        return None

    def mark_call_unadopted(
        self, world_id: str, *, step: int, agent_id: str, stage: str, reason: str
    ) -> None:
        return None

    def annotate_recorded_call(
        self, world_id: str, *, step: int, agent_id: str, stage: str, **fields: Any
    ) -> None:
        return None

    async def flush(self, world_id: str, step: int | None = None) -> None:
        return None

    def delete_world(self, world_id: str) -> None:
        return None

    def delete_run_traces(self, world_id: str) -> None:
        return None

    def read_calls(
        self,
        world_id: str,
        *,
        step: int | None = None,
        stage: str | None = None,
        agent_id: str | None = None,
    ) -> list[LLMCallTrace]:
        return []

    def read_step_summaries(self, world_id: str) -> list[StepTrace]:
        return []

    def list_steps(self, world_id: str) -> list[int]:
        return []

    def list_agents(self, world_id: str) -> list[str]:
        return []

    def list_stages(self, world_id: str) -> list[str]:
        return []
