"""Snapshot-backed replay helper."""

from __future__ import annotations

from core.interfaces.snapshot import SnapshotProvider
from engine.injection import Author

from interaction.models import InterventionRecord, StepEvent


class Replayer:
    """Read historical world state from persisted snapshots for one world."""

    def __init__(
        self,
        world_id: str,
        snapshot_provider: SnapshotProvider,
    ) -> None:
        self._world_id = world_id
        self._snapshot_provider = snapshot_provider

    async def list_steps(self) -> list[int]:
        """Return all persisted snapshot steps for the world."""

        return await self._snapshot_provider.list_steps(self._world_id)

    async def get_step(self, step: int) -> StepEvent:
        """Read a single replayable step as an observation-ready StepEvent."""

        snapshot = await self._snapshot_provider.load(self._world_id, step)
        if snapshot is None:
            raise ValueError(f"Snapshot not found for world={self._world_id} step={step}")
        return StepEvent.from_snapshot(snapshot)

    async def list_interventions(self, *, after: int = -1) -> list[InterventionRecord]:
        """Every director intervention this world has recorded, oldest first.

        ``after`` reads only steps beyond a point already scanned (one snapshot load per step),
        so a polling caller can extend its earlier answer. The LLM editor's injections are
        skipped: this is the human's own record.
        """
        records: list[InterventionRecord] = []
        for step in await self.list_steps():
            if step <= after:
                continue
            snapshot = await self._snapshot_provider.load(self._world_id, step)
            if snapshot is None:
                continue
            for record in snapshot.event_summaries:
                if str(record.get("authored_by", "")) != Author.DIRECTOR.value:
                    continue
                records.append(InterventionRecord(
                    step=snapshot.step,
                    time_label=snapshot.time_label,
                    directive_text=str(record.get("directive_text", "")),
                    narrative=str(record.get("narrative_desc", "")),
                ))
        return records
