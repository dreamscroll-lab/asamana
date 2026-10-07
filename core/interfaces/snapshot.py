"""Snapshot contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.interfaces.message import Message
from core.interfaces.perception import Broadcast


@dataclass
class WorldSnapshot:
    """Persisted step snapshot."""

    world_id: str
    step: int
    timestamp: datetime
    # The world's clock, in the same shape the live event-bus payload carries:
    #
    #   {"hour": 19, "minute": 0,                           ← CODE layer, structured
    #    "iso":   "step=0013 time=19:00 elapsed=68400s",    ← CODE layer, log string
    #    "label": "武德九年，六月初一，酉时，夕阳西下"}          ← NARRATIVE layer
    #
    # ``iso`` is never parsed. ``label``'s shape belongs to the world's calendar, so it is only
    # passed through whole. Keep the clock under this one key: a second copy in ``metadata`` lets
    # the same name mean different clocks. See WorldTime.clock_payload.
    world_time: dict[str, Any] = field(default_factory=dict)
    agent_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    agent_relations: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_messages: list[Message] = field(default_factory=list)
    # Must be saved: death notices queue for the next step, straddling the snapshot boundary, and
    # unsaved, survivors would assume the dead alive forever after a restore. Not
    # metadata["broadcasts"], which holds what this step already delivered (for replay).
    pending_broadcasts: list[Broadcast] = field(default_factory=list)
    events_this_step: list[dict[str, Any]] = field(default_factory=list)
    actions_this_step: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def time_label(self) -> str:
        """The world's own name for this moment — the ONLY clock that may enter narrative."""
        return str(self.world_time.get("label", ""))

    @property
    def clock(self) -> str:
        """The machine clock as a log string (``iso_label``). Code layer; nothing parses it."""
        return str(self.world_time.get("iso", ""))

    @property
    def agent_summaries(self) -> list[dict[str, Any]]:
        """The actions taken this step. Empty when nobody acted — and that is a fact.

        **Never fall back to ``agent_states``**: that renders a non-event as blank deeds
        (CLAUDE.md Rule 1) and hides runs whose decisions were all dropped behind what looks
        like a quiet night.
        """

        return list(self.actions_this_step)

    @property
    def event_summaries(self) -> list[dict[str, Any]]:
        return list(self.events_this_step)


class SnapshotProvider(ABC):
    """Abstract snapshot persistence."""

    @abstractmethod
    async def save(self, world_id: str, step: int, snapshot: WorldSnapshot) -> None:
        """Persist a world snapshot."""

    @abstractmethod
    async def load(self, world_id: str, step: int) -> WorldSnapshot | None:
        """Read a single snapshot."""

    @abstractmethod
    async def list_steps(self, world_id: str) -> list[int]:
        """List persisted steps for a world."""

    @abstractmethod
    async def load_latest(self, world_id: str) -> WorldSnapshot | None:
        """Load the latest snapshot for a world."""

    @abstractmethod
    async def delete_steps_after(self, world_id: str, step: int) -> None:
        """Delete all snapshots with step number greater than *step*."""

    @abstractmethod
    async def delete_world(self, world_id: str) -> None:
        """Purge every persisted snapshot and world-config asset for *world_id*.

        Called by whole-world deletion; irreversible. Concrete providers must
        also drop any in-memory state they keep for the world.
        """

    @abstractmethod
    async def save_world_config(self, world_id: str, data: dict[str, Any]) -> None:
        """Persist the world's serialized world-config asset (written once at build)."""

    @abstractmethod
    async def load_world_config(self, world_id: str) -> dict[str, Any] | None:
        """Read the world's serialized world-config asset, or None if absent."""

    @abstractmethod
    async def save_world_map(self, world_id: str, data: dict[str, Any]) -> None:
        """Persist the world's frozen render-map (.tmj) asset (written once at build).

        This is the renderer's geometry, frozen per world so later template edits
        never shift a built world's rendered map. Distinct from the world config,
        which the engine simulates on.
        """

    @abstractmethod
    async def load_world_map(self, world_id: str) -> dict[str, Any] | None:
        """Read the world's frozen render-map asset, or None if absent."""

    @abstractmethod
    async def save_world_asset(self, world_id: str, rel_path: str, data: bytes) -> None:
        """Persist one image the world's frozen map names (written once at build).

        The map and the pixels it points at must be frozen together: the template's tilesets
        keep being edited, and a moved tile would render garbage in an old world.
        `rel_path` is the path exactly as written in the .tmj.
        """

    @abstractmethod
    async def load_world_asset(self, world_id: str, rel_path: str) -> bytes | None:
        """Read one frozen image, or None if this world has none (pre-freeze worlds)."""
