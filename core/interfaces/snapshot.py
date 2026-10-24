"""Snapshot contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.interfaces.message import Message
from core.interfaces.perception import Broadcast


@dataclass
class AgentState:
    """One agent's committed state: the part of its ``WorldState.agents`` record a restore
    decodes (``Agent.committed_state`` / ``Agent.apply_stored_state``). It holds what a restore
    needs to bring back the agent as it was; short-lived bookkeeping whose loss soon heals stays
    out (see ``Agent``). Nested values are JSON-native dicts and lists."""

    world_id: str
    agent_id: str
    step: int
    # ``EmotionState`` as a dict: primary (an ``EmotionType`` value), intensity, valence, triggered_by.
    emotion: dict[str, object]
    active_needs: list[str]
    dominant_need: str | None
    long_term_goals: list[str]
    short_term_goals: list[str]
    current_location: str
    activity_status: str
    activity_target: str | None
    action_status: str
    current_action: str | None
    action_remaining_steps: int
    last_action: str | None
    last_action_result: str | None
    last_action_succeeded: bool | None
    short_term_goal_entities: list[dict[str, object]] = field(default_factory=list)
    need_intensities: dict[str, float] = field(default_factory=dict)
    vitality: float = 1.0
    # A dict, not a BodyCondition, so the file and in-memory snapshots agree (see
    # condition_to_dict); agent.state_codec.decode_agent_state rebuilds it.
    condition: dict[str, object] | None = None
    # 0 = never decided; the scheduler's cadence gate reads that as starved and admits it, so a
    # fresh world's step 1 isn't empty.
    last_decision_step: int = 0
    # Others' acts landed on him since he last perceived (StateLayer.acted_upon).
    acted_upon: list[str] = field(default_factory=list)
    long_term_goal_entities: list[dict[str, object]] = field(default_factory=list)
    # The memory id counter. Restarted, it would hand out ids already in use and overwrite memories.
    memory_seq: int = 0


@dataclass
class WorldState:
    """The world at the end of a step, whole: every body, thing and relation.

    Records are JSON-native dicts in the shapes their producers write (``snapshot_agent_state``,
    ``EnvironmentSystem`` and the relation snapshot). Nothing is left out for being private: who
    may see what is decided by the views built from a snapshot, not here. Places are not here:
    they are fixed at build and live in the world's stored config.
    """

    # Agent id → ``snapshot_agent_state`` (every ``AgentState`` field plus identity) and where the
    # body is (``location_id``: a place, or ``IN_TRANSIT`` between two waypoints), its ``transit``
    # while mid-move and ``displaced``. The only committed copy of the agent: restore reads it.
    agents: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Npc id → the whole Npc, with ``location_id``, ``displaced`` and this beat's ``outcome``.
    npcs: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Entity id → every field, tombstones included (restore must not resurrect a destroyed seed).
    entities: dict[str, dict[str, Any]] = field(default_factory=dict)
    # ``"from->to"`` → the full relation record (``relation_to_snapshot_dict``), numbers included.
    # The only committed copy: restore hands each agent its own back.
    relations: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class EngineState:
    """What the engine needs to resume from this step and nothing else reads."""

    # Queued, not yet delivered: a message in flight, a death notice due next step. Unsaved, a
    # restore would drop them and survivors would assume the dead alive forever.
    pending_messages: list[Message] = field(default_factory=list)
    pending_broadcasts: list[Broadcast] = field(default_factory=list)
    # Next ordinal to assign on the emission timeline; restore resumes strictly increasing.
    event_seq: int = 0
    # Multi-step acts still under way (``ActionExecutionState.to_payload``). Unsaved, a restore
    # would drop them: half-done acts vanish with no result and nobody remembers them.
    active_executions: list[dict[str, Any]] = field(default_factory=list)
    # ``EnvironmentSystem.engine_state``: in-transit origins and the ambient annotations
    # (this step's and the ones carried to the next).
    environment: dict[str, Any] = field(default_factory=dict)
    # The scheduler's batches for the step (diagnostics).
    schedule: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldSnapshot:
    """One step as the engine records it: the world it ended in, what happened in it, and what
    the engine needs to resume from it.

    Complete and engine-shaped. Views for readers (``interaction.models.StepView``) are built
    from it and decide what each reader may see.
    """

    world_id: str
    step: int
    timestamp: datetime
    # The world's clock:
    #
    #   {"hour": 19, "minute": 0,                           ← CODE layer, structured
    #    "iso":   "step=0013 time=19:00 elapsed=68400s",    ← CODE layer, log string
    #    "label": "武德九年，六月初一，酉时，夕阳西下"}          ← NARRATIVE layer
    #
    # ``iso`` is never parsed. ``label``'s shape belongs to the world's calendar, so it is only
    # passed through whole. See WorldTime.clock_payload.
    world_time: dict[str, Any] = field(default_factory=dict)
    state: WorldState = field(default_factory=WorldState)
    # What happened this step, in emission order, each event once (see engine.step_record):
    # ``action`` (one execution's beat, every participant's part folded in), ``message`` (a
    # delivery), ``broadcast`` and ``injection``. Every event carries its ``seq``.
    events: list[dict[str, Any]] = field(default_factory=list)
    engine: EngineState = field(default_factory=EngineState)
    # The actor's thread in this step (``core.interfaces.actor.ActorEntry`` as JSON), in order;
    # empty when nobody performs a character.
    actor_thread: list[dict[str, Any]] = field(default_factory=list)
    # Step 0 only: the world's setup (``world.setup.WorldSetup`` as a dict), frozen when the world
    # was confirmed. Restore and reset rebuild the agents from it.
    setup: dict[str, Any] = field(default_factory=dict)

    @property
    def time_label(self) -> str:
        """The world's own name for this moment — the ONLY clock that may enter narrative."""
        return str(self.world_time.get("label", ""))

    @property
    def clock(self) -> str:
        """The machine clock as a log string (``iso_label``). Code layer; nothing parses it."""
        return str(self.world_time.get("iso", ""))

    def events_of(self, kind: str) -> list[dict[str, Any]]:
        """This step's events of one kind, in emission order. Empty when none happened — and
        that is a fact: never fill a quiet step from the state."""
        return [e for e in self.events if e.get("kind") == kind]


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
    async def save_world_setup(self, world_id: str, data: dict[str, Any]) -> None:
        """Persist the world's setup (``world.setup.WorldSetup`` as a dict) in its own file:
        written at build and rewritten by every revision until the world is confirmed."""

    @abstractmethod
    async def load_world_setup(self, world_id: str) -> dict[str, Any] | None:
        """Read the world's setup file, or None if absent (never built, or confirmed)."""

    @abstractmethod
    async def delete_world_setup(self, world_id: str) -> None:
        """Delete the setup file once the setup is frozen into step 0."""

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
        """Read one frozen image, or None if this world has none."""
