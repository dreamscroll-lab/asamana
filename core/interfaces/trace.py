"""LLM observability contracts.

``LLMRouter`` records each call as an ``LLMCallTrace`` tagged with its **cognition stage**; each
runtime step emits a ``StepTrace``.

``stage`` can't be derived from ``scene``: one scene spans several stages (``AGENT_DECISION_*``
covers perception / decision / feedback), so it is set at each cognition seam via
``core.context.observe_stage``.

Pure code-layer records, never in a prompt or embedding, so ids are legal here.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Stage taxonomy
# ---------------------------------------------------------------------------

class Stage(str, Enum):
    """Cognition stage an LLM call ran in — the authoritative observability bucket.

    Values are English snake_case and used directly as UI labels (no separate
    label mapping). ``COGNITION_ORDER`` below fixes their display order along the
    cognition loop.
    """

    WORLD_INIT = "world_init"
    PRESSURE = "pressure"
    PERCEPTION = "perception"
    INTERRUPT = "interrupt"
    MOTIVATION = "motivation"
    DECISION = "decision"
    ACTION = "action"
    FEEDBACK = "feedback"
    MEMORY = "memory"
    COMPRESS = "compress"
    REFLECTION = "reflection"
    RELATION_EVOLUTION = "relation_evolution"
    LONG_TERM_GOALS = "long_term_goals"
    # Separate slots for the world's two authors: "did a human cause this beat?" is the first
    # question when observing (see engine/injection.Author).
    DIRECTOR = "director"    # human director: free text → injection plan / suggestions
    EVENT = "event"          # LLM event editor: pacing gate + authoring
    # No stage set on the context: a missing seam shows up here rather than vanishing.
    UNKNOWN = "unknown"


# Fixed display order along the cognition loop. Stages absent here (only UNKNOWN)
# sort last.
COGNITION_ORDER: tuple[Stage, ...] = (
    Stage.WORLD_INIT,
    Stage.PRESSURE,
    Stage.PERCEPTION,
    Stage.INTERRUPT,
    Stage.MOTIVATION,
    Stage.DECISION,
    Stage.ACTION,
    Stage.FEEDBACK,
    Stage.MEMORY,
    Stage.COMPRESS,
    Stage.REFLECTION,
    Stage.RELATION_EVOLUTION,
    Stage.LONG_TERM_GOALS,
    Stage.DIRECTOR,
    Stage.EVENT,
)


def stage_sort_key(stage_value: str) -> int:
    """Return the cognition-order index for a stage value (UNKNOWN / unseen last)."""
    for index, stage in enumerate(COGNITION_ORDER):
        if stage.value == stage_value:
            return index
    return len(COGNITION_ORDER)


# The call order of `WorldBuilder.build`. Build calls have no step and all share WORLD_INIT, so
# scene is the only axis that orders them (alphabetical would scramble the pipeline).
# Plain strings, not ``LLMScene``: importing it would be a cycle. A test pins each to a real scene.
BUILD_SCENE_ORDER: tuple[str, ...] = (
    "world_template_selection",
    "world_building",       # theme analysis
    "cast_design",
    "persona_generation",   # agent generation
    "need_goal_generation",  # initializer: needs + opening goals
)


def build_scene_sort_key(scene: str) -> int:
    """Return the build-pipeline index for a scene (unlisted scenes sort last)."""
    try:
        return BUILD_SCENE_ORDER.index(scene)
    except ValueError:
        return len(BUILD_SCENE_ORDER)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def to_jsonable(obj: Any) -> Any:
    """Recursively convert *obj* into JSON-serialisable primitives.

    Handles dataclasses (preferring a custom ``as_dict`` when present), Enums,
    datetimes, mappings, and sequences.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    as_dict = getattr(obj, "as_dict", None)
    if callable(as_dict) and not isinstance(obj, type):
        return to_jsonable(as_dict())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return str(obj)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LLMCallTrace:
    """One LLM call captured at the router boundary.

    Three failure fields, each a strictly later question about the same call:

    - ``ok``: did the provider answer? A raising call is still recorded before re-raising.
    - ``parse_ok``: did structured data come out? None = no parse attempted; set via
      ``core.context.note_active_call_parse``.
    - ``adopted``: did the **consumer use** it? Only this catches a parseable payload the engine
      couldn't act on (e.g. a missing required field). None = no verdict filed; False carries
      ``reject_reason``. A deliberate no-op the engine honoured (``act:false``) is adopted.

    Mutable so ``parse_ok`` / ``adopted`` can be filled in before the step is flushed.
    Observability only — never read back into logic, a prompt, or an embedding.
    """

    world_id: str
    stage: str
    scene: str
    prompt_messages: list[dict[str, str]]
    response_content: str
    temperature: float
    max_tokens: int
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float
    timestamp: str
    agent_id: str | None = None
    step: int | None = None
    # A call parameter like temperature (here only because defaulted fields come last); the prompt
    # debugger needs it to reproduce a call.
    json_mode: bool = False
    # Kept apart from response_content: the product vs how it came about.
    thinking: str = ""
    thinking_tokens: int = 0
    ok: bool = True
    error: str = ""
    parse_ok: bool | None = None
    # See the class docstring.
    adopted: bool | None = None
    # A short code-layer slug (e.g. "missing_action_description") that makes the discard diagnosable.
    reject_reason: str = ""
    # Also logged on "llm_complete", to cross-reference a call across web / jsonl / logs.
    call_id: str = ""
    # Diagnostic annotations from core.context.annotate_call: one open channel, rendered
    # generically, so a new field needs no schema/router/UI change. JSON-serialisable;
    # observability only.
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclasses.dataclass(frozen=True)
class StepTrace:
    """One runtime step's wall-clock total plus its per-phase breakdown.

    ``phase_ms`` maps each step-loop phase (event_check / message / pressure /
    perceive / interrupt / executor / plan / exec / cognition / snapshot) to its
    wall-clock ms. Empty in older traces.
    """

    world_id: str
    step: int
    world_time: dict[str, Any]  # WorldTime.clock_payload() — the same shape the snapshot carries
    wall_ms: float
    timestamp: str
    phase_ms: dict[str, float] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sink contract
# ---------------------------------------------------------------------------

class TraceSink(ABC):
    """Destination for observability records.

    Write side is called from the runtime/router; read side backs the web
    dashboard's three viewing dimensions (step / stage / agent).
    """

    # -- write side --------------------------------------------------------

    @abstractmethod
    def record_llm_call(self, trace: LLMCallTrace) -> None: ...

    @abstractmethod
    def record_step(self, trace: StepTrace) -> None: ...

    @abstractmethod
    def mark_call_unadopted(
        self, world_id: str, *, step: int, agent_id: str, stage: str, reason: str
    ) -> None:
        """File a not-adopted verdict on an already-recorded call, addressed by coordinates.

        For a verdict reached after the call's own task is gone (arbitration discarding a
        step's cognition), when ``note_active_call_adoption``'s pointer has moved on. No match →
        no-op. Observability-only: business logic must never read it back.
        """

    @abstractmethod
    def annotate_recorded_call(
        self, world_id: str, *, step: int, agent_id: str, stage: str, **fields: Any
    ) -> None:
        """Merge diagnostic *fields* into an already-recorded call's ``extra``, by coordinates.

        The coordinate-addressed ``annotate_active_call``, as ``mark_call_unadopted`` is to
        ``note_active_call_adoption``; same rules.
        """

    @abstractmethod
    async def flush(self, world_id: str, step: int | None = None) -> None:
        """Persist buffered records for a world (and optionally one step).

        Call it on the event loop, never from a worker thread: records are still being added and
        annotated on the loop, so only the sink may decide what part of the work leaves it.
        """

    @abstractmethod
    def delete_world(self, world_id: str) -> None:
        """Purge every buffered and persisted trace for *world_id*.

        Called by whole-world deletion; irreversible. Concrete sinks must also
        drop any in-memory buffers / read caches they keep for the world.
        """

    @abstractmethod
    def delete_run_traces(self, world_id: str) -> None:
        """Purge a world's *runtime step* traces, keeping its world-build traces.

        Called by ``reset_session``: the clock restarts at 1, and append-only sinks would
        silently merge two runs into one step. Build traces survive: a reset doesn't rebuild
        the world. Irreversible.
        """

    # -- read side ---------------------------------------------------------

    @abstractmethod
    def read_calls(
        self,
        world_id: str,
        *,
        step: int | None = None,
        stage: str | None = None,
        agent_id: str | None = None,
    ) -> list[LLMCallTrace]:
        """Return LLM calls for a world, filtered by any combination of dimensions."""

    @abstractmethod
    def read_step_summaries(self, world_id: str) -> list[StepTrace]:
        """Return every recorded step's wall-clock summary, step-ascending."""

    @abstractmethod
    def list_steps(self, world_id: str) -> list[int]:
        """Return the runtime step numbers that have traces, ascending."""

    @abstractmethod
    def list_agents(self, world_id: str) -> list[str]:
        """Return the agent ids that appear in this world's traces."""

    @abstractmethod
    def list_stages(self, world_id: str) -> list[str]:
        """Return the stage values present in this world's traces, cognition-ordered."""
