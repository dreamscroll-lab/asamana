"""Interaction-facing read models for observation and replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from agent.need import NeedType
from agent.personality import AgentActivityStatus, EmotionType
from core.coerce import (
    coerce_datetime,
    coerce_dict,
    coerce_int,
    coerce_list,
    coerce_optional_bool,
    coerce_optional_float,
    coerce_optional_int,
    coerce_optional_str,
)
from core.interfaces.agent_store import relation_has_substance
from core.interfaces.snapshot import WorldSnapshot
from engine.environment import IN_TRANSIT
from world.identity_color import MINDLESS_BODY_COLOR


class WorldStatus(str, Enum):
    """How far the world's story has progressed, as recorded on disk.

    It says nothing about whether the world is running right now; that is ``RunController``
    state, exposed as the API's ``run_state`` field. Snapshots and catalog entries can't tell
    whether another process is driving the world, so don't add values like running / building:
    they would be wrong whenever the world actually runs.
    """

    IN_PROGRESS = "in_progress"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class WorldMeta:
    """Summary metadata for a world."""

    world_id: str
    theme: str = ""
    world_name: str = ""
    description: str = ""
    # Story progress, not liveness or run state. See WorldStatus.
    status: WorldStatus = WorldStatus.UNKNOWN
    created_at: datetime | None = None
    current_step: int = 0
    main_agent_names: list[str] = field(default_factory=list)
    # World time per step, in seconds; authored by the theme analysis and frozen at build. On
    # the creation-review card because it decides whether the story runs in hours or days.
    # Seconds so only the display layer assumes a granularity. 0 when unknown.
    seconds_per_step: int = 0
    # Whether the user has reviewed and locked initialization. Unconfirmed worlds
    # sit in the creation-review stage and cannot start their narrative run.
    confirmed: bool = False


@dataclass(frozen=True)
class GraphNode:
    """One agent in the relationship graph (GET /worlds/{id}/graph)."""

    id: str
    name: str
    role: str
    is_main_character: bool
    color: str


@dataclass(frozen=True)
class GraphEdge:
    """One directed relationship edge (A→B and B→A are separate)."""

    from_id: str
    to_id: str
    trust: float
    affection: float
    labels: list[str] = field(default_factory=list)
    interaction_count: int = 0
    history_summary: str = ""


@dataclass(frozen=True)
class WorldGraph:
    """The relationship graph payload: agents as nodes, relations as edges."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)


@dataclass(frozen=True)
class AgentProfile:
    """Static agent identity persisted at step 0 (GET …/agents/{id}/profile).

    Distinct from ``AgentStateSummary`` (the mutable per-step state) — this is the
    unchanging identity used by the creation-review cards and the detail panel.
    """

    agent_id: str
    name: str
    role: str
    age: int | None
    gender: str
    background: str
    appearance: str
    color: str
    is_main_character: bool
    core_traits: list[str] = field(default_factory=list)
    core_values: list[str] = field(default_factory=list)
    self_image: str = ""
    life_goal: str = ""
    secret: str = ""


def _target_view(raw: Any) -> "ActionTargetView":
    """The wire's target block → its typed view.

    Plain read (see ActionSummary's note on producers handing this layer JSON-native values).
    A missing block is an action that named nothing — a real and common answer, not a defect.
    An entry with no id is dropped: it points at nothing, and downstream an empty id reads as
    a real target the map would turn a figure toward.
    """
    block = coerce_dict(raw)

    def refs(key: str) -> list[TargetRef]:
        return [
            TargetRef(kind=str(r.get("kind") or ""), id=str(r.get("id") or ""))
            for r in (coerce_dict(item) for item in coerce_list(block.get(key)))
            if str(r.get("id") or "").strip()
        ]

    return ActionTargetView(acts_on=refs("acts_on"), claims=refs("claims"), reaches=refs("reaches"))


@dataclass(frozen=True)
class TargetRef:
    """One thing an action names, and what kind of thing it is.

    The kind rides with the id because an id alone does not say what it names, and the
    renderer routes on it: a figure turns toward a person, walks to a place, swings at a
    thing. Mirrors ``core.interfaces.action.Ref``.
    """

    kind: str = ""  # agent | npc | item | landmark | object | location
    id: str = ""


@dataclass(frozen=True)
class ActionTargetView:
    """An action's three relations to the world, as the observer receives them.

    The same three the cognition layer files (``core.interfaces.action.ActionTarget``),
    unflattened: a people/things split would lose a journey's destination and whose turn the
    act spends. All three carry kinded refs, since an id alone doesn't say what it names.
    Within ``acts_on`` every entry shares one kind (the cognition layer guarantees it), so a
    consumer may read the kind off the first entry.
    """

    acts_on: list[TargetRef] = field(default_factory=list)   # who/what the act is done to
    claims: list[TargetRef] = field(default_factory=list)    # whose turn it uses up
    reaches: list[TargetRef] = field(default_factory=list)   # passively receive it; turn unaffected


@dataclass(frozen=True)
class DialogueTurn:
    """One line of a conversation: who said it, and what."""

    speaker_id: str  # code layer — the join key; never rendered
    speaker: str     # narrative layer — the name to print
    line: str


@dataclass(frozen=True)
class ActionSummary:
    """User-facing summary of a single agent action."""

    agent_id: str
    agent_name: str
    action_description: str
    outcome: str
    succeeded: bool
    # What happened in one sentence: `outcome` without the detail it may append (a TALK
    # transcript, an interrupter's thought). See core.interfaces.action.ActionResult.
    gist: str = ""
    # Why it failed, as a short third-person phrase. Set only on failure (empty on success,
    # interruption or failed adjudication). Renderers show it as its own element (a line by the
    # ✗, a tooltip) rather than parsing the outcome prose, for the same reason `deed` exists.
    # See the viewpoint contract on core.interfaces.action.ActionResult.
    failure_reason: str = ""
    # Whether a COVERT act was noticed. Independent of succeeded (whether it achieved its aim):
    # a theft can succeed and be seen, or fail unnoticed.
    detected: bool = False
    # True when the action never engaged the world (precondition unmet: target/
    # co-participant absent or busy, no path). A NON-EVENT, not an attempted-and-failed
    # outcome — the renderer mutes it (no miss VFX / no ✗) rather than showing a defeat.
    not_executed: bool = False
    action_type: str = ""  # talk/move/physical/covert/work/rest/send_message/errand
    # WHICH BEAT of an execution this record reports — the only thing that says whether its
    # ``outcome`` is an opening or a result:
    #   begin            — the opening: the act is under way and nothing has come of it yet, so
    #                      ``outcome`` only restates the intent. Carries ``duration_label``.
    #   ongoing_tick     — still at it. Progress (``elapsed_steps``/``total_steps``), no result.
    #   ongoing_complete — the closing beat: the result of something begun steps ago.
    #   settled          — opened and closed in one beat: a real result, but with no opening of
    #                      its own to conclude, so neither of the two above.
    #   interrupt        — cut short. Terminal like a completion; ``interrupted_by`` says by whom.
    #   initialization   — step-0 placement, the one value that is not an execution's beat.
    phase: str = ""
    # Render-neutral semantic membrane (see engine _action_semantics): the renderer
    # differentiates effects from these facts (e.g. hit a person vs smash an object).
    deed: str = ""  # the observable VERB — see core.interfaces.action.Deed. Empty = nothing was done.
    # A joint action yields one record per participant (each writes their own memory), but an
    # observer's feed is a stream of events. To fold them back into one deed, group by
    # execution_id and take the outcome from the record whose agent_id == initiator_id (a
    # conscripted participant's outcome only restates the intent). Code-layer ids, never rendered.
    execution_id: str = ""
    initiator_id: str = ""
    # How long the actor expects it to take, as a natural duration ("about 2 hours") — already
    # rendered by the producer, because a step count is a code-layer coordinate and must not
    # reach a narrative surface. Empty unless ``phase`` is "begin".
    duration_label: str = ""
    # How far along a multi-step act is — for a PROGRESS BAR. Code-layer integers, consumed by
    # renderers only (Transit sets the precedent: it drives the walk animation off the same two).
    # They must never be rendered as text: "step N" on a narrative surface is a membrane breach.
    elapsed_steps: int = 0
    total_steps: int = 0
    # Who cut short an in-progress action (phase="interrupt"), "" otherwise: "A talked with B
    # …until B walked out of it". The why rides in ``outcome``, quoted by the executor, not in
    # inner_monologue.
    interrupted_by: str = ""
    # The act's three relations, as the cognition layer filed them — see ActionTargetView,
    # core.interfaces.action.ActionTarget and engine.executors.base.target_semantics.
    target: "ActionTargetView" = field(default_factory=ActionTargetView)
    # What this step actually CHANGED, as against what ``target.acts_on`` was AIMED at. A
    # renderer with only this cannot point a man at the gate he failed to force: ``affected``
    # is empty in exactly that case, which is when the aim matters most.
    affected_entity_ids: list[str] = field(default_factory=list)
    # Present for it, party to none of it: who overheard this exchange. Kept out of
    # participant_ids (they spent no turn on it) — without its own field the memory they
    # walked away with traces back to no event at all, replay included.
    overheard_by: list[str] = field(default_factory=list)

    # The first-person deliberation that produced this action. Set only on the beat the decision
    # was made (the begin record, or a born-zero act's single record); empty on ticks, later
    # completions, conscripted participants, and interrupts (abandoning is a different thought).
    inner_monologue: str = ""
    is_main_character: bool = False
    dialogue: list[DialogueTurn] = field(default_factory=list)
    # Monotonic emission ordinal stamped by the runtime: the event's position on the world's
    # emission timeline (an engine property, not a render concept), used to order events within
    # a step. Code-layer only, never rendered. 0 for step-0 initialization records.
    seq: int = 0


@dataclass(frozen=True)
class MessageSummary:
    """User-facing summary of a delivered message."""

    message_id: str
    sender_id: str
    sender_name: str
    receiver_ids: list[str]
    perceived_summary: str
    # The part the sender actually said; the rest of the body is attached material (a runner's
    # one-line report plus a description of the scene). The sender marks the split in the
    # message metadata, so displays needn't cut by length. Equal to the body for a plain letter.
    spoken: str
    # How it was addressed (MessageSystem resolves receivers from recipients × location_scope):
    #   direct — named recipients: it reached them, by name.
    #   place  — no recipients, a location: an announcement to a room.
    #   world  — no recipients, no location: a proclamation to everyone.
    # Without it a proclamation reads as a pile of identical private letters. ``place`` is the
    # location's name (never its id), empty unless scope == "place".
    scope: str = "direct"
    place: str = ""
    # ``place`` as its id, for consumers that look the place up (the map hangs an announcement
    # on the room). Never render it. Same name/id pair as BroadcastSummary's location_name /
    # location_scope: reverse-resolving the name would add a second id→name→id path.
    place_id: str = ""
    # See ActionSummary.seq — same monotonic ordinal, same channel semantics.
    seq: int = 0


@dataclass(frozen=True)
class AgentStateSummary:
    """User-facing state summary for an agent."""

    agent_id: str
    agent_name: str
    location: str      # narrative name, for text
    emotion: str
    activity_status: str
    # The narrative-layer names of ``emotion`` / ``activity_status`` / ``dominant_need``; "" when
    # the value is empty or not a known member.
    emotion_label: str = ""
    activity_label: str = ""
    dominant_need: str | None = None
    dominant_need_label: str = ""
    is_main_character: bool = False
    color: str = ""  # fixed identity colour (#RRGGBB); "" until a world is rebuilt
    long_term_goals: list[str] = field(default_factory=list)
    short_term_goals: list[str] = field(default_factory=list)
    emotion_intensity: float | None = None
    emotion_valence: float | None = None
    vitality: float | None = None
    short_term_goal_entities: list[dict[str, Any]] = field(default_factory=list)
    is_active: bool = True
    # Code-layer id for ``location``, for consumers that look up or group by place. Mid-move it
    # is the waypoint room, or "" between waypoints (when ``location`` is "途中", "on the way").
    # As with MessageSummary's place/place_id, the name is for display and the id for lookups.
    location_id: str = ""
    # Their current ongoing condition as one line of narrative text ("hands tied behind their
    # back"); "" for none. since_step / until_step / source_agent_id are code-layer details the
    # observer doesn't need.
    condition: str = ""
    # Movement transit contract: for an agent mid-move, {from_location_id,
    # to_location_id, path, arrivals, elapsed_steps, total_steps} so a 2D renderer walks
    # the token along `path` (the real waypoint sequence), standing on path[i] at
    # elapsed == arrivals[i]. None when not travelling.
    transit: dict[str, Any] | None = None
    # The trip that ended this step: the route walked to get here, in transit's shape with
    # elapsed_steps == total_steps. None unless he arrived this step.
    arrival: dict[str, Any] | None = None
    # The transit contract's opposite: this step a force outside the world (the human director)
    # put him here and he did not walk. Without it a renderer sees only a changed location and
    # draws a teleport as a stroll; this tells it to cut instead.
    displaced: bool = False


@dataclass(frozen=True)
class WorldEventSummary:
    """User-facing summary of a world event — something INJECTED into the world.

    Two authors produce these and the reader must be able to tell them apart, which is
    what ``authored_by`` is for: ``"system"`` is the LLM narrative editor inventing a
    twist, ``"director"`` is the human reaching in. Same card, different hand.
    """

    id: str
    authored_by: str
    narrative: str
    affected_names: list[str] = field(default_factory=list)
    location_label: str | None = None
    is_positive: bool | None = None
    # The sentence a human typed to cause this (director-authored events only), closing the
    # loop: what I said → what the engine made of it → what the world did about it.
    directive_text: str = ""
    # Only a director's injection carries one: who it landed on, who it actually pressed,
    # who re-decided because of it, who dropped what they were doing. Without it an
    # intervention that was correctly delivered but judged unimportant is indistinguishable
    # from one that silently failed.
    receipt: dict[str, Any] | None = None
    # See ActionSummary.seq — same monotonic ordinal, same channel semantics.
    seq: int = 0


@dataclass(frozen=True)
class InterventionRecord:
    """One row of the index of things a human has told this world to do.

    Deliberately thinner than the feed's 🎬 card. The card answers "what came of this one"
    (delivered to, pressed, re-decided, cut off), readings that need the surrounding step to
    mean anything. This answers "was I understood": the sentence and the engine's one-line
    account of what it dispatched, without the receipt, since a verdict without its step is
    noise.

    The feed holds only watched steps, so this reads the snapshots. Don't add a side-file
    appended at commit: ``delete_steps_after`` would not roll it back. Refusals must stay
    absent: they changed nothing in the world.
    """

    step: int
    time_label: str            # the narrative calendar label, never the machine clock
    directive_text: str        # the sentence that was typed
    narrative: str             # what the engine dispatched off it — the other half of the pair


@dataclass(frozen=True)
class NpcStateSummary:
    """An actor with a body but no mind, as an observer sees it.

    Kept out of ``agent_states`` because it isn't an agent, so the relation graph, character
    cards and agent profiles exclude it without any filtering.

    The frontend picks the body sprite from ``(gender, age)``, a simple lookup. ``color`` comes
    from the backend because it only marks "no mind" if the character palette never uses grey,
    and ``world.identity_color`` owns that guarantee; it is supplied the same way as
    ``AgentStateSummary.color``.
    """

    npc_id: str
    name: str
    location: str                    # narrative name; "途中" (on the way) while travelling
    location_id: str = ""            # see AgentStateSummary.location_id
    # One colour shared by all of them. See ``world.identity_color.MINDLESS_BODY_COLOR``.
    color: str = MINDLESS_BODY_COLOR
    gender: str = ""
    age: int | None = None
    description: str = ""
    condition: str = ""
    # What happened to them this beat, as one observable third-person sentence ("handed over
    # the letter"); empty when nothing happened. Same meaning as ``ActionSummary.outcome``.
    # It always happens at ``location`` (the same-place invariant in ``NpcRunner._advance_one``),
    # so the sentence names a place only when they failed to reach it.
    outcome: str = ""
    # True when the outcome describes something still under way, like ``phase="begin"`` on
    # ``ActionSummary`` (this tier has no beats). A world fact, not a render instruction.
    ongoing: bool = False
    # True when they were moved to ``location`` this beat by something outside the world
    # rather than walking there; a position change alone can't tell the two apart. Same as
    # ``AgentStateSummary.displaced``.
    displaced: bool = False


@dataclass(frozen=True)
class EntityView:
    """A non-agent world object (item / landmark) as it stands at this step — the god view.

    ``presence`` places it on one axis: ``at_location`` (``presence_ref`` = location id),
    ``held`` (``presence_ref`` = holder agent id) or ``destroyed`` (``presence_ref`` = None;
    kept as a tombstone). ``is_public`` says whether someone standing there can perceive it.

    ``created_step`` is the step it came into the world on. A client can't derive it by
    diffing entity tables: it lacks the step before the earliest one it holds.
    """

    name: str
    entity_type: str
    state: str
    presence: str
    presence_ref: str | None
    description: str = ""
    is_public: bool = True
    content: str = ""
    created_step: int = 0


@dataclass(frozen=True)
class BroadcastSummary:
    """A senderless world announcement — a death, an injected change.

    ``location_scope`` is the place's id (None = the whole world) and ``location_name`` its
    narrative name ("" = the whole world); same name/id pair as MessageSummary.place/place_id.
    ``phenomenon`` is what the change visibly looks like (``core.interfaces.phenomenon``),
    orthogonal to ``severity``, which is how big it is.
    """

    content: str
    broadcast_type: str = ""
    severity: str = ""
    location_scope: str | None = None
    location_name: str = ""
    phenomenon: str = ""
    # See ActionSummary.seq — same monotonic ordinal, same channel semantics.
    seq: int = 0


@dataclass(frozen=True)
class WorldTimeView:
    """The world's clock at this step — one fact, two layers.

    ``label`` is the narrative layer: the world's own name for the moment, shaped by its
    calendar and only ever shown whole — never parsed. ``hour`` / ``minute`` are the code
    layer: theme-neutral numbers a renderer may read (the map's day/night wash); None when the
    step was recorded without them.
    """

    label: str
    hour: int | None
    minute: int | None

    @classmethod
    def from_payload(cls, payload: object) -> "WorldTimeView":
        """Read a ``WorldTime.clock_payload()`` dict. Anything else reads as an unknown time —
        never a guessed one."""
        data = coerce_dict(payload)
        return cls(
            label=str(data.get("label") or ""),
            hour=coerce_optional_int(data.get("hour")),
            minute=coerce_optional_int(data.get("minute")),
        )


@dataclass(frozen=True)
class StepEvent:
    """Observation-ready view of one completed runtime step."""

    world_id: str
    step: int
    world_time: WorldTimeView
    actions: list[ActionSummary] = field(default_factory=list)
    messages: list[MessageSummary] = field(default_factory=list)
    world_events: list[WorldEventSummary] = field(default_factory=list)
    agent_states: dict[str, AgentStateSummary] = field(default_factory=dict)
    npcs: list[NpcStateSummary] = field(default_factory=list)
    entities: dict[str, EntityView] = field(default_factory=dict)
    broadcasts: list[BroadcastSummary] = field(default_factory=list)
    # Directed relations as they stand at the end of this step — the same edges GET /graph
    # serves, so a live observer needs no per-step round trip for them.
    relations: list[GraphEdge] = field(default_factory=list)

    @classmethod
    def from_runtime_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "StepEvent":
        """Build a step event from the runtime event-bus payload."""

        metadata = {
            "schedule": coerce_dict(payload.get("schedule")),
            "messages": coerce_dict(payload.get("messages")),
            "environment": coerce_dict(payload.get("environment")),
            # The live payload carries broadcasts (with their per-event ``seq``);
            # forward them through the transient WorldSnapshot so from_snapshot
            # surfaces them exactly as the persisted-snapshot path does.
            "broadcasts": coerce_list(payload.get("broadcasts")),
        }
        snapshot = WorldSnapshot(
            world_id=str(payload["world_id"]),
            step=int(payload["step"]),
            timestamp=datetime.now(timezone.utc),
            # The wire already speaks the snapshot's shape — read the keys straight, no
            # per-path reconciliation.
            world_time=coerce_dict(payload.get("world_time")),
            agent_states=_agent_state_payload(payload),
            agent_relations={
                str(key): dict(rel)
                for key, rel in coerce_dict(payload.get("agent_relations")).items()
                if isinstance(rel, Mapping)
            },
            events_this_step=coerce_list(payload.get("events")),
            actions_this_step=coerce_list(payload.get("actions")),
            metadata=metadata,
        )
        return cls.from_snapshot(snapshot)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: WorldSnapshot,
    ) -> "StepEvent":
        """Build a step event from a persisted snapshot."""

        actions = [_action_from_record(record) for record in snapshot.agent_summaries]
        messages = _messages_from_snapshot(snapshot)
        world_events = [_world_event_from_record(record) for record in snapshot.event_summaries]
        agent_states = _agent_states_from_snapshot(snapshot, actions=actions)
        return cls(
            world_id=snapshot.world_id,
            step=snapshot.step,
            world_time=WorldTimeView.from_payload(snapshot.world_time),
            actions=actions,
            messages=messages,
            world_events=world_events,
            agent_states=agent_states,
            npcs=_npcs_from_snapshot(snapshot),
            entities=_entities_from_snapshot(snapshot),
            relations=relations_from_snapshot(snapshot),
            # The producer already baked location_name in (EventSequencer.assign_event_seqs),
            # so there is nothing for a read model to translate.
            broadcasts=[
                _broadcast_from_record(coerce_dict(b))
                for b in coerce_list(snapshot.metadata.get("broadcasts"))
            ],
        )


def build_world_meta(
    *,
    world_id: str,
    snapshots: list[WorldSnapshot],
    catalog_entry: Mapping[str, Any] | None = None,
) -> WorldMeta:
    """Derive world metadata from persisted snapshots and optional catalog hints.

    ``snapshots`` is in ascending step order and need only hold step 0 (identity, frozen at
    build) and the last step (progress); callers pass just those two.
    """

    latest_snapshot = snapshots[-1] if snapshots else None
    main_agent_names = sorted(
        {
            str(record.get("agent_name", record.get("agent_id", "")))
            for snapshot in snapshots
            for record in snapshot.agent_summaries
            if record.get("is_main_character")
        }
        - {""}
    )
    # World lore for the creation-review card: pulled from the step-0 snapshot's
    # world block (narrative pitch preferred, era description as fallback).
    description = ""
    seconds_per_step = 0
    step0 = next((s for s in snapshots if s.step == 0), snapshots[0] if snapshots else None)
    if step0 is not None and isinstance(step0.metadata, dict):
        world_block = step0.metadata.get("world")
        if isinstance(world_block, dict):
            description = str(
                world_block.get("narrative_pitch") or world_block.get("era_description") or ""
            )
            seconds_per_step = coerce_optional_int(world_block.get("seconds_per_step")) or 0

    current_step = latest_snapshot.step if latest_snapshot else 0
    status = WorldStatus.IN_PROGRESS if latest_snapshot is not None else WorldStatus.UNKNOWN

    return WorldMeta(
        world_id=world_id,
        theme=str(catalog_entry.get("theme", "")) if catalog_entry else "",
        world_name=str(catalog_entry.get("world_name", world_id)) if catalog_entry else world_id,
        description=description,
        status=status,
        created_at=coerce_datetime(catalog_entry.get("created_at")) if catalog_entry else None,
        current_step=current_step,
        main_agent_names=main_agent_names,
        seconds_per_step=seconds_per_step,
        confirmed=bool(catalog_entry.get("confirmed", False)) if catalog_entry else False,
    )


def _action_from_record(record: Mapping[str, Any]) -> ActionSummary:
    # inner_monologue is the agent's actual thought (set by the decision engine) or
    # empty — never synthesised from anything else; a fabricated thought under a real
    # name is worse than none.
    inner_monologue = str(record.get("inner_monologue") or "")
    dialogue = [
        DialogueTurn(
            speaker_id=str(d.get("speaker_id") or ""),
            speaker=str(d.get("speaker") or ""),
            line=str(d.get("line") or ""),
        )
        for d in coerce_list(record.get("dialogue"))
        if isinstance(d, Mapping)
    ]
    return ActionSummary(
        agent_id=str(record.get("agent_id", "")),
        agent_name=str(record.get("agent_name") or record.get("agent_id") or ""),
        action_description=str(
            record.get("action_description") or record.get("summary") or record.get("action_type") or ""
        ),
        # `or ""`, not `.get(k, "")`: a key present and null slips past a default, and
        # `str(None)` silently yields "None", which prose prints. Every field here guards alike.
        outcome=str(record.get("outcome") or ""),
        gist=str(record.get("gist") or ""),
        succeeded=bool(record.get("succeeded", True)),
        failure_reason=str(record.get("failure_reason") or ""),
        detected=bool(record.get("detected", False)),
        not_executed=bool(record.get("not_executed", False)),
        # Plain read: producers hand this layer JSON-native values, guarded at one boundary by
        # tests/unit/test_state_payload_is_jsonable.py. Don't re-defend here: coercing one field
        # hides a producer leak that then shows up raw ("EmotionType.FRUSTRATION") in the next.
        action_type=str(record.get("action_type") or ""),
        phase=str(record.get("phase") or ""),
        target=_target_view(record.get("target")),
        affected_entity_ids=[str(e) for e in coerce_list(record.get("affected_entity_ids"))],
        overheard_by=[str(e) for e in coerce_list(record.get("overheard_by"))],
        deed=str(record.get("deed") or ""),
        execution_id=str(record.get("execution_id") or ""),
        initiator_id=str(record.get("initiator_id") or ""),
        duration_label=str(record.get("duration_label") or ""),
        elapsed_steps=int(coerce_optional_int(record.get("elapsed_steps")) or 0),
        total_steps=int(coerce_optional_int(record.get("total_steps")) or 0),
        interrupted_by=str(record.get("interrupted_by") or ""),
        inner_monologue=inner_monologue,
        is_main_character=bool(record.get("is_main_character", False)),
        dialogue=dialogue,
        seq=int(coerce_optional_int(record.get("seq")) or 0),
    )


def _messages_from_snapshot(snapshot: WorldSnapshot) -> list[MessageSummary]:
    """Build MessageSummary list from snapshot.metadata['messages'].

    Messages are lossless: message.content is byte-identical to what was sent, and
    perceived_summary is the original text.

    ``recipients``/``location_scope`` form MessageSystem's addressing matrix; here it folds into
    ``scope``. The observe layer must tell "a named letter was delivered" from "an edict was
    proclaimed publicly", or the latter would unfold into a pile of identical private letters.
    """
    messages = coerce_dict(snapshot.metadata.get("messages"))
    delivered = coerce_list(messages.get("delivered"))
    inboxes = {
        str(agent_id): [str(message_id) for message_id in coerce_list(message_ids)]
        for agent_id, message_ids in coerce_dict(messages.get("inboxes")).items()
    }

    summaries: list[MessageSummary] = []
    for record in delivered:
        message_id = str(record.get("id", ""))
        receiver_ids = sorted(
            agent_id
            for agent_id, message_ids in inboxes.items()
            if message_id and message_id in message_ids
        )
        # recipients=None means broadcast (directed delivery is always a list); location_scope then
        # splits place-wide from world-wide.
        if record.get("recipients") is None:
            scope = "place" if record.get("location_scope") else "world"
        else:
            scope = "direct"
        summaries.append(
            MessageSummary(
                message_id=message_id,
                sender_id=str(record.get("sender_id", "")),
                sender_name=str(record.get("sender_name", "")),
                receiver_ids=receiver_ids,
                perceived_summary=str(record.get("content", "")),  # no rendering, the original text
                spoken=(
                    str(coerce_dict(record.get("metadata")).get("spoken") or "")
                    or str(record.get("content", ""))
                ),
                scope=scope,
                # The name is filled in at write time by the producer (runtime, which holds
                # WorldDirectory), like sender_name. The read model doesn't look it up: that would
                # be a second id->name translation path besides WorldDirectory.
                place=str(record.get("location_name", "")) if scope == "place" else "",
                place_id=str(record.get("location_scope", "") or "") if scope == "place" else "",
                seq=int(coerce_optional_int(record.get("seq")) or 0),
            )
        )
    return summaries


def _world_event_from_record(record: Mapping[str, Any]) -> WorldEventSummary:
    """Build WorldEventSummary from a serialized event dict.

    Key-for-key with ``engine.injection.serialize_world_event`` — that function is the
    only producer, so this reader has no aliases to reconcile and no permanently-empty
    fields to carry.

    References stay in the narrative layer: ``affected_names`` and ``location_label`` are
    names, not ids, because they were resolved through the directory at injection time.
    """
    receipt = record.get("receipt")
    return WorldEventSummary(
        id=str(record.get("id", "")),
        authored_by=str(record.get("authored_by", "system")),
        narrative=str(record.get("narrative_desc", "")),
        affected_names=[str(item) for item in coerce_list(record.get("affected_names"))],
        location_label=coerce_optional_str(record.get("location_label")),
        is_positive=coerce_optional_bool(record.get("is_positive")),
        directive_text=str(record.get("directive_text", "")),
        receipt=dict(receipt) if isinstance(receipt, Mapping) else None,
        seq=int(coerce_optional_int(record.get("seq")) or 0),
    )


def relations_from_snapshot(snapshot: WorldSnapshot) -> list[GraphEdge]:
    """The snapshot's relations as graph edges — only those with narrative substance.

    The store also persists a baseline record the moment two agents merely co-occur in a
    memory; that is not a relationship. Same rule RelationEvolution filters candidates by.
    """
    edges: list[GraphEdge] = []
    for rel in snapshot.agent_relations.values():
        from_id, to_id = rel.get("from_id"), rel.get("to_id")
        if not (from_id and to_id):
            continue
        labels = [str(label) for label in coerce_list(rel.get("labels"))]
        interaction_count = int(coerce_optional_int(rel.get("interaction_count")) or 0)
        history_summary = str(rel.get("history_summary") or "")
        if not relation_has_substance(
            labels=labels,
            history_summary=history_summary,
            interaction_count=interaction_count,
        ):
            continue
        edges.append(GraphEdge(
            from_id=str(from_id),
            to_id=str(to_id),
            trust=round(float(rel.get("trust_objective", 0.5)), 2),
            affection=round(float(rel.get("affection_objective", 0.0)), 2),
            labels=labels,
            interaction_count=interaction_count,
            history_summary=history_summary,
        ))
    return edges


def _placed_location_id(location_id: str) -> str:
    """A body's location id as the wire carries it: the transit sentinel is the engine's
    own bookkeeping, not a place, so it goes out as "" (``transit`` says where he is going)."""
    return "" if location_id == IN_TRANSIT else location_id


def _broadcast_from_record(record: Mapping[str, Any]) -> BroadcastSummary:
    """Key-for-key with ``EventSequencer.assign_event_seqs``, its only producer."""
    return BroadcastSummary(
        content=str(record.get("content") or ""),
        broadcast_type=str(record.get("broadcast_type") or ""),
        severity=str(record.get("severity") or ""),
        location_scope=coerce_optional_str(record.get("location_scope")) or None,
        location_name=str(record.get("location_name") or ""),
        phenomenon=str(record.get("phenomenon") or ""),
        seq=int(coerce_optional_int(record.get("seq")) or 0),
    )


def _entities_from_snapshot(snapshot: WorldSnapshot) -> dict[str, EntityView]:
    """The environment's entity table, projected to what an observer needs.

    The snapshot block is the restore payload (``EnvironmentSystem.snapshot_state``); a field
    goes out only if it means something to an observer: ``is_takeable`` (seed-reconstruction
    authority) stays out, ``created_step`` (a world fact the observer can't derive) goes in.
    """
    table = coerce_dict(coerce_dict(snapshot.metadata.get("environment")).get("entity_states"))
    out: dict[str, EntityView] = {}
    for entity_id, raw in sorted(table.items()):
        payload = coerce_dict(raw)
        out[str(entity_id)] = EntityView(
            name=str(payload.get("name") or ""),
            entity_type=str(payload.get("entity_type") or ""),
            state=str(payload.get("state") or ""),
            presence=str(payload.get("presence") or ""),
            presence_ref=coerce_optional_str(payload.get("presence_ref")),
            description=str(payload.get("description") or ""),
            is_public=bool(payload.get("is_public", True)),
            content=str(payload.get("content") or ""),
            created_step=coerce_int(payload.get("created_step"), default=0),
        )
    return out


def _npcs_from_snapshot(snapshot: WorldSnapshot) -> list[NpcStateSummary]:
    """Project the environment's Npc roster into the observer's read model.

    Joins the three snapshot tables (``body_locations`` / ``npc_states`` / ``npc_outcomes``) so
    the display needn't, in id order so renders of a step are identical. Composes no sentence
    (see ``NpcStateSummary.outcome``); it only translates place ids to names.
    """
    environment = coerce_dict(snapshot.metadata.get("environment"))
    roster = coerce_dict(environment.get("npc_states"))
    if not roster:
        return []
    placements = coerce_dict(environment.get("body_locations"))
    names = coerce_dict(environment.get("location_names"))
    outcomes = coerce_dict(environment.get("npc_outcomes"))
    displaced = {str(nid) for nid in environment.get("npc_displaced") or []}
    out: list[NpcStateSummary] = []
    for npc_id in sorted(roster):
        payload = coerce_dict(roster.get(npc_id))
        happened = coerce_dict(outcomes.get(npc_id))
        where = str(placements.get(npc_id, "") or "")
        condition = coerce_dict(payload.get("condition"))
        out.append(NpcStateSummary(
            npc_id=str(npc_id),
            name=str(payload.get("name", "") or ""),
            # Narrative name. Unknown places fall back to the descriptive "某地" (somewhere);
            # a bare id would leak across the layer boundary.
            location=str(names.get(where, "") or ("途中" if where else "") or "某地"),
            location_id=_placed_location_id(where),
            gender=str(payload.get("gender", "") or ""),
            age=payload.get("age") if isinstance(payload.get("age"), int) else None,
            description=str(payload.get("description", "") or ""),
            condition=str(condition.get("description", "") or ""),
            outcome=str(happened.get("text", "") or ""),
            ongoing=bool(happened.get("ongoing", False)),
            displaced=str(npc_id) in displaced,
        ))
    return out


def _member_label(
    enum_cls: type[EmotionType] | type[AgentActivityStatus] | type[NeedType], value: str,
) -> str:
    try:
        return enum_cls(value).label
    except ValueError:
        return ""


def _agent_states_from_snapshot(
    snapshot: WorldSnapshot,
    *,
    actions: list[ActionSummary],
) -> dict[str, AgentStateSummary]:
    states: dict[str, AgentStateSummary] = {}
    actions_by_id = {action.agent_id: action for action in actions if action.agent_id}
    environment = coerce_dict(snapshot.metadata.get("environment"))
    bodies = coerce_dict(environment.get("body_locations"))
    place_names = coerce_dict(environment.get("location_names"))

    for agent_id, record in snapshot.agent_states.items():
        if not isinstance(record, Mapping):
            continue
        action = actions_by_id.get(str(agent_id))
        transit = dict(record["transit"]) if isinstance(record.get("transit"), Mapping) else None
        arrival = dict(record["arrival"]) if isinstance(record.get("arrival"), Mapping) else None
        emotion = record.get("emotion", "")
        emotion_value = str(emotion.get("primary", "")) if isinstance(emotion, Mapping) else str(emotion)
        activity_value = str(record.get("activity_status", ""))
        states[str(agent_id)] = AgentStateSummary(
            agent_id=str(record.get("agent_id", agent_id)),
            agent_name=str(record.get("agent_name") or "某人"),
            # Mid-move, don't read the agent's own current_location: it changes only on arrival,
            # so it shows the origin the whole way. The environment owns where the body is now:
            # at a waypoint, that room; between points, "途中" (on the way).
            # Narrative-layer field: prefer the place name written into the snapshot; if missing,
            # fall back to the descriptive "某地" (somewhere), never the bare location_id.
            location=(
                str(place_names.get(bodies.get(agent_id), "") or "途中") if transit
                else str(record.get("location_name") or "某地")
            ),
            location_id=_placed_location_id(
                str((bodies.get(agent_id) if transit else record.get("location_id")) or "")
            ),
            emotion=emotion_value,
            activity_status=activity_value,
            emotion_label=_member_label(EmotionType, emotion_value),
            activity_label=_member_label(AgentActivityStatus, activity_value),
            dominant_need=coerce_optional_str(record.get("dominant_need")),
            dominant_need_label=_member_label(NeedType, str(record.get("dominant_need") or "")),
            is_main_character=bool(record.get("is_main_character", False)),
            color=str(record.get("color") or ""),
            long_term_goals=list(record.get("long_term_goals") or []),
            short_term_goals=list(record.get("short_term_goals") or []),
            emotion_intensity=coerce_optional_float(record.get("emotion_intensity")),
            emotion_valence=coerce_optional_float(record.get("emotion_valence")),
            vitality=coerce_optional_float(record.get("vitality")),
            # record["condition"] is the dict (or None) snapshot_agent_state wrote. Read
            # defensively: live gets the in-memory dict, replay gets one round-tripped through
            # json.dumps(default=str), and either may yield something that isn't a Mapping.
            condition=str(
                (record.get("condition") or {}).get("description", "")
                if isinstance(record.get("condition"), Mapping) else ""
            ),
            short_term_goal_entities=list(record.get("short_term_goal_entities") or []),
            is_active=bool(record.get("is_active", True)),
            transit=transit,
            arrival=arrival,
            displaced=bool(record.get("displaced", False)),
        )

    for action in actions:
        states.setdefault(
            action.agent_id,
            AgentStateSummary(
                agent_id=action.agent_id,
                agent_name=action.agent_name,
                location="某地",
                emotion="",
                activity_status="",
                dominant_need=None,
                is_main_character=action.is_main_character,
                color="",  # ActionSummary has no color; the frontend colours this by agent identity
                long_term_goals=[],
                short_term_goals=[],
                short_term_goal_entities=[],
            ),
        )
    return dict(sorted(states.items()))




def _agent_state_payload(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(agent_id): dict(record)
        for agent_id, record in coerce_dict(payload.get("agent_states")).items()
        if isinstance(record, Mapping)
    }


