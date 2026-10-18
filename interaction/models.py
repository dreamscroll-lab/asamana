"""Interaction-facing read models for observation and replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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
from core.interfaces.action import DialogueTurn
from core.interfaces.actor import ActorPace
from core.interfaces.agent_store import relation_has_substance
from core.interfaces.snapshot import WorldSnapshot
from engine.environment import IN_TRANSIT
from world.models import EntityPresence
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
    # The character an actor performs (chosen at confirmation), None if nobody was cast. Once
    # the actor exits, ``actor_exited`` is set and the character is back with AI cognition.
    actor_character_id: str | None = None
    actor_exited: bool = False
    # When the actor is asked to decide while the character is idle.
    actor_pace: ActorPace = ActorPace.EVERY_STEP
    # The whole cast has been shown to the user as the director; the world takes no actor after.
    director_reviewed: bool = False


@dataclass(frozen=True)
class CharacterIdentity:
    """A character's public identity, identical for every viewer.

    Inner life, relations and narrative tier belong to viewpoint-filtered contracts.
    """

    agent_id: str
    name: str
    gender: str
    age: int | None
    role: str
    background: str
    appearance: str
    color: str


@dataclass(frozen=True)
class PlaceIdentity:
    """A place's public identity, fixed at build and identical for every viewer."""

    location_id: str
    name: str
    description: str


@dataclass(frozen=True)
class GraphNode:
    """One agent in the relationship graph (GET /worlds/{id}/graph)."""

    id: str
    name: str
    role: str
    # None: not known to this viewer (the actor's view, ``interaction.actor_view``).
    is_main_character: bool | None


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

    Distinct from ``AgentView`` (the mutable per-step state) — this is the
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
    # None: not known to this viewer (the actor's view, ``interaction.actor_view``).
    is_main_character: bool | None
    core_traits: list[str] = field(default_factory=list)
    core_values: list[str] = field(default_factory=list)
    self_image: str = ""
    life_goal: str = ""
    secret: str = ""


@dataclass(frozen=True)
class InjectionSummary:
    """User-facing summary of an injection — something put INTO the world from outside it.

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
    # The sentence a human typed to cause this (director-authored only), closing the
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



# ----- one step, by scope ---------------------------------------------------------
#
# A StepView is what happened in the world this step and the state it ended in, grouped by
# where it happened. Four scopes hold everything a character could perceive:
#
#   locations  — one per place with anything in it: the bodies and things there at the end of
#                the step, and what happened there.
#   transit    — bodies between two waypoints (the engine's IN_TRANSIT), and what happened to them
#                on the way. A body standing on a waypoint is in that location, not here.
#   world      — what reached every place at once (a world-wide broadcast or proclamation).
#
# ``observer`` sits outside them: an injection's record, which only the one watching from outside
# the world reads. What the injection did inside the world reaches the scopes as broadcasts,
# messages and observations of the changes it made.
#
# Every body, thing and action splits into what anyone present perceives (``public``, or the
# top-level fields of an event) and what only some know (``private``, with its audience). The
# split follows what the engine lets each character perceive; a view for one character is a
# filter over it, never a second visibility rule.
#
# Every event carries ``seq``, its place on the world's emission timeline (``EventSequencer``):
# unique across steps, so an event that spans steps or scopes keeps one id. Its parts are tied
# by their own ids: ``execution_id`` for the beats of one act, ``message_id`` for one message.


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
class ObservedView:
    """What onlookers at one place saw of an action (``core.interfaces.action.Observed``)."""

    text: str
    strength: float | None = None


@dataclass(frozen=True)
class ActionPrivateView:
    """What only those inside an action know of it.

    ``audience`` is everyone the act involved: its participants, whoever it landed on and the
    listeners it named. ``outcome`` is the complete account (a TALK's transcript, an
    interrupter's thought); ``gist`` the same in one sentence. See the viewpoint contract on
    ``core.interfaces.action.ActionResult``.
    """

    audience: list[str]
    outcome: str
    gist: str
    dialogue: list[DialogueTurn] = field(default_factory=list)
    # The doer's own words for what he set out to do; onlookers see only ``public.text``.
    action_description: str = ""
    # How long the doer expects it to take, as a natural phrase rendered by the producer. Empty
    # unless phase="begin".
    duration_label: str = ""
    # Progress of a multi-step act. Code-layer integers for renderers; never rendered as text.
    elapsed_steps: int = 0
    total_steps: int = 0
    # The way walked on the beat a MOVE ended: transit's shape with elapsed_steps == total_steps,
    # so a renderer walks him in along the waypoints the engine reported. None otherwise.
    route: dict[str, Any] | None = None


@dataclass(frozen=True)
class ActionSummary:
    """One beat of one execution, every participant's record folded into one event.

    A joint act yields a record per participant (each writes their own memory); this keeps the
    initiator's, so the act is told once. It sits in the scope where its doer stood when the beat
    was recorded. Onlookers at other places get an ``ObservationSummary`` of it instead (a move's
    origin sees him leave), under the same ``seq``.
    """

    seq: int
    execution_id: str = ""
    initiator_id: str = ""
    # Everyone who spent their turn on it, initiator included.
    participant_ids: list[str] = field(default_factory=list)
    action_type: str = ""  # talk/move/physical/covert/work/rest/send_message/errand
    # WHICH BEAT of an execution this is — the only thing that says whether ``private.outcome``
    # is an opening or a result:
    #   begin            — the opening: under way, nothing has come of it yet. Carries
    #                      ``duration_label``.
    #   ongoing_tick     — still at it. Progress (``elapsed_steps``/``total_steps``), no result.
    #   ongoing_complete — the closing beat: the result of something begun steps ago.
    #   settled          — opened and closed in one beat.
    #   interrupt        — cut short. Terminal like a completion; ``interrupted_by`` says by whom.
    phase: str = ""
    deed: str = ""  # the observable VERB — see core.interfaces.action.Deed. Empty = nothing was done.
    target: ActionTargetView = field(default_factory=ActionTargetView)
    # What this beat actually CHANGED, as against what ``target.acts_on`` was AIMED at: empty when
    # a man fails to force a gate, which is when the aim matters most.
    affected_entity_ids: list[str] = field(default_factory=list)
    # Listeners the initiator named, standing there: party to none of it, so kept out of
    # participant_ids (they spent no turn on it).
    overheard_by: list[str] = field(default_factory=list)
    # Who cut it short (phase="interrupt"); "" when nobody inside chose to stop.
    interrupted_by: str = ""
    succeeded: bool = True
    # Why it failed, as a short third-person phrase; set only on failure.
    failure_reason: str = ""
    # Whether a COVERT act was noticed. Independent of succeeded: a theft can succeed and be seen.
    detected: bool = False
    # The act never engaged the world (precondition unmet). A non-event, not a defeat.
    not_executed: bool = False
    # What onlookers here saw; None when the act showed nothing here (SEND_MESSAGE, an unexposed
    # COVERT act, a beat recorded on the way).
    public: ObservedView | None = None
    private: ActionPrivateView = field(
        default_factory=lambda: ActionPrivateView(audience=[], outcome="", gist="")
    )
    kind: str = "action"


@dataclass(frozen=True)
class ObservationSummary:
    """Something seen at a place, from one of two sources:

    - an action seen from a place other than where it is recorded: the origin watching a man
      leave, a waypoint watching him pass. ``seq`` is the action's own.
    - a change an injection made, seen where it landed (a seal on a decree, a man struck down).
      ``seq`` is the injection's own; nobody in the world did it, so ``initiator_id`` is "" and
      ``participant_ids`` is empty.
    """

    seq: int
    initiator_id: str
    participant_ids: list[str]
    text: str
    strength: float | None = None
    kind: str = "observation"


@dataclass(frozen=True)
class MessageSummary:
    """A message reaching the people in one scope.

    ``scope`` is how it was addressed (MessageSystem resolves receivers from recipients ×
    location_scope):
      direct — named recipients. A letter to people in several places appears in each of their
               scopes, under one ``seq``. Private: ``audience`` is the receivers in this scope,
               the only ones who lived this delivery. Don't add the sender or receivers
               elsewhere: they would learn where the letter was received. The sender knows the
               words from their own SEND_MESSAGE act.
      place  — no recipients, a location: an announcement to the room it sits in. ``audience``
               is whoever heard it there; someone who arrives by the step's end did not.
      world  — no recipients, no location: a proclamation, in the world scope. Public.
    ``audience`` is None when the message reached everyone (world scope).
    """

    seq: int
    message_id: str
    sender_id: str
    sender_name: str
    # Who received it in this scope.
    receiver_ids: list[str]
    scope: str
    content: str
    # The part the sender actually said; the rest of ``content`` is attached material (a runner's
    # report plus a description of the scene). The sender marks the split in the message
    # metadata, so displays needn't cut by length. Equal to ``content`` for a plain letter.
    spoken: str
    audience: list[str] | None = None
    kind: str = "message"


@dataclass(frozen=True)
class BroadcastSummary:
    """A senderless change everyone in its scope perceives — a death, an injected change.

    ``phenomenon`` is what the change visibly looks like (``core.interfaces.phenomenon``),
    orthogonal to ``severity``, which is how big it is. ``audience`` is who heard a place's
    broadcast (it reached them as the step began; someone who arrives by the step's end did
    not); None for a world broadcast, which reached everyone.
    """

    seq: int
    content: str
    broadcast_type: str = ""
    severity: str = ""
    phenomenon: str = ""
    audience: list[str] | None = None
    kind: str = "broadcast"


ScopeEvent = ActionSummary | ObservationSummary | MessageSummary | BroadcastSummary


@dataclass(frozen=True)
class EntityPrivateView:
    """What only whoever holds a thing can read: the words in a letter, the figures in a ledger.
    Seeing a thing is not reading it."""

    content: str = ""


@dataclass(frozen=True)
class EntityView:
    """A non-agent world object (item / landmark) as it stands at the end of this step.

    Where it is follows from where it is listed: in a location or in a holder's belongings. A
    destroyed thing is no longer in the world and is not listed; the act that destroyed it names
    it in ``affected_entity_ids``. ``is_public`` says whether someone in its presence can
    perceive it.

    ``created_step`` is the step it came into the world on. A client can't derive it by
    diffing entity tables: it lacks the step before the earliest one it holds.
    """

    name: str
    entity_type: str
    state: str
    description: str = ""
    is_public: bool = True
    created_step: int = 0
    private: EntityPrivateView = field(default_factory=lambda: EntityPrivateView())


@dataclass(frozen=True)
class AgentPublicView:
    """What anyone in the agent's presence can see of them."""

    # What he is visibly doing; "" when it doesn't show (someone acting covertly looks like
    # someone doing nothing in particular). His actual activity is ``private.activity_status``.
    activity_status: str
    activity_label: str = ""
    # Their current ongoing condition as one line of narrative text ("hands tied behind their
    # back"); "" for none.
    condition: str = ""
    # Put here this step by a force outside the world (the director), not walked here.
    displaced: bool = False
    # Belongings others can see (``EntityView.is_public``).
    carrying: dict[str, EntityView] = field(default_factory=dict)


@dataclass(frozen=True)
class ThoughtView:
    """The deliberation behind one of the agent's acts, set on the beat it was decided. It is
    the decision engine's actual thought or absent: never synthesised."""

    seq: int
    execution_id: str
    text: str


@dataclass(frozen=True)
class GoalView:
    """One short-term goal: its text, where it stands, and where it came from.

    ``origin`` is ``cognitive`` (something they decided to pursue) or ``residue`` (something an
    action left them owing); see ``GoalOrigin``.
    """

    text: str
    status: str
    origin: str = "cognitive"


@dataclass(frozen=True)
class AgentPrivateView:
    """What only the agent knows: their mind, where they are headed, and hidden belongings."""

    emotion: str
    emotion_label: str = ""
    emotion_intensity: float | None = None
    emotion_valence: float | None = None
    dominant_need: str | None = None
    dominant_need_label: str = ""
    vitality: float | None = None
    activity_status: str = ""
    activity_label: str = ""
    # Mid-move: {from_location_id, to_location_id, path, arrivals, elapsed_steps, total_steps},
    # so a renderer walks the body along ``path``, standing on path[i] at elapsed ==
    # arrivals[i]. None when not traveling. Private: a passer-by is seen, his route is not.
    transit: dict[str, Any] | None = None
    long_term_goals: list[str] = field(default_factory=list)
    # The whole queue, finished goals included; ``status`` says which are still live.
    short_term_goals: list[GoalView] = field(default_factory=list)
    # Their relations to others, as they stand at the end of the step.
    relations: list[GraphEdge] = field(default_factory=list)
    thoughts: list[ThoughtView] = field(default_factory=list)
    # Belongings nobody else can perceive (``EntityView.is_public`` false).
    carrying: dict[str, EntityView] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentView:
    """One agent at the end of the step, in the scope their body is in."""

    agent_id: str
    name: str
    # Visible at a glance, like the name. "" / None when the world never fixed them.
    gender: str = ""
    age: int | None = None
    # None: not known to this viewer (the actor's view, ``interaction.actor_view``).
    is_main_character: bool | None = False
    is_active: bool = True
    public: AgentPublicView = field(default_factory=lambda: AgentPublicView(activity_status=""))
    private: AgentPrivateView = field(default_factory=lambda: AgentPrivateView(emotion=""))


@dataclass(frozen=True)
class NpcPublicView:
    """What anyone in an Npc's presence can see of it."""

    condition: str = ""
    # Put here this beat by something outside the world rather than walked here.
    displaced: bool = False
    carrying: dict[str, EntityView] = field(default_factory=dict)


@dataclass(frozen=True)
class NpcPrivateView:
    """What nobody in the world reads of an Npc: it has no mind and no actor performs it, so this
    is the observer's alone.

    ``outcome`` is what happened to it this beat, as one third-person sentence written by its
    runner; "" when nothing did. The engine keeps an Npc's doings out of every onlooker's
    perception and off the emission timeline, which is why it is here and not an event. It
    always happens where the body ends the step (the same-place invariant in
    ``NpcRunner._advance_one``).
    """

    outcome: str = ""
    # The outcome describes something still under way, like ``phase="begin"`` on an action.
    ongoing: bool = False
    # Hidden belongings (a sealed letter it carries).
    carrying: dict[str, EntityView] = field(default_factory=dict)


@dataclass(frozen=True)
class NpcView:
    """A body with no mind, in the scope it is in.

    Kept apart from agents because it isn't one, so the relation graph, character cards and
    agent profiles exclude it without any filtering. ``color`` comes from the backend because it
    only marks "no mind" if the character palette never uses gray, and ``world.identity_color``
    owns that guarantee.
    """

    npc_id: str
    name: str
    color: str = MINDLESS_BODY_COLOR
    gender: str = ""
    age: int | None = None
    description: str = ""
    public: NpcPublicView = field(default_factory=NpcPublicView)
    private: NpcPrivateView = field(default_factory=NpcPrivateView)


@dataclass(frozen=True)
class LocationScope:
    """One place at the end of the step: who and what is there, and what happened there.

    Referenced by id; its name comes from the world's ``PlaceIdentity`` table."""

    location_id: str
    agents: list[AgentView] = field(default_factory=list)
    npcs: list[NpcView] = field(default_factory=list)
    entities: dict[str, EntityView] = field(default_factory=dict)
    events: list[ScopeEvent] = field(default_factory=list)  # in seq order


@dataclass(frozen=True)
class TransitScope:
    """Bodies between two waypoints, and what happened to them on the way. No place to watch
    from, so nothing here is public."""

    agents: list[AgentView] = field(default_factory=list)
    npcs: list[NpcView] = field(default_factory=list)
    events: list[ScopeEvent] = field(default_factory=list)


@dataclass(frozen=True)
class WorldScope:
    """What reached every place at once."""

    events: list[ScopeEvent] = field(default_factory=list)


@dataclass(frozen=True)
class ObserverView:
    """What only the one watching from outside the world reads; no character perceives it."""

    # Injections, with the sentence a director typed and what came of it. What each did inside
    # the world reaches the scopes as broadcasts and messages.
    injections: list[InjectionSummary] = field(default_factory=list)


@dataclass(frozen=True)
class StepView:
    """One completed runtime step: what happened and the state it ended in, by scope."""

    world_id: str
    step: int
    world_time: WorldTimeView
    locations: dict[str, LocationScope] = field(default_factory=dict)
    transit: TransitScope = field(default_factory=TransitScope)
    world: WorldScope = field(default_factory=WorldScope)
    observer: ObserverView = field(default_factory=ObserverView)

    @classmethod
    def from_runtime_payload(cls, payload: Mapping[str, Any]) -> "StepView":
        """Build a step view from the runtime event-bus payload, which carries the very
        snapshot the step saved."""

        return cls.from_snapshot(payload["snapshot"])

    @classmethod
    def from_snapshot(cls, snapshot: WorldSnapshot) -> "StepView":
        """Build a step view from a snapshot."""

        return _StepAssembler(snapshot).build()


class _StepAssembler:
    """Projects one snapshot into scopes and splits what onlookers perceive from what only some
    know. Composes no text and decides nothing new: every sentence is the producer's, and every
    split follows a fact the engine recorded (observations, participants, ``is_public``)."""

    def __init__(self, snapshot: WorldSnapshot) -> None:
        self._snapshot = snapshot
        self._state = snapshot.state
        self._events: dict[str, list[ScopeEvent]] = {}
        self._transit_events: list[ScopeEvent] = []
        self._world_scope_events: list[ScopeEvent] = []
        self._thoughts: dict[str, list[ThoughtView]] = {}

    def build(self) -> StepView:
        located, held = self._entities()
        self._actions()
        self._messages()
        self._broadcasts()
        self._injections()

        agents: dict[str, list[AgentView]] = {}
        transit_agents: list[AgentView] = []
        relations = _relations_by_holder(self._snapshot)
        for agent_id, record in sorted(self._state.agents.items()):
            if not isinstance(record, Mapping):
                continue
            agent_id = str(agent_id)
            view = self._agent(agent_id, record, held.get(agent_id, {}), relations.get(agent_id, []))
            where = str(record.get("location_id") or "")
            if _is_place(where):
                agents.setdefault(where, []).append(view)
            else:
                transit_agents.append(view)

        npcs: dict[str, list[NpcView]] = {}
        transit_npcs: list[NpcView] = []
        for npc_id, record in sorted(self._state.npcs.items()):
            if not isinstance(record, Mapping):
                continue
            view = self._npc(str(npc_id), record, held.get(str(npc_id), {}))
            where = str(record.get("location_id") or "")
            if _is_place(where):
                npcs.setdefault(where, []).append(view)
            else:
                transit_npcs.append(view)

        place_ids = sorted({*agents, *npcs, *located, *self._events})
        return StepView(
            world_id=self._snapshot.world_id,
            step=self._snapshot.step,
            world_time=WorldTimeView.from_payload(self._snapshot.world_time),
            locations={
                place_id: LocationScope(
                    location_id=place_id,
                    agents=agents.get(place_id, []),
                    npcs=npcs.get(place_id, []),
                    entities=located.get(place_id, {}),
                    events=_by_seq(self._events.get(place_id, [])),
                )
                for place_id in place_ids
            },
            transit=TransitScope(
                agents=transit_agents, npcs=transit_npcs, events=_by_seq(self._transit_events),
            ),
            world=WorldScope(events=_by_seq(self._world_scope_events)),
            observer=ObserverView(
                injections=[
                    _injection_from_record(r) for r in self._snapshot.events_of("injection")
                ],
            ),
        )

    # ----- events ------------------------------------------------------------

    def _put(self, location_id: str, event: ScopeEvent) -> None:
        if _is_place(location_id):
            self._events.setdefault(location_id, []).append(event)
        else:
            self._transit_events.append(event)

    def _actions(self) -> None:
        """Each action event where it was recorded, and what each other place observed of it.

        The snapshot already tells each execution's beat once (``engine.step_record``); the
        initiator's part speaks for it, every participant's thought goes to its owner.
        """
        for record in self._snapshot.events_of("action"):
            seq = int(coerce_optional_int(record.get("seq")) or 0)
            initiator_id = str(record.get("initiator_id") or "")
            execution_id = str(record.get("execution_id") or "")
            parts = [coerce_dict(p) for p in coerce_list(record.get("parts"))]
            lead = next((p for p in parts if str(p.get("agent_id") or "") == initiator_id), {})
            for part in parts:
                thought = str(part.get("inner_monologue") or "")
                if thought:
                    self._thoughts.setdefault(str(part.get("agent_id") or ""), []).append(
                        ThoughtView(seq=seq, execution_id=execution_id, text=thought)
                    )
            participant_ids = [str(i) for i in coerce_list(record.get("participant_ids"))]
            overheard_by = [str(i) for i in coerce_list(record.get("overheard_by"))]
            acted_upon = [str(i) for i in coerce_list(record.get("acted_upon"))]
            home = str(record.get("location_id") or "")
            observed: dict[str, ObservedView] = {}
            for o in (coerce_dict(o) for o in coerce_list(record.get("observations"))):
                place, text = str(o.get("location_id") or ""), str(o.get("text") or "")
                if text and place not in observed:
                    observed[place] = ObservedView(
                        text=text, strength=coerce_optional_float(o.get("strength")),
                    )
            route = record.get("route")

            self._put(home, ActionSummary(
                seq=seq,
                execution_id=execution_id,
                initiator_id=initiator_id,
                participant_ids=participant_ids,
                # Plain read: producers hand this layer JSON-native values, guarded at one
                # boundary by tests/unit/test_state_payload_is_jsonable.py. Don't re-defend here.
                action_type=str(record.get("action_type") or ""),
                phase=str(record.get("phase") or ""),
                deed=str(record.get("deed") or ""),
                target=_target_view(record.get("target")),
                affected_entity_ids=[str(e) for e in coerce_list(record.get("affected_entity_ids"))],
                overheard_by=overheard_by,
                interrupted_by=str(record.get("interrupted_by") or ""),
                succeeded=bool(record.get("succeeded", True)),
                failure_reason=str(record.get("failure_reason") or ""),
                detected=bool(record.get("detected", False)),
                not_executed=bool(record.get("not_executed", False)),
                public=observed.get(home) if _is_place(home) else None,
                private=ActionPrivateView(
                    audience=_unique([*participant_ids, *acted_upon, *overheard_by]),
                    # `or ""`, not `.get(k, "")`: a key present and null slips past a default,
                    # and `str(None)` silently yields "None", which prose prints.
                    outcome=str(lead.get("outcome") or ""),
                    gist=str(lead.get("gist") or lead.get("outcome") or ""),
                    dialogue=[
                        DialogueTurn(
                            speaker_id=str(d.get("speaker_id") or ""),
                            speaker=str(d.get("speaker") or ""),
                            line=str(d.get("line") or ""),
                        )
                        for d in coerce_list(record.get("dialogue"))
                        if isinstance(d, Mapping)
                    ],
                    action_description=str(lead.get("action_description") or ""),
                    duration_label=str(record.get("duration_label") or ""),
                    elapsed_steps=int(coerce_optional_int(record.get("elapsed_steps")) or 0),
                    total_steps=int(coerce_optional_int(record.get("total_steps")) or 0),
                    route=dict(route) if isinstance(route, Mapping) else None,
                ),
            ))
            for place, view in observed.items():
                # The way between waypoints is no place to watch from (see
                # NarrativeRuntime._carry_step_observations).
                if place == home or not _is_place(place):
                    continue
                self._put(place, ObservationSummary(
                    seq=seq, initiator_id=initiator_id, participant_ids=participant_ids,
                    text=view.text, strength=view.strength,
                ))

    def _injections(self) -> None:
        """What onlookers at each place perceived of an injection's world changes, under the
        injection's seq. The injection itself is the observer's (``ObserverView``)."""
        for record in self._snapshot.events_of("injection"):
            seq = int(coerce_optional_int(record.get("seq")) or 0)
            for o in (coerce_dict(o) for o in coerce_list(record.get("observations"))):
                text = str(o.get("text") or "")
                if text:
                    self._put(str(o.get("location_id") or ""), ObservationSummary(
                        seq=seq, initiator_id="", participant_ids=[], text=text,
                        strength=coerce_optional_float(o.get("strength")),
                    ))

    def _messages(self) -> None:
        """Each delivered message, in the scope(s) it reached.

        ``recipients``/``location_scope`` form MessageSystem's addressing matrix. A direct
        message lands where each receiver stood when it reached them, as the engine recorded:
        a letter reaches a man, not a room.
        """
        for record in self._snapshot.events_of("message"):
            receivers = [coerce_dict(r) for r in coerce_list(record.get("receivers"))]
            receiver_ids = [str(r.get("agent_id") or "") for r in receivers]
            sender_id = str(record.get("sender_id") or "")
            content = str(record.get("content") or "")

            def summary(ids: list[str], scope: str, audience: list[str] | None) -> MessageSummary:
                return MessageSummary(
                    seq=int(coerce_optional_int(record.get("seq")) or 0),
                    message_id=str(record.get("message_id") or ""),
                    sender_id=sender_id,
                    sender_name=str(record.get("sender_name") or ""),
                    receiver_ids=ids,
                    scope=scope,
                    content=content,
                    spoken=str(coerce_dict(record.get("metadata")).get("spoken") or "") or content,
                    audience=audience,
                )

            # recipients=None means broadcast (directed delivery is always a list); location_scope
            # then splits place-wide from world-wide.
            if record.get("recipients") is None:
                place = str(record.get("location_scope") or "")
                if place:
                    self._put(place, summary(receiver_ids, "place", receiver_ids))
                else:
                    self._world_scope_events.append(summary(receiver_ids, "world", None))
                continue
            by_place: dict[str, list[str]] = {}
            for receiver in receivers:
                where = str(receiver.get("location_id") or "")
                by_place.setdefault(where if _is_place(where) else "", []).append(
                    str(receiver.get("agent_id") or "")
                )
            for place, ids in by_place.items():
                self._put(place, summary(ids, "direct", _unique(ids)))

    def _broadcasts(self) -> None:
        for record in self._snapshot.events_of("broadcast"):
            event = BroadcastSummary(
                seq=int(coerce_optional_int(record.get("seq")) or 0),
                content=str(record.get("content") or ""),
                broadcast_type=str(record.get("broadcast_type") or ""),
                severity=str(record.get("severity") or ""),
                phenomenon=str(record.get("phenomenon") or ""),
                audience=(
                    [str(a) for a in coerce_list(record.get("receivers"))]
                    if record.get("receivers") is not None else None
                ),
            )
            place = str(record.get("location_scope") or "")
            if place:
                self._put(place, event)
            else:
                self._world_scope_events.append(event)

    # ----- state -------------------------------------------------------------

    def _entities(self) -> tuple[dict[str, dict[str, EntityView]], dict[str, dict[str, EntityView]]]:
        """The world's entities sorted by presence: (by location, by holder). Tombstones stay
        out (the engine keeps them only so restore can't resurrect a destroyed seed), as does
        ``is_takeable`` (seed-reconstruction authority)."""
        located: dict[str, dict[str, EntityView]] = {}
        held: dict[str, dict[str, EntityView]] = {}
        for entity_id, raw in sorted(self._state.entities.items()):
            payload = coerce_dict(raw)
            presence = str(payload.get("presence") or "")
            if presence == EntityPresence.DESTROYED.value:
                continue
            view = EntityView(
                name=str(payload.get("name") or ""),
                entity_type=str(payload.get("entity_type") or ""),
                state=str(payload.get("state") or ""),
                description=str(payload.get("description") or ""),
                is_public=bool(payload.get("is_public", True)),
                created_step=coerce_int(payload.get("created_step"), default=0),
                private=EntityPrivateView(content=str(payload.get("content") or "")),
            )
            ref = str(payload.get("presence_ref") or "")
            bucket = held if presence == EntityPresence.HELD.value else located
            bucket.setdefault(ref, {})[str(entity_id)] = view
        return located, held

    def _agent(
        self,
        agent_id: str,
        record: Mapping[str, Any],
        belongings: dict[str, EntityView],
        relations: list[GraphEdge],
    ) -> AgentView:
        emotion = record.get("emotion", "")
        emotion_value = str(emotion.get("primary", "")) if isinstance(emotion, Mapping) else str(emotion)
        activity_value = str(record.get("activity_status", ""))
        visible_activity = "" if activity_value == AgentActivityStatus.COVERT.value else activity_value
        need = coerce_optional_str(record.get("dominant_need"))
        return AgentView(
            agent_id=agent_id,
            name=str(record.get("agent_name") or "某人"),
            gender=str(record.get("gender") or ""),
            age=record.get("age") if isinstance(record.get("age"), int) and record.get("age") else None,
            is_main_character=bool(record.get("is_main_character", False)),
            is_active=bool(record.get("is_active", True)),
            public=AgentPublicView(
                activity_status=visible_activity,
                activity_label=_member_label(AgentActivityStatus, visible_activity),
                # record["condition"] is the dict (or None) snapshot_agent_state wrote; replay
                # round-trips it through json.dumps(default=str), so read defensively.
                condition=str(
                    (record.get("condition") or {}).get("description", "")
                    if isinstance(record.get("condition"), Mapping) else ""
                ),
                displaced=bool(record.get("displaced", False)),
                carrying={k: v for k, v in belongings.items() if v.is_public},
            ),
            private=AgentPrivateView(
                emotion=emotion_value,
                emotion_label=_member_label(EmotionType, emotion_value),
                emotion_intensity=coerce_optional_float(record.get("emotion_intensity")),
                emotion_valence=coerce_optional_float(record.get("emotion_valence")),
                dominant_need=need,
                dominant_need_label=_member_label(NeedType, need or ""),
                vitality=coerce_optional_float(record.get("vitality")),
                activity_status=activity_value,
                activity_label=_member_label(AgentActivityStatus, activity_value),
                transit=dict(record["transit"]) if isinstance(record.get("transit"), Mapping) else None,
                long_term_goals=list(record.get("long_term_goals") or []),
                short_term_goals=[
                    GoalView(
                        text=str(g.get("text") or ""),
                        status=str(g.get("status") or ""),
                        origin=str(g.get("origin") or "cognitive"),
                    )
                    for g in (coerce_dict(item) for item in coerce_list(record.get("short_term_goal_entities")))
                ],
                relations=relations,
                thoughts=self._thoughts.get(agent_id, []),
                carrying={k: v for k, v in belongings.items() if not v.is_public},
            ),
        )

    def _npc(self, npc_id: str, record: Mapping[str, Any], belongings: dict[str, EntityView]) -> NpcView:
        happened = coerce_dict(record.get("outcome"))
        return NpcView(
            npc_id=npc_id,
            name=str(record.get("name", "") or ""),
            gender=str(record.get("gender", "") or ""),
            age=record.get("age") if isinstance(record.get("age"), int) else None,
            description=str(record.get("description", "") or ""),
            public=NpcPublicView(
                condition=str(coerce_dict(record.get("condition")).get("description", "") or ""),
                displaced=bool(record.get("displaced", False)),
                carrying={k: v for k, v in belongings.items() if v.is_public},
            ),
            private=NpcPrivateView(
                outcome=str(happened.get("text") or ""),
                ongoing=bool(happened.get("ongoing", False)),
                carrying={k: v for k, v in belongings.items() if not v.is_public},
            ),
        )


def _is_place(location_id: str) -> bool:
    """Whether a body or event at this id is somewhere: the transit sentinel is the engine's own
    bookkeeping for "between two waypoints", not a place."""
    return bool(location_id) and location_id != IN_TRANSIT


def _by_seq(events: list[ScopeEvent]) -> list[ScopeEvent]:
    return sorted(events, key=lambda e: e.seq)


def _unique(items: Any) -> list[str]:
    return [i for i in dict.fromkeys(items) if i]


def _relations_by_holder(snapshot: WorldSnapshot) -> dict[str, list[GraphEdge]]:
    by_holder: dict[str, list[GraphEdge]] = {}
    for edge in relations_from_snapshot(snapshot):
        by_holder.setdefault(edge.from_id, []).append(edge)
    return by_holder


def _injection_from_record(record: Mapping[str, Any]) -> InjectionSummary:
    """Build InjectionSummary from a step's ``injection`` event.

    Key-for-key with ``engine.injection.serialize_injection`` — that function is the
    only producer, so this reader has no aliases to reconcile and no permanently-empty
    fields to carry.

    References stay in the narrative layer: ``affected_names`` and ``location_label`` are
    names, not ids, because they were resolved through the directory at injection time.
    """
    receipt = record.get("receipt")
    return InjectionSummary(
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
            str(record.get("agent_name") or "")
            for snapshot in snapshots
            for record in snapshot.state.agents.values()
            if record.get("is_main_character")
        }
        - {""}
    )
    # World lore for the creation-review card: pulled from the step-0 snapshot's
    # world block (narrative pitch preferred, era description as fallback).
    description = ""
    seconds_per_step = 0
    step0 = next((s for s in snapshots if s.step == 0), snapshots[0] if snapshots else None)
    if step0 is not None:
        world_block = step0.build.get("world")
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
        actor_character_id=(catalog_entry.get("actor_character_id") or None) if catalog_entry else None,
        actor_exited=bool(catalog_entry.get("actor_exited", False)) if catalog_entry else False,
        actor_pace=ActorPace(catalog_entry.get("actor_pace", ActorPace.EVERY_STEP.value))
        if catalog_entry else ActorPace.EVERY_STEP,
        director_reviewed=bool(catalog_entry.get("director_reviewed", False)) if catalog_entry else False,
    )



def relations_from_snapshot(snapshot: WorldSnapshot) -> list[GraphEdge]:
    """The snapshot's relations as graph edges — only those with narrative substance.

    The store also persists a baseline record the moment two agents merely co-occur in a
    memory; that is not a relationship. Same rule RelationEvolution filters candidates by.
    """
    edges: list[GraphEdge] = []
    for rel in snapshot.state.relations.values():
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



def _member_label(
    enum_cls: type[EmotionType] | type[AgentActivityStatus] | type[NeedType], value: str,
) -> str:
    try:
        return enum_cls(value).label
    except ValueError:
        return ""
