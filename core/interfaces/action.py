"""Action contracts shared across cognition, engine, and arbitration.

No imports from ``agent/``, ``engine/`` or ``world/``: ``core/`` stays the dependency root.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields
from enum import Enum
from typing import Any, Iterable, List

from core.coerce import coerce_enum
from core.interfaces.condition import BodyCondition
from core.interfaces.urgency import Urgency


# What a ``Ref`` may point at. ``AGENT`` and ``LOCATION`` are the world's two structural
# kinds; the rest are entities — ``ITEM``/``LANDMARK`` come from the environment's own
# ``entity_type`` (its table is the only authority on which a listed entity is), and
# ``OBJECT`` is an off-roster one a decision named in free text ("the ground").
KIND_AGENT = "agent"
KIND_LOCATION = "location"
# A body that acts but does not think: something with a position and hands that can't be talked
# to, remembered at, or made to feel. Not AGENT: those paths all assume an ``Agent`` object behind
# the id.
KIND_NPC = "npc"
KIND_ITEM = "item"
KIND_LANDMARK = "landmark"
# A source marker, not an ontology class: only the free-text branch of ``_bind_physical`` sets it.
# ``Ref.id`` then holds the original phrase, not an id: the judge must read that phrase. Looking it
# up in the directory always misses and falls back to "某物", discarding the only information.
KIND_OBJECT = "object"


@dataclass(frozen=True)
class Ref:
    """One thing an action names: WHAT KIND it is, and WHICH one.

    ``kind`` is a ROUTING key, never taken from an LLM: it is derived from which decision slot
    was filled (CLAUDE.md §6). A wrong kind doesn't miss the target, it routes the act into
    another namespace ("pin him down" becomes "clutch the edict").
    """

    kind: str
    id: str

    @staticmethod
    def agent(agent_id: str) -> "Ref":
        return Ref(KIND_AGENT, agent_id)

    @staticmethod
    def place(location_id: str) -> "Ref":
        return Ref(KIND_LOCATION, location_id)

    @staticmethod
    def npc(npc_id: str) -> "Ref":
        return Ref(KIND_NPC, npc_id)

    @staticmethod
    def entity(entity_id: str, kind: str = KIND_ITEM) -> "Ref":
        """Keep the kind as given; don't filter it through an allowlist.

        ``core/`` can't import ``world/`` to know the real kinds, so an allowlist would only
        rewrite a newly added entity type to ``object``. The reading layer decides.
        """
        return Ref(kind or KIND_ITEM, entity_id)

    @property
    def is_agent(self) -> bool:
        return self.kind == KIND_AGENT


def agents_of(refs: "Iterable[Ref]") -> list[str]:
    """The people among some refs, in order. The one way to ask; don't filter by hand."""
    return [ref.id for ref in refs if ref.is_agent]


@dataclass
class ActionTarget:
    """How an action relates to the rest of the world — three relations, nothing else.

    A ref is filed by **what the system will do with that relation**, never by the action type
    or the data's shape. Three relations, because there are three mechanisms:

    1. ``acts_on`` — the act is DONE TO them.  Read by the observer, ``_foiled_key``'s merging
       of repeated intents, memory rewrites, and which way a figure turns on screen.
    2. ``claims``  — the act SPENDS THEIR TURN.  Read by arbitration
       (``ActionExecutor.conscription``) and becomes ``participant_ids``.
    3. ``reaches`` — the act LANDS ON them passively, their turn untouched (feedback layer).

    They are independent: a TALK's interlocutor is both acted on and claimed. Don't file two as
    one: a carried companion (claimed, not acted on) in the aim reports a journey as an act upon
    a man with the destination dropped.

    **A NEW RELATION IS ONE OF THESE THREE, OR IT IS A NEW MECHANISM** with a consumer. A
    per-type field for one executor's convenience (a "recipient", a "listener") is ``reaches``.

    Which relation carries the real subject is the action's business: a MOVE acts on a PLACE and
    claims those it drags along; a handover acts on the THING and reaches the receiver, or the
    record reads as an assault with no object.
    """

    acts_on: list[Ref] = field(default_factory=list)
    claims: list[Ref] = field(default_factory=list)
    reaches: list[Ref] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Unfiltered, an unresolved slot arrives downstream as a REAL target with an empty id.
        # Dropped rather than raised, so the binding guards (talk_without_target /
        # physical_without_target / move_without_destination) reject the step.
        self.acts_on = [ref for ref in self.acts_on if str(ref.id or "").strip()]
        self.claims = [ref for ref in self.claims if str(ref.id or "").strip()]
        self.reaches = [ref for ref in self.reaches if str(ref.id or "").strip()]
        # One act, one kind of object, so consumers can read the kind off the first entry.
        # Several entries of ONE kind is fine (a group message's recipients).
        kinds = {ref.kind for ref in self.acts_on}
        if len(kinds) > 1:
            raise ValueError(f"acts_on mixes kinds: {sorted(kinds)}")

    @property
    def acted_on_kind(self) -> str:
        """The ROUTING kind of what this acts on; "none" when it acts on nothing."""
        return self.acts_on[0].kind if self.acts_on else "none"

    @property
    def acted_on_ids(self) -> list[str]:
        return [ref.id for ref in self.acts_on]

    @property
    def acted_on_agents(self) -> list[str]:
        return agents_of(self.acts_on)

    @property
    def acted_on_place(self) -> str | None:
        """The place this acts on — a MOVE's destination, and nothing else's."""
        return self.acts_on[0].id if self.acted_on_kind == KIND_LOCATION else None

    @property
    def single_acted_on_agent(self) -> str | None:
        agents = self.acted_on_agents
        return agents[0] if len(agents) == 1 else None

    @property
    def claimed_agents(self) -> list[str]:
        return agents_of(self.claims)

    @property
    def reached_agents(self) -> list[str]:
        return agents_of(self.reaches)


class ActionType(str, Enum):
    """Action types of the cognition loop.

    Each type must change persistent state, change who knows what, or advance goals, with an
    effect no other type can produce (CLAUDE.md Action Type Rules); otherwise merge it.

    What only each type can do:
    - TALK         — create knowledge shared by both sides; also changes the relation both ways
    - SEND_MESSAGE — convey intent asynchronously across time and space
    - MOVE         — change the agent's location
    - WORK         — turn personal will into a tangible result
    - PHYSICAL     — apply immediate force to a person or object
    - COVERT       — gather information about others or other places unnoticed
    - REST         — actively restore the agent's vitality
    - ERRAND       — have another body act elsewhere on your behalf and bring back what it saw.
                     Unlike MOVE (only your own body), SEND_MESSAGE (words one way, no return,
                     no objects) and PHYSICAL (force here and now). The cost is travel time,
                     being seen, and the chance it isn't done.
    """

    TALK = "talk"
    SEND_MESSAGE = "send_message"
    REST = "rest"
    WORK = "work"
    MOVE = "move"
    PHYSICAL = "physical"
    COVERT = "covert"
    ERRAND = "errand"


class Deed(str, Enum):
    """What a bystander SEES a body do — the observable verb.

    ``ActionType`` is the kind of intent; ``Deed`` is the observable act. They coincide for
    every type but PHYSICAL, which packs seven unrelated verbs into one type. Neither
    ``action_type`` nor the consequence (harm/alter/subdue/none) tells them apart: "an object's
    state changed" fits taking, using and destroying alike, so picking up a letter would render
    as smashing it.

    Declared by the adjudicator from what the actor was DOING, so it survives failure: a snatch
    that misses is still a snatch. Structural only, no theme (CLAUDE.md Rule 7).
    """

    # PHYSICAL's seven verbs — the axis ActionType cannot express.
    STRIKE     = "strike"      # violence upon a person
    RESTRAIN   = "restrain"    # hands laid on a person, drawing no blood
    SEIZE      = "seize"       # take an object into one's possession
    RELINQUISH = "relinquish"  # let a held object go — set it down, or into another's hands
    OPERATE    = "operate"     # work an object where it stands (open / use / tear)
    DESTROY    = "destroy"     # wreck an object out of the world
    EXERT      = "exert"       # physical force with no target at all
    # Every other type IS its own deed; the two axes coincide, so they need no extra one.
    TALK         = "talk"
    SEND_MESSAGE = "send_message"
    MOVE         = "move"
    WORK         = "work"
    REST         = "rest"
    COVERT       = "covert"
    ERRAND       = "errand"


#: Action types whose hidden layer a covert action can uncover at most; an allowlist, so a new
#: type is excluded by default. A coarse outer gate only: each beat is also granted per result
#: via ``ActionResult.happening`` and both must pass (a TALK's interrupt beat holds the
#: interrupter's private thoughts, which a type-only check would let through).
COVERTABLE_ACTION_TYPES: "frozenset[ActionType]" = frozenset({
    ActionType.TALK,     # dialogue transcript (onlookers only see that two people talked)
    ActionType.WORK,     # the work product (onlookers only see him busy at his desk)
    ActionType.ERRAND,   # what the errand was (onlookers only see him stop someone for a word)
})


#: The deeds PHYSICAL may resolve to — the closed set the adjudicator chooses from.
PHYSICAL_DEEDS: tuple[Deed, ...] = (
    Deed.STRIKE, Deed.RESTRAIN, Deed.SEIZE, Deed.RELINQUISH,
    Deed.OPERATE, Deed.DESTROY, Deed.EXERT,
)


def parse_deed(label: str, *, default: Deed) -> Deed:
    """Coerce an adjudicator's deed label to the enum; anything unknown → ``default``, silently
    (a field-level coercion, like ``parse_emotion_type``)."""
    return coerce_enum(Deed, label, default=default)


@dataclass
class AgentAction:
    """Action chosen by the agent.

    ``status`` and ``intent`` are ``Any`` because their types (``ActionStatus`` in
    ``agent.personality``, ``ActionIntent`` in ``agent.decision``) live above this layer.
    """

    action_type: "ActionType | str"
    action_description: str = ""
    agent_id: str = ""
    step: int = 0
    target: ActionTarget = field(default_factory=ActionTarget)
    inner_monologue: str = ""
    expected_outcome: str = ""
    estimated_steps: int = 1
    urgency: Urgency = Urgency.NORMAL   # SEND_MESSAGE only; see URGENCY_SCALE_DESCRIPTION
    remaining_steps: int = 1
    status: Any = None
    llm_model: str = "rule"
    reason: str = ""
    content: str = ""
    intent: Any = None

    def __post_init__(self) -> None:
        if not self.action_description:
            self.action_description = self.content
        if not self.content:
            self.content = self.action_description
        if isinstance(self.action_type, str) and not isinstance(self.action_type, ActionType):
            self.action_type = ActionType(self.action_type)


@dataclass(frozen=True)
class Observed:
    """What onlookers at one place see: where, what, and how noticeable.

    Per location because one action can read differently in several places (a MOVE's origin
    sees him leave, the way sees him pass): N viewpoints, not one view delivered N times.

    ``strength`` feeds ``EnvironmentSystem.record_carry_observation`` (None = ordinary ambient),
    so covert exposure takes the same delivery path.

    No free-form extra field: a ``notes: list[str]`` gets used to encode strings for engine code
    to match, invisible to the read model and renderer. Add a typed, named field instead.
    """

    location_id: str          # code-layer id, not a narrative name; from observed_here / path ids
    text: str                 # third-person onlooker text (the part visible at this place)
    strength: float | None = None


@dataclass
class TargetAgentEffect:
    """Effects an action lands on somebody other than its actor.

    The one sanctioned channel for that (CLAUDE.md Executor/Feedback): an executor declares the
    deltas, the feedback layer applies them via ``Agent.apply_target_effect``. Every field past
    ``factual_memory`` is opt-in and defaults to a no-op.
    """

    agent_id: str
    factual_memory: str
    # True: I heard/saw this because I was there, not because it was done to me.
    overheard: bool = False
    emotion_type: str | None = None     # None → skip emotion injection
    emotion_intensity: float = 0.0
    emotion_valence: float = 0.0
    relation_toward_actor: tuple[str, float, float] | None = None
    # (actor_id, trust_delta, affection_delta) — None → skip relation update
    vitality_damage: float = 0.0
    # Change to the target's ongoing condition. Two flat fields give three states:
    #   condition_set non-empty     → impose / replace
    #   condition_cleared=True      → lift (someone untied him, woke him up)
    #   both default                → leave his current condition alone
    # The default must be "unchanged", not "clear": punching a bound man shouldn't untie him.
    condition_set: "BodyCondition | None" = None
    condition_cleared: bool = False
    death_cause: str | None = None
    # Third-person cause of death, from which death handling builds the global death notice.
    # Narrative-layer text: names and natural wording only, never ids or steps.
    #
    # Includes its own closing punctuation: death handling only prepends a "name (role)" tag, so
    # it must not append a period. Don't remove that tag: the notice is a global broadcast and
    # most readers don't know the deceased. Accepted cost: in the director's case the name
    # appears twice, since code shouldn't rewrite a sentence the director wrote.


@dataclass
class EntityStateChange:
    """The one command form for changing an entity already in the world, applied only by
    EnvironmentSystem.change_entity_state() (which also propagates perception).
    """

    entity_id:        str
    # Empty means keep, not clear: fields change independently, and making callers copy unchanged
    # ones invites silent edits. Rename only for a new identity (wood carved into a chair), not a
    # new look: others' memories refer to it by name.
    new_name:         str = ""
    new_state:        str = ""
    new_description:  str = ""
    new_content:      str = ""
    owner_id:         str | None = None   # set → the entity's NEW HOLDER (presence → HELD): the actor when seized, the recipient when handed over; gated on is_takeable
    location_id:      str | None = None   # set → drop/move semantics (presence → AT_LOCATION)
    destroyed:        bool = False        # set → removal (presence → DESTROYED, terminal); wins over owner/location
    perception:       str = ""            # perception text broadcast to observers; empty → no perception triggered
    perception_scope: str | None = None   # None → use actor's current location


@dataclass
class EntitySpawn:
    """A world entity that did not exist until this action produced it.

    Unlike ``EntityStateChange`` (whose ``entity_id`` must already resolve), a spawn carries the
    whole thing and gets its id on landing, only via ``EnvironmentSystem.spawn_entity``.

    ``is_public=False`` keeps it private to its holder (``spatial_for`` admits
    ``owner_id == viewer or is_public``). Use it for anything the action's contract calls
    privileged, e.g. a WORK product: born public, the whole room reads it next step.
    """

    name: str
    description: str = ""
    # WorldEntityType string value; ``str`` so ``core/`` keeps no dependency on ``world/``
    # (same reason as ``VisibleEntity.entity_type``).
    entity_type: str = "item"
    state: str = "intact"
    content: str = ""
    # Set → born in this agent's hands (HELD); None → born at the actor's location.
    holder_id: str | None = None
    is_public: bool = True
    # Third-person onlooker text, same channel as ``EntityStateChange.perception``. Empty
    # (onlookers notice nothing) is the fail-safe default: better to miss a perception than leak one.
    perception: str = ""
    # Filled in by ``EnvironmentSystem.spawn_entity`` on landing (uniqueness is its job). Empty
    # means the spawn was rejected (world full or empty name), so the observer reports nothing.
    entity_id: str = ""


@dataclass(frozen=True)
class ErrandOrder:
    """An errand assigned to an Npc: go somewhere, act there along these axes, look around, come back and report.

    The capabilities aren't enumerated; they come from combining four axes. "Hand something to
    someone", "leave something there", "carry a message", "announce publicly" and "go and look"
    aren't five separate capabilities. They are readings of combinations of the axes below.
    ``destination_id`` is required, so the other three axes give 2³ = 8 cells, all valid:

    ======  ======  =======  ======================================
    item    person  message  reads as
    ======  ======  =======  ======================================
    –       –       –        go and look
    –       –       ✓        go and say something publicly there
    –       ✓       –        go and see whether that person is there
    –       ✓       ✓        carry a message to someone
    ✓       –       –        leave the item there
    ✓       –       ✓        leave the item and say something publicly
    ✓       ✓       –        hand the item to someone
    ✓       ✓       ✓        hand it to someone and tell them something
    ======  ======  =======  ======================================

    No cell may silently collapse into another. The third row is the one most easily dropped as
    "nothing filled in": a person alone means "check whether they're there".

    The axes are the world's four kinds of things: places, things, people, words. Fetching isn't
    a fifth (the assigner can't see what's there to bind against), nor is acting on an object
    (that takes discretion, i.e. cognition).

    Don't turn this into a table of action variants: combinations explode, each needing a
    variant, a serialization tag and a prompt slot, while the behavior is one piece of code
    reading what's filled in.

    The skeleton is fixed: go → act → look → return → report.

    Landing semantics (``NpcRunner`` follows these literally, with no interpretation):

    - ``destination_id`` is required: an errand acts elsewhere; right here he can reach it himself.
    - ``recipient_id`` addresses both item and message: set, it targets that person; empty, it
      lands on the place. If the person isn't found, item and message come back as is. Never
      redirect to someone else: that would tell a room what was meant for one.
    - ``item_id`` empty = carry nothing.
    - ``message`` is the sentence delivered verbatim at the other end, any subject. It must be
      something that can be said aloud: it lands in the listener's memory and gets embedded, so
      an instruction for your own side ("go scout the guard posts") read out in public is
      permanent contamination (CLAUDE.md Rule 1). It is not the errand's purpose, which is
      ``AgentAction.action_description``.

    Produced by decision binding (``agent.decision``), landed by ``ErrandExecutor.start``. No
    adjudication: an Npc has no will, and whether it's free is a rule (``ErrandExecutor``'s
    feasibility gate).
    """

    npc_id: str
    destination_id: str
    item_id: str = ""
    recipient_id: str = ""
    message: str = ""


def errand_to_dict(order: "ErrandOrder | None") -> dict[str, object] | None:
    """ErrandOrder → JSON-native dict (None passes through). The only serializer for snapshots and the read model."""
    if order is None:
        return None
    return {f.name: getattr(order, f.name) for f in dataclass_fields(ErrandOrder)}


def errand_from_dict(payload: object) -> "ErrandOrder | None":
    """dict → ErrandOrder; None, non-dicts and payloads missing ``npc_id``/``destination_id``
    return None.

    Never raises: this runs during environment restore, where one bad errand would fail the
    whole world's restore. Unknown keys are dropped for the same reason.
    """
    if not isinstance(payload, dict):
        return None
    if not payload.get("npc_id") or not payload.get("destination_id"):
        return None
    known = {f.name for f in dataclass_fields(ErrandOrder)}
    return ErrandOrder(**{
        k: ("" if v is None else str(v)) for k, v in payload.items() if k in known
    })


@dataclass
class NpcEffect:
    """The effect of an action on an Npc, applied only by the feedback layer.

    Much smaller than ``TargetAgentEffect``: an Npc has no emotion, relations, vitality or
    memory, so only a condition lands on it (same three-state pair as ``TargetAgentEffect``).
    Taking what it holds goes through ``EntityStateChange.owner_id``.

    Deliberately no "cancel errand" field: once untied, the errand continues. Voiding it would
    leave the sender never hearing back, so memory and world diverge for good. The condition
    alone decides whether it stops (``NpcRunner`` checks it every step).
    """

    npc_id: str
    condition_set: "BodyCondition | None" = None
    condition_cleared: bool = False


@dataclass
class ActionResult:
    """Result of an executed action.

    Viewpoint contract: each text channel serves the consumers entitled to it; never reuse one
    string for two of them.

    - ``outcome``: the authoritative, complete, third-person account (actor, target, act,
      result), for full-information readers: the actor's action log and the god-view web /
      snapshot. Never first person: a subjectless sentence loses who did what. Names, not ids.
    - ``gist``: what happened in one sentence (who, where, what, how it ended): ``outcome``
      without the detail some outcomes append after it (a TALK transcript, an interrupter's
      thought). Same viewpoint as ``outcome``. Defaults to ``outcome``; a producer that appends
      detail must set it.
    - ``observations``: the third-person onlooker channel, one entry per place, perceived via
      ``runtime._carry_step_observations`` → ``environment.spatial_for``. Privileged content is
      removed (dialogue transcripts, work products, covert purposes, an interrupted action's
      ``purpose``/``reason``). Matches ``outcome`` for fully public acts (PHYSICAL / MOVE); empty
      for private ones (unexposed COVERT, SEND_MESSAGE).
    - ``factual_memory``: first person, the actor's private memory ("I tried X…"); never
      delivered to onlookers.
    - ``failure_reason``: third person, authoritative side, ≤20 characters, so the renderer can
      show the reason without parsing prose. Only externally statable reasons, never private
      thoughts. Never put it in ``observations``, which alone decides what onlookers learn.
    - ``entity_spawns``: ``name``/``description`` are the item's narrative reference;
      ``perception`` is onlooker text. ``is_public`` decides who can reach it from now on,
      ``perception`` who sees it appear; a private item sets neither.

    TALK is the model case: ``outcome`` = the named transcript; ``observations`` = "A and B
    talked and parted on bad terms"; ``factual_memory`` = each participant's own summary.

    Every new text field must state its viewpoint and consumers here first; one outside the
    contract drifts until a third-person template lands in a first-person memory channel.
    """

    action: AgentAction
    expected_outcome: str
    outcome: str            # third-person, complete and authoritative (actor log + god-view web/snapshot); see class docstring
    gist: str = ""          # see class docstring
    succeeded: bool = True
    # See Observed. Never derive it from outcome: that leaks privileged content. Empty by default
    # is fail-safe: better to miss a perception than to leak one.
    observations: list[Observed] = field(default_factory=list)
    # Third-person and complete, for someone hiding and watching: the layer only deliberate
    # watching catches (a transcript, the real result of some work). Empty = no such layer, so new
    # types are safe by default. Like observations, the executor grants it explicitly; nothing
    # downstream infers it from outcome. It often equals outcome and must still be written out.
    # Only consumer: EnvironmentSystem.record_happening, read by covert-action adjudication.
    happening: str = ""
    factual_memory: str = ""  # first person (the actor's own memory channel); see class docstring
    failure_reason: str = ""
    # third-person authoritative reason for failure, ≤20 chars; see class docstring
    # Whether a COVERT action was exposed; read directly by exposure propagation and
    # interrupt_coordinator, never parsed from text.
    detected: bool = False
    dialogue: List[dict] = field(default_factory=list)
    relation_updates: list[tuple[str, float, float]] = field(default_factory=list)
    # (target_id, trust_delta, affection_delta) — applied to the actor's relations via _apply_feedback
    target_effects: list[TargetAgentEffect] = field(default_factory=list)
    # Inbound effects to route to target agents after commit (PHYSICAL only)
    vitality_damage: float = 0.0
    # Change to the actor's own ongoing condition; same three states as TargetAgentEffect's pair.
    # Separate from the target's: an ownerless slot would have to guess whose condition it is.
    actor_condition_set: "BodyCondition | None" = None
    actor_condition_cleared: bool = False
    entity_state_changes: list[EntityStateChange] = field(default_factory=list)
    # See EntitySpawn. Applied via ``EnvironmentSystem.spawn_entity`` in the feedback layer.
    entity_spawns: list[EntitySpawn] = field(default_factory=list)
    # A receipt, not a command: ``ErrandExecutor.start`` already landed the errand. Don't land it
    # again: checking the body is free and occupying it must be one operation. Empty never means
    # the Npc refused (it has no will); only rules reject an errand.
    errand_orders: list[ErrandOrder] = field(default_factory=list)
    # See NpcEffect.
    npc_effects: list[NpcEffect] = field(default_factory=list)
    # What the actor was SEEN doing (see Deed). Only PHYSICAL sets it — every other type
    # is its own deed, so action_semantics falls back to the action_type there.
    deed: str = ""
    # True only when adjudication never happened (infra failure). The feedback layer treats it as
    # a null step with no writeback (Rule 1 tier 1). A genuine in-world failure leaves this False
    # and IS remembered. Never set on a fabricated/“successful” result.
    adjudication_failed: bool = False
    # True when a precondition was unmet (target absent or busy, no path, prerequisite missing);
    # set by ActionExecutionState.create_failed, always with succeeded=False. Unlike
    # adjudication_failed it IS remembered: a foiled plan is a real experience. It only lets
    # consumers tell a non-event from an in-world failure (the renderer mutes the former).
    not_executed: bool = False

    def __post_init__(self) -> None:
        if not self.gist:
            self.gist = self.outcome
