"""Contracts for actor mode: a person performing one character, in exchange with the world.

An actor supplies the motivation and decision stages for one character; the engine still runs
perception, adjudication and feedback for it. At those two stages the engine asks through an
``ActorTurn`` and continues the step with the actor's answer: an ``ActorDecision`` for a
DECIDE turn, an ``ActorInterruptAnswer`` for an INTERRUPT turn. The actor speaks only when
asked: words are read against an open DECIDE turn, into an ``ActorDecision`` the actor confirms
before it is submitted. Explicit operations without supplementary interpretation submit directly. Outside a turn
the actor may only break off an act in progress, which lands at the next interrupt stage.

A DECIDE turn opens whenever the character is idle, at the actor's pace (``ActorPace``). An
INTERRUPT turn opens when something reached the character mid-action. An INVITE turn opens at
arbitration, when someone's act would enlist the free character by invitation; it is answered by an
``ActorInvitationAnswer``. Compulsion is never asked. A SPEAK turn opens in a live conversation (spoken
dialogue turn by dialogue turn, ``Conversation``) whenever it is the character's dialogue turn; it
is answered by an ``ActorDialogueTurn``.

A DECIDE turn carries an ``ActorBriefing``: the lists an AI character's decision binds
against. An ``ActorAction`` names its targets by the ids in those lists and is bound through the
same per-type rules, so the actor can reach exactly what the character can, nothing more.

The exchange is recorded as the actor's thread (``ActorEntry``), one entry per thing that passes
between actor and world, in order: what reached the character, what the world asked, what the
actor said and answered, the dialogue turns of its live conversations, and what the character set out to
do and how it ended, in the words its own memory keeps. Each step's entries are kept with that step's snapshot.

Apart from the thread the actor amends the character's mind (``ActorMind``, ``ActorAmendments``)
at any time. The actor is a person and remembers for themselves, so the character's memories are
not shown. Its body, belongings and surroundings are the world's to show (the map), not the
actor's.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from core.interfaces.action import (
    KIND_AGENT, KIND_ITEM, KIND_LANDMARK, KIND_LOCATION, KIND_NPC, ActionType, DialogueTurn, Ref,
)
from core.interfaces.urgency import Urgency
from core.prompts import AFFECTION_RANGE, EMOTION_INTENSITY_RANGE, EMOTION_VALENCE_RANGE, TRUST_RANGE
from core.validate import require_at_least, require_in_range, require_member, require_text


class TurnKind(str, Enum):
    # Decision stage: the character is idle and needs something to do.
    DECIDE = "decide"
    # Interrupt stage: something reached the character mid-action; break off or carry on.
    INTERRUPT = "interrupt"
    # Arbitration: someone invites the free character into their act; accept or refuse.
    INVITE = "invite"
    # A live conversation: the character's dialogue turn, or the end of the conversation.
    SPEAK = "speak"


@dataclass(frozen=True)
class BriefPerson:
    """Someone the character can reach: present, or known and alive elsewhere."""

    agent_id: str
    name: str
    gender: str = ""
    present: bool = False


@dataclass(frozen=True)
class BriefNpc:
    """Someone here with no mind of their own, who runs errands for whoever asks."""

    npc_id: str
    name: str


@dataclass(frozen=True)
class BriefPlace:
    location_id: str
    name: str
    travel_seconds: int = 0


@dataclass(frozen=True)
class BriefEntity:
    """A thing within reach: lying here, or in someone's hands here, the character's own included."""

    entity_id: str
    name: str
    entity_type: str = "item"
    holder_id: str | None = None
    is_takeable: bool = False


@dataclass(frozen=True)
class BriefMessage:
    """A message delivered to the character."""

    # The sender as the message names them; empty when it names nobody.
    sender_name: str
    content: str
    # A ``Urgency`` value.
    urgency: str


@dataclass(frozen=True)
class BriefRelation:
    """The character's own side of a relation."""

    agent_id: str
    name: str
    gender: str = ""
    trust: float = 0.0
    affection: float = 0.0
    labels: tuple[str, ...] = ()
    summary: str = ""
    deceased: bool = False


@dataclass(frozen=True)
class ActorBriefing:
    """What a decision can name: the lists an ``ActorAction`` picks its targets from, as an AI
    character's decision binds against them, by the names the character knows.

    It names the ids in a decision summary (``ActorDecisionSummary``), so the actor can see
    what the words were bound to before confirming; binding itself checks the character's own
    lists, never this. ``people`` are everyone the character can reach, present or not; ``npcs``
    and ``entities`` are those within reach. The scene itself is the map's to show.
    """

    people: tuple[BriefPerson, ...]
    npcs: tuple[BriefNpc, ...]
    places: tuple[BriefPlace, ...]
    entities: tuple[BriefEntity, ...]
    location: BriefPlace | None = None


@dataclass(frozen=True)
class Invitation:
    """Someone inviting the character to join their act, named as the character perceives them."""

    inviter: BriefPerson
    # The kind of act it is invited into.
    action_type: ActionType
    # A TALK's topic, if the inviter stated one (``AgentAction.talk_topic``).
    talk_topic: str = ""


@dataclass(frozen=True)
class Conversation:
    """A live conversation the character is in, at the character's dialogue turn: whom with, and
    what has been said so far.

    ``turns_left`` counts the dialogue turns left, this one included; ``last``: the conversation
    ends with this one whatever is said. ``max_turn_chars`` caps what the actor says in it.
    """

    other: BriefPerson
    dialogue: tuple[DialogueTurn, ...]
    turns_left: int
    last: bool
    max_turn_chars: int
    listener_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActorTurn:
    agent_id: str
    step: int
    kind: TurnKind
    # DECIDE only.
    briefing: ActorBriefing | None = None
    # INTERRUPT only: the action the character is in the middle of.
    doing: str = ""
    # INVITE only: the invitation, and the act the character decided on this step, which
    # accepting sets aside ("" when it decided none).
    invitation: Invitation | None = None
    decided: str = ""
    # SPEAK only.
    conversation: Conversation | None = None
    # Why the character is asked: what reached it this step, summed up in a sentence for the
    # actor. Empty when nothing did (a decision at every step, or one after its own act ended).
    reason: str = ""
    # Names this turn and no other. Step and kind don't: a step replayed after a restore, or
    # renumbered by a reset, asks the same (step, kind) again as a new turn.
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(frozen=True)
class ActorAction:
    """The actor's action, in the slots an AI character's decision fills, with ids where the
    model gives list indices.

    Every id must come from the turn's briefing list named in its comment. Which slots count
    depends on ``action_type``, by the same rules as for an AI character (``agent.decision``);
    a slot that doesn't apply to the type is ignored.
    """

    action_type: ActionType
    # First person: what I do.
    action_description: str
    expected_outcome: str = ""
    estimated_steps: int = 1
    # SEND_MESSAGE recipients, or TALK: the first is whom I talk with, the rest listen. (people)
    person_ids: tuple[str, ...] = ()
    # SEND_MESSAGE to everyone here instead of to named people.
    message_announce: bool = False
    anonymous: bool = False
    # SEND_MESSAGE: the words delivered. A TALK's words the actor speaks turn by turn.
    message_content: str = ""
    # TALK: what it is about, known to everyone in it from the start; "" keeps it to myself.
    talk_topic: str = ""
    # SEND_MESSAGE: how pressing the words are to whoever receives them.
    urgency: Urgency = Urgency.NORMAL
    destination_id: str | None = None                # MOVE (places)
    move_carry_ids: tuple[str, ...] = ()             # MOVE: people present I take along (people)
    # PHYSICAL acts on exactly one of these four.
    physical_person_id: str | None = None            # (people, present)
    physical_npc_id: str | None = None               # (npcs)
    physical_entity_id: str | None = None            # (entities)
    item_name: str = ""                              # something on no list ("the door")
    physical_recipient_id: str | None = None         # handing the thing to someone present (people)
    errand_npc_id: str | None = None                 # ERRAND: who runs it (npcs)
    errand_destination_id: str | None = None         # (places)
    errand_item_id: str | None = None                # something in my hands (entities)
    errand_recipient_id: str | None = None           # (people)
    errand_announce: bool = False
    errand_message: str = ""

    def __post_init__(self) -> None:
        require_text(self.action_description, "action_description")
        require_at_least(self.estimated_steps, 1, "estimated_steps")


class ActorOperation(str, Enum):
    MOVE = "move"
    BROADCAST = "broadcast"
    TALK = "talk"
    ERRAND = "errand"
    TAKE = "take"
    USE = "use"
    DESTROY = "destroy"
    PHYSICAL = "physical"
    WORK = "work"
    REST = "rest"
    COVERT = "covert"
    SEND_MESSAGE = "send_message"


@dataclass(frozen=True)
class ActorOperationParameters:
    destination_id: str | None = None
    item_id: str | None = None
    recipient_id: str | None = None
    message: str | None = None
    announce: bool | None = None
    anonymous: bool | None = None

    def __post_init__(self) -> None:
        for name in ("destination_id", "item_id", "recipient_id"):
            value = getattr(self, name)
            if value is not None:
                require_text(value, name)


_INTENT_TARGET_KINDS = (KIND_AGENT, KIND_LOCATION, KIND_NPC, KIND_ITEM, KIND_LANDMARK)


def require_intent_target(target: Ref) -> None:
    """Raise ``ValueError`` unless ``target`` is something an actor intent may name."""
    require_member(target.kind, _INTENT_TARGET_KINDS, "target.kind")
    require_text(target.id, "target.id")


@dataclass(frozen=True)
class ActorIntent:
    """User intent; execution slots and parsing policy remain server-owned."""

    turn_id: str
    text: str | None = None
    operation: ActorOperation | None = None
    target: Ref | None = None
    parameters: ActorOperationParameters = field(default_factory=ActorOperationParameters)
    description: str = ""

    def __post_init__(self) -> None:
        require_text(self.turn_id, "turn_id")
        if self.text is not None:
            require_text(self.text, "text")
            if self.operation is not None or self.target is not None or self.description or self.parameters != ActorOperationParameters():
                raise ValueError("Text and operation intents are mutually exclusive")
        elif self.operation is None or self.target is None:
            raise ValueError("An operation intent requires operation and target")
        if self.target is not None:
            require_intent_target(self.target)
        if self.operation is not None:
            if self.operation is ActorOperation.BROADCAST:
                if self.description.strip():
                    raise ValueError("A broadcast intent takes no description; its text is the message")
            allowed = {
                ActorOperation.ERRAND: {"destination_id", "item_id", "recipient_id", "message", "announce"},
                ActorOperation.BROADCAST: {"message", "anonymous"},
                ActorOperation.SEND_MESSAGE: {"anonymous"},
            }.get(self.operation, set())
            for name in ("destination_id", "item_id", "recipient_id", "message", "announce", "anonymous"):
                if getattr(self.parameters, name) is not None and name not in allowed:
                    raise ValueError(f"Parameter {name} does not apply to {self.operation.value}")


@dataclass(frozen=True)
class ActorDecisionSummary:
    """A public decision summary, without executor slots or executable authority: an action, or
    ``operation`` ``IDLE_OPERATION`` for doing nothing this step (no description, target or
    duration)."""

    decision_summary_id: str
    description: str
    operation: str
    target: Ref | None
    estimated_duration_seconds: int
    parameters: dict[str, str | bool | list[str]] = field(default_factory=dict)


# The ``operation`` of a decision summary that does nothing this step.
IDLE_OPERATION = "idle"


@dataclass(frozen=True)
class ActorIntentResult:
    status: Literal["submitted", "needs_confirmation", "rejected"]
    reason: str | None = None
    decision_summary: ActorDecisionSummary | None = None


@dataclass(frozen=True)
class ActorOperationCapability:
    operation: ActorOperation
    available: bool
    reason: str | None = None
    required_parameters: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActorOperations:
    turn_id: str
    target: Ref
    operations: tuple[ActorOperationCapability, ...]


@dataclass(frozen=True)
class ActorConfirmation:
    decision_summary_id: str

    def __post_init__(self) -> None:
        require_text(self.decision_summary_id, "decision_summary_id")


@dataclass(frozen=True)
class EmotionAmendment:
    # An ``EmotionType`` value.
    emotion: str
    intensity: float
    valence: float

    def __post_init__(self) -> None:
        require_in_range(self.intensity, EMOTION_INTENSITY_RANGE, "intensity")
        require_in_range(self.valence, EMOTION_VALENCE_RANGE, "valence")


@dataclass(frozen=True)
class RelationAmendment:
    """How the character now regards one listed person; ``None`` keeps that value.

    Labels and the relation's summary are not here: they are a slow third-person reading of
    the relation (``RelationEvolution``), not the character's own feeling.
    """

    agent_id: str
    trust: float | None = None
    affection: float | None = None

    def __post_init__(self) -> None:
        if self.trust is not None:
            require_in_range(self.trust, TRUST_RANGE, "trust")
        if self.affection is not None:
            require_in_range(self.affection, AFFECTION_RANGE, "affection")


@dataclass(frozen=True)
class GoalAmendment:
    """One live short-term goal's text; no id means a new goal. Other fields are read-only."""

    text: str
    id: str | None = None

    def __post_init__(self) -> None:
        require_text(self.text, "goal.text")
        if self.id is not None:
            require_text(self.id, "goal.id")


@dataclass(frozen=True)
class BriefGoal:
    """A goal record visible only to its character's actor; time hints use world time."""

    id: str
    text: str
    status: str
    origin: str
    created_step: int
    due_step: int | None
    progress_summary: str
    time_hint: str
    attention: bool


@dataclass(frozen=True)
class ActorAmendments:
    """The actor's changes to the character's mind, each an outright value over what the engine
    wrote; whatever is left out keeps its value. ``insights`` are added: judgments the character
    has come to, kept among its memories like those its reflection forms.

    They land where nothing else is writing the character: at once while a DECIDE or INTERRUPT
    turn is open (the step waits on the actor), otherwise at the start of the next step. An INVITE
    turn counts as otherwise: it opens after the step's plan is laid, and committing the plan
    writes the goals it carries. So does a SPEAK turn: the conversation is spoken while the step's
    other acts are adjudicated, before its own feedback lands. Never between steps: a
    write outside a step belongs to no snapshot, and a restore would roll it back.
    """

    emotion: EmotionAmendment | None = None
    relations: tuple[RelationAmendment, ...] = ()
    # Long-term goals replace the list. Short-term edits preserve unmentioned goals.
    long_term_goals: tuple[str, ...] | None = None
    short_term_goals: tuple[GoalAmendment, ...] | None = None
    removed_short_term_goal_ids: tuple[str, ...] = ()
    insights: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for text in self.long_term_goals or ():
            require_text(text, "goal")
        for goal_id in self.removed_short_term_goal_ids:
            require_text(goal_id, "removed_short_term_goal_id")
        if self.removed_short_term_goal_ids and self.short_term_goals is None:
            raise ValueError("Goal removals require short_term_goals")
        for text in self.insights:
            require_text(text, "insight")

    def then(self, later: "ActorAmendments") -> "ActorAmendments":
        """These amendments with ``later`` saved over them: a value given in both is ``later``'s;
        insights from both are kept, in order."""
        relations = {r.agent_id: r for r in self.relations}
        for r in later.relations:
            earlier = relations.get(r.agent_id)
            relations[r.agent_id] = RelationAmendment(
                r.agent_id,
                trust=r.trust if r.trust is not None else (earlier.trust if earlier else None),
                affection=r.affection if r.affection is not None else (earlier.affection if earlier else None),
            )
        return ActorAmendments(
            emotion=later.emotion or self.emotion,
            relations=tuple(relations.values()),
            long_term_goals=later.long_term_goals if later.long_term_goals is not None else self.long_term_goals,
            short_term_goals=(
                later.short_term_goals if later.short_term_goals is not None else self.short_term_goals
            ),
            removed_short_term_goal_ids=(
                later.removed_short_term_goal_ids
                if later.short_term_goals is not None else self.removed_short_term_goal_ids
            ),
            insights=self.insights + later.insights,
        )


@dataclass(frozen=True)
class ActorMind:
    """The character's mind as its actor sees it at any time, and amends it
    (``ActorAmendments``).

    ``short_term_goals`` includes the live queue and recent history, separated by status.
    ``relations`` are everyone the character has a relation with. ``pending`` holds amendments
    saved but not landed yet; the values above don't include them.
    """

    # An ``EmotionType`` value.
    emotion: str
    emotion_intensity: float
    emotion_valence: float
    long_term_goals: tuple[str, ...]
    short_term_goals: tuple[BriefGoal, ...]
    relations: tuple[BriefRelation, ...]
    pending: ActorAmendments | None = None


class ActorAnswerRejected(ValueError):
    """The answer doesn't fit: an action the character can't take this step, or an amendment
    to something the character's mind doesn't list.

    ``reason`` is a code-layer slug (``talk_target_not_present``, ``person_ids_not_perceived``,
    ``relation_not_listed``...).
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# The reason given when the actor's words name nothing a decision can take.
UNREADABLE = "unreadable"


@dataclass(frozen=True)
class ActorDecision:
    """The answer to a DECIDE turn: exactly one of ``action`` or ``idle`` (the character does
    nothing this step). Goals are not part of it: the actor amends them in the mind
    (``ActorAmendments``) at any time."""

    action: ActorAction | None = None
    idle: bool = False

    def __post_init__(self) -> None:
        if (self.action is None) != self.idle:
            raise ValueError("A decision is exactly one of an action or idling")


@dataclass(frozen=True)
class ActorInterruptAnswer:
    """The answer to an INTERRUPT turn: break off or carry on. What to do instead the actor says
    at the DECIDE turn the break-off leads into. ``thought`` explains the choice in character."""

    interrupt: bool
    thought: str = ""


@dataclass(frozen=True)
class ActorInvitationAnswer:
    """The answer to an INVITE turn: join the inviter's act or not. Refusing turns the inviter
    away; whatever the character decided this step goes ahead as if nobody had asked."""

    accept: bool


@dataclass(frozen=True)
class ActorDialogueTurn:
    """The answer to a SPEAK turn: the character's dialogue turn, what it says (``line``) and
    whether the conversation ends with it. Ending may leave the line empty."""

    line: str = ""
    end: bool = False

    def __post_init__(self) -> None:
        if not self.end:
            require_text(self.line, "line")


@dataclass(frozen=True)
class ActorIdleAnswer:
    idle: Literal[True] = True

    def __post_init__(self) -> None:
        # ``is``, not equality: ``1 == True`` would let a non-bool through.
        if self.idle is not True:
            raise ValueError("An idle answer requires idle to be true")


# Resolved internal answers consumed by the runtime channel, including executable decisions.
ActorAnswer = ActorDecision | ActorInterruptAnswer | ActorInvitationAnswer | ActorDialogueTurn


# Client submissions and public records; decisions are referenced by decision summary id.
ActorPublicAnswer = ActorConfirmation | ActorIdleAnswer | ActorInterruptAnswer | ActorInvitationAnswer | ActorDialogueTurn


class ActorPace(str, Enum):
    """When an idle character's actor is asked to decide."""

    # Every step it is idle.
    EVERY_STEP = "every_step"
    # Only when something reached it, or an act of its own has just ended.
    WHEN_REACHED = "when_reached"


# ----- the actor's thread ----------------------------------------------------------------------


class EntryKind(str, Enum):
    PERCEIVED = "perceived"
    ASKED = "asked"
    SAID = "said"
    ANSWERED = "answered"
    BREAK_OFF = "break_off"
    BEGAN = "began"
    PROGRESSED = "progressed"
    ENDED = "ended"
    DIALOGUE_TURN = "dialogue_turn"


def _entry_id() -> str:
    return uuid.uuid4().hex


# Every entry carries ``step`` and ``time_label``: the step it belongs to (code layer) and the
# world's own name for that moment (``WorldTime.time_label``). The coordinator stamps
# ``time_label`` as it records the entry; constructors leave it empty.


@dataclass(frozen=True)
class PerceivedEntry:
    """What reached the character: the letters and broadcasts arriving as a step begins, or an
    act of someone else landing on it (``acted_upon``, as its own memory keeps it). Names come
    from perception, as for an AI character's prompts."""

    step: int
    messages: tuple[BriefMessage, ...] = ()
    broadcasts: tuple[str, ...] = ()
    acted_upon: tuple[str, ...] = ()
    time_label: str = ""
    kind: EntryKind = field(default=EntryKind.PERCEIVED, init=False)
    entry_id: str = field(default_factory=_entry_id)


@dataclass(frozen=True)
class AskedEntry:
    """The world asks: a turn opened."""

    step: int
    turn: ActorTurn
    time_label: str = ""
    kind: EntryKind = field(default=EntryKind.ASKED, init=False)
    entry_id: str = field(default_factory=_entry_id)


@dataclass(frozen=True)
class SaidEntry:
    """An intention submitted at a DECIDE turn, its decision summary or rejection, and the original
    request. Explicit operations retain their selected target and parameters even when rejected.
    ``reading`` is the reading's reasoning when it found the words unclear: where it got stuck,
    for the words to be put another way. A summary needs none: it shows what was read."""

    step: int
    turn_id: str
    words: str
    decision_summary: ActorDecisionSummary | None = None
    reason: str | None = None
    reading: str = ""
    intent: ActorIntent | None = None
    time_label: str = ""
    kind: EntryKind = field(default=EntryKind.SAID, init=False)
    entry_id: str = field(default_factory=_entry_id)


@dataclass(frozen=True)
class AnsweredEntry:
    """The actor answered the turn ``turn_id``."""

    step: int
    turn_id: str
    answer: ActorPublicAnswer
    time_label: str = ""
    kind: EntryKind = field(default=EntryKind.ANSWERED, init=False)
    entry_id: str = field(default_factory=_entry_id)


@dataclass(frozen=True)
class BreakOffEntry:
    """The actor asked to break off the act in progress, outside any turn."""

    step: int
    time_label: str = ""
    kind: EntryKind = field(default=EntryKind.BREAK_OFF, init=False)
    entry_id: str = field(default_factory=_entry_id)


@dataclass(frozen=True)
class BeganEntry:
    """The character set out on an act of its own, or was drawn into someone else's, expected to
    take ``total_steps`` steps, this one the first. It moves on with each ``ProgressedEntry`` and
    ends with the next ``EndedEntry``."""

    step: int
    description: str
    total_steps: int
    time_label: str = ""
    kind: EntryKind = field(default=EntryKind.BEGAN, init=False)
    entry_id: str = field(default_factory=_entry_id)


@dataclass(frozen=True)
class ProgressedEntry:
    """The act in progress went on another step: ``elapsed_steps`` of ``total_steps`` are done."""

    step: int
    elapsed_steps: int
    total_steps: int
    time_label: str = ""
    kind: EntryKind = field(default=EntryKind.PROGRESSED, init=False)
    entry_id: str = field(default_factory=_entry_id)


@dataclass(frozen=True)
class EndedEntry:
    """How the act in progress ended, as the character's memory keeps it (``fact``, first
    person). ``happened`` is False for an act that never engaged the world, which leaves no
    memory; ``fact`` is empty when the act was never adjudicated."""

    step: int
    fact: str
    succeeded: bool
    happened: bool = True
    time_label: str = ""
    kind: EntryKind = field(default=EntryKind.ENDED, init=False)
    entry_id: str = field(default_factory=_entry_id)


@dataclass(frozen=True)
class DialogueTurnEntry:
    """A dialogue turn of a live conversation, either side's. ``end``: the conversation ends with
    it, by the speaker's choice or because the turns ran out; an ending turn's line may be empty.
    SPEAK turns leave no ``AskedEntry`` or ``AnsweredEntry``: these entries are their record."""

    step: int
    dialogue_turn: DialogueTurn
    end: bool = False
    time_label: str = ""
    kind: EntryKind = field(default=EntryKind.DIALOGUE_TURN, init=False)
    entry_id: str = field(default_factory=_entry_id)


ActorEntry = (
    PerceivedEntry | AskedEntry | SaidEntry | AnsweredEntry | BreakOffEntry | BeganEntry
    | ProgressedEntry | EndedEntry | DialogueTurnEntry
)
