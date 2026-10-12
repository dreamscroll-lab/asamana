"""Contracts for actor-mode turns.

An actor supplies the motivation and decision stages for one character; the engine still runs
perception, adjudication and feedback for it. At those two stages the engine asks through an
``ActorTurn`` and continues the step with the actor's answer: an ``ActorDecision`` for a
DECIDE turn, an ``ActorInterruptAnswer`` for an INTERRUPT turn.

The actor answers in their own words. The engine interprets them into an answer (for a
decision: an ``ActorAction`` or a wait, plus any goals the actor stated) and the actor confirms
it before it is submitted; nothing the engine read into the words acts unconfirmed.

A DECIDE turn carries an ``ActorBriefing``: what the character has in front of it, in the same
lists an AI character's decision binds against. An ``ActorAction`` names its targets by the
ids in those lists and is bound through the same per-type rules, so the actor can reach exactly
what the character can, nothing more.

Feedback is the engine's: it lands as for any character, as soon as it is computed. A DECIDE
turn lists the subjective part written since the actor last decided (``ActorFeedback``), and
the actor may amend it with the decision (``ActorAmendments``). What happened stays as the
engine recorded it: factual memories, outcomes, body and belongings are not amendable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum

from core.interfaces.action import ActionType
from core.interfaces.urgency import Urgency
from core.prompts import AFFECTION_RANGE, EMOTION_INTENSITY_RANGE, EMOTION_VALENCE_RANGE, TRUST_RANGE
from core.validate import require_at_least, require_in_range, require_text


class TurnKind(str, Enum):
    # Decision stage: the character is idle and needs something to do.
    DECIDE = "decide"
    # Interrupt stage: something reached the character mid-action; break off or carry on.
    INTERRUPT = "interrupt"


class TriggerKind(str, Enum):
    """What reached the character this step, from the same perception an AI character gets.

    An AI character's signals go to the world-pressure LLM, which judges what they mean to it;
    for the actor's character that judgment is the actor's, so every signal is passed on.
    """

    MESSAGE = "message"
    BROADCAST = "broadcast"
    AMBIENT = "ambient"
    # Someone else's action landed on the character. This reaches a character through feedback
    # (its own memory of the effect), not perception, so it is noted separately.
    TARGETED = "targeted"


@dataclass(frozen=True)
class TurnTrigger:
    kind: TriggerKind
    # The perceived text (message, broadcast, onlooker's view of an act); empty for TARGETED,
    # whose details reach the character through its own memory.
    text: str = ""


@dataclass(frozen=True)
class BriefPerson:
    """Someone the character can reach: present, or known and alive elsewhere."""

    agent_id: str
    name: str
    gender: str = ""
    present: bool = False
    # How they look right now; only someone present can be seen.
    condition: str = ""


@dataclass(frozen=True)
class BriefNpc:
    """Someone here with no mind of their own, who runs errands for whoever asks."""

    npc_id: str
    name: str
    description: str = ""
    busy: bool = False
    condition: str = ""


@dataclass(frozen=True)
class BriefPlace:
    location_id: str
    name: str
    description: str = ""
    travel_seconds: int = 0


@dataclass(frozen=True)
class BriefEntity:
    """A thing within reach: lying here, or in someone's hands here."""

    entity_id: str
    name: str
    description: str = ""
    # What is written on it, if anything and readable.
    content: str = ""
    held_by_me: bool = False
    # The holder as the character knows them; empty for something lying here.
    holder_name: str = ""


class BriefMemoryKind(str, Enum):
    FACTUAL = "factual"            # what happened
    EXPERIENTIAL = "experiential"  # how the character took it
    INSIGHT = "insight"            # a judgment formed over time
    PERIOD = "period"              # the impression a stretch of time left


@dataclass(frozen=True)
class BriefMemory:
    memory_id: str
    kind: BriefMemoryKind
    text: str
    created_step: int


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
    """What the character has in front of it at a decision, as an AI character's decision sees it.

    ``people``, ``npcs``, ``places`` and ``entities`` are the lists an ``ActorAction`` picks its
    targets from. Memories are those this moment called up, not everything the character
    remembers.
    """

    time_label: str
    location_id: str
    location_name: str
    # An ``EmotionType`` value.
    emotion: str
    emotion_intensity: float
    emotion_valence: float
    vitality: float
    # The character's own condition (tied up, wounded...), as it knows it; empty if none.
    condition: str
    long_term_goals: tuple[str, ...]
    short_term_goals: tuple[str, ...]
    people: tuple[BriefPerson, ...]
    npcs: tuple[BriefNpc, ...]
    places: tuple[BriefPlace, ...]
    entities: tuple[BriefEntity, ...]
    memories: tuple[BriefMemory, ...]
    relations: tuple[BriefRelation, ...]
    # Recent attempts that didn't land; the world did not change.
    foiled_attempts: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActorFeedback:
    """The subjective feedback written for the character since the actor last decided.

    ``memories`` are how the character took things (experiential memories, insights, period
    impressions); ``relations`` are the people whose standing with the character changed. The
    current emotion is in the briefing.
    """

    memories: tuple[BriefMemory, ...] = ()
    relations: tuple[BriefRelation, ...] = ()


@dataclass(frozen=True)
class ActorTurn:
    agent_id: str
    step: int
    kind: TurnKind
    triggers: tuple[TurnTrigger, ...] = ()
    # DECIDE only.
    briefing: ActorBriefing | None = None
    feedback: ActorFeedback | None = None
    # DECIDE only: the engine's reading of what the actor already said for this step (with a
    # break-off), for them to confirm or replace. None when there was nothing, or it can't be
    # taken here.
    proposal: "ActorDecision | None" = None
    # INTERRUPT only: the action the character is in the middle of.
    doing: str = ""
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
    inner_monologue: str = ""
    expected_outcome: str = ""
    estimated_steps: int = 1
    # SEND_MESSAGE recipients, or TALK: the first is whom I talk with, the rest listen. (people)
    person_ids: tuple[str, ...] = ()
    # SEND_MESSAGE to everyone here instead of to named people.
    message_announce: bool = False
    # SEND_MESSAGE: the words delivered; TALK: what I mean to say.
    message_content: str = ""
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
class MemoryAmendment:
    """Rewrite one listed memory, or forget it (``text`` None)."""

    memory_id: str
    text: str | None

    def __post_init__(self) -> None:
        # Forgetting is ``None``, never an empty rewrite.
        if self.text is not None:
            require_text(self.text, "text")


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
class ActorAmendments:
    """The actor's changes to the feedback listed in the turn it answers."""

    emotion: EmotionAmendment | None = None
    memories: tuple[MemoryAmendment, ...] = ()
    relations: tuple[RelationAmendment, ...] = ()


class ActorAnswerRejected(ValueError):
    """The answer doesn't fit the step it answers: an action the character can't take, or an
    amendment to something the turn didn't list.

    ``reason`` is a code-layer slug (``talk_target_not_present``, ``person_ids_not_perceived``,
    ``memory_not_listed``...).
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ActorWait:
    """Stay idle until a trigger arrives, or until ``steps`` world steps pass if given."""

    steps: int | None = None

    def __post_init__(self) -> None:
        if self.steps is not None:
            require_at_least(self.steps, 1, "steps")


@dataclass(frozen=True)
class ActorDecision:
    """The answer to a DECIDE turn: exactly one of ``action`` or ``wait``.

    The goal lists are the actor's motivation and replace the character's goals when given;
    ``None`` keeps them unchanged. ``amendments`` change the feedback the turn listed.
    """

    action: ActorAction | None = None
    wait: ActorWait | None = None
    long_term_goals: tuple[str, ...] | None = None
    short_term_goals: tuple[str, ...] | None = None
    amendments: ActorAmendments | None = None

    def __post_init__(self) -> None:
        if (self.action is None) == (self.wait is None):
            raise ValueError("A decision is exactly one of an action or a wait")
        for goals in (self.long_term_goals, self.short_term_goals):
            for text in goals or ():
                require_text(text, "goal")


@dataclass(frozen=True)
class ActorInterruptAnswer:
    """The answer to an INTERRUPT turn: whether to break off, and why.

    ``then`` keeps the actor's words about what to do next, said with the break-off. They are
    read at that step's decision stage, against what the character then has in front of it, and
    come back as the DECIDE turn's ``proposal`` to confirm.
    """

    interrupt: bool
    thought: str = ""
    then: str = ""

    def __post_init__(self) -> None:
        if self.then.strip() and not self.interrupt:
            raise ValueError("Only a break-off leads into a next decision")

