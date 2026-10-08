"""Structured perception contracts for the agent cognition loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from core.interfaces.phenomenon import Phenomenon, parse_phenomenon
from core.interfaces.severity import Severity, parse_severity


class BroadcastType(str, Enum):
    """World-state broadcast categories.

    BroadcastChannel carries only sender-less changes in world state. Communication (messages
    with a sender_id) goes through MessageSystem instead. Add new categories here; don't use bare
    strings.
    """

    WORLD_EVENT = "world_event"


@dataclass
class VisibleEntity:
    """An entity an agent can reach from where they stand.

    **Co-presence, not ground-presence**: lying here, in the hands of someone here, or in my own
    hands are all equally here. Being held changes affordance and ownership, not presence;
    otherwise a held item would vanish from the world, even for its holder.

    ``holder_id`` is a code-layer key and never enters a prompt: renderers resolve it to a
    narrative referent (a name, or "我").
    """

    entity_id:   str
    name:        str
    entity_type: str      # WorldEntityType string value; str to avoid core→world dep
    state:       str = "intact"
    description: str = ""
    is_takeable: bool = False
    holder_id:   str | None = None   # the agent carrying it; None = lying here
    # Filled only when this perceiver can read it (see ``WorldEntity.readable_by``). Empty means no
    # content or unreadable; to the perceiver these are the same.
    content:     str = ""


@dataclass(frozen=True)
class LocationView:
    """A perceived location's display info, for the agent's current location and (inside a
    ``ReachableLocation``) each place it can travel to. New perceived attributes extend this
    object, not every signature it flows through.

    ``is_public`` is a perceived fact, NOT a gate: whether THIS person may enter turns on who he
    is and what he's doing, a judgment (CLAUDE.md §5). A boolean bar would shut a palace against
    the emperor who lives in it.
    """

    name:        str
    description: str = ""
    is_public:   bool = True


@dataclass(frozen=True)
class ReachableLocation:
    """One place the agent can travel to, with the cost of getting there.

    ``destination_index`` binds to this list's position; ``travel_seconds`` is rendered as a
    natural travel duration.
    """

    location_id:    str            # code-layer binding key; never enters prompts
    view:           LocationView   # narrative-layer display (name + description)
    travel_seconds: int            # shortest-path walking time from the agent's location


@dataclass(frozen=True)
class Situation:
    """An agent's narrative situation right now: when and where. The input to the situation header.

    Narrative-layer values only, never steps or ids. The agent caches one after each perceive,
    for paths that don't hold the spatial perception.
    """

    location_view: LocationView | None = None
    time_label: str = ""

    @classmethod
    def from_spatial(cls, spatial: "SpatialPerception") -> "Situation":
        return cls(location_view=spatial.location_view, time_label=spatial.world_time_label)


@dataclass(frozen=True)
class AmbientEvent:
    """A public, location-level ambient event; the element type of SpatialPerception.ambient_events.

    content: the text onlookers see (already includes any readable prefix, e.g. "[秘密行动暴露] ...")
    strength:
        None          → PerceptionLayer uses the default ambient strength (0.2, a weak background signal)
        float ∈ [0,1] → strength set by the producer, e.g. a strong social signal like covert exposure
    actor_ids:
        ()            → a generic trace in the environment (every agent at the location sees it)
        non-empty     → produced by the acting body these agents belong to (one id for a solo
                        action; all participants for a group action like TALK), plus anyone the
                        act landed on (struck, restrained; they already got their own memory from
                        the effect). spatial_for filters them out, so nobody reads their own
                        action as a third-party observation. A set, because a group action can
                        be recorded under another participant's id (a TALK passive-join).

    agent_actor_ids:
        The subset of ``actor_ids`` with cognition, derived in ``record_carry_observation``
        (``agent/`` can't reach ``EnvironmentSystem._npcs``). A memory's ``related_agents`` may
        only come from this: an Npc there would get relation labels, though it has no relations.

    PerceptionLayer is driven entirely by strength, never by content type.
    """

    content: str
    strength: float | None = None
    actor_ids: tuple[str, ...] = ()
    agent_actor_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PerceivedIdentity:
    """How a person appears in perception: what I can call them.

    Gender is more visible than a name, so any channel that provides a name may provide gender.
    Render with ``core.prompts.person_referent``; missing values are omitted. "Perceived", not
    "View": unlike a place, what I can call a person depends on who I am.
    """

    name: str = ""
    gender: str = ""


@dataclass(frozen=True)
class PerceivedPresence:
    """Everything I see of someone present: what I can call them, and their current state.

    Nested because the halves have different lifetimes: ``identity`` is fixed and may enter the
    persistent name cache (``Agent.remember_agent``); ``condition`` is derived each step and must
    never be cached, or he would stay bound long after being untied. The type enforces it.

    Both may be empty; presence itself is the key existing.
    """

    identity: "PerceivedIdentity" = field(default_factory=lambda: PerceivedIdentity())
    condition: str = ""


@dataclass(frozen=True)
class NpcIdentity:
    """An Npc's immutable identity, fixed at build time, so ``WorldDirectory`` supplies it.

    Not ``PerceivedIdentity``: the assigner also needs age and ``description`` to pick from the
    roster.
    """

    name: str = ""
    gender: str = ""
    age: int | None = None
    description: str = ""   # who they are, what they do, their strengths and weaknesses; the main basis for admission


@dataclass(frozen=True)
class PerceivedNpc:
    """Everything I see of an Npc at my location; nested like ``PerceivedPresence``.

    ``busy`` / ``condition`` only warn the assigner; what actually blocks an errand is
    ``ErrandExecutor``'s feasibility gate.
    """

    identity: "NpcIdentity" = field(default_factory=lambda: NpcIdentity())
    busy: bool = False       # already on an errand
    condition: str = ""      # rendered condition text (pinned down / broken leg…); empty = none


@dataclass
class SpatialPerception:
    """Category 1: location perception. The agent must be physically present to receive it."""

    location_id: str                  # code-layer binding key (routing/snapshot); never enters prompts
    location_view: LocationView       # narrative-layer display (name + description) for in-character text
    world_time_label: str
    current_step: int
    world_time_hour: int = -1  # hour-of-day [0,23] from the clock; -1 = unknown
    # agent_id → PerceivedPresence: the single source of truth for who is here and how they appear.
    # Don't split it into parallel structures keyed by the same ids: they drift. Assembled only by
    # ``engine.presence.attach_presence``. Key order is contract: the roster's #N binds to it.
    visible_agents: dict = field(default_factory=dict)
    # npc_id → PerceivedNpc, same key-order contract. A separate table: consumers of the one above
    # assume every id resolves to an ``Agent``, and a mixed-in Npc becomes a silent null step.
    visible_npcs: dict = field(default_factory=dict)
    visible_entities: List[VisibleEntity] = field(default_factory=list)
    ambient_events: List[AmbientEvent] = field(default_factory=list)      # public location-level events and action results
    # Every location reachable from here via shortest paths over the connection graph (not just
    # neighbors), sorted by travel cost. destination_index binds to positions in this list.
    # Empty while IN_TRANSIT.
    reachable_locations: List["ReachableLocation"] = field(default_factory=list)

    @property
    def visible_agent_ids(self) -> List[str]:
        """Ids of those present, in presence order. Read-only: who is here is only replaced
        wholesale (``engine.presence.reset_visible``)."""
        return list(self.visible_agents)

    @property
    def visible_npc_ids(self) -> List[str]:
        return list(self.visible_npcs)


@dataclass
class Broadcast:
    """Carries a change in world state to perception (category 3).

    Rules:
    - source is always "system"; there is no meaningful sender.
    - It represents something that objectively happened (weather, resources, death…), perceived
      passively.
    - It doesn't trigger memory writes, relation updates or interrupt checks; the agent's
      cognition loop decides whether to respond.
    - Communication from someone goes through MessageSystem, not here.
    """

    content: str
    source: str           # always "system"
    broadcast_type: BroadcastType
    deliver_step: int     # step it becomes perceivable (like Message.deliver_step): injections = same step; death notices = next step
    location_scope: str | None = None  # None = global; str = only agents at that location_id receive it
    severity: Severity = Severity.LOW  # set by the producer when publishing; consumers must not infer it
    # What the change looks like (fire / rain / …), orthogonal to severity (how big). A fact, not a
    # render instruction; NONE = nothing visible. Never in content: it is a code-layer classification.
    phenomenon: Phenomenon = Phenomenon.NONE

    def __post_init__(self) -> None:
        # Accept strings from snapshots; boundary normalization only, no business rules.
        if not isinstance(self.severity, Severity):
            self.severity = parse_severity(self.severity)
        if not isinstance(self.phenomenon, Phenomenon):
            self.phenomenon = parse_phenomenon(self.phenomenon)
