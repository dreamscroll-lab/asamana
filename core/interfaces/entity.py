"""Runtime bodies and things placed in the world: placed entities, NPCs and their errands.

Every layer reads these: the environment owns their live state, the world build creates them,
executors and the API read them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.interfaces.action import ErrandOrder
from core.interfaces.condition import BodyCondition
from core.interfaces.perception import DEFAULT_ENTITY_STATE


class WorldEntityType(str, Enum):
    """Ontological category of a placed entity.

    There is deliberately no ``LOCATION`` value, and that is the guard. A location is a ``Place``, a
    different type; the two share no mutable state (a location isn't somewhere else, can't be taken,
    can't be destroyed; a thing has no routes and no capacity). Without the value, "create a
    location" or "destroy a palace" can't be expressed on this channel at all. That is far stronger
    than call-site checks, where one missed check fails silently.
    """

    ITEM     = "item"        # portable item
    LANDMARK = "landmark"    # stronghold / marker / notice: not portable, but its state can change

    @property
    def is_takeable(self) -> bool:
        """Whether this kind of thing can be taken. This is the only place that answers it.

        A new kind puts the question right next to its value, so nobody has to remember to patch it
        elsewhere. Missing it fails silently: ``is_takeable`` always False means the PHYSICAL judge prompt
        never offers ``seize``, and the contract "PHYSICAL is the only way to take an item" breaks.
        """
        return self is WorldEntityType.ITEM


class EntityPresence(str, Enum):
    """Where a placed entity is — a closed, discriminated placement axis.

    An entity's placement is a sum type, not two orthogonal coordinates: it is
    at a location, held by a body, or destroyed (terminal, no longer in the
    world). Encoding it as one discriminator + one ref makes illegal states
    (both/neither of a location and an owner) unrepresentable and gives
    destruction a first-class terminal, rather than leaving the invariant to be
    hand-maintained across scattered mutation sites.

    ``presence_ref`` carries the target: a location_id for AT_LOCATION, the id of the
    holding body for HELD, and is unused (None) for DESTROYED. ``state`` (open/torn/used…)
    is an orthogonal, reversible condition and is *not* a placement — destruction
    never rides on the ``state`` string.
    """

    AT_LOCATION = "at_location"   # presence_ref = location_id
    HELD        = "held"          # presence_ref = the body holding it (either tier)
    DESTROYED   = "destroyed"     # presence_ref = None; terminal, gone from world


@dataclass
class WorldEntity:
    """Something that gets placed: an item, a stronghold marker.

    It can't hold a location. A location is a ``Place``, which answers "where can I get to from
    here"; this answers "where am I". Neither asks the other's question; they share only an identity
    header.

    Who is standing here isn't stored here either, for the same reason as ``Place``; ask
    ``EnvironmentSystem.bodies_at``.

    Placement lives on a single discriminated axis (``presence`` + ``presence_ref``);
    ``location_id`` / ``owner_id`` / ``is_destroyed`` are derived read-only views
    over it, so they can never desync from the single source of truth.
    """

    entity_id:      str
    name:           str
    entity_type:    WorldEntityType
    state:          str = DEFAULT_ENTITY_STATE
    description:    str = ""
    presence:       EntityPresence = EntityPresence.AT_LOCATION
    presence_ref:   str | None = None   # location_id (AT_LOCATION) | agent_id (HELD) | None
    is_public:      bool = True
    # The step it entered the world. Build-time seeds are 0: they exist from the start, so nothing
    # "appeared". Things made at runtime record their step so they sort ahead of older things in the
    # reachable list (see EnvironmentSystem.reachable_in_order). What just appeared is what the scene
    # is about, and registration order would push it behind every seed.
    created_step:   int = 0
    # What it carries: the words in a letter, the figures in a ledger, the inscription on a stele;
    # description only says what it looks like. Who can read it is decided by ``readable_by``, not
    # ``is_public``: seeing that it's there isn't reading what it says.
    content:        str = ""

    # -- derived read-only placement views (single source of truth = presence) --
    @property
    def location_id(self) -> str | None:
        """The location this entity sits in, or None when held/destroyed."""
        return self.presence_ref if self.presence == EntityPresence.AT_LOCATION else None

    @property
    def owner_id(self) -> str | None:
        """The body carrying this entity, or None when at-location/destroyed.

        Either tier of body can hold it: a thing can be in an Npc's hands (an errand delivery). Typing it
        as an agent would make the signature lie; the holding axis, like the location axis, doesn't
        distinguish tiers.
        """
        return self.presence_ref if self.presence == EntityPresence.HELD else None

    @property
    def is_destroyed(self) -> bool:
        return self.presence == EntityPresence.DESTROYED

    @property
    def is_takeable(self) -> bool:
        return self.entity_type.is_takeable

    def readable_by(self, viewer_id: str) -> bool:
        """Whether this person can read ``content`` right now.

        Whoever holds it can read it; anything fixed at a location (a stele, a notice) can be read by
        everyone present. Something in someone else's hands or lying on the ground has to be picked up
        first: seeing a letter from across the room isn't seeing what's written on it.
        """
        if self.owner_id is not None:
            return self.owner_id == viewer_id
        return self.location_id is not None and not self.is_takeable


@dataclass
class ActiveErrand:
    """An errand in progress: the only "plan" on an ``Npc`` that changes.

    This isn't cognition. It isn't an intent the body formed itself, but instructions someone else
    gave, carried out to the letter, plus how far along they are. Its behavior is still purely a
    function of what others told it to do.

    ``seen`` is narrative-layer text (the scene captured on arrival, which goes verbatim into the
    report Message and gets embedded), so it contains no ids and no step counts;
    ``assemble_scene_context`` guarantees this. ``done`` holds a short result line per task at the
    far end, used to report truthfully ("delivered" / "nobody home").
    """

    order: ErrandOrder
    requester_id: str                  # whoever assigned it; the report goes to them
    origin_id: str                     # where the errand was picked up; return here when done
    # The beat the errand was accepted. It doesn't set off on that beat: if it were already halfway
    # across town the instant the order was given, the distance would be free. So taking the order
    # uses a beat and travel starts on the next (see ``NpcRunner._advance_one``).
    #
    # No default on purpose: every errand is given on some beat. Allowing it to be omitted allows an
    # errand nobody remembers dispatching, which would silently swallow the beat it was accepted on.
    assigned_step: int
    outbound: bool = True              # True = outbound; False = returning
    # Seconds of walking left on the current edge. An edge can outlast a beat's walking budget, so
    # unfinished progress has to be stored; otherwise every edge counts as one beat and the body
    # walks on different physics from an agent on the same route (see NpcRunner's NPC_PACE).
    # 0 = standing at a location, not partway along.
    leg_remaining: int = 0
    seen: str = ""
    done: tuple[str, ...] = ()


class BodyKind(str, Enum):
    """Which kind a body in the world is: a positive test, not a complement.

    "Has cognition" must be written ``kind is AGENT``, never "not an Npc". The two are equivalent
    today because there are only two kinds of body, but they diverge the day a third kind arrives:
    the complement would treat the newcomer as an agent and let it flow into TALK admission,
    arbitration and relation evolution, all of which assume an ``Agent`` can be looked up behind
    every id, and which silently turn a missing one into an empty step.

    So adding a kind of body = a new value here + its own roster and presence view, with no change
    to any existing check.
    """

    AGENT = "agent"   # can act + has cognition
    NPC   = "npc"     # can act, no cognition


@dataclass
class Npc:
    """NPC: a body with no cognition, the middle of the world's three tiers.

    It takes up space, is perceived as a "who", can hold things, and can take a task and see it
    through. It doesn't perceive, remember, form needs, build relations or decide.

    Two properties; without either it falls apart:

    1. It has no cognition; it isn't a cheap agent. This class has no vitality / emotion / needs /
       relations / memory, and must not. That isn't unfinished work, it's the definition: an
       Agent's behavior is a function of who it is, this one's behavior is a function of what
       others tell it to do. Any field that makes its behavior depend on its own past turns it into
       a cheap Agent, and a world where some beings "think by different rules" splits into two
       kinds of causality (CLAUDE.md §5). Its cognition isn't simplified; it's zero.

    2. It obeys absolutely. What an agent does to it is assigning, not asking, requesting or
       begging; those words assume the other side can refuse, refusing needs will, and will needs
       cognition. An errand can only fail to go out for objective reasons (it's already running one
       for someone else, it can't move, there's no such person). Those are all rules, checked at
       ``ErrandExecutor``'s feasibility gate, never adjudicated.

    The two depend on each other: drop the second and something has to decide whether it's willing,
    which needs the inner life the first rules out.

    Both are code-layer facts, like ids and steps; the narrative layer has no "tool person" tier. To
    agents it is a person who does what he's told, doesn't decide for himself, doesn't refuse. The
    term "NPC 工具人" must never enter any prompt: it would shape ``action_description`` /
    ``inner_monologue`` / ``outcome``, which are embedded into memory and can't heal (guarded by
    ``tests/unit/test_npc_is_not_an_agent.py``).

    Position isn't stored here; it's in ``EnvironmentSystem``, the same table as agent positions. A
    body is a body.
    """

    npc_id:    str
    name:      str
    gender:    str = ""
    age:       int = 30
    description: str = ""
    condition: BodyCondition | None = None
    errand:    ActiveErrand | None = None

    @property
    def busy(self) -> bool:
        return self.errand is not None
