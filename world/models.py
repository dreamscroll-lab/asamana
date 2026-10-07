"""Typed models for world construction and initialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from agent.need import NeedState, NeedType, build_innate_needs
from agent.personality import SECRET_LABEL, EmotionState, EmotionType, SoulLayer, parse_emotion_type
from agent.relation import NEUTRAL_AFFECTION, NEUTRAL_TRUST
from core.interfaces.action import ErrandOrder
from core.interfaces.condition import BodyCondition
from core.interfaces.directory import WorldDirectory
from core.interfaces.snapshot import WorldSnapshot
from core.interfaces.world_config import WorldConfig
from core.text import slugify
from engine.clock import WorldTimeConfig

if TYPE_CHECKING:
    from agent.agent import Agent
    from engine.environment import EnvironmentSystem


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
    state:          str = "intact"
    description:    str = ""
    presence:       EntityPresence = EntityPresence.AT_LOCATION
    presence_ref:   str | None = None   # location_id (AT_LOCATION) | agent_id (HELD) | None
    is_takeable:    bool = False
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

    def readable_by(self, viewer_id: str) -> bool:
        """Whether this person can read ``content`` right now.

        Whoever holds it can read it; anything fixed at a location (a stele, a notice) can be read by
        everyone present. Something in someone else's hands or lying on the ground has to be picked up
        first: seeing a letter from across the room isn't seeing what's written on it.
        """
        if self.owner_id is not None:
            return self.owner_id == viewer_id
        return self.location_id is not None and not self.is_takeable


@dataclass(frozen=True)
class WorldEntitySeed:
    """A narrative-specific world object generated by ThemeAnalyzer.

    Instantiated by WorldInitializer into a DYNAMIC WorldEntity placed
    in the environment before step 0.
    """

    name:          str
    entity_type:   str            # WorldEntityType string value
    description:   str = ""
    initial_state: str = "intact"
    location_name: str = ""       # resolved to location_id by WorldInitializer
    content:       str = ""

    @property
    def entity_id(self) -> str:
        """The id in the world, derived from name rather than a separate field (like is_takeable below).

        When two seeds map to the same id, the later one silently replaces the earlier
        (register_entity assigns unconditionally): the world is short one catalyst while the log says
        both were built. So "which seeds are the same thing" must be decided on this derived value, not
        on name: ``Letter`` and ``letter`` differ as names but share a slug. Dedup is in
        _distinct_entity_seeds.
        """
        return f"seed_{slugify(self.name, fallback_prefix='seed')}"

    @property
    def is_takeable(self) -> bool:
        """Whether someone can take it (change hands): derived from entity_type, not a separate field.

        Don't make it a field the build LLM fills: asking twice lets the answers contradict (models copy
        the schema's literal `false` and produce item + is_takeable=false), and an always-false
        ``is_takeable`` means the PHYSICAL judge prompt never offers ``seize``. To make something
        untakeable, classify it as a landmark.

        The rule lives on ``WorldEntityType``. An unrecognized category string falls back to "can't be
        taken", matching how ``spawn_entity`` handles unknown categories.
        """
        try:
            return WorldEntityType(self.entity_type).is_takeable
        except ValueError:
            return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "initial_state": self.initial_state,
            "location_name": self.location_name,
            "is_takeable": self.is_takeable,
            "content": self.content,
        }


@dataclass(frozen=True)
class NpcSeed:
    """A body without a mind, generated by ThemeAnalyzer alongside the cast.

    Instantiated by WorldInitializer through ``EnvironmentSystem.spawn_npc`` before step 0.
    Sibling of ``WorldEntitySeed`` in every structural way — theme-authored, placed by name,
    rebuilt from the manifest on restore.

    All four identity fields are required and none of them is decoration:
    ``description`` is a SHORT line on who this one is and what they are good and bad at —
    written against nothing in particular, which is the point: whatever an adjudicator is
    asked at runtime, it reads this same line. Tying it to one act's verbs would freeze every
    built world against the verb set of the day it was built. Not a spec sheet either: a judge
    reasons fine from "strong" and has no scale to compare "can lift 30kg" against.
    ``age`` feeds that judgement AND picks a body sheet, ``gender`` is forced into the open by
    Chinese narration (pronouns, forms of address) AND picks a body sheet, ``name`` is the
    narrative referent.
    """

    name:          str
    gender:        str = ""
    age:           int = 30
    description:   str = ""       # What kind of person this is, what they do, their skills or failings; the main basis for admission in adjudication
    location_name: str = ""       # resolved to location_id by WorldInitializer

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "gender": self.gender,
            "age": self.age,
            "description": self.description,
            "location_name": self.location_name,
        }


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

    order: "ErrandOrder"
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
    """NPC: an actor with a body and no cognition, the middle of the world's three tiers.

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
    condition: "BodyCondition | None" = None
    errand:    "ActiveErrand | None" = None

    @property
    def busy(self) -> bool:
        return self.errand is not None


class AgentTier(str, Enum):
    """World-building complexity tier for an agent."""

    MAIN = "main"
    BACKGROUND = "background"


@dataclass(frozen=True)
class ThemeFigure:
    """A key figure extracted from a theme analysis."""

    name: str
    role: str = ""
    importance: str = AgentTier.BACKGROUND.value
    brief: str = ""
    age: int = 30
    gender: str = ""
    secret: str = ""

    @property
    def tier(self) -> AgentTier:
        return AgentTier.MAIN if self.importance == AgentTier.MAIN.value else AgentTier.BACKGROUND

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "importance": self.importance,
            "brief": self.brief,
            "age": self.age,
            "gender": self.gender,
            "secret": self.secret,
        }

    def as_prompt(self) -> str:
        """Single-line self-describing render for prompt injection; Chinese field names make each field's meaning clear without a legend."""
        importance_cn = "叙事主角" if self.tier is AgentTier.MAIN else "背景角色"
        secret = f"、{SECRET_LABEL}：{self.secret}" if self.secret else ""
        return (
            f"名字：{self.name}、角色定位：{self.role}{secret}、"
            f"当前处境与动机：{self.brief}、年龄：{self.age}、"
            f"性别：{self.gender}、重要性：{importance_cn}"
        )


@dataclass(frozen=True)
class RelationSeed:
    """Initial relationship state between two named figures.

    Bilateral by construction: a single seed describes both directions of one
    relationship. The forward fields (trust/affection/labels) describe
    `source → target`; the `reverse_*` fields describe `target → source` and
    may diverge — e.g. asymmetric kinship (``labels=["父子:儿子"]`` vs
    ``reverse_labels=["父子:父亲"]``) or asymmetric sentiment (one side trusts
    more than the other). When a `reverse_*` field is empty / unset, the
    corresponding forward value is used as the symmetric default.
    """

    source_name: str
    target_name: str
    trust: float = NEUTRAL_TRUST
    affection: float = NEUTRAL_AFFECTION
    labels: list[str] = field(default_factory=list)
    reverse_trust: float | None = None
    reverse_affection: float | None = None
    reverse_labels: list[str] = field(default_factory=list)

    def effective_reverse_trust(self) -> float:
        return self.trust if self.reverse_trust is None else self.reverse_trust

    def effective_reverse_affection(self) -> float:
        return self.affection if self.reverse_affection is None else self.reverse_affection

    def effective_reverse_labels(self) -> list[str]:
        return list(self.reverse_labels) if self.reverse_labels else list(self.labels)

    def as_dict(self) -> dict[str, Any]:
        return {
            "from": self.source_name,
            "to": self.target_name,
            "trust": self.trust,
            "affection": self.affection,
            "labels": list(self.labels),
            "reverse_trust": self.reverse_trust,
            "reverse_affection": self.reverse_affection,
            "reverse_labels": list(self.reverse_labels),
        }


@dataclass(frozen=True)
class HistoricalEventSeed:
    """A world event that happened before runtime step 0.

    ``importance`` is a 0.0-1.0 score aligned with the memory subsystem's
    importance scale (see ``MEMORY_IMPORTANCE_SCORE_DEFINITION`` in
    ``core/prompts.py`` and ``agent.memory_types.importance_level`` for the
    threshold anchors).
    """

    event: str
    step_offset: int = -1
    related_figures: list[str] = field(default_factory=list)
    importance: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "step_offset": self.step_offset,
            "related_figures": list(self.related_figures),
            "importance": self.importance,
        }


@dataclass(frozen=True)
class LocationSeed:
    """A named narrative location from theme analysis.

    Used by AgentGenerator to tell the LLM which locations are available
    for initial agent placement. Separate from WorldEntitySeed.
    """

    name: str
    description: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
        }


@dataclass(frozen=True)
class FigureCastRole:
    """Narrative role assignment for one figure in the cast."""

    name: str
    narrative_role: str = ""
    arc_summary: str = ""
    key_relationships: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "narrative_role": self.narrative_role,
            "arc_summary": self.arc_summary,
            "key_relationships": list(self.key_relationships),
        }


@dataclass(frozen=True)
class CastDesign:
    """Cross-agent narrative consistency layer produced by one shared LLM call."""

    roles: list[FigureCastRole] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"roles": [r.as_dict() for r in self.roles]}

    def role_for(self, name: str) -> FigureCastRole | None:
        return next((r for r in self.roles if r.name == name), None)


@dataclass
class ThemeAnalysis:
    """Structured world knowledge derived from a theme input.

    Holds three orthogonal build-time narrative invariants, distinguished by
    temporal scope. All three are theme-agnostic *slots* — content is produced
    by the LLM per theme, but the slot shapes do not encode any narrative
    theory (e.g. they do not assume opposition / conflict).

    - ``core_tension`` (structural / standing): the load-bearing unresolved
      pull that *drives* this story — what makes the current configuration
      unstable, giving characters reasons to act. Answers "what drives", not
      "what it is about" (the latter is ``narrative_theme``). Predates step 0.
      May take the form of opposition, longing, ambiguity, irreconcilability,
      becoming, inevitability, information asymmetry, or combinations thereof.
      Shapes agent identity (background, traits, values, hard_constraints,
      life_goal).
    - ``narrative_theme`` (thematic / pervasive): one plain sentence summarizing
      what kind of story this is — what it is *about*. Shapes the meaning and
      texture of generation (value orientation, trait color, emotion register,
      entity symbolism); does not enumerate plot routes or set dynamics.
    - ``narrative_pitch`` (situational / step-0 only): one sentence describing
      the situational focus right now — what is unsettled / awaited / unfolding
      at step 0. Must describe the present, not predict the future. Shapes
      ``initial_emotion`` only. Step-0 short-term goals are NOT seeded from the
      global pitch directly — they emerge per-agent from identity, that agent's
      own situation (location / relations / history), and that emotion, so the
      authorial pitch is not broadcast uniformly (preserves information
      asymmetry).

    All three are hard anchors during build (injected into every parallel LLM
    call to keep generation coordinated) and persisted into the step-0
    snapshot manifest. They are deliberately **not** consumed by runtime
    cognition prompts — runtime emergence must be free of these anchors so
    agent decisions are driven only by personality, need, perception, and
    memory.
    """

    theme_input: str
    world_name: str
    era_description: str
    core_tension: str
    narrative_theme: str
    narrative_pitch: str = ""
    world_time_config: dict[str, Any] = field(default_factory=dict)
    key_figures: list[ThemeFigure] = field(default_factory=list)
    initial_relations: list[RelationSeed] = field(default_factory=list)
    historical_events: list[HistoricalEventSeed] = field(default_factory=list)
    key_locations: list[LocationSeed] = field(default_factory=list)
    world_entity_seeds: list[WorldEntitySeed] = field(default_factory=list)
    npc_seeds: list[NpcSeed] = field(default_factory=list)
    source_material: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "theme_input": self.theme_input,
            "world_name": self.world_name,
            "era_description": self.era_description,
            "core_tension": self.core_tension,
            "narrative_theme": self.narrative_theme,
            "narrative_pitch": self.narrative_pitch,
            "world_time_config": dict(self.world_time_config),
            "key_figures": [figure.as_dict() for figure in self.key_figures],
            "initial_relations": [relation.as_dict() for relation in self.initial_relations],
            "historical_events": [event.as_dict() for event in self.historical_events],
            "key_locations": [location.as_dict() for location in self.key_locations],
            "world_entity_seeds": [seed.as_dict() for seed in self.world_entity_seeds],
            "npc_seeds": [seed.as_dict() for seed in self.npc_seeds],
            "source_material": self.source_material,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThemeAnalysis:
        return cls(
            theme_input=data["theme_input"],
            world_name=data["world_name"],
            era_description=data["era_description"],
            core_tension=data["core_tension"],
            narrative_theme=data["narrative_theme"],
            narrative_pitch=data["narrative_pitch"],
            world_time_config=dict(data.get("world_time_config") or {}),
            key_figures=[
                ThemeFigure(**f) for f in data.get("key_figures", [])
            ],
            initial_relations=[
                RelationSeed(
                    source_name=r["from"], target_name=r["to"],
                    trust=r.get("trust", NEUTRAL_TRUST), affection=r.get("affection", NEUTRAL_AFFECTION),
                    labels=list(r.get("labels", []) or []),
                    reverse_trust=r.get("reverse_trust"),
                    reverse_affection=r.get("reverse_affection"),
                    reverse_labels=list(r.get("reverse_labels", []) or []),
                )
                for r in data.get("initial_relations", [])
            ],
            historical_events=[
                HistoricalEventSeed(**e) for e in data.get("historical_events", [])
            ],
            key_locations=[
                LocationSeed(**loc) for loc in data.get("key_locations", [])
            ],
            world_entity_seeds=[
                WorldEntitySeed(
                    name=s["name"],
                    entity_type=s.get("entity_type", "item"),
                    description=s.get("description", ""),
                    initial_state=s.get("initial_state", "intact"),
                    location_name=s.get("location_name", ""),
                    content=s.get("content", ""),
                )
                for s in data.get("world_entity_seeds", [])
                if isinstance(s, dict) and s.get("name")
            ],
            npc_seeds=[
                NpcSeed(
                    name=s["name"],
                    gender=s.get("gender", ""),
                    age=s.get("age", 30),
                    description=s.get("description", ""),
                    location_name=s.get("location_name", ""),
                )
                for s in data.get("npc_seeds", [])
                if isinstance(s, dict) and s.get("name")
            ],
            source_material=data.get("source_material", ""),
        )


@dataclass
class AgentDefinition:
    """WorldBuilder output consumed by WorldInitializer."""

    agent_id: str
    name: str
    tier: AgentTier
    soul: SoulLayer
    initial_location: str
    initial_emotion: EmotionState
    # Static need endowment lives in soul.innate_needs (immutable identity); the long-term goal seed is a top-level field (it evolves at runtime).
    long_term_goals: list[str] = field(default_factory=list)
    initial_relations: list[RelationSeed] = field(default_factory=list)
    historical_memories: list[HistoricalEventSeed] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_main_character(self) -> bool:
        return self.tier == AgentTier.MAIN

    @property
    def soul_data(self) -> dict[str, Any]:
        return {
            "name": self.soul.name,
            "role": self.soul.role,
            "agent_id": self.soul.agent_id,
            "age": self.soul.age,
            "gender": self.soul.gender,
            "core_traits": list(self.soul.core_traits),
            "core_values": list(self.soul.core_values),
            "self_image": self.soul.self_image,
            "background": self.soul.background,
            "appearance": self.soul.appearance,
            "color": self.soul.color,
            "life_goal": self.soul.life_goal,
            "secret": self.soul.secret,
            "hard_constraints": list(self.soul.hard_constraints),
        }

    @property
    def initial_needs(self) -> list[dict[str, Any]]:
        return [_need_state_to_dict(n) for n in self.soul.innate_needs if not n.is_hidden]

    @property
    def hidden_needs(self) -> list[dict[str, Any]]:
        return [_need_state_to_dict(n) for n in self.soul.innate_needs if n.is_hidden]

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "tier": self.tier.value,
            "soul_data": self.soul_data,
            "initial_location": self.initial_location,
            "initial_emotion": {
                "primary": self.initial_emotion.primary.value,
                "intensity": self.initial_emotion.intensity,
                "valence": self.initial_emotion.valence,
                "triggered_by": self.initial_emotion.triggered_by,
            },
            "long_term_goals": self.long_term_goals,
            "initial_needs": self.initial_needs,
            "hidden_needs": self.hidden_needs,
            "initial_relations": [relation.as_dict() for relation in self.initial_relations],
            "historical_memories": [event.as_dict() for event in self.historical_memories],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentDefinition:
        sd = data["soul_data"]
        # Rebuild the static endowment from persisted initial_needs/hidden_needs and attach it to soul.innate_needs
        # (build_innate_needs already runs the baseline ensure, filling in the universal Maslow needs).
        active = [
            NeedState(
                type=NeedType(n["type"]),
                label=n.get("label", ""),
                intensity=float(n.get("intensity", 0.5)),
                weight=float(n.get("weight", 1.0)),
            )
            for n in data.get("initial_needs", [])
        ]
        hidden = [
            NeedState(
                type=NeedType(n["type"]),
                label=n.get("label", ""),
                intensity=float(n.get("intensity", 0.4)),
                weight=float(n.get("weight", 1.0)),
                is_hidden=True,
            )
            for n in data.get("hidden_needs", [])
        ]
        soul = SoulLayer(
            name=sd["name"],
            role=sd.get("role", ""),
            agent_id=sd.get("agent_id", data["agent_id"]),
            age=sd.get("age", 30),
            gender=sd.get("gender", ""),
            core_traits=sd.get("core_traits", []),
            core_values=sd.get("core_values", []),
            self_image=sd.get("self_image", ""),
            background=sd.get("background", ""),
            appearance=sd.get("appearance", ""),
            color=sd.get("color", ""),
            life_goal=sd.get("life_goal"),
            secret=sd.get("secret", ""),
            hard_constraints=sd.get("hard_constraints", []),
            innate_needs=build_innate_needs(active, hidden),
        )
        ie = data.get("initial_emotion") or {}
        initial_emotion = EmotionState(
            primary=parse_emotion_type(ie.get("primary", EmotionType.NEUTRAL.value)),
            intensity=float(ie.get("intensity", 0.3)),
            valence=float(ie.get("valence", 0.0)),
            triggered_by=ie.get("triggered_by"),
        )
        return cls(
            agent_id=data["agent_id"],
            name=data["name"],
            tier=AgentTier(data["tier"]),
            soul=soul,
            initial_location=data["initial_location"],
            initial_emotion=initial_emotion,
            long_term_goals=list(data.get("long_term_goals", [])),
            initial_relations=[
                RelationSeed(
                    source_name=r["from"], target_name=r["to"],
                    trust=r.get("trust", NEUTRAL_TRUST), affection=r.get("affection", NEUTRAL_AFFECTION),
                    labels=list(r.get("labels", []) or []),
                    reverse_trust=r.get("reverse_trust"),
                    reverse_affection=r.get("reverse_affection"),
                    reverse_labels=list(r.get("reverse_labels", []) or []),
                )
                for r in data.get("initial_relations", [])
            ],
            historical_memories=[
                HistoricalEventSeed(**e) for e in data.get("historical_memories", [])
            ],
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class World:
    """Initialized world state ready for runtime orchestration."""

    world_id: str
    theme: str
    analysis: ThemeAnalysis
    agent_definitions: list[AgentDefinition]
    agents: dict[str, Agent]
    world_config: WorldConfig
    clock_config: WorldTimeConfig
    environment: EnvironmentSystem
    step_zero_snapshot: WorldSnapshot
    directory: WorldDirectory
    current_step: int = 0  # step the world resumes at (latest snapshot); 0 for a fresh build

    def agent_list(self) -> list[Agent]:
        return list(self.agents.values())


def _need_state_to_dict(n: NeedState) -> dict[str, Any]:
    # The key "intensity" carries the build-time seed intensity; the world-definition JSON contract is stable, so don't rename it.
    return {
        "type": n.type.value,
        "label": n.label,
        "intensity": n.intensity,
        "weight": n.weight,
        "is_hidden": n.is_hidden,
    }
