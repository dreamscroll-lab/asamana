"""A built world's setup: what the world starts from, kept in a shape made for revising.

The build produces it (``WorldSetup.from_build``) and persists it in its own file
(``SnapshotProvider.save_world_setup``). Until the world is confirmed that file is the only copy:
the review shows it and every revision rewrites it, and nothing is derived from it yet. Confirming
materializes it (``WorldInitializer.initialize``: agents, relations, backstory memories, initial
states, the step-0 snapshot) and freezes it into step 0 (``WorldSnapshot.setup``), after which the
file is deleted; restore and reset read the frozen copy. So at any moment exactly one copy is
authoritative (``WorldSetup.load``).

People are referred to by id throughout, so nothing joins on a name. A character's name is still
set once, when it is added: an event's text names people as it was written.

Every setup the build or a revision produces passes ``WorldSetup.validate``. Its rules are
structural (references resolve, values lie in their ranges, one relation per pair...) and never
about a theme.
"""

from __future__ import annotations

from collections.abc import Sequence, Set as AbstractSet
from dataclasses import dataclass, fields, replace
from typing import Any

from agent.memory_types import IMPORTANCE_RANGE
from agent.need import NEED_INTENSITY_RANGE, NEED_WEIGHT_RANGE, NeedState, build_innate_needs
from agent.personality import GENDERS, EmotionState, SoulLayer
from core.interfaces.snapshot import SnapshotProvider
from core.logging import get_logger
from core.prompts import (
    AFFECTION_RANGE, EMOTION_INTENSITY_RANGE, EMOTION_VALENCE_RANGE, TRUST_RANGE,
    canonical_relation_label, relation_labels,
)
from core.validate import require_at_least, require_in_range, require_member, require_text
from engine.clock import STEP_SECONDS_RANGE, resolve_step_seconds
from world.identity_color import free_identity_color
from world.models import (
    AgentDefinition, AgentTier, NpcSeed, ThemeAnalysis, WorldEntitySeed, name_key,
    unique_agent_id, unique_entity_id, unique_npc_id,
)

logger = get_logger(__name__)



class SetupRejected(Exception):
    """A revision the setup refuses. Not a ``ValueError``, so callers can tell a bad revision
    apart from a world whose setup is closed to revision."""


@dataclass(frozen=True)
class WorldLore:
    """What the world is, for every character alike: the build's narrative anchors (also read by
    the runtime's world pressure) and the clock the story starts on."""

    world_name: str
    era_description: str
    core_tension: str
    narrative_theme: str
    narrative_pitch: str
    # The calendar anchor (era, start date and hour) and ``seconds_per_step``.
    world_time_config: dict[str, Any]

    @property
    def seconds_per_step(self) -> int:
        return resolve_step_seconds(self.world_time_config.get("seconds_per_step"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "world_name": self.world_name,
            "era_description": self.era_description,
            "core_tension": self.core_tension,
            "narrative_theme": self.narrative_theme,
            "narrative_pitch": self.narrative_pitch,
            "world_time_config": dict(self.world_time_config),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorldLore:
        return cls(
            world_name=str(data["world_name"]),
            era_description=str(data.get("era_description", "")),
            core_tension=str(data.get("core_tension", "")),
            narrative_theme=str(data.get("narrative_theme", "")),
            narrative_pitch=str(data.get("narrative_pitch", "")),
            world_time_config=dict(data.get("world_time_config") or {}),
        )


@dataclass(frozen=True)
class SetupRelation:
    """One relation the world starts with: both directions of one pair. ``trust``/``affection``/
    ``labels`` are ``source_id``'s view of ``target_id``, the ``reverse_*`` fields the other way.
    Labels say who the other is to the holder, as ``AgentRelation.labels`` do."""

    source_id: str
    target_id: str
    trust: float
    affection: float
    labels: tuple[str, ...]
    reverse_trust: float
    reverse_affection: float
    reverse_labels: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id, "target_id": self.target_id,
            "trust": self.trust, "affection": self.affection, "labels": list(self.labels),
            "reverse_trust": self.reverse_trust, "reverse_affection": self.reverse_affection,
            "reverse_labels": list(self.reverse_labels),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SetupRelation:
        return cls(
            source_id=str(data["source_id"]), target_id=str(data["target_id"]),
            trust=float(data["trust"]), affection=float(data["affection"]),
            labels=tuple(data.get("labels") or ()),
            reverse_trust=float(data["reverse_trust"]), reverse_affection=float(data["reverse_affection"]),
            reverse_labels=tuple(data.get("reverse_labels") or ()),
        )


@dataclass(frozen=True)
class SetupHistoricalEvent:
    """One historical event: something that happened before the story starts. Its memory is
    what the people in it start with (the backstory memories). ``hours_before_start`` is a
    narrative duration, never a
    step count: the step length may still change, and the initializer converts it only when it
    writes the memory. ``figure_ids`` took part, and so remember it."""

    event: str
    hours_before_start: int
    figure_ids: tuple[str, ...]
    importance: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": self.event, "hours_before_start": self.hours_before_start,
            "figure_ids": list(self.figure_ids), "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SetupHistoricalEvent:
        return cls(
            event=str(data["event"]), hours_before_start=int(data["hours_before_start"]),
            figure_ids=tuple(data.get("figure_ids") or ()), importance=float(data["importance"]),
        )


@dataclass(frozen=True)
class CharacterRevision:
    """The fields of a character a revision may set; ``None`` leaves a field as it is.

    ``needs`` replaces the character's innate needs whole; a need type it leaves out gets the
    generic baseline (``build_innate_needs``).
    """

    role: str | None = None
    age: int | None = None
    gender: str | None = None
    appearance: str | None = None
    background: str | None = None
    self_image: str | None = None
    life_goal: str | None = None
    secret: str | None = None
    core_traits: tuple[str, ...] | None = None
    core_values: tuple[str, ...] | None = None
    hard_constraints: tuple[str, ...] | None = None
    needs: tuple[NeedState, ...] | None = None
    is_main_character: bool | None = None
    initial_location: str | None = None
    initial_emotion: EmotionState | None = None
    long_term_goals: tuple[str, ...] | None = None

    def changes(self) -> dict[str, Any]:
        return {f.name: value for f in fields(self) if (value := getattr(self, f.name)) is not None}


# The revision fields that live on the soul; the rest are the definition's.
_SOUL_FIELDS = frozenset({
    "role", "age", "gender", "appearance", "background", "self_image", "life_goal", "secret",
    "core_traits", "core_values", "hard_constraints",
})


def _revised(definition: AgentDefinition, revision: CharacterRevision) -> AgentDefinition:
    # A built character may have no gender (the build never guesses one); a revision sets one.
    if revision.gender is not None:
        require_member(revision.gender, GENDERS, "gender")
    soul_changes = {k: v for k, v in revision.changes().items() if k in _SOUL_FIELDS}
    if soul_changes.get("life_goal") == "":
        soul_changes["life_goal"] = None
    if revision.needs is not None:
        soul_changes["innate_needs"] = build_innate_needs(
            [n for n in revision.needs if not n.is_hidden], [n for n in revision.needs if n.is_hidden],
        )
    changes: dict[str, Any] = {"soul": replace(definition.soul, **soul_changes)}
    if revision.is_main_character is not None:
        changes["tier"] = AgentTier.MAIN if revision.is_main_character else AgentTier.BACKGROUND
    if revision.initial_location is not None:
        changes["initial_location"] = revision.initial_location
    if revision.initial_emotion is not None:
        changes["initial_emotion"] = revision.initial_emotion
    if revision.long_term_goals is not None:
        changes["long_term_goals"] = list(revision.long_term_goals)
    return replace(definition, **changes)


@dataclass(frozen=True)
class WorldSetup:
    theme: str
    lore: WorldLore
    characters: tuple[AgentDefinition, ...]
    relations: tuple[SetupRelation, ...]
    historical_events: tuple[SetupHistoricalEvent, ...]
    # Built with the world and not revisable: what stands in its places, and who keeps them.
    entity_seeds: tuple[WorldEntitySeed, ...] = ()
    npc_seeds: tuple[NpcSeed, ...] = ()

    # ----- the build ---------------------------------------------------------

    @classmethod
    def from_build(
        cls, theme: str, analysis: ThemeAnalysis, definitions: Sequence[AgentDefinition],
    ) -> WorldSetup:
        """The setup a build produced. The build speaks of people by name; here they become ids.

        Repairs only what is mechanical: a reference to someone not in the cast, a second relation
        for a pair (the first stands), a label out of form (put in canonical form, or dropped when
        it has none), an event nobody is left in, a need listed twice (the first stands). Names arrive distinct (``ThemeAnalyzer`` decided that on the
        LLM's own names); here every NPC and thing gets its id.
        Anything else is the caller's ``validate`` to refuse.
        """
        ids = {d.soul.name: d.agent_id for d in definitions}
        # Every NPC and thing gets its id here, once, from a name the analysis already made
        # distinct: the world uses it from step 0 on, whatever later happens to the name.
        used_ids = {d.agent_id for d in definitions}
        relations: list[SetupRelation] = []
        pairs: set[frozenset[str]] = set()
        for seed in analysis.initial_relations:
            source, target = ids.get(seed.source_name), ids.get(seed.target_name)
            pair = frozenset((source or "", target or ""))
            if source is None or target is None or len(pair) == 1 or pair in pairs:
                logger.debug(
                    "relation_seed_dropped",
                    extra={"source_name": seed.source_name, "target_name": seed.target_name},
                )
                continue
            pairs.add(pair)
            relations.append(SetupRelation(
                source_id=source,
                target_id=target,
                trust=seed.trust,
                affection=seed.affection,
                labels=tuple(relation_labels(seed.labels)),
                reverse_trust=seed.effective_reverse_trust(),
                reverse_affection=seed.effective_reverse_affection(),
                reverse_labels=tuple(relation_labels(seed.effective_reverse_labels())),
            ))
        events: list[SetupHistoricalEvent] = []
        for seed in analysis.historical_events:
            figure_ids = tuple(dict.fromkeys(ids[n] for n in seed.related_figures if n in ids))
            if not figure_ids:
                logger.debug("historical_event_dropped", extra={"event": seed.event[:100]})
                continue
            events.append(SetupHistoricalEvent(
                event=seed.event, hours_before_start=seed.hours_before_start,
                figure_ids=figure_ids, importance=seed.importance,
            ))
        return cls(
            theme=theme,
            lore=WorldLore(
                world_name=analysis.world_name,
                era_description=analysis.era_description,
                core_tension=analysis.core_tension,
                narrative_theme=analysis.narrative_theme,
                narrative_pitch=analysis.narrative_pitch,
                world_time_config=dict(analysis.world_time_config),
            ),
            characters=tuple(_distinct_needs(d) for d in definitions),
            relations=tuple(relations),
            historical_events=tuple(events),
            npc_seeds=tuple(
                replace(seed, npc_id=unique_npc_id(seed.name, used_ids))
                for seed in analysis.npc_seeds
            ),
            entity_seeds=tuple(
                replace(seed, entity_id=unique_entity_id(seed.name, used_ids))
                for seed in analysis.world_entity_seeds
            ),
        )

    # ----- persistence -------------------------------------------------------

    @classmethod
    async def load(cls, snapshots: SnapshotProvider, world_id: str) -> WorldSetup | None:
        """A world's setup wherever it is authoritative: the setup file until the world is
        confirmed, frozen in step 0 after; None for a world that has neither.

        The file goes first: confirming deletes it last, so a confirmation that fails after
        writing step 0 leaves the file, still the one revisions edit, beside a stale step 0.
        A file left by a failed delete holds what step 0 froze, and a confirmed setup takes no
        revision.
        """
        stored = await snapshots.load_world_setup(world_id)
        if stored:
            return cls.from_dict(stored)
        step_zero = await snapshots.load(world_id, 0)
        return cls.from_dict(step_zero.setup) if step_zero is not None and step_zero.setup else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "theme": self.theme,
            "lore": self.lore.as_dict(),
            "characters": [d.as_dict() for d in self.characters],
            "relations": [r.as_dict() for r in self.relations],
            "historical_events": [e.as_dict() for e in self.historical_events],
            "entity_seeds": [s.as_dict() for s in self.entity_seeds],
            "npc_seeds": [s.as_dict() for s in self.npc_seeds],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorldSetup:
        return cls(
            theme=str(data.get("theme", "")),
            lore=WorldLore.from_dict(data["lore"]),
            characters=tuple(AgentDefinition.from_dict(d) for d in data["characters"]),
            relations=tuple(SetupRelation.from_dict(r) for r in data.get("relations", [])),
            historical_events=tuple(SetupHistoricalEvent.from_dict(e) for e in data.get("historical_events", [])),
            entity_seeds=tuple(
                WorldEntitySeed(
                    name=s["name"],
                    entity_id=s["entity_id"],
                    entity_type=s.get("entity_type", "item"),
                    description=s.get("description", ""),
                    initial_state=s.get("initial_state", "intact"),
                    location=s.get("location", ""),
                    content=s.get("content", ""),
                )
                for s in data.get("entity_seeds", [])
            ),
            npc_seeds=tuple(
                NpcSeed(
                    name=s["name"],
                    npc_id=s["npc_id"],
                    gender=s.get("gender", ""),
                    age=s.get("age", 30),
                    description=s.get("description", ""),
                    location=s.get("location", ""),
                )
                for s in data.get("npc_seeds", [])
            ),
        )

    # ----- the rules ---------------------------------------------------------

    def validate(self, places: AbstractSet[str]) -> None:
        """Refuse (``ValueError``) a setup a world can't start from. ``places``: the location
        ids of the world's map."""
        require_text(self.lore.world_name, "world_name")
        # The authored value, not ``lore.seconds_per_step``: that falls back to the default.
        seconds = self.lore.world_time_config.get("seconds_per_step")
        if isinstance(seconds, bool) or not isinstance(seconds, int):
            raise ValueError("seconds_per_step must be an integer")
        require_in_range(seconds, STEP_SECONDS_RANGE, "seconds_per_step")
        if not self.characters:
            raise ValueError("A world needs at least one character")
        # The scheduler and the narrative lean on main characters; a cast without one has no lead.
        if not any(d.is_main_character for d in self.characters):
            raise ValueError("A world needs at least one main character")
        ids = [d.agent_id for d in self.characters]
        # One namespace: a reference to an id must not have to guess what kind of thing it is.
        every_id = [*ids, *(n.npc_id for n in self.npc_seeds), *(e.entity_id for e in self.entity_seeds)]
        for one_id in every_id:
            require_text(one_id, "id")
        if len(set(every_id)) != len(every_id):
            raise ValueError("Two characters, NPCs or things share an id")
        # Narrative text and every prompt tell people and things apart by name, and an NPC's or a
        # thing's id comes from its name: one name for two of them is one referent for two.
        seen: set[str] = set()
        for name in self._names():
            if name_key(name) in seen:
                raise ValueError(f"The name {name!r} is used twice among characters, NPCs and things")
            seen.add(name_key(name))
        for definition in self.characters:
            _validate_character(definition, places)
        for seed in (*self.npc_seeds, *self.entity_seeds):
            require_member(seed.location, places, f"location of {seed.name!r}")

        cast = set(ids)
        pairs: set[frozenset[str]] = set()
        for relation in self.relations:
            require_member(relation.source_id, cast, "relation source")
            require_member(relation.target_id, cast, "relation target")
            pair = frozenset((relation.source_id, relation.target_id))
            if len(pair) == 1:
                raise ValueError("A relation needs two different characters")
            # One relation holds both directions; a second for the pair would contradict it.
            if pair in pairs:
                raise ValueError("A pair of characters has more than one relation")
            pairs.add(pair)
            require_in_range(relation.trust, TRUST_RANGE, "trust")
            require_in_range(relation.affection, AFFECTION_RANGE, "affection")
            require_in_range(relation.reverse_trust, TRUST_RANGE, "reverse_trust")
            require_in_range(relation.reverse_affection, AFFECTION_RANGE, "reverse_affection")
            for labels in (relation.labels, relation.reverse_labels):
                for label in labels:
                    if canonical_relation_label(label) != label:
                        raise ValueError(f"Relation label {label!r} is not in canonical form")
                if len(set(labels)) != len(labels):
                    raise ValueError("A relation lists a label twice")

        for event in self.historical_events:
            require_text(event.event, "event")
            require_at_least(event.hours_before_start, 1, "hours_before_start")
            require_in_range(event.importance, IMPORTANCE_RANGE, "importance")
            # An event nobody in the cast took part in is remembered by nobody.
            if not event.figure_ids:
                raise ValueError("A historical event needs at least one character who took part")
            if len(set(event.figure_ids)) != len(event.figure_ids):
                raise ValueError("A historical event lists a character twice")
            for agent_id in event.figure_ids:
                require_member(agent_id, cast, "related figure")

    # ----- revisions ---------------------------------------------------------
    #
    # Each returns a new setup that has passed ``validate``; ``places`` is the map's location ids.

    def character(self, agent_id: str) -> AgentDefinition:
        for definition in self.characters:
            if definition.agent_id == agent_id:
                return definition
        raise ValueError(f"No character {agent_id!r} in this world")

    def with_character(
        self, agent_id: str, revision: CharacterRevision, *, places: AbstractSet[str],
    ) -> WorldSetup:
        revised = _revised(self.character(agent_id), revision)
        characters = tuple(revised if d.agent_id == agent_id else d for d in self.characters)
        return replace(self, characters=characters)._validated(places)

    def with_character_added(
        self, name: str, revision: CharacterRevision, *, places: AbstractSet[str], cast_limit: int,
    ) -> tuple[WorldSetup, str]:
        """Add a character written by hand; returns the setup and the new character's id.

        Name, role, age, gender, tier and initial location are required. Its id comes from its
        name, its identity color is one nobody else wears, and it starts with no relations and
        no part in any historical event.
        """
        require_text(name, "name")
        if len(self.characters) >= cast_limit:
            raise ValueError(f"A world has at most {cast_limit} characters")
        for field_name in ("role", "age", "gender", "is_main_character", "initial_location"):
            if getattr(revision, field_name) is None:
                raise ValueError(f"A new character needs {field_name}")
        require_text(revision.role or "", "role")
        agent_id = unique_agent_id(name, {
            *(d.agent_id for d in self.characters),
            *(n.npc_id for n in self.npc_seeds),
            *(e.entity_id for e in self.entity_seeds),
        })
        blank = AgentDefinition(
            agent_id=agent_id,
            tier=AgentTier.BACKGROUND,
            soul=SoulLayer(
                name=name,
                agent_id=agent_id,
                color=free_identity_color(d.soul.color for d in self.characters),
                innate_needs=build_innate_needs([], []),
            ),
            initial_location="",
            initial_emotion=EmotionState(),
            metadata={"source": "revision"},
        )
        setup = replace(self, characters=(*self.characters, _revised(blank, revision)))
        return setup._validated(places), agent_id

    def without_character(self, agent_id: str, *, places: AbstractSet[str]) -> WorldSetup:
        """Remove a character with its relations; historical events keep their text but no longer
        list it, and an event nobody is left in goes too."""
        self.character(agent_id)
        events = (
            replace(event, figure_ids=tuple(f for f in event.figure_ids if f != agent_id))
            for event in self.historical_events
        )
        return replace(
            self,
            characters=tuple(d for d in self.characters if d.agent_id != agent_id),
            relations=tuple(r for r in self.relations if agent_id not in (r.source_id, r.target_id)),
            historical_events=tuple(e for e in events if e.figure_ids),
        )._validated(places)

    def with_seconds_per_step(self, seconds: int, *, places: AbstractSet[str]) -> WorldSetup:
        """Set how much world time one step spans. Durations in the setup are kept in narrative
        units (``SetupHistoricalEvent.hours_before_start``), so nothing else needs rewriting."""
        time_config = {**self.lore.world_time_config, "seconds_per_step": seconds}
        return replace(self, lore=replace(self.lore, world_time_config=time_config))._validated(places)

    def with_history(self, events: Sequence[SetupHistoricalEvent], *, places: AbstractSet[str]) -> WorldSetup:
        return replace(self, historical_events=tuple(events))._validated(places)

    def with_relations(self, relations: Sequence[SetupRelation], *, places: AbstractSet[str]) -> WorldSetup:
        return replace(self, relations=tuple(relations))._validated(places)

    def _names(self) -> list[str]:
        return [
            *(d.soul.name for d in self.characters),
            *(n.name for n in self.npc_seeds),
            *(e.name for e in self.entity_seeds),
        ]

    def _validated(self, places: AbstractSet[str]) -> WorldSetup:
        self.validate(places)
        return self


def _validate_character(definition: AgentDefinition, places: AbstractSet[str]) -> None:
    soul = definition.soul
    require_text(soul.name, "name")
    # "" is a gender the build left unknown rather than guessed (see ``_revised``).
    require_member(soul.gender, ("", *GENDERS), "gender")
    require_at_least(soul.age, 0, "age")
    for name, items in (
        ("core_traits", soul.core_traits), ("core_values", soul.core_values),
        ("hard_constraints", soul.hard_constraints), ("long_term_goals", definition.long_term_goals),
    ):
        for item in items:
            require_text(item, name)
    require_member(definition.initial_location, places, "initial_location")
    emotion = definition.initial_emotion
    require_in_range(emotion.intensity, EMOTION_INTENSITY_RANGE, "emotion intensity")
    require_in_range(emotion.valence, EMOTION_VALENCE_RANGE, "emotion valence")
    types = [need.type for need in soul.innate_needs]
    if len(set(types)) != len(types):
        raise ValueError("A character has each need once")
    for need in soul.innate_needs:
        require_text(need.label, "need label")
        require_in_range(need.intensity, NEED_INTENSITY_RANGE, "need intensity")
        require_in_range(need.weight, NEED_WEIGHT_RANGE, "need weight")


def _distinct_needs(definition: AgentDefinition) -> AgentDefinition:
    seen: set[Any] = set()
    needs = []
    for need in definition.soul.innate_needs:
        if need.type not in seen:
            seen.add(need.type)
            needs.append(need)
    return replace(definition, soul=replace(definition.soul, innate_needs=tuple(needs)))
