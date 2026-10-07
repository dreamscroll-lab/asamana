"""World space state and physics query service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping

from core.interfaces.action import (
    EntitySpawn, EntityStateChange, ErrandOrder, NpcEffect,
    errand_from_dict, errand_to_dict,
)
from core.interfaces.condition import condition_from_dict, condition_to_dict
from core.interfaces.perception import (
    PerceivedNpc, PerceivedPresence,
    AmbientEvent, LocationView, ReachableLocation, SpatialPerception, VisibleEntity,
)
from core.logging import get_logger
from core.text import slugify
from engine.clock import WorldTime
from engine.space import SpaceManager
from world.models import (
    ActiveErrand,
    BodyKind,
    EntityPresence,
    Npc,
    NpcSeed,
    WorldEntity,
    WorldEntityType,
)
from core.interfaces.place import Place

logger = get_logger(__name__)

IN_TRANSIT = "__in_transit__"

# Pseudo-location for a body not in the world (never registered, or removed at death);
# IN_TRANSIT means "between two places". Onlooker checks must exclude both.
UNPLACED = "unknown"

# Ambient strength for a strong social signal (vs. the 0.2 default): it becomes a
# high-importance memory and clears background agents' perception threshold. Reuse it for
# anything of that weight (an exposed covert act, a visible injury, a restraint).
SALIENT_AMBIENT_STRENGTH = 0.7

# Recent happenings a location keeps, for covert adjudication only. Both bounds are needed:
# beats alone leave a busy location unbounded; entries alone let a few busy beats crowd out a
# long wait. The beat window is deliberately tight: one finding can't use more material, and
# more tempts the judge to dump everything in the room.
HAPPENINGS_WINDOW_STEPS = 6
HAPPENINGS_MAX_ENTRIES = 20

# How many things one place can hold: a room's floor, or one person's hands, each counted
# separately. A narrative bound, not a performance one: too many things in reach turn the
# decision list into an inventory. Per place because that list is per place; a world-wide total
# would let map size decide who can still make things.
MAX_ENTITIES_PER_PLACE = 12

# The most things one person sees at once: an overlong binding menu (``physical_entity_index``)
# offers items in the weak-attention middle that never get picked. A loose guardrail; the
# per-place cap normally bounds it already.
MAX_VISIBLE_ENTITIES = 15

# Larger than the actor's menu: a judge missing a thing rules as if it weren't in the room.
MAX_SCENE_ENTITIES = 30

# The most Npcs a world can hold (prompt space: each crowds the co-located list). Per world
# because they move. Default only; the real value comes from ``engine.max_npcs``.
DEFAULT_MAX_NPCS = 6


def _npc_payload(npc: "Npc") -> dict[str, object]:
    """Npc → JSON-native dict. Identity is stored too so restore needn't match against seed
    names, which is where ids and identities drift apart."""
    errand = npc.errand
    return {
        "name": npc.name,
        "gender": npc.gender,
        "age": npc.age,
        "description": npc.description,
        "condition": condition_to_dict(npc.condition),
        "errand": None if errand is None else {
            "order": errand_to_dict(errand.order),
            "requester_id": errand.requester_id,
            "origin_id": errand.origin_id,
            "outbound": errand.outbound,
            "assigned_step": errand.assigned_step,
            "leg_remaining": errand.leg_remaining,
            "seen": errand.seen,
            "done": list(errand.done),
        },
    }


def _ambient_payload(event: AmbientEvent) -> dict[str, object]:
    return {
        "content": event.content,
        "strength": event.strength,
        "actor_ids": list(event.actor_ids),
        "agent_actor_ids": list(event.agent_actor_ids),
    }


def _ambient_from_payload(payload: Mapping[str, object]) -> AmbientEvent:
    strength = payload.get("strength")
    return AmbientEvent(
        content=str(payload.get("content", "")),
        strength=float(strength) if isinstance(strength, (int, float)) else None,
        actor_ids=tuple(str(a) for a in payload.get("actor_ids") or ()),
        agent_actor_ids=tuple(str(a) for a in payload.get("agent_actor_ids") or ()),
    )


def _npc_from_payload(npc_id: str, payload: Mapping[str, object]) -> "Npc":
    """Inverse of ``_npc_payload``. An unreadable errand is treated as not in progress: standing
    still beats being stuck halfway forever."""
    raw_errand = payload.get("errand")
    errand = None
    if isinstance(raw_errand, Mapping):
        order = errand_from_dict(raw_errand.get("order"))
        if order is not None:
            errand = ActiveErrand(
                order=order,
                requester_id=str(raw_errand.get("requester_id", "")),
                origin_id=str(raw_errand.get("origin_id", "")),
                outbound=bool(raw_errand.get("outbound", True)),
                # -1 matches no real step, so a corrupt payload's errand continues this beat.
                assigned_step=int(raw_errand.get("assigned_step", -1) or -1),
                leg_remaining=int(raw_errand.get("leg_remaining", 0) or 0),
                seen=str(raw_errand.get("seen", "")),
                done=tuple(str(d) for d in (raw_errand.get("done") or [])),
            )
    return Npc(
        npc_id=npc_id,
        name=str(payload.get("name", "")),
        gender=str(payload.get("gender", "")),
        age=int(payload.get("age", 30) or 30),
        description=str(payload.get("description", "")),
        condition=condition_from_dict(payload.get("condition")),
        errand=errand,
    )


@dataclass(frozen=True)
class FeasibilityResult:
    """Outcome of a physical feasibility check."""

    ok: bool
    reason: str
    resolved_id: str = ""  # resolved location or entity ID returned to callers
    # For a feasible MOVE: the ordered waypoint sequence [origin, …, destination]
    # along the shortest path (endpoints included). Empty for non-MOVE / infeasible checks.
    path: tuple[str, ...] = ()
    # For a feasible MOVE: how long the whole route takes, in seconds. 0 otherwise.
    travel_seconds: int = 0


@dataclass(frozen=True)
class _Happening:
    """The full ``ActionResult.outcome`` of something that surfaced at a place; ``AmbientEvent``
    is the degraded string onlookers perceive of it. Outcomes nobody could see aren't included.
    ``actor_ids`` already have it and are excluded on read.
    """

    step: int
    content: str
    actor_ids: tuple[str, ...]


class EnvironmentSystem:
    """World space state and physics query service.

    Responsibilities:
    - Agent location tracking and space mutations
    - Physical feasibility checks (reachability, capacity, co-location)
    - Duration queries from world configuration
    - Carry-observation recording for next-step ambient perception
    - SpatialPerception construction for the cognition loop
    - Per-location recent happenings (full-fidelity, for covert adjudication only)

    Does NOT execute actions or produce ActionResult objects.

    The only perception write is record_carry_observation, which lands in next step's
    ambient_events; same-step signals must go through BroadcastChannel or MessageSystem.
    _step_annotations is only flipped in by begin_step().

    record_happening() is not perception: it keeps the full string for covert adjudication as a
    bounded per-beat history, and no spatial_for reads it.
    """

    def __init__(
        self, world_config: object | None = None, *,
        max_entities_per_place: int = MAX_ENTITIES_PER_PLACE,
        max_npcs: int = DEFAULT_MAX_NPCS,
    ) -> None:
        self.world_config = world_config
        self._max_entities_per_place = max_entities_per_place
        self._max_npcs = max_npcs
        self.space = SpaceManager()
        # Location is kind-agnostic: walking, transit and co-location work the same for both kinds.
        self._locations: Dict[str, str] = {}           # body id → location_id
        # The single source of truth for "is this an agent", written and deleted together with
        # ``_locations``. Not in snapshots: restore re-registers every body through placement.
        self._bodies: Dict[str, BodyKind] = {}         # body id → BodyKind
        # Npc data only. Ask ``_bodies`` whether something is an Npc; the complement of this
        # table would count a third kind of body as an agent.
        self._npcs: Dict[str, "Npc"] = {}
        # strength None = unmarked (PerceptionMemoryLayer applies its default).
        self._step_annotations: Dict[str, List[AmbientEvent]] = {}
        # Observer-only: one finished line per Npc per beat (``NpcRunner`` merges it). No location
        # stored: it happened where he stands now. Not ambient: errand lines would crowd its
        # 1-2 slots and push out real events.
        self._npc_outcomes: Dict[str, tuple[str, bool]] = {}
        # Observer-only: bodies teleported this beat, so the map doesn't draw it as a walk.
        self._npc_displaced: set[str] = set()
        self._carry_annotations: Dict[str, List[AmbientEvent]] = {}
        # Full-fidelity happenings for covert adjudication (see _Happening). No carry/flip: the
        # writer (runtime._carry_step_observations) already runs after the beat's adjudications.
        # Must never reach spatial_for (breaks information asymmetry), snapshot_state (leaks
        # into the director's briefing and replay) or assemble_scene_context (would show up in
        # broadcast PHYSICAL outcomes).
        self._happenings: Dict[str, List[_Happening]] = {}
        # Tombstones stay in ``_entities`` (names must resolve and snapshots must keep them, or a
        # destroyed seed comes back on restore) but leave ``_items``. Ask "what's it called" of
        # the former, "what's here" of the latter.
        self._entities: Dict[str, WorldEntity] = {}    # all of them, tombstones included
        self._items: Dict[str, WorldEntity] = {}       # the ones still alive
        # agent_id → origin. Without it an agent in transit at snapshot time is stranded after
        # restore; restore sends it back to the origin.
        self._transit_origins: Dict[str, str] = {}
        self._current_step = 0
        self._current_world_time_label = ""
        self._current_world_time_hour = -1
        self._load_places(world_config)

    def place_agent(self, *, agent_id: str, location_id: str) -> None:
        """The only placement entry point for agents. A body's kind follows from its entry point
        (Npcs come through ``spawn_npc``/restore), never from the caller's say-so."""
        self._place_body(agent_id, location_id, BodyKind.AGENT)

    def _place_body(self, body_id: str, location_id: str, kind: BodyKind) -> None:
        """Registers location and kind together."""
        self._bodies[body_id] = kind
        self._locations[body_id] = location_id
        if location_id != IN_TRANSIT:
            self._transit_origins.pop(body_id, None)

    def move_body(self, *, body_id: str, location_id: str) -> None:
        old_location_id = self._locations.get(body_id)
        self._locations[body_id] = location_id
        if location_id == IN_TRANSIT:
            if old_location_id and old_location_id != IN_TRANSIT:
                self._transit_origins[body_id] = old_location_id
        else:
            self._transit_origins.pop(body_id, None)

    def remove_body(self, body_id: str) -> None:
        """Take a body out of the world's space; it no longer appears in ``spatial_for`` or
        ``bodies_at``.

        Location only, not the roster: the dead stay in ``agents_dict`` and Npcs in ``_npcs``.
        Releasing held items is the caller's job.
        """
        self._locations.pop(body_id, None)
        self._bodies.pop(body_id, None)
        self._transit_origins.pop(body_id, None)

    def transit_origin(self, body_id: str) -> str | None:
        """Return the departure location of an in-transit body, if registered."""
        return self._transit_origins.get(body_id)

    def get_body_location(self, body_id: str) -> str:
        return self._locations.get(body_id, UNPLACED)

    def narrative_location_name(self, location_id: str) -> str:
        """The single location-name resolver for narrative text. IN_TRANSIT → "途中"; unknown →
        "此处", never a bare id (that would leak into embedded prose)."""
        if location_id == IN_TRANSIT:
            return "途中"
        return self.space.name_of(location_id) or "此处"

    def spawn_npc(self, seed: NpcSeed, *, location_id: str) -> bool:
        """The one way to put a body-without-a-mind into the world. Returns False when nothing
        was created (no name, not a real place, or at quota); that's not a build failure.

        The quota is enforced here, not on restore, or a world that grew to the cap would lose
        people on restore. ids are a deterministic name slug, not a uuid, so replay reproduces
        the same world.
        """
        name = seed.name.strip()
        if not name:
            return False
        if location_id in (IN_TRANSIT, UNPLACED) or not self.space.has(location_id):
            logger.warning(
                "npc_spawn_without_ground",
                extra={"npc_name": name, "ground": location_id},
            )
            return False
        if len(self._npcs) >= self._max_npcs:
            logger.warning(
                "npc_quota_reached",
                extra={"cap": self._max_npcs, "npc_name": name},
            )
            return False

        base = f"npc_{slugify(name, fallback_prefix='npc')}"
        npc_id = base
        suffix = 2
        while npc_id in self._npcs:
            npc_id = f"{base}-{suffix}"
            suffix += 1
        self._npcs[npc_id] = Npc(
            npc_id=npc_id, name=name, gender=seed.gender, age=seed.age,
            description=seed.description,
        )
        self._place_body(npc_id, location_id, BodyKind.NPC)
        return True

    def assign_errand(self, order: "ErrandOrder", *, requester_id: str) -> bool:
        """Hand an errand to a body. Returns False when it can't take it (already on an errand,
        or restrained); an Npc never refuses.

        Called from ``ErrandExecutor.start``. Don't move it to the landing phase: with check and
        occupation split, two agents in one beat would both see the body free.
        """
        npc = self._npcs.get(order.npc_id)
        if npc is None or npc.errand is not None or npc.condition is not None:
            return False
        origin = self._locations.get(requester_id, "")
        if not origin or origin in (IN_TRANSIT, UNPLACED):
            return False
        npc.errand = ActiveErrand(
            order=order, requester_id=requester_id, origin_id=origin,
            assigned_step=self._current_step,
        )
        return True

    def clear_errand(self, npc_id: str) -> None:
        """Free the errand slot. Occupancy is written only here and in ``assign_errand``;
        ``NpcRunner`` owns progress within ``ActiveErrand``, not whether there is one."""
        npc = self._npcs.get(npc_id)
        if npc is not None:
            npc.errand = None

    def note_npc_outcome(self, npc_id: str, text: str, /, *, ongoing: bool) -> None:
        """Record this body's one observer line for this beat; last write wins.

        Written by its producer (``NpcRunner``), never re-derived from the errand state: a second
        reader guessing the state machine's position fails silently. ``ongoing`` means the same
        as ``ActionSummary``'s ``phase="begin"``.
        """
        self._npc_outcomes[npc_id] = (text, ongoing)

    def displace_npc(self, npc_id: str, location_id: str) -> bool:
        """Teleport this body (not by walking). Returns whether one existed.

        The errand stays but ``leg_remaining`` is zeroed: the unfinished edge belongs to the
        place it left.
        """
        npc = self._npcs.get(npc_id)
        if npc is None:
            return False
        self.move_body(body_id=npc_id, location_id=location_id)
        if npc.errand is not None:
            npc.errand.leg_remaining = 0
        self._npc_displaced.add(npc_id)
        return True

    def expire_npc_condition(self, npc_id: str, step: int) -> bool:
        """Lift an expired self-limiting condition. Returns whether one lifted.

        The condition slot is written only in this class (here and ``apply_npc_effect``); the
        runner only picks the timing. ``until_step is None`` never lifts on its own; only a
        PHYSICAL adjudication can lift it. No perception: code must not narrate the recovery.
        """
        npc = self._npcs.get(npc_id)
        if npc is None or npc.condition is None:
            return False
        if npc.condition.until_step is None or step < npc.condition.until_step:
            return False
        npc.condition = None
        return True

    def apply_npc_effect(self, effect: "NpcEffect") -> bool:
        """Land what an action did to a body; condition is the only thing it can take. The
        errand is untouched: once freed he carries on (see ``NpcEffect``)."""
        npc = self._npcs.get(effect.npc_id)
        if npc is None:
            return False
        if effect.condition_set is not None:
            npc.condition = effect.condition_set
        elif effect.condition_cleared:
            npc.condition = None
        return True

    def get_npc(self, npc_id: str) -> "Npc | None":
        return self._npcs.get(npc_id)

    def all_npcs(self) -> List["Npc"]:
        """Every Npc, ordered by id — stable so numbered menus are reproducible."""
        return [self._npcs[nid] for nid in sorted(self._npcs)]

    def bodies_at(self, location_id: str) -> List[str]:
        """Every body standing here, sorted by id, derived from ``_locations`` (places must not
        keep their own presence list). Pseudo-locations are always empty.

        Don't use it as "the people present": ask ``agents_at`` / ``npcs_at``. Using it to
        deliver messages or build relations silently treats a mindless body as a person.
        """
        if location_id in (IN_TRANSIT, UNPLACED):
            return []
        return sorted(bid for bid, loc in self._locations.items() if loc == location_id)

    def has_cognition(self, body_id: str) -> bool:
        """The single positive cognition test: ``is AGENT``, not "is not an Npc" (a complement
        would let a third body kind into code that assumes every id is an ``Agent``).

        Unrecognized ids (dead, unplaced, not a body) never have cognition: writing one record
        less beats a record pointing at nothing.
        """
        return self._bodies.get(body_id) is BodyKind.AGENT

    def is_npc(self, body_id: str) -> bool:
        """Positive ``is NPC`` test; not ``not has_cognition``, for the same reason."""
        return self._bodies.get(body_id) is BodyKind.NPC

    def agents_at(self, location_id: str) -> List[str]:
        """The ids standing here that have cognition, in presence order. Use this for "can anyone
        here hear / receive / relate"; don't filter ``bodies_at`` yourself."""
        return [bid for bid in self.bodies_at(location_id) if self.has_cognition(bid)]

    def npcs_at(self, location_id: str) -> List["Npc"]:
        """The Npcs standing here, in presence order (an index contract)."""
        return [
            self._npcs[bid]
            for bid in self.bodies_at(location_id)
            if self.is_npc(bid)
        ]

    def get_entity(self, entity_id: str) -> WorldEntity | None:
        """Return any registered world entity (location, item, landmark) by ID."""
        return self._entities.get(entity_id)

    def all_live_entities(self) -> List[WorldEntity]:
        """Every non-destroyed thing, in stable id order. Places live in ``space``."""
        return [self._items[eid] for eid in sorted(self._items)]

    def register_entity(self, entity: WorldEntity) -> None:
        """Register a thing (places go through ``space.register_place``). A DESTROYED one is
        kept only as a tombstone in ``_entities``."""
        self._entities[entity.entity_id] = entity
        if not entity.is_destroyed:
            self._items[entity.entity_id] = entity

    def spawn_entity(
        self, spawn: EntitySpawn, *, ground: str, actor_id: str | None = None,
    ) -> bool:
        """The one way to put a new thing into the world. Returns False when nothing was
        created (no name, no real ground, or the place is full); that's not an error.

        ``ground`` is where it lands unless born in someone's hands. ``actor_id`` is only for
        self-filtering. The quota is enforced here, not in ``register_entity``, so restore never
        loses things. ids are deterministic name slugs for replay. An empty ``perception`` means
        nobody saw it (a private item), not a missed delivery.
        """
        name = spawn.name.strip()
        if not name:
            return False

        holder_id = spawn.holder_id
        try:
            entity_type = WorldEntityType(spawn.entity_type)
        except ValueError:
            # ``"location"`` lands here too: places must not be created this way.
            logger.warning(
                "spawn_unknown_kind",
                extra={"entity_name": name, "kind": spawn.entity_type, "actor_id": actor_id},
            )
            return False
        if holder_id and entity_type.is_takeable:
            presence, presence_ref = EntityPresence.HELD, holder_id
        else:
            # A thing landed on a pseudo-location would be out of reach forever.
            if not ground or ground in (IN_TRANSIT, UNPLACED):
                logger.warning(
                    "spawn_without_ground",
                    extra={"entity_name": name, "actor_id": actor_id, "ground": ground},
                )
                return False
            presence, presence_ref = EntityPresence.AT_LOCATION, ground

        # Counts only the destination (hands or that floor) and only live things.
        crowd = (
            len(self.get_items_of(presence_ref)) if presence is EntityPresence.HELD
            else len(self.get_items_at(presence_ref))
        )
        if crowd >= self._max_entities_per_place:
            # extra keys must avoid LogRecord's reserved attributes (name / module / args…):
            # a collision raises KeyError.
            logger.warning(
                "entity_place_full",
                extra={
                    "cap": self._max_entities_per_place, "entity_name": name,
                    "actor_id": actor_id, "place": presence_ref,
                },
            )
            return False

        base = f"made_{slugify(name)}"
        entity_id = base
        suffix = 2
        while entity_id in self._entities:
            entity_id = f"{base}-{suffix}"
            suffix += 1
        self.register_entity(
            WorldEntity(
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                state=spawn.state or "intact",
                description=spawn.description,
                content=spawn.content,
                presence=presence,
                presence_ref=presence_ref,
                is_takeable=entity_type.is_takeable,
                is_public=spawn.is_public,
                created_step=self._current_step,
            )
        )
        # The observer needs the assigned id.
        spawn.entity_id = entity_id
        if spawn.perception and ground and ground not in (IN_TRANSIT, UNPLACED):
            # actor_ids keeps the maker from perceiving his own product as a third party.
            self.record_carry_observation(
                location_id=ground, observation=spawn.perception,
                actor_ids=(actor_id,) if actor_id else (),
            )
        return True

    def change_entity_state(
        self,
        change: EntityStateChange,
        *,
        acting_agent_id: str | None = None,
    ) -> bool:
        """The ONLY runtime path for mutating entity placement/state, so the placement invariant
        lives in one place. Returns True if the entity was found.

        ``perception``, if given, is delivered with the change. ``WorldMutationChannel`` omits it
        because it delivers through its own ``_deliver``; both would double the perception.
        """
        entity = self._entities.get(change.entity_id)
        if entity is None:
            return False

        # Capture the scope before mutation: a held item that gets destroyed
        # loses its location, so the observation must fall back to where it was.
        prior_location = entity.location_id
        # Empty = leave it alone (see EntityStateChange).
        if change.new_name:
            entity.name = change.new_name
        if change.new_state:
            entity.state = change.new_state
        if change.new_description:
            entity.description = change.new_description
        if change.new_content:
            entity.content = change.new_content
        if change.destroyed:
            entity.presence = EntityPresence.DESTROYED
            entity.presence_ref = None
            self._items.pop(entity.entity_id, None)
        elif change.owner_id is not None and entity.is_takeable:
            entity.presence = EntityPresence.HELD
            entity.presence_ref = change.owner_id
        elif change.location_id is not None:
            entity.presence = EntityPresence.AT_LOCATION
            entity.presence_ref = change.location_id
            # Must be public: visibility is "in my hands or public", so a non-public thing on
            # the ground would be invisible to everyone forever.
            entity.is_public = True

        if change.perception:
            scope = (
                change.perception_scope
                or (acting_agent_id and self._locations.get(acting_agent_id))
                or prior_location
                or ""
            )
            if scope:
                # Carry, not the current layer: this step's perception is already taken.
                # actor_ids keeps the actor from perceiving his own change.
                self.record_carry_observation(
                    location_id=scope,
                    observation=change.perception,
                    actor_ids=(acting_agent_id,),
                )

        return True

    def get_items_at(self, location_id: str) -> List[WorldEntity]:
        """Return the entities lying at a location — not the ones in somebody's hands."""
        return [
            e for e in self._items.values()
            if e.location_id == location_id and e.owner_id is None
        ]

    def get_items_of(self, owner_id: str) -> List[WorldEntity]:
        """Return entities carried by one body — **either tier**; a Npc holds entities too."""
        return [e for e in self._items.values() if e.owner_id == owner_id]

    @staticmethod
    def _as_visible(entity: WorldEntity, viewer_id: str) -> VisibleEntity:
        """One entity → what ``viewer_id`` perceives of it. ``holder_id`` stays a code-layer
        id here; the name is resolved by whoever renders it (membrane)."""
        return VisibleEntity(
            entity_id=entity.entity_id,
            name=entity.name,
            entity_type=entity.entity_type.value,
            state=entity.state,
            description=entity.description,
            is_takeable=entity.is_takeable,
            holder_id=entity.owner_id,
            content=entity.content if entity.readable_by(viewer_id) else "",
        )

    def reachable_in_order(
        self, entities: List[WorldEntity], *, viewer_id: str,
        limit: int = MAX_VISIBLE_ENTITIES,
    ) -> List[WorldEntity]:
        """Reachable things, ordered by the cost of acting on them, then cut to the menu cap.

        The only place order and cap are decided: perception and the adjudication scene share
        it, and ``physical_entity_index`` binds to it, so any extra cut elsewhere misaligns
        indices and ids.

        Tiers: in my hands, lying here, in someone else's hands. Within a tier, newest first
        (otherwise build-time seeds outrank everything made at runtime), then registration
        order. No randomness: replay must resolve recorded indices to the same things.
        """
        def _tier(e: WorldEntity) -> int:
            if e.owner_id == viewer_id:
                return 0
            return 1 if e.owner_id is None else 2
        # sorted is stable, so items in the same tier and step keep registration order.
        ordered = sorted(entities, key=lambda e: (_tier(e), -e.created_step))
        return ordered[:limit]

    def items_present_at(self, location_id: str) -> List[WorldEntity]:
        """Every live entity co-present at a location: lying here, or in the hands of someone
        here. God view; ``spatial_for`` layers ``is_public`` on top. Shared by perception and
        the executors' scene context so "what is here" has one definition."""
        return [
            e for e in self._items.values()
            if e.location_id == location_id
            or (e.owner_id is not None and self._locations.get(e.owner_id) == location_id)
        ]

    def find_item(self, item_id_or_name: str, location_id: str | None = None) -> WorldEntity | None:
        """Find an entity by exact ID, or by name among what is co-present. The name branch must
        not reach across the map, or a namesake elsewhere would answer for the one here."""
        if item_id_or_name in self._items:
            return self._items[item_id_or_name]
        for entity in self._items.values():
            if entity.name == item_id_or_name:
                if location_id is None or entity.location_id == location_id or (
                    entity.owner_id is not None
                    and self._locations.get(entity.owner_id) == location_id
                ):
                    return entity
        return None

    def check_physical_feasibility(
        self,
        agent_id: str,
        target_id: str | None,
        item_type: str | None = None,
    ) -> FeasibilityResult:
        """Check only whether a PHYSICAL target is within reach.

        ``item_type="agent"/"npc"``: a co-location check for both tiers. Everything else: the
        thing is lying here or in the hands of someone here; unknown ids pass, for the judge.
        Whether a thing can be taken from someone is the judge's call (§5), not a rule's.
        Never add an unconditional pass for unknown types: a new type would then count things
        across the world as right here.
        """
        if item_type in ("agent", "npc"):
            if not target_id:
                return FeasibilityResult(ok=False, reason="没有找到指定的人物目标")
            my_loc = self.get_body_location(agent_id)
            their_loc = self.get_body_location(target_id)
            # Two travellers with equal location strings aren't in the same place.
            if my_loc == IN_TRANSIT or their_loc == IN_TRANSIT:
                return FeasibilityResult(ok=False, reason="赶路途中，无法执行物理行动")
            if my_loc != their_loc:
                # Never name the other's location: that would be free reconnaissance.
                return FeasibilityResult(
                    ok=False,
                    reason=f"对方不在{self.narrative_location_name(my_loc)}，无法执行物理行动",
                )
            return FeasibilityResult(ok=True, reason="", resolved_id=target_id)
        if not target_id:
            return FeasibilityResult(ok=True, reason="", resolved_id="")
        agent_loc = self.get_body_location(agent_id)
        entity = self.find_item(target_id, agent_loc)
        if entity is None:
            return FeasibilityResult(ok=True, reason="", resolved_id="")
        if entity.owner_id == agent_id:
            return FeasibilityResult(ok=True, reason="", resolved_id=entity.entity_id)
        here = (
            self._locations.get(entity.owner_id) == agent_loc and agent_loc != IN_TRANSIT
            if entity.owner_id is not None
            else entity.location_id == agent_loc
        )
        if not here:
            # Don't say where it is (free reconnaissance).
            return FeasibilityResult(
                ok=False,
                reason=f"{entity.name}不在{self.narrative_location_name(agent_loc)}",
            )
        return FeasibilityResult(ok=True, reason="", resolved_id=entity.entity_id)

    def check_move_feasibility(
        self,
        agent_id: str,
        target_location: str | None,
    ) -> FeasibilityResult:
        """Check whether agent can move to target_location right now."""
        to_id = self.space.resolve(target_location or "")
        if to_id is None:
            # Don't echo target_location: it is whatever the caller passed (an id, or nothing), and
            # this reason reaches memory prose.
            return FeasibilityResult(ok=False, reason="找不到地点")

        from_id = self.get_body_location(agent_id)
        # Multi-hop allowed; one edge may cost >1 step.
        path = self.space.shortest_path(from_id, to_id)
        to_place = self.space.get(to_id)
        to_name = self.space.name_of(to_id) or "某地"
        if path is None:
            return FeasibilityResult(
                ok=False,
                reason=f"从{self.narrative_location_name(from_id)}没有通路可抵达{to_name}",
            )

        capacity = to_place.capacity if to_place is not None else 0
        if len(self.bodies_at(to_id)) >= capacity:
            return FeasibilityResult(
                ok=False,
                reason=f"{to_name}人满为患，无法进入",
            )

        return FeasibilityResult(
            ok=True, reason="", resolved_id=to_id, path=tuple(path),
            travel_seconds=self.space.reachable_from(from_id).get(to_id, 0),
        )

    def check_talk_feasibility(
        self,
        initiator_id: str,
        target_ids: list[str],
        *,
        target_label: str = "对方",
    ) -> FeasibilityResult:
        """Check whether initiator can speak with all targets right now.

        ``target_label`` is resolved by the caller (env has no directory). The reason never
        names the target's actual location (no free recon).
        """
        if not target_ids:
            return FeasibilityResult(ok=False, reason="想要对话，但是不知道对话目标。")

        self_location = self.get_body_location(initiator_id)
        if self_location == IN_TRANSIT:
            return FeasibilityResult(ok=False, reason="自己正在赶路途中，无法交谈")
        my_loc_name = self.narrative_location_name(self_location)
        for tid in target_ids:
            target_location = self.get_body_location(tid)
            if target_location == IN_TRANSIT:
                return FeasibilityResult(ok=False, reason=f"{target_label}正在赶路途中，无法交谈")
            if self_location != target_location:
                return FeasibilityResult(ok=False, reason=f"{target_label}不在{my_loc_name}，无法交谈")

        return FeasibilityResult(ok=True, reason="", resolved_id=self_location)

    @property
    def current_world_time_label(self) -> str:
        """The calendar label for the current world time, cached by begin_step, for time
        anchors in adjudication prompts."""
        return self._current_world_time_label

    def begin_step(self, *, step: int, world_time: WorldTime) -> None:
        """Flip carry buffers into the active step layer (see the class docstring)."""
        self._current_step = step
        self._current_world_time_label = world_time.time_label
        self._current_world_time_hour = world_time.hour_of_day
        self._npc_outcomes = {}
        self._npc_displaced = set()
        self._step_annotations = {k: list(v) for k, v in self._carry_annotations.items()}
        self._carry_annotations = {}

    def record_carry_observation(
        self,
        *,
        location_id: str,
        observation: str,
        strength: float | None = None,
        actor_ids: tuple[str, ...] = (),
    ) -> None:
        """Queue a location-scoped AmbientEvent for the NEXT step's ambient_events.

        strength:
            None  → ordinary ambient (PerceptionMemoryLayer's default)
            float → used as is, for something that matters to onlookers
        actor_ids:
            Every member of the producing execution plus whoever the act landed on; spatial_for
            hides the ambient from them so nobody reads a third-person account of their own act.
            ``()`` for environment / system ambient.
        """
        if observation is None or len(observation.strip()) == 0:
            return None
        self._carry_annotations.setdefault(location_id, []).append(
            AmbientEvent(
                content=observation, strength=strength, actor_ids=tuple(actor_ids),
                agent_actor_ids=self._cognizant(actor_ids),
            )
        )

    def record_happening(
        self, *, location_id: str, outcome: str, step: int, actor_ids: tuple[str, ...] = (),
    ) -> None:
        """Record the full-fidelity version of something that surfaced at ``location_id``, for
        lurkers; ``record_carry_observation`` gives onlookers the degraded one. The caller
        decides what surfaced; this only stores and trims to both bounds.
        """
        if not outcome or not outcome.strip() or not location_id:
            return None
        bucket = self._happenings.setdefault(location_id, [])
        bucket.append(_Happening(
            step=step, content=outcome, actor_ids=tuple(actor_ids),
        ))
        cutoff = step - HAPPENINGS_WINDOW_STEPS
        if bucket[0].step <= cutoff:
            bucket = [t for t in bucket if t.step > cutoff]
        if len(bucket) > HAPPENINGS_MAX_ENTRIES:
            bucket = bucket[-HAPPENINGS_MAX_ENTRIES:]
        self._happenings[location_id] = bucket

    def recent_happenings(
        self, location_id: str, *, since_step: int, exclude_ids: tuple[str, ...] = (),
    ) -> List[tuple[int, str]]:
        """``(beat number, full string)`` for what happened at ``location_id`` since
        ``since_step`` (inclusive), sorted by beat, write order kept within a beat.

        ``exclude_ids`` drops what those people already have, or adjudication would rule success
        on something he already knew. The caller must translate the beat into a time reference
        (``core.prompts.recency_prefix``) before it enters a prompt.
        """
        if not location_id:
            return []
        excluded = {i for i in exclude_ids if i}
        return [
            (t.step, t.content)
            for t in sorted(self._happenings.get(location_id, []), key=lambda t: t.step)
            if t.step >= since_step and not excluded.intersection(t.actor_ids)
        ]

    def _cognizant(self, ids: tuple[str, ...]) -> tuple[str, ...]:
        """The ids that have cognition. Derived here rather than declared by producers, where a
        forgotten declaration silently lets a tool-body into relation indexes."""
        return tuple(i for i in ids if i and self.has_cognition(i))

    def location_view(self, location_id: str) -> LocationView:
        """The single renderer for a location's narrative appearance. Unknown → "某地", never
        the bare id. ``IN_TRANSIT`` is handled here, not by callers, so the wording lives in
        one place."""
        if location_id == IN_TRANSIT:
            return LocationView(name="赶路途中", description="正在两地之间的路途上。")
        # Unknown places fall back to True: "not open to the public" is a claim about a real
        # place, and saying an unknown place has restricted access would invent a fact.
        return self.space.view_of(location_id) or LocationView(name="某地", is_public=True)

    def spatial_for(
        self,
        *,
        agent_id: str,
        step: int | None = None,
        world_time: str | None = None,
    ) -> SpatialPerception:
        """Build location-based perception for one agent."""
        location_id = self._locations.get(agent_id, UNPLACED)

        # Travellers don't share a room: they can't see each other or hear location ambient.
        if location_id == IN_TRANSIT:
            return SpatialPerception(
                location_id=location_id,
                location_view=self.location_view(location_id),
                world_time_label=world_time or self._current_world_time_label,
                current_step=step if step is not None else self._current_step,
                world_time_hour=self._current_world_time_hour,
                visible_agents={},
                # What he carries; empty would make carried things vanish on the road.
                visible_entities=[
                    self._as_visible(e, agent_id) for e in self.reachable_in_order(
                        self.get_items_of(agent_id), viewer_id=agent_id,
                    )
                ],
                ambient_events=[],
                reachable_locations=[],
            )

        location_view = self.location_view(location_id)
        # Empty presences: engine.presence.attach_presence fills in names and state. Split by
        # kind: consumers of visible_agents assume every id resolves to an ``Agent``.
        here = [bid for bid in self.bodies_at(location_id) if bid != agent_id]
        visible_agents = {
            other_id: PerceivedPresence() for other_id in here if self.has_cognition(other_id)
        }
        visible_npcs = {
            other_id: PerceivedNpc() for other_id in here if self.is_npc(other_id)
        }
        # Never show an agent a third-person account of an act he was part of.
        ambient_events: list[AmbientEvent] = [
            ev for ev in self._step_annotations.get(location_id, []) if agent_id not in ev.actor_ids
        ]
        # is_public hides things in other people's hands, never in one's own.
        visible_entities = [
            self._as_visible(e, agent_id) for e in self.reachable_in_order(
                [
                    e for e in self.items_present_at(location_id)
                    if e.owner_id == agent_id or e.is_public
                ],
                viewer_id=agent_id,
            )
        ]
        # destination_index binds to this order, so it must be deterministic.
        reachable_seconds = (
            self.space.reachable_from(location_id)
            if self.space.has(location_id) else {}
        )
        reachable_ids = sorted(reachable_seconds, key=lambda lid: (reachable_seconds[lid], lid))
        reachable_locations = [
            ReachableLocation(
                location_id=nid,
                view=self.location_view(nid),
                travel_seconds=reachable_seconds[nid],
            )
            for nid in reachable_ids
        ]
        return SpatialPerception(
            location_id=location_id,
            location_view=location_view,
            world_time_label=world_time or self._current_world_time_label,
            current_step=step if step is not None else self._current_step,
            world_time_hour=self._current_world_time_hour,
            visible_agents=visible_agents,
            visible_npcs=visible_npcs,
            visible_entities=visible_entities,
            ambient_events=ambient_events,
            reachable_locations=reachable_locations,
        )

    def snapshot_state(self) -> dict[str, object]:
        """Expose environment state for snapshots and observation tooling."""
        return {
            # The agent half is write-only (agent_store is authoritative); the Npc half is read
            # back on restore.
            "body_locations": dict(sorted(self._locations.items())),
            "transit_origins": dict(sorted(self._transit_origins.items())),
            "npc_outcomes": {
                nid: {"text": text, "ongoing": ongoing}
                for nid, (text, ongoing) in sorted(self._npc_outcomes.items())
            },
            "npc_displaced": sorted(self._npc_displaced),
            # The whole Npc: nothing else records its condition or errand.
            "npc_states": {
                nid: _npc_payload(npc) for nid, npc in sorted(self._npcs.items())
            },
            "step_annotations": {
                scope: list(entries) for scope, entries in sorted(self._step_annotations.items())
            },
            # What onlookers perceive next step. Unsaved, a restore would drop that beat.
            "carry_annotations": {
                scope: [_ambient_payload(e) for e in entries]
                for scope, entries in sorted(self._carry_annotations.items())
            },
            "location_names": {
                place.place_id: place.name for place in self.space.all_places()
            },
            # Includes tombstones, so restore doesn't resurrect a destroyed seed. Every field:
            # this is the only record of runtime-made things (a missing is_public makes a hidden
            # item public on restore).
            "entity_states": {
                eid: {
                    "name": e.name,
                    "entity_type": e.entity_type.value,
                    "state": e.state,
                    "description": e.description,
                    "presence": e.presence.value,
                    "presence_ref": e.presence_ref,
                    "is_takeable": e.is_takeable,
                    "is_public": e.is_public,
                    "created_step": e.created_step,
                    "content": e.content,
                }
                for eid, e in sorted(self._entities.items())
            },
        }

    def restore_state(self, state: Mapping[str, object]) -> None:
        """Inverse of :meth:`snapshot_state` for the mutable parts of the world.

        The caller rebuilds places and agent placement; this restores entity state (re-registering
        runtime-made ones, re-killing tombstoned seeds), ``transit_origins`` (so the caller can
        send mid-MOVE agents back), next step's carried observations, and Npcs whole, position
        included.
        """
        carry = state.get("carry_annotations")
        if isinstance(carry, Mapping):
            self._carry_annotations = {
                str(scope): [_ambient_from_payload(e) for e in entries if isinstance(e, Mapping)]
                for scope, entries in carry.items() if isinstance(entries, list)
            }
        transit_origins = state.get("transit_origins")
        if isinstance(transit_origins, Mapping):
            self._transit_origins = {
                str(agent_id): str(origin) for agent_id, origin in transit_origins.items()
            }
        npc_states = state.get("npc_states")
        if isinstance(npc_states, Mapping):
            placements = state.get("body_locations")
            placements = placements if isinstance(placements, Mapping) else {}
            for npc_id, payload in npc_states.items():
                if not isinstance(payload, Mapping):
                    continue
                npc_id = str(npc_id)
                self._npcs[npc_id] = _npc_from_payload(npc_id, payload)
                where = str(placements.get(npc_id, "") or "")
                if where and where != IN_TRANSIT:
                    self._place_body(npc_id, where, BodyKind.NPC)
                else:
                    # Restore can't put it back mid-road: back to the origin, errand resumes.
                    origin = self._transit_origins.get(npc_id, "")
                    if origin:
                        self._place_body(npc_id, origin, BodyKind.NPC)
        entity_states = state.get("entity_states")
        if not isinstance(entity_states, Mapping):
            return
        for entity_id, payload in entity_states.items():
            if not isinstance(payload, Mapping):
                continue
            presence = EntityPresence(str(payload.get("presence", EntityPresence.AT_LOCATION.value)))
            presence_ref = payload.get("presence_ref")
            existing = self._entities.get(str(entity_id))
            if existing is not None:
                # Seeds rebuild only the original state; overlay runtime changes.
                existing.name = str(payload.get("name", existing.name))
                existing.state = str(payload.get("state", existing.state))
                existing.description = str(payload.get("description", existing.description))
                existing.content = str(payload.get("content", existing.content))
                existing.presence = presence
                existing.presence_ref = presence_ref
                existing.is_public = bool(payload.get("is_public", existing.is_public))
                if presence == EntityPresence.DESTROYED:
                    self._items.pop(str(entity_id), None)
                continue
            # Runtime-made: no seed, so restore field by field.
            entity_type = WorldEntityType(str(payload.get("entity_type", WorldEntityType.ITEM.value)))
            self.register_entity(
                WorldEntity(
                    entity_id=str(entity_id),
                    name=str(payload.get("name", entity_id)),
                    entity_type=entity_type,
                    state=str(payload.get("state", "intact")),
                    description=str(payload.get("description", "")),
                    presence=presence,
                    presence_ref=presence_ref,
                    is_takeable=bool(payload.get("is_takeable", entity_type.is_takeable)),
                    is_public=bool(payload.get("is_public", True)),
                    created_step=int(payload.get("created_step", 0) or 0),
                    content=str(payload.get("content", "")),
                )
            )

    def _load_places(self, world_config: object | None) -> None:
        """Load places from the config: the one time places enter this world."""
        if world_config is None or not hasattr(world_config, "get_places"):
            return
        places = world_config.get_places()
        if not isinstance(places, dict):
            return
        for place in places.values():
            if isinstance(place, Place):
                self.space.register_place(place)
