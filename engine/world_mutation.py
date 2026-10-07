"""World-state mutation channel: the channel behind EventSystem's private-effect rule.

Private effects must go through this one channel, not scattered state edits in the author layer.

Two authors change the world through it with asymmetric permissions, checked per author at
apply time (``_permits``):

- The human director (``Author.DIRECTOR``): every kind, on people and on errand bodies
  (``Npc``) alike.
- The LLM event editor (``Author.SYSTEM``): may only place a new thing somewhere
  (``SpawnMutation``), or change / destroy a thing lying on the ground, held by nobody. Anything
  agents have their own causal claim on (life, injuries, location, held things) is off limits.

Don't move this table back into ``EventSystem``: the editor generates in the background, and a
thing on the ground at generation time may be picked up by landing time. Only the check at
landing time is accurate.

Invariant: every mutation carries its own observable description (``observation``)
================================================================================
A world change nobody knows about splits world state from everyone's beliefs, so
``observation`` is a required field on the base class. It also serves as ``cause`` when
relocate tears down in-flight execution.

The other half: the description must actually reach people (``Audience``)
==========================================================================
Delivery belongs to the channel, not to each kind: a missed per-kind delivery is silent. Each
mutation only declares its ``Audience`` and ``_deliver`` does the delivery; agent-internal deltas
are applied by the feedback layer (the Executor/Feedback boundary one level up). Declaring no
audience must be an explicit, justified choice (e.g. KILL leaves the account to death handling).

Where it lands
==============
Injections land in the event-injection phase of ``NarrativeRuntime.run_step``: before
``agent_spatials`` is built (everyone sees the move or death the same step), and after
``pre_step_active`` is sampled (``DeathHandler.process_new_deaths`` handles a director kill as a
normal death). So this module leaves death cleanup alone; redoing it would send two notices.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict

from core.interfaces.action import (
    EntitySpawn, EntityStateChange, NpcEffect, Observed, TargetAgentEffect,
)
from core.interfaces.condition import BodyCondition, npc_condition_fallback_steps
from core.interfaces.llm import IndexedRef
from core.interfaces.perception import Situation
from core.logging import get_logger
from engine.environment import IN_TRANSIT, UNPLACED, EnvironmentSystem
from engine.execution_processor import ExecutionProcessor
from engine.injection import Author

if TYPE_CHECKING:
    # Type-only: no runtime engine→world dependency.
    from agent.agent import Agent
    from world.models import WorldEntity

logger = get_logger(__name__)


@dataclass(frozen=True)
class _MutationBase:
    """What every mutation shares: how it looks to observers. Don't give it a default (see the
    module docstring's invariant)."""

    observation: str


@dataclass(frozen=True)
class EntityMutation(_MutationBase):
    """Change or destroy a world entity (item / landmark). Empty fields are left unchanged."""

    entity_id: str
    new_state: str = ""
    new_description: str = ""
    new_content: str = ""
    destroyed: bool = False


@dataclass(frozen=True)
class SpawnMutation(_MutationBase):
    """Place something new at a location. It lands at a place, not in anyone's hands."""

    location_id: str
    name: str
    entity_type: str        # a WorldEntityType value; unknown ones are rejected by ``spawn_entity``
    description: str = ""
    content: str = ""


class VitalityEffect(str, Enum):
    """What to do to a person's life: three well-defined levels, not a float.

    A decimal from upstream would be unanchored: the same "wound him" could parse to 0.2 or 0.8.
    Finer granularity needs a defined scale first, not more numbers here.
    """

    KILL  = "kill"
    WOUND = "wound"
    HEAL  = "heal"

    @classmethod
    def prompt_choices(cls) -> str:
        return " 或 ".join(e.value for e in cls)


# Positive is damage, same direction as ``PersonalityLayer.apply_vitality_damage``. Severity
# belongs to this layer; parsing only recognizes the level.
_VITALITY_DELTA: dict[VitalityEffect, float] = {
    VitalityEffect.KILL:  1.0,    # always fatal: vitality is clamped to 0
    VitalityEffect.WOUND: 0.35,
    VitalityEffect.HEAL:  -0.35,
}


@dataclass(frozen=True)
class VitalityMutation(_MutationBase):
    """Wound, heal or kill a person. Never applies to an Npc: it has no vitality axis."""

    body_id: str
    effect: VitalityEffect


@dataclass(frozen=True)
class RelocateMutation(_MutationBase):
    """Move a person straight to a place, regardless of route or travel time; the director can."""

    body_id: str
    location_id: str


@dataclass(frozen=True)
class ConditionMutation(_MutationBase):
    """Add a lasting condition to a person, or lift theirs. Empty ``description`` = lift."""

    body_id: str
    description: str = ""


Mutation = (
    EntityMutation | SpawnMutation | VitalityMutation | RelocateMutation | ConditionMutation
)


def parse_spawn(raw: Any, location_ref: IndexedRef) -> SpawnMutation | None:
    """LLM output → ``SpawnMutation``. Both authors' prompts use the same keys; one parser.

    Anything missing a location, name or observation is dropped (observation is a mutation
    invariant). Unknown categories aren't filtered here; ``spawn_entity`` rejects them on landing.
    """
    if not isinstance(raw, dict):
        return None
    where = location_ref.resolve([raw.get("location")])
    name = str(raw.get("name", "")).strip()
    observation = str(raw.get("observation", "")).strip()
    if not where or not name or not observation:
        return None
    return SpawnMutation(
        observation=observation,
        location_id=where[0],
        name=name,
        entity_type=str(raw.get("entity_type", "")).strip().lower(),
        description=str(raw.get("description", "")).strip(),
        content=str(raw.get("content", "")).strip(),
    )


def _departure_line(name: str) -> str:
    """The line seen at the place someone left: a person walked away, nothing more.

    Needed because ``SpatialPerception`` has no cross-step diff, so departures are otherwise silent.
    Wording matches the unknown-direction branch of ``MovementExecutor._departure_line``. Don't
    write "忽然不见了踪影" (suddenly vanished): code adding interpretation bleeds into bystanders'
    appraisal and memory. Don't reuse the director's sentence: it describes the arrival.
    No shared helper with that one: it also handles direction and ``scene_line``. The name
    comes from the object the caller holds, not ``WorldDirectory`` (usage rule, boundary 2).
    """
    return f"{name}离开了此地。"


@dataclass(frozen=True)
class Audience:
    """Who should know about this change; the one question every mutation must answer.

    - ``observed``: third person, per location (a relocation reads "he appeared" at the
      destination and "he's gone" at the origin).
    - ``effect``: first person, received through the feedback layer, which is also the only
      writer of the agent-internal deltas it carries.

    ``world_changed``: has the world-side change already landed? relocate / entity change the
    world in ``_apply_*``; vitality lives entirely in ``effect``, so if that fails nothing
    happened. Keeps ``apply``'s receipt honest in both directions.
    """

    observed: tuple[Observed, ...] = ()
    effect: TargetAgentEffect | None = None
    world_changed: bool = True
    displaced: tuple[str, ...] = ()
    """Agents moved by the mutation rather than walking there themselves (only relocate): a new
    ``location_id`` alone doesn't say whether the body walked the route or was put there."""

    @property
    def principals(self) -> tuple[str, ...]:
        """Who this change is about; also the self-exclusion list, since the person has their own
        first-person record."""
        return (self.effect.agent_id,) if self.effect is not None else ()


@dataclass(frozen=True)
class MutationOutcome:
    """Everything the caller needs to know once a mutation has landed; ``apply``'s return value.

    Not a bare ``tuple[str, ...] | None``: "landed, nobody saw" (``()``) and "didn't land"
    (``None``) would look nearly the same. Getting an object means it landed.
    """

    reached: tuple[str, ...] = ()
    """The person concerned plus onlookers at each location; the receipt's ``delivered_to``."""

    displaced: tuple[str, ...] = ()
    """Agents moved without walking. See ``Audience.displaced``."""


class WorldMutationChannel:
    """Apply a validated ``Mutation`` to the world.

    Only changes world state and the perception of those present; the ledger entry is
    ``InjectionDispatcher``'s job, death cleanup is death handling's.
    """

    def __init__(
        self,
        *,
        environment: EnvironmentSystem,
        processor: ExecutionProcessor,
        seconds_per_step: int,
    ) -> None:
        self._environment = environment
        self._processor = processor
        # Only for ``npc_condition_fallback_steps``.
        self._seconds_per_step = seconds_per_step

    async def apply(
        self,
        mutation: Mutation,
        *,
        step: int,
        agents: Dict[str, "Agent"],
        author: Author,
    ) -> MutationOutcome | None:
        """Land a mutation: change the world side, then deliver it to those who should know.

        ``None`` means it didn't land (including permission failures); never raises. ``reached=()``
        is a valid result: the world changed with nobody there to see it.

        Only this method knows whom it reached; callers must not guess via
        ``getattr(..., "body_id")``.

        ``author`` has no default: a default would silently grant a permission level.
        """
        if not self._permits(author, mutation):
            logger.warning(
                "world_mutation_forbidden",
                extra={"kind": type(mutation).__name__, "author": author.value, "step": step},
            )
            return None
        try:
            audience = await self._change_world(mutation, step=step, agents=agents)
            if audience is None:
                return None
            return await self._deliver(audience, step=step, agents=agents)
        except Exception as exc:  # noqa: BLE001 — a failed intervention must not stop the step loop
            logger.warning(
                "world_mutation_failed",
                extra={"kind": type(mutation).__name__, "step": step, "error": str(exc)},
            )
            return None

    def _permits(self, author: Author, mutation: Mutation) -> bool:
        """The only copy of the permission table (see module docstring)."""
        if author is Author.DIRECTOR:
            return True
        if isinstance(mutation, SpawnMutation):
            return True
        if isinstance(mutation, EntityMutation):
            entity = self._environment.get_entity(mutation.entity_id)
            # ``location_id`` is set only when AT_LOCATION: held and destroyed items are rejected.
            return entity is not None and entity.location_id is not None
        return False

    async def _change_world(
        self, mutation: Mutation, *, step: int, agents: Dict[str, "Agent"],
    ) -> Audience | None:
        """``None`` = nothing happened."""
        if isinstance(mutation, EntityMutation):
            return self._apply_entity(mutation)
        if isinstance(mutation, SpawnMutation):
            return self._apply_spawn(mutation)
        if isinstance(mutation, VitalityMutation):
            return self._apply_vitality(mutation, agents=agents)
        if isinstance(mutation, RelocateMutation):
            return await self._apply_relocate(mutation, step=step, agents=agents)
        if isinstance(mutation, ConditionMutation):
            return self._apply_condition(mutation, step=step, agents=agents)
        logger.warning("world_mutation_unknown_kind", extra={"kind": type(mutation).__name__})
        return None

    async def _deliver(
        self, audience: Audience, *, step: int, agents: Dict[str, "Agent"],
    ) -> MutationOutcome | None:
        """The only place that delivers a mutation's perception. ``None`` means nothing landed.

        First person goes first: it may be the entire change (vitality), and whether the third
        person is sent depends on whether it was fatal. The first-person account carries
        ``_here_and_now`` for every kind.
        """
        landed = audience.world_changed
        reached: list[str] = []
        effect = audience.effect
        if effect is not None:
            target = agents.get(effect.agent_id)
            if target is None:
                logger.warning("mutation_effect_target_missing", extra={"agent_id": effect.agent_id})
            else:
                try:
                    # No person did this to him; the world did.
                    await target.apply_target_effect(
                        effect, from_agent_id=None, step=step,
                        situation=self._here_and_now(effect.agent_id),
                    )
                    landed = True
                    reached.append(effect.agent_id)
                except Exception as exc:  # noqa: BLE001 — Rule 1: must not kill the step loop
                    logger.warning(
                        "mutation_effect_failed",
                        extra={"agent_id": effect.agent_id, "step": step, "error": str(exc)},
                    )

        if not landed:
            return None
        displaced = tuple(aid for aid in audience.displaced if aid in agents)

        # If the person died, death handling sends the notice; another account here would be
        # perceived twice. Checked at delivery, not per kind: a WOUND can be fatal too.
        if any(
            (agent := agents.get(pid)) is not None and not agent.is_active
            for pid in audience.principals
        ):
            return MutationOutcome(reached=tuple(reached), displaced=displaced)

        for observed in audience.observed:
            location = observed.location_id
            # Pseudo-locations have no onlookers.
            if not location or location in (IN_TRANSIT, UNPLACED):
                continue
            self._environment.record_carry_observation(
                location_id=location,
                observation=observed.text,
                strength=observed.strength,
                actor_ids=audience.principals,
            )
            reached.extend(self._onlookers_at(location, audience.principals, agents))
        return MutationOutcome(
            reached=tuple(dict.fromkeys(reached)), displaced=displaced,
        )

    def _here_and_now(self, agent_id: str) -> Situation:
        """Where and when this change happened; handed to the person along with the
        first-person effect.

        ``Agent._situation`` assumes this step's perceive already ran, but injections land before
        perceive; without this the memory gets last step's header, and a relocation a
        self-contradictory place ("在东宫：他出现在玄武门").

        Hand it over; don't overwrite ``Agent._situation``: only ``perceive_step`` and ``Agent.refresh_situation`` write it.
        """
        return Situation(
            location_view=self._environment.location_view(
                self._environment.get_body_location(agent_id)
            ),
            time_label=self._environment.current_world_time_label,
        )

    def _onlookers_at(
        self, location_id: str, principals: tuple[str, ...], agents: Dict[str, "Agent"],
    ) -> list[str]:
        """The living people standing there who will read this ambient next step.

        The criterion must match ``spatial_for``'s ambient delivery (at the location, not in
        ``actor_ids``). The dead don't count. Whether he remembers it (write threshold) is a
        separate question; don't anticipate it here.
        """
        return [
            agent_id
            for agent_id, agent in agents.items()
            if agent_id not in principals
            and agent.is_active
            and self._environment.get_body_location(agent_id) == location_id
        ]

    # ------------------------------------------------------------------

    def _apply_entity(self, mutation: EntityMutation) -> Audience | None:
        """Change or destroy a thing.

        Compute the scope explicitly: a held thing has no location_id, and
        ``change_entity_state``'s default chain would let nobody perceive the change.

        No person concerned: the holder sees it through ambient like everyone else; a separate
        first-person channel would write it twice.
        """
        entity = self._environment.get_entity(mutation.entity_id)
        if entity is None:
            logger.warning("mutation_entity_missing", extra={"entity_id": mutation.entity_id})
            return None
        scope = self._entity_scope(entity)
        changed = self._environment.change_entity_state(
            EntityStateChange(
                entity_id=mutation.entity_id,
                new_state=mutation.new_state,   # empty = unchanged (destroy / perceive only)
                new_description=mutation.new_description,
                new_content=mutation.new_content,
                destroyed=mutation.destroyed,
                # perception stays empty: delivery goes only through ``_deliver``.
            )
        )
        if not changed:
            return None
        return Audience(observed=(Observed(location_id=scope or "", text=mutation.observation),))

    def _apply_spawn(self, mutation: SpawnMutation) -> Audience | None:
        """Place a new thing somewhere. If it can't land (place full, bad name/category), no-op."""
        landed = self._environment.spawn_entity(
            EntitySpawn(
                name=mutation.name,
                description=mutation.description,
                entity_type=mutation.entity_type,
                content=mutation.content,
                # perception stays empty: delivery goes only through ``_deliver``.
            ),
            ground=mutation.location_id,
        )
        if not landed:
            return None
        return Audience(observed=(Observed(location_id=mutation.location_id, text=mutation.observation),))

    def _entity_scope(self, entity: "WorldEntity") -> str | None:
        """Where this thing's change is visible right now. ``None`` = nobody can see it."""
        if entity.location_id:
            return entity.location_id
        owner_id = entity.owner_id
        return self._environment.get_body_location(owner_id) if owner_id else None

    def _apply_vitality(
        self, mutation: VitalityMutation, *, agents: Dict[str, "Agent"]
    ) -> Audience | None:
        """Wound / heal / kill, entirely through the first-person effect (``world_changed=False``).

        Don't call ``apply_vitality_damage`` directly: ``apply_target_effect`` also writes the
        memory and flips ``is_active`` on death (a plain bool; skipping it leaves a corpse that
        keeps acting).

        An Npc has no vitality; wounding or binding it is a ``ConditionMutation``.
        """
        if self._environment.is_npc(mutation.body_id):
            logger.warning(
                "mutation_npc_without_vitality",
                extra={"npc_id": mutation.body_id, "effect": mutation.effect.value},
            )
            return None
        agent = agents.get(mutation.body_id)
        if agent is None or not agent.is_active:
            logger.warning("mutation_agent_unavailable", extra={"agent_id": mutation.body_id})
            return None
        return Audience(
            observed=(Observed(
                location_id=self._environment.get_body_location(mutation.body_id),
                text=mutation.observation,
            ),),
            effect=TargetAgentEffect(
                agent_id=mutation.body_id,
                # Same sentence onlookers see: the director wrote one.
                factual_memory=mutation.observation,
                vitality_damage=_VITALITY_DELTA[mutation.effect],
                # Death handling builds the notice from this narrative phrase.
                death_cause=mutation.observation,
            ),
            world_changed=False,
        )

    async def _apply_relocate(
        self, mutation: RelocateMutation, *, step: int, agents: Dict[str, "Agent"]
    ) -> Audience | None:
        """Move someone directly. Both location records must be written: the world side
        (``EnvironmentSystem``) and ``personality.state.current_location``.

        Tear down in-flight execution first, or on completion it treats him as still at the old
        place. Three audiences: arrival, departure, and his own memory. ``displaced`` tells the
        map he was put there, not walking.

        An Npc's errand isn't cancelled: ``NpcRunner`` re-routes from where it stands next tick.
        """
        if not self._environment.space.has(mutation.location_id):
            logger.warning("mutation_location_missing", extra={"location_id": mutation.location_id})
            return None
        if self._environment.is_npc(mutation.body_id):
            return self._relocate_npc(mutation)
        agent = agents.get(mutation.body_id)
        if agent is None or not agent.is_active:
            logger.warning("mutation_agent_unavailable", extra={"agent_id": mutation.body_id})
            return None

        # Force teardown, not interrupt: he didn't decide to drop it (see force_teardown).
        await self._processor.force_teardown(
            agent, agents, step, cause=mutation.observation, trigger="relocate",
        )

        # Taken after teardown: MOVE's interrupt places someone in transit at a real waypoint,
        # which is the spot he actually vacates.
        origin_id = self._environment.get_body_location(mutation.body_id)

        self._environment.move_body(body_id=mutation.body_id, location_id=mutation.location_id)
        agent.personality.update_location(step=step, location=mutation.location_id)

        origin_name = (
            self._environment.narrative_location_name(origin_id)
            if origin_id not in (IN_TRANSIT, UNPLACED) else ""
        )
        return Audience(
            observed=(
                # The director's sentence is written for the destination's viewpoint.
                Observed(location_id=mutation.location_id, text=mutation.observation),
                # Skipped if moved to where he already is: nobody should see him arrive and leave.
                *([Observed(location_id=origin_id, text=_departure_line(agent.personality.soul.name))]
                  if origin_id != mutation.location_id else []),
            ),
            effect=TargetAgentEffect(
                agent_id=mutation.body_id,
                factual_memory=(
                    mutation.observation + (f"此前还在{origin_name}。" if origin_name else "")
                ),
            ),
            displaced=(mutation.body_id,),
        )

    def _relocate_npc(self, mutation: RelocateMutation) -> Audience | None:
        """Move an errand body; onlookers at both ends as for agents."""
        npc = self._environment.get_npc(mutation.body_id)
        if npc is None:
            return None
        origin_id = self._environment.get_body_location(mutation.body_id)
        self._environment.displace_npc(mutation.body_id, mutation.location_id)
        return Audience(observed=(
            Observed(location_id=mutation.location_id, text=mutation.observation),
            *([Observed(location_id=origin_id, text=_departure_line(npc.name or "某人"))]
              if origin_id != mutation.location_id else []),
        ))

    def _apply_condition(
        self, mutation: ConditionMutation, *, step: int, agents: Dict[str, "Agent"],
    ) -> Audience | None:
        """Apply or lift a lasting condition. Lifting when there's no condition = nothing happens.

        No one is recorded as imposing it. For a person ``until_step=None`` (he can free himself);
        an Npc can't, so ``None`` would be permanent and it falls back to a fixed duration.
        """
        description = mutation.description.strip()
        where = self._environment.get_body_location(mutation.body_id)
        observed = (Observed(location_id=where, text=mutation.observation),)
        if self._environment.is_npc(mutation.body_id):
            npc = self._environment.get_npc(mutation.body_id)
            if npc is None or (not description and npc.condition is None):
                return None
            self._environment.apply_npc_effect(NpcEffect(
                npc_id=mutation.body_id,
                condition_set=BodyCondition(
                    description=description,
                    since_step=step,
                    until_step=step + npc_condition_fallback_steps(self._seconds_per_step),
                ) if description else None,
                condition_cleared=not description,
            ))
            return Audience(observed=observed)

        agent = agents.get(mutation.body_id)
        if agent is None or not agent.is_active:
            logger.warning("mutation_agent_unavailable", extra={"agent_id": mutation.body_id})
            return None
        if not description and agent.personality.state.condition is None:
            return None
        return Audience(
            observed=observed,
            effect=TargetAgentEffect(
                agent_id=mutation.body_id,
                factual_memory=mutation.observation,
                condition_set=BodyCondition(description=description, since_step=step)
                if description else None,
                condition_cleared=not description,
            ),
            # Like vitality: carried entirely in the effect.
            world_changed=False,
        )

