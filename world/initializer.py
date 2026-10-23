"""World initialization for runtime-ready Asamana worlds."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from agent.agent import Agent, snapshot_agent_state
from agent.decision import DecisionEngine
from agent.memory import MemorySystem
from agent.memory_types import importance_level
from agent.goals import text_to_long_term_goal_entity
from agent.need import NeedEngine
from agent.personality import ActionStatus, AgentActivityStatus, PersonalityLayer, StateLayer
from agent.relation import RelationSystem
from agent.state_codec import encode_agent_state
from core.container import Container
from core.interfaces.agent_store import AgentRelation, AgentState, relations_snapshot
from core.interfaces.snapshot import EngineState, WorldSnapshot, WorldState
from core.logging import get_logger
from engine.clock import (
    SECONDS_PER_HOUR,
    WorldTime,
    WorldTimeConfig,
    parse_calendar_style,
    resolve_step_seconds,
)
from engine.directory import LiveWorldDirectory
from engine.environment import DEFAULT_MAX_NPCS, IN_TRANSIT, EnvironmentSystem
from engine.executors.base import ActionExecutionState
from core.interfaces.world_config import WorldConfig
from worlds.tiled import CAST_ART_FIELDS, CHARACTERS_FILENAME

from core.coerce import coerce_int
from world.catalog import ActorCasting
from world.models import AgentDefinition, AgentTier, NpcSeed, World, WorldEntitySeed
from core.interfaces.entity import Npc, WorldEntity, WorldEntityType
from world.setup import MapPlaces, SetupHistoricalEvent, WorldLore, WorldSetup
from world.stored_config import StoredWorldConfig, serialize_world_config

logger = get_logger(__name__)



class WorldInitializer:
    """Instantiate runtime agents and persist step-0 world state."""

    def __init__(self, container: Container, *, max_npcs: int = DEFAULT_MAX_NPCS) -> None:
        self._container = container
        # How many cognition-less doers a world can have. The default is a module constant; application
        # injects the real value from config. How busy a world should be is tuned per world, not a universal law.
        self._max_npcs = max_npcs

    async def prepare(self, world_id: str, setup: WorldSetup, world_config: WorldConfig) -> WorldSetup:
        """Freeze what a built world stands on and persist its setup; nothing is materialized.

        ``world_config`` is the map this world was built on, chosen per world. Its data, render
        map and cast art are frozen under the world's id, so later template edits never touch
        the world: ``initialize`` and ``restore`` read only those copies and the setup. Initial
        Every location in the setup is already a location id (the build resolved them), and the setup
        must pass ``WorldSetup.validate`` (else ``ValueError``: a build that can't start a world
        fails, Rule 2). Returns the setup as persisted.
        """
        setup.validate(MapPlaces.of(world_config))
        await self._container.snapshot.save_world_config(
            world_id, serialize_world_config(world_config)
        )
        # The render map (.tmj) is frozen too, so the 2D renderer keeps serving the map this world
        # was built on after the template file is edited. Configs with no map art have none.
        render_map = world_config.render_map()
        if render_map is not None:
            await self._container.snapshot.save_world_map(world_id, render_map)
            await self._freeze_map_assets(world_id, world_config, render_map)
        # The cast is frozen on the same terms as the ground it walks on: a body is
        # dressed for the map's period, so restyling that art later would restyle
        # every past world's REPLAY along with it.
        await self._freeze_character_assets(world_id, world_config)
        await self._container.snapshot.save_world_setup(world_id, setup.as_dict())
        return setup

    async def initialize(self, world_id: str, *, casting: ActorCasting | None = None) -> World:
        """Materialize a world from its persisted setup and write step 0.

        Creates the agents and writes what the setup derives: relations, backstory memories and
        initial agent states, then step 0 with the setup frozen into it. Runs once, when the world
        is confirmed, from the setup file; the caller clears whatever an earlier, failed attempt
        left behind and deletes the file afterwards. Raises on any failure, like the build (Rule 2/6).
        ``casting`` as in ``restore``.
        """
        config_data = await self._container.snapshot.load_world_config(world_id)
        stored = await self._container.snapshot.load_world_setup(world_id)
        if config_data is None or stored is None:
            raise ValueError(f"World {world_id!r} has no setup to confirm")
        world_config = StoredWorldConfig(config_data)
        setup = WorldSetup.from_dict(stored)
        # It passed every revision's check; this is the last line before anything is written.
        setup.validate(MapPlaces.of(world_config))
        clock_config = self._clock_config_for(setup.lore, world_config)
        environment = EnvironmentSystem(world_config, max_npcs=self._max_npcs)
        await self._instantiate_entity_seeds(setup.entity_seeds, environment)
        self._instantiate_npc_seeds(setup.npc_seeds, environment)
        agents: dict[str, Agent] = {}
        for definition in setup.characters:
            agent = self._instantiate_agent(
                world_id=world_id,
                definition=definition,
                clock_config=clock_config,
                casting=casting,
            )
            agents[agent.agent_id] = agent
            environment.place_agent(agent_id=agent.agent_id, location_id=definition.initial_location)
        # Concurrent but fail-fast: any failure makes the world unusable, so raise and let the
        # user retry instead of degrading silently (Rule 2/3). return_exceptions=True only keeps
        # siblings from being cancelled (Rule 4).
        for result in await asyncio.gather(
            *(agent.memory_system.ensure_collections() for agent in agents.values()),
            return_exceptions=True,
        ):
            if isinstance(result, BaseException):
                raise result

        directory = LiveWorldDirectory.from_agents(agents, environment)
        await self._initialize_relations(world_id=world_id, agents=agents, setup=setup)
        await self._write_historical_memories(
            agents=agents,
            historical_events=setup.historical_events,
            seconds_per_step=clock_config.seconds_per_step,
        )
        # Warming recent is a read/cache warm-up: failure isn't fatal (the memories are still on disk), so log a warning and continue (Rule 6 read).
        for result in await asyncio.gather(
            *(agent.memory_system.warm_recent_memories(current_step=0) for agent in agents.values()),
            return_exceptions=True,
        ):
            if isinstance(result, BaseException):
                logger.warning("warm_recent_memories_failed", extra={"error": str(result)})
        # Short-term goals aren't generated here; NeedEngine.run is their sole owner. Each agent's
        # first cognition step finds an empty queue (_should_update_goals) and generates them from real
        # perception, warmed history and perceived relations. One generation path keeps quality uniform.
        # A failed initial state write must raise (Rule 6 write): swallowing it would quietly corrupt restore/reset.
        for result in await asyncio.gather(
            *(self._persist_initial_agent_state(agent) for agent in agents.values()),
            return_exceptions=True,
        ):
            if isinstance(result, BaseException):
                raise result
        step_zero_snapshot = await self._save_initial_snapshot(
            world_id=world_id,
            setup=setup,
            agents=agents,
            environment=environment,
            clock_config=clock_config,
        )

        logger.info(
            "world_initialized",
            extra={
                "world_id": world_id,
                "world_name": setup.lore.world_name,
                "agent_count": len(agents),
            },
        )

        return World(
            world_id=world_id,
            setup=setup,
            agents=agents,
            world_config=world_config,
            clock_config=clock_config,
            environment=environment,
            step_zero_snapshot=step_zero_snapshot,
            directory=directory,
        )

    def _instantiate_agent(
        self,
        *,
        world_id: str,
        definition: AgentDefinition,
        clock_config: WorldTimeConfig = WorldTimeConfig(),
        casting: ActorCasting | None = None,
    ) -> Agent:
        seconds_per_step = clock_config.seconds_per_step
        # Closure from step → the world calendar's time label, injected into MemorySystem to render time
        # spans in compressed summaries, so agent/ doesn't import engine.clock (the layer boundary holds;
        # the translation membrane lives in the world layer).
        def _time_label_for(step: int) -> str:
            return WorldTime.from_step(step, clock_config).time_label
        # The day-boundary side of the same membrane: the clock time at the world's start. Together with
        # seconds_per_step it defines the world's timeline, from which memory prefixes count how many
        # midnights have passed. "今日 / 昨日" is a difference in day boundaries, which can't be derived by
        # dividing a duration (see core.prompts.render_memory). The start date isn't needed: it cancels out of the day difference.
        world_start_second_of_day = WorldTime.from_step(0, clock_config).elapsed_seconds
        # Static needs live in soul.innate_needs (baseline-ensured in build/from_dict, including all universal needs).
        active_innate = [n for n in definition.soul.innate_needs if not n.is_hidden]
        personality = PersonalityLayer(
            soul=definition.soul,
            state=StateLayer(
                agent_id=definition.agent_id,
                step=0,
                emotion=definition.initial_emotion,
                active_needs=[n.type.value for n in active_innate],
                # Leave dominant_need empty: it's derived solely by NeedEngine.run()'s argmax(scores), computed on
                # the agent's first cognition step from real perception. Don't pre-guess it at init with an
                # intensity×weight formula that differs from run (it gets overwritten on the first step and may
                # contradict it); record "no cognition yet" honestly as None.
                dominant_need=None,
                long_term_goals=list(definition.long_term_goals),
                long_term_goal_entities=[
                    text_to_long_term_goal_entity(t) for t in definition.long_term_goals
                ],
                # The short-term goal queue starts empty (see initialize). Don't seed rule templates here: restore
                # would rebuild them into the FIFO queue as permanent placeholders. If the LLM fails, the queue
                # stays empty and the next step tries again.
                short_term_goals=[],
                short_term_goal_entities=[],
                current_location=definition.initial_location,
                activity_status=AgentActivityStatus.IDLE,
                action_status=ActionStatus.IDLE,
                # Need intensity seed: active needs get innate's seed intensity; hidden needs stay out of
                # need_intensities (their intensity falls back to innate.intensity in current_needs() when scoring).
                need_intensities={n.type.value: n.intensity for n in active_innate},
            ),
        )
        cast = casting is not None and casting.character_id == definition.agent_id
        # The character an actor plays is main tier from then on, exit or not: the user lives the
        # story through it, so it gets full perception fidelity, memory and retries.
        is_main = definition.tier == AgentTier.MAIN or cast
        memory_system = MemorySystem(
            self._container.llm_router,
            self._container.embedding,
            self._container.vector_store,
            world_id=world_id,
            agent_id=definition.agent_id,
            is_main_character=is_main,
            time_label_for=_time_label_for,
            world_start_second_of_day=world_start_second_of_day,
            seconds_per_step=seconds_per_step,
        )
        from agent.reflection import ReflectionEngine

        reflection_engine = ReflectionEngine(
            self._container.llm_router,
            memory_system,
            personality,
            agent_id=definition.agent_id,
            seconds_per_step=seconds_per_step,
            time_label_for=_time_label_for,
            world_start_second_of_day=world_start_second_of_day,
        )
        relation_system = RelationSystem(
            self._container.agent_store,
            world_id=world_id,
            agent_id=definition.agent_id,
        )
        from agent.relation_evolution import RelationEvolution

        relation_evolution = RelationEvolution(
            self._container.llm_router,
            memory_system,
            relation_system,
            personality,
            agent_id=definition.agent_id,
        )
        return Agent(
            world_id=world_id,
            agent_id=definition.agent_id,
            personality=personality,
            decision_engine=DecisionEngine(
                self._container.llm_router,
                seconds_per_step=seconds_per_step,
                world_start_second_of_day=world_start_second_of_day,
            ),
            memory_system=memory_system,
            need_engine=NeedEngine(llm_router=self._container.llm_router, seconds_per_step=seconds_per_step,
                                   world_start_second_of_day=world_start_second_of_day),
            relation_system=relation_system,
            agent_store=self._container.agent_store,
            llm_router=self._container.llm_router,
            is_main_character=is_main,
            seconds_per_step=seconds_per_step,
            world_start_second_of_day=world_start_second_of_day,
            reflection_engine=reflection_engine,
            relation_evolution=relation_evolution,
        )

    async def _persist_initial_agent_state(self, agent: Agent) -> None:
        initial_state = encode_agent_state(agent.world_id, agent.agent_id, agent.personality.state)
        await agent.agent_store.save_agent_state(agent.world_id, agent.agent_id, initial_state)
        await agent.agent_store.save_initial_agent_state(agent.world_id, agent.agent_id, initial_state)

    async def _initialize_relations(
        self,
        *,
        world_id: str,
        agents: dict[str, Agent],
        setup: WorldSetup,
    ) -> None:
        # Each setup relation holds both directions of one pair (``WorldSetup.validate`` admits one
        # per pair), so each becomes two records.
        tasks = []
        for relation in setup.relations:
            for from_id, to_id, trust, affection, labels in (
                (relation.source_id, relation.target_id,
                 relation.trust, relation.affection, list(relation.labels)),
                (relation.target_id, relation.source_id,
                 relation.reverse_trust, relation.reverse_affection, list(relation.reverse_labels)),
            ):
                target_soul = agents[to_id].personality.soul
                initial_relation = AgentRelation(
                    world_id=world_id,
                    from_id=from_id,
                    to_id=to_id,
                    trust_objective=trust,
                    affection_objective=affection,
                    updated_step=0,
                    labels=labels,
                    to_name=target_soul.name,
                    # to_gender shares to_name's source and lifetime: people in seed relations are usually not
                    # present, so this record is their only reference in the prompt, and a name alone still leaves the
                    # LLM guessing gender.
                    to_gender=target_soul.gender,
                )
                tasks.append(self._container.agent_store.save_relation(initial_relation))
                tasks.append(self._container.agent_store.save_initial_relation(initial_relation))
        if tasks:
            # A build-time write failure must raise (Rule 2/6), same rule as _persist_initial_agent_state.
            # Initial relation labels are never rewritten (rule 3), so dropping a father-son label here loses it for
            # good: the world still reports ready, the user still confirms, and the two treat each other as
            # strangers from step one with no way to recover it.
            # return_exceptions=True only keeps siblings from being cancelled (Rule 4); raise on any exception after collecting.
            for result in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(result, BaseException):
                    logger.error(
                        "relation_save_failed",
                        extra={"world_id": world_id, "error": str(result)},
                    )
                    raise result

    async def _write_historical_memories(
        self,
        *,
        agents: dict[str, Agent],
        historical_events: Sequence[SetupHistoricalEvent],
        seconds_per_step: int,
    ) -> None:
        tasks = []
        for event in historical_events:
            related_ids = list(event.figure_ids)
            # Quantized to the world's step length here, where the step is first needed; an event
            # within one step of the start still lands before it.
            created_step = -max(1, round(event.hours_before_start * SECONDS_PER_HOUR / seconds_per_step))
            importance = importance_level(event.importance)
            decay_score = max(0.3, 1.0 - min(abs(created_step), 70) * 0.01)
            for agent_id in related_ids:
                agent = agents[agent_id]
                tasks.append(
                    agent.memory_system.seed_factual_memory(
                        current_step=created_step,
                        raw_content=event.event,
                        related_agents=[rid for rid in related_ids if rid != agent_id],
                        triggered_by="historical_init",
                        importance=importance,
                        metadata={
                            "kind": "historical_init",
                            "importance": event.importance,
                        },
                        decay_score=decay_score,
                    )
                )
        if tasks:
            # Same as _initialize_relations: raise on a build-time write failure (Rule 2/6). Backstory is a
            # one-off the LLM wrote for this world and won't come in again. An agent missing it has no memory
            # of an event they were at, and can't talk with the others who were there.
            # return_exceptions=True only keeps siblings from being cancelled (Rule 4).
            for result in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(result, BaseException):
                    logger.error(
                        "historical_memory_write_failed",
                        extra={
                            "error": str(result),
                            "error_type": type(result).__name__,
                            "error_repr": repr(result),
                        },
                    )
                    raise result

    async def restore(self, world_id: str, *, casting: ActorCasting | None = None) -> World:
        """Restore a previously built world from persisted storage.

        Loads the world's persisted setup and config asset, reconstructs agents
        from their definitions, applies the latest AgentState from agent_store,
        and returns a runtime-ready World.

        ``casting`` is kept outside the world's stores (in the catalog), so the caller passes it;
        the cast character is built main tier. Who performs it is the runtime's to wire
        (``NarrativeRuntime.cast_actor``).
        """
        config_data = await self._container.snapshot.load_world_config(world_id)
        if config_data is None:
            raise ValueError(f"No persisted world config found for world {world_id!r}")
        active_world_config = StoredWorldConfig(config_data)
        step_zero = await self._container.snapshot.load(world_id, 0)
        if step_zero is None or not step_zero.setup:
            raise ValueError(f"No step-0 snapshot with a setup found for world {world_id!r}")
        setup = WorldSetup.from_dict(step_zero.setup)
        agent_definitions = list(setup.characters)
        if casting is not None and casting.character_id not in {d.agent_id for d in agent_definitions}:
            raise ValueError(
                f"World {world_id!r} has no character {casting.character_id!r} to cast the actor as"
            )

        environment = EnvironmentSystem(active_world_config, max_npcs=self._max_npcs)
        await self._instantiate_entity_seeds(setup.entity_seeds, environment)
        # Npcs aren't rebuilt here, deliberately unlike entity seeds: the npc_states payload is
        # self-contained (identity + position + condition + current errand) and _restore_environment_state
        # loads it in one go. Building them on both paths would collide on ids, and since ids are slugs of
        # names, a collision splits one person into two in the world.
        clock_config = self._clock_config_for(setup.lore, active_world_config)
        agents: dict[str, Agent] = {}

        # The step the world resumes at = the latest persisted snapshot. This is the
        # world's authoritative current step (not any per-agent counter, which lags for
        # idle agents); the runtime clock starts here and the next plan runs at
        # resume_step+1. reset deletes snapshots > 0 before restoring, so this is 0 there.
        resume_step = max(
            (s for s in await self._container.snapshot.list_steps(world_id) if s > 0),
            default=0,
        )
        await self._rollback_relations(world_id, resume_step)
        latest = await self._container.snapshot.load(world_id, resume_step)
        stored_states = {
            definition.agent_id: await self._container.agent_store.load_agent_state(world_id, definition.agent_id)
            for definition in agent_definitions
        }
        active_executions = _resumable_executions(world_id, latest, stored_states)
        resumed_participants = {pid for execution in active_executions for pid in execution.participant_ids}

        for definition in agent_definitions:
            agent = self._instantiate_agent(
                world_id=world_id,
                definition=definition,
                clock_config=clock_config,
                casting=casting,
            )
            agents[agent.agent_id] = agent
            await agent.memory_system.ensure_collections()
            # Before warming: an uncommitted step's memories must not come back with the rest.
            await agent.memory_system.purge_memories_after(resume_step)

            stored = stored_states[agent.agent_id]
            if stored is not None:
                agent.apply_stored_state(
                    stored, in_resumed_execution=agent.agent_id in resumed_participants
                )
                # Don't place the dead: death handling already removed them from the environment (current_location
                # isn't cleared, it only records where they fell). Placing them unconditionally would stand the
                # corpse back up in the room, visible to everyone there forever.
                # Check liveness with is_active (just derived by apply_stored_state from vitality, the single source of truth), not a raw vitality comparison.
                if agent.is_active:
                    environment.place_agent(
                        agent_id=agent.agent_id,
                        location_id=_body_location(latest, agent.agent_id, stored, resumed_participants),
                    )
                await agent.memory_system.warm_recent_memories(current_step=stored.updated_step)
            else:
                environment.place_agent(agent_id=agent.agent_id, location_id=definition.initial_location)
                await agent.memory_system.warm_recent_memories(current_step=0)

        self._restore_environment_state(latest, environment)
        self._repatriate_transit_agents(agents, agent_definitions, environment, resumed_participants=resumed_participants)
        directory = LiveWorldDirectory.from_agents(agents, environment)

        logger.info(
            "world_restored",
            extra={
                "world_id": world_id,
                "world_name": setup.lore.world_name,
                "agent_count": len(agents),
            },
        )
        return World(
            world_id=world_id,
            setup=setup,
            agents=agents,
            world_config=active_world_config,
            clock_config=clock_config,
            environment=environment,
            step_zero_snapshot=step_zero,
            directory=directory,
            current_step=resume_step,
            active_executions=active_executions,
        )

    async def _rollback_relations(self, world_id: str, resume_step: int) -> None:
        """Make the relation store match the snapshot the world resumes from.

        Relations are written during a step (perception, feedback), but only a completed step's
        snapshot is authoritative; a step that never committed would otherwise leave its
        relation changes behind and apply them twice. An empty snapshot table is the read-failure
        fallback of the step that wrote it, not "no relations", so it is left alone.
        """
        snapshot = await self._container.snapshot.load(world_id, resume_step)
        if snapshot is None or not snapshot.state.relations:
            logger.warning(
                "relation_rollback_skipped",
                extra={"world_id": world_id, "step": resume_step},
            )
            return
        store = self._container.agent_store
        await store.clear_relations(world_id)
        for payload in snapshot.state.relations.values():
            await store.save_relation(AgentRelation(**payload))

    def _repatriate_transit_agents(
        self,
        agents: dict[str, Agent],
        agent_definitions: Sequence[AgentDefinition],
        environment: EnvironmentSystem,
        *,
        resumed_participants: set[str],
    ) -> None:
        """Land a body left in transit with no trip to carry it.

        A body on a resumed MOVE (in ``resumed_participants``) stays where the snapshot left it and
        walks on. One with no trip (its snapshot saved no MOVE for it, or the MOVE wasn't resumed)
        stands on the IN_TRANSIT pseudo-location, which has no connections and no entities, so
        every move feasibility check fails and it is stuck forever. Matches movement's interrupt
        turn-back (<50%): return to the origin registered in the environment, else the initial
        location.
        """
        initial_by_id = {d.agent_id: d.initial_location for d in agent_definitions}
        for agent_id, agent in agents.items():
            if agent_id in resumed_participants or environment.get_body_location(agent_id) != IN_TRANSIT:
                continue
            origin = environment.transit_origin(agent_id)
            landed = origin or initial_by_id.get(agent_id, "")
            if not landed:
                continue
            environment.place_agent(agent_id=agent_id, location_id=landed)
            agent.personality.update_location(location=landed)
            logger.info(
                "in_transit_agent_repatriated",
                extra={
                    "agent_id": agent_id,
                    "landed_at": landed,
                    "from_transit_origin": bool(origin),
                },
            )

    @staticmethod
    def _restore_environment_state(
        snapshot: WorldSnapshot | None, environment: EnvironmentSystem
    ) -> None:
        """Re-apply the persisted environment state onto a freshly seeded world.

        ``restore`` rebuilds the environment from the step-0 entity seeds, which leaves item
        state/location/ownership at their initial values. The resume step's snapshot records
        the live environment in its ``state`` and ``engine``; without this the world's items
        silently revert to step 0 while agents carry their latest state.

        **STEP 0 IS A RESUME STEP LIKE ANY OTHER — do not skip it.** For entities the re-apply
        is a no-op, but mindless bodies have no seeds and ``restore`` deliberately doesn't
        rebuild them (ids would collide with this payload), so this call is the ONLY thing that
        puts them back. Skipping step 0 (every fresh world's first run, every reset) leaves the
        world with none: ERRAND silently out of reach.
        """
        if snapshot is None:
            return
        environment.restore_state(
            npcs=snapshot.state.npcs,
            entities=snapshot.state.entities,
            engine=snapshot.engine.environment,
        )

    async def _freeze_map_assets(
        self, world_id: str, world_config: WorldConfig, render_map: dict[str, object]
    ) -> None:
        """Copy the images the frozen map names into the world's own storage.

        Freezing the .tmj alone leaves the freeze half-done: the map addresses its
        art by path, and that art sits in a template directory the author keeps
        editing. Without this, a world renders its own geometry through today's
        tilesets — and a tile that moved within a sheet renders as garbage.

        Best-effort and never fatal (see ``_freeze_assets``).
        """
        source_dir = world_config.render_assets_dir()
        if source_dir is None:
            return
        tilesets = render_map.get("tilesets")
        if not isinstance(tilesets, list):
            return
        await self._freeze_assets(
            world_id,
            source_dir,
            [
                image
                for tileset in tilesets
                if isinstance(tileset, dict) and (image := str(tileset.get("image") or ""))
            ],
        )

    async def _freeze_character_assets(
        self, world_id: str, world_config: WorldConfig
    ) -> None:
        """Copy the cast's manifest and every sheet it names into the world.

        Same reasoning as the map's art, applied to the figures: a template's
        character art goes on being edited, and a world that read today's copy
        would replay its own history in whatever the cast is wearing now.

        The manifest is stored as an asset under its own relative path rather than
        through a bespoke channel — it IS a file in the template's asset tree, and
        the world reads it back the same way the renderer reads a sheet.
        """
        manifest = world_config.render_characters()
        source_dir = world_config.render_assets_dir()
        if manifest is None or source_dir is None:
            return
        atlases = manifest.get("atlases")
        named: list[str] = []
        if isinstance(atlases, dict):
            for entry in atlases.values():
                if not isinstance(entry, dict):
                    continue
                named.extend(
                    path for key in CAST_ART_FIELDS if (path := str(entry.get(key) or ""))
                )
        await self._freeze_assets(world_id, source_dir, named)
        try:
            await self._container.snapshot.save_world_asset(
                world_id,
                CHARACTERS_FILENAME,
                json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
            )
        except OSError as exc:
            logger.warning(
                "character_manifest_freeze_failed",
                extra={"world_id": world_id, "error": str(exc)},
            )

    async def _freeze_assets(
        self, world_id: str, source_dir: Path, relative_paths: Iterable[str]
    ) -> None:
        """Copy each named file into the world's own storage, by relative path.

        Best-effort per file, and never fatal: a world that builds is worth more
        than a perfectly frozen one. A file that fails to freeze is served as missing,
        so only that image is absent from the map.
        """
        for relative in relative_paths:
            path = source_dir / relative
            try:
                data = await asyncio.to_thread(path.read_bytes)
                await self._container.snapshot.save_world_asset(world_id, relative, data)
            except OSError as exc:
                logger.warning(
                    "render_asset_freeze_failed",
                    extra={"world_id": world_id, "asset": relative, "error": str(exc)},
                )

    async def _save_initial_snapshot(
        self,
        *,
        world_id: str,
        setup: WorldSetup,
        agents: dict[str, Agent],
        environment: EnvironmentSystem,
        clock_config: WorldTimeConfig,
    ) -> WorldSnapshot:
        world_time = WorldTime.from_step(0, clock_config)
        environment_state = environment.world_state()
        # Step 0 has no events: everyone simply stands where the world put them, which the state
        # says. Nobody is mid-move yet, so the body location is the agent's own.
        snapshot = WorldSnapshot(
            world_id=world_id,
            step=0,
            timestamp=datetime.now(),
            world_time=world_time.clock_payload(),
            state=WorldState(
                agents={agent_id: snapshot_agent_state(agent) for agent_id, agent in agents.items()},
                npcs=environment_state["npcs"],
                entities=environment_state["entities"],
                relations=await relations_snapshot(self._container.agent_store, world_id, agents),
            ),
            engine=EngineState(
                pending_messages=await self._container.message_provider.peek_pending(world_id),
                environment=environment.engine_state(),
            ),
            setup=setup.as_dict(),
        )
        await self._container.snapshot.save(world_id, 0, snapshot)
        return snapshot

    def _instantiate_npc_seeds(self, seeds: Sequence[NpcSeed], environment: EnvironmentSystem) -> None:
        """Put the theme's mindless bodies into the world, each where the setup places it. Sibling
        of the entity seeds. Slots and quotas belong to ``spawn_npc`` (the single point where they
        come into being)."""
        for seed in seeds:
            environment.spawn_npc(
                Npc(
                    npc_id=seed.npc_id, name=seed.name, gender=seed.gender, age=seed.age,
                    description=seed.description,
                ),
                location_id=seed.location,
            )

    async def _instantiate_entity_seeds(
        self, seeds: Sequence[WorldEntitySeed], environment: EnvironmentSystem,
    ) -> None:
        for seed in seeds:
            location_id = seed.location
            entity_id = seed.entity_id
            entity = WorldEntity(
                entity_id=entity_id,
                name=seed.name,
                entity_type=WorldEntityType(seed.entity_type),
                state=seed.initial_state,
                description=seed.description,
                presence_ref=location_id,   # presence defaults to AT_LOCATION
                is_public=True,
                content=seed.content,
            )
            environment.register_entity(entity)
            logger.info(
                "entity_seed_instantiated",
                extra={
                    "entity_id": entity_id,
                    "seed_name": seed.name,
                    "entity_type": seed.entity_type,
                    "location_id": location_id,
                },
            )

    def _clock_config_for(
        self,
        lore: WorldLore,
        world_config: WorldConfig,
    ) -> WorldTimeConfig:
        runtime_context = world_config.to_runtime_context()
        time_config = dict(lore.world_time_config)
        return WorldTimeConfig(
            # The era name falls back to the map's era label (e.g. a dynasty name) and never to the world name: the
            # world name is the story's title, not an era, and using it as one would render dates like
            # "Year One of Neon and Bills" that mix the two layers.
            era_name=str(time_config.get("era_name") or runtime_context.get("era_name") or ""),
            # Date NAMING follows the world (its map), never the theme payload: it is a
            # property of the setting, and it is frozen with the world's config so a
            # built world keeps saying dates the way it always did.
            calendar=parse_calendar_style(runtime_context.get("calendar")),
            # The year can only come from theme analysis: maps carry no specific year (see the WorldConfig
            # contract), so there's no runtime_context to fall back to. It's missing only in test stubs and
            # saves predating world building; fall back to year one.
            start_year=coerce_int(time_config.get("start_year"), default=1, minimum=1),
            start_month=coerce_int(
                time_config.get("start_month"),
                default=int(runtime_context.get("start_month", 1)),
                minimum=1,
                maximum=12,
            ),
            start_day=coerce_int(
                time_config.get("start_day"),
                default=int(runtime_context.get("start_day", 1)),
                minimum=1,
                maximum=30,
            ),
            start_hour=coerce_int(
                time_config.get("start_hour"),
                default=int(runtime_context.get("start_hour", 6)),
                minimum=0,
                maximum=23,
            ),
            # Step duration and the calendar anchor share a source: both are fixed by the world-building theme
            # analysis and both are read from this world_time_config (already quantized from hours to seconds
            # at the LLM boundary in _analysis_from_payload; this only reads it back). So it's naturally
            # frozen: the lore is persisted with the world's setup and restore reads the same copy back, so
            # a world's time pricing never drifts with deployment config, and no second frozen copy is needed.
            seconds_per_step=resolve_step_seconds(time_config.get("seconds_per_step")),
        )


def _resumable_executions(
    world_id: str, snapshot: WorldSnapshot | None, stored_states: Mapping[str, AgentState | None],
) -> list[ActionExecutionState]:
    """The multi-step acts the resume step's snapshot left under way, each kept only if every
    participant's saved state is still in it.

    Both are written at the step's commit point, so they agree unless an agent state write
    failed (Rule 6). An act that can't be read, or that someone in it has no record of, is
    dropped whole: half an act would tick a character who isn't in it. Its participants are then
    left idle by ``Agent.apply_stored_state``, as is anyone whose snapshot saved no act at all.
    """
    if snapshot is None:
        return []
    resumed: list[ActionExecutionState] = []
    for payload in snapshot.engine.active_executions:
        try:
            execution = ActionExecutionState.from_payload(payload)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "active_execution_unreadable",
                extra={"world_id": world_id, "execution_id": payload.get("execution_id"), "error": str(exc)},
            )
            continue
        states = [stored_states.get(pid) for pid in execution.participant_ids]
        if not all(
            state is not None
            and state.action_status == ActionStatus.IN_PROGRESS.value
            and state.vitality > 0.0
            for state in states
        ):
            logger.warning(
                "active_execution_dropped",
                extra={
                    "world_id": world_id, "execution_id": execution.execution_id,
                    "participant_ids": execution.participant_ids,
                },
            )
            continue
        resumed.append(execution)
    return resumed


def _body_location(
    snapshot: WorldSnapshot | None, agent_id: str, stored: AgentState, resumed_participants: set[str],
) -> str:
    """Where to stand a living body on restore.

    A body in a resumed act stands where the snapshot saw it: a MOVE in progress is on a
    waypoint or IN_TRANSIT, while the character's own ``current_location`` changes only on
    arrival, so that would put him back at his origin for the step before his trip ticks on.
    Everyone else stands where they last knew themselves to be.
    """
    if agent_id in resumed_participants and snapshot is not None:
        where = (snapshot.state.agents.get(agent_id) or {}).get("location_id")
        if where:
            return str(where)
    return stored.current_location
