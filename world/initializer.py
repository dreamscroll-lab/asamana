"""World initialization for runtime-ready Asamana worlds."""

from __future__ import annotations

import asyncio
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from agent.agent import Agent, snapshot_agent_state
from agent.decision import DecisionEngine
from agent.memory import MemorySystem
from agent.memory_types import importance_level
from agent.goals import GoalEntity, GoalOrigin, GoalStatus, text_to_long_term_goal_entity
from agent.need import NeedEngine, NeedType
from agent.personality import ActionStatus, AgentActivityStatus, EmotionState, PersonalityLayer, StateLayer, parse_emotion_type
from agent.relation import RelationSystem
from core.container import Container
from core.interfaces.agent_store import AgentRelation, AgentState, relations_snapshot
from core.interfaces.condition import condition_from_dict, condition_to_dict
from core.interfaces.snapshot import WorldSnapshot
from core.logging import get_logger
from engine.clock import (
    WorldTime,
    WorldTimeConfig,
    parse_calendar_style,
    resolve_step_seconds,
)
from core.interfaces.directory import WorldDirectory
from engine.directory import LiveWorldDirectory
from engine.environment import DEFAULT_MAX_NPCS, IN_TRANSIT, EnvironmentSystem
from core.interfaces.world_config import WorldConfig
from worlds.tiled import CAST_ART_FIELDS, CHARACTERS_FILENAME

from core.coerce import coerce_int
from world.models import AgentDefinition, AgentTier, HistoricalEventSeed, ThemeAnalysis, World, WorldEntity, WorldEntityType
from world.stored_config import StoredWorldConfig, serialize_world_config

logger = get_logger(__name__)



class WorldInitializer:
    """Instantiate runtime agents and persist step-0 world state."""

    def __init__(self, container: Container, *, max_npcs: int = DEFAULT_MAX_NPCS) -> None:
        self._container = container
        # How many cognition-less actors a world can have. The default is a module constant; application
        # injects the real value from config. How busy a world should be is tuned per world, not a universal law.
        self._max_npcs = max_npcs

    async def initialize(
        self,
        *,
        theme: str,
        analysis: ThemeAnalysis,
        agent_definitions: Sequence[AgentDefinition],
        world_id: str,
        world_config: WorldConfig,
    ) -> World:
        """Create runtime agents, persist initialization state, and write step 0.

        ``world_config`` is the map this particular world was built on — an argument,
        not initializer state, because it is chosen per world: two worlds built by the
        same process sit on different maps. ``restore`` takes none, and that asymmetry
        is the contract: a restored world reads the copy persisted under its own id.
        """

        # This world's private copy of the template: environment placement mutates
        # entity state by reference, and that must never reach the shared template.
        world_config = copy.deepcopy(world_config)
        clock_config = self._clock_config_for(analysis, world_config)
        environment = EnvironmentSystem(world_config, max_npcs=self._max_npcs)
        await self._instantiate_entity_seeds(analysis, environment, world_config)
        self._instantiate_npc_seeds(analysis, environment, world_config)
        agents: dict[str, Agent] = {}

        for definition in agent_definitions:
            definition.initial_location = self._resolve_initial_location(definition, world_config)
            agent = self._instantiate_agent(
                world_id=world_id,
                definition=definition,
                clock_config=clock_config,
            )
            agents[agent.agent_id] = agent
            # Vector collections are created per agent → run concurrently after the loop.
            environment.place_agent(agent_id=agent.agent_id,
                location_id=definition.initial_location,
            )
        # Concurrent but fail-fast: collection creation is build-time infrastructure, and any failure
        # makes the world unusable, so raise and let the user retry instead of degrading silently (Rule
        # 2/3). return_exceptions=True only keeps siblings from being cancelled (Rule 4); after
        # collecting, handle per slot and raise on any exception.
        for result in await asyncio.gather(
            *(agent.memory_system.ensure_collections() for agent in agents.values()),
            return_exceptions=True,
        ):
            if isinstance(result, BaseException):
                raise result

        directory = LiveWorldDirectory.from_agents(agents, environment)
        await self._initialize_relations(world_id=world_id, agents=agents, analysis=analysis)
        await self._write_historical_memories(agents=agents, historical_events=analysis.historical_events)
        # Warming recent is a read/cache warm-up: failure isn't fatal (the memories are still on disk), so log a warning and continue (Rule 6 read).
        for result in await asyncio.gather(
            *(agent.memory_system.warm_recent_memories(current_step=0) for agent in agents.values()),
            return_exceptions=True,
        ):
            if isinstance(result, BaseException):
                logger.warning("warm_recent_memories_failed", extra={"error": str(result)})
        # Short-term goals aren't generated at build time; NeedEngine.run is their sole owner. Each agent's
        # first cognition step finds an empty queue (_should_update_goals) and generates them from real
        # perception, warmed history and perceived relations. One generation path keeps quality uniform.
        # A failed initial state write must raise (Rule 6 write): swallowing it would quietly corrupt restore/reset.
        for result in await asyncio.gather(
            *(self._persist_initial_agent_state(agent) for agent in agents.values()),
            return_exceptions=True,
        ):
            if isinstance(result, BaseException):
                raise result
        # Persist the world's private config asset: restore rebuilds the map from
        # this data alone, so later template-code edits never touch built worlds.
        await self._container.snapshot.save_world_config(
            world_id, serialize_world_config(world_config)
        )
        # Freeze the render map (.tmj) per world too, so the 2D renderer keeps
        # serving the map this world was built on even after the template file is
        # edited — the same decoupling the config freeze gives the engine. Configs
        # with no map art (render_map None) simply have nothing to freeze.
        render_map = world_config.render_map()
        if render_map is not None:
            await self._container.snapshot.save_world_map(world_id, render_map)
            await self._freeze_map_assets(world_id, world_config, render_map)
        # The cast is frozen on the same terms as the ground it walks on: a body is
        # dressed for the map's period, so restyling that art later would restyle
        # every past world's REPLAY along with it.
        await self._freeze_character_assets(world_id, world_config)
        step_zero_snapshot = await self._save_initial_snapshot(
            world_id=world_id,
            theme=theme,
            analysis=analysis,
            agent_definitions=list(agent_definitions),
            agents=agents,
            environment=environment,
            directory=directory,
            clock_config=clock_config,
        )

        logger.info(
            "world_initialized",
            extra={
                "world_id": world_id,
                "world_name": analysis.world_name,
                "agent_count": len(agents),
            },
        )

        return World(
            world_id=world_id,
            theme=theme,
            analysis=analysis,
            agent_definitions=list(agent_definitions),
            agents=agents,
            world_config=world_config,
            clock_config=clock_config,
            environment=environment,
            step_zero_snapshot=step_zero_snapshot,
            directory=directory,
        )

    def _instantiate_agent(
        self, *, world_id: str, definition: AgentDefinition, clock_config: WorldTimeConfig = WorldTimeConfig()
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
        is_main = definition.tier == AgentTier.MAIN
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
        state = agent.personality.state
        initial_state = AgentState(
            world_id=agent.world_id,
            agent_id=agent.agent_id,
            updated_step=0,
            current_emotion=state.emotion.primary,
            emotion_intensity=state.emotion.intensity,
            emotion_valence=state.emotion.valence,
            emotion_triggered_by=state.emotion.triggered_by,
            active_needs=list(state.active_needs),
            dominant_need=state.dominant_need,
            long_term_goals=list(state.long_term_goals),
            short_term_goals=list(state.short_term_goals),
            current_location=state.current_location,
            activity_status=state.activity_status.value,
            activity_target=state.activity_target,
            action_status=state.action_status.value,
            current_action=state.current_action,
            action_remaining_steps=state.action_remaining_steps,
            last_action=state.last_action,
            last_action_result=state.last_action_result,
            last_action_succeeded=state.last_action_succeeded,
            need_intensities=dict(state.need_intensities),
            vitality=state.vitality,
            condition=condition_to_dict(state.condition),
            # 0 at world init = "never decided" → the scheduler's cadence gate treats every agent
            # as starved on step 1, so a fresh world does not open on a step where nobody thought.
            last_decision_step=state.last_decision_step,
        )
        await agent.agent_store.save_agent_state(agent.world_id, agent.agent_id, initial_state)
        await agent.agent_store.save_initial_agent_state(agent.world_id, agent.agent_id, initial_state)

    async def _initialize_relations(
        self,
        *,
        world_id: str,
        agents: dict[str, Agent],
        analysis: ThemeAnalysis,
    ) -> None:
        name_to_id = {
            agent.personality.soul.name: agent_id
            for agent_id, agent in agents.items()
        }
        # Each RelationSeed is bilateral by design — emit both directions.
        # Forward uses (trust/affection/labels); reverse uses (reverse_trust/
        # reverse_affection/reverse_labels), each defaulting to the forward
        # value when not explicitly set (handled by RelationSeed accessors).
        tasks = []
        emitted: set[tuple[str, str]] = set()
        for relation in analysis.initial_relations:
            source_agent_id = name_to_id.get(relation.source_name)
            target_agent_id = name_to_id.get(relation.target_name)
            if not source_agent_id or not target_agent_id:
                logger.warning(
                    "relation_seed_name_unresolved",
                    extra={
                        "world_id": world_id,
                        "source_name": relation.source_name,
                        "target_name": relation.target_name,
                    },
                )
                continue
            for from_id, to_id, trust, affection, labels, to_name in (
                (source_agent_id, target_agent_id,
                 relation.trust, relation.affection, list(relation.labels), relation.target_name),
                (target_agent_id, source_agent_id,
                 relation.effective_reverse_trust(),
                 relation.effective_reverse_affection(),
                 relation.effective_reverse_labels(), relation.source_name),
            ):
                if (from_id, to_id) in emitted:
                    continue
                emitted.add((from_id, to_id))
                initial_relation = AgentRelation(
                    world_id=world_id,
                    from_id=from_id,
                    to_id=to_id,
                    trust_objective=trust,
                    affection_objective=affection,
                    updated_step=0,
                    labels=labels,
                    to_name=to_name,
                    # to_gender shares to_name's source and lifetime: people in seed relations are usually not
                    # present, so this record is their only reference in the prompt, and a name alone still leaves the
                    # LLM guessing gender. to_id always comes from name_to_id.values(), i.e. the agents' keys, so it's read directly without a fallback.
                    to_gender=agents[to_id].personality.soul.gender,
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
        historical_events: Sequence[HistoricalEventSeed],
    ) -> None:
        name_to_id = {
            agent.personality.soul.name: agent_id
            for agent_id, agent in agents.items()
        }
        tasks = []
        for event in historical_events:
            resolved: dict[str, str] = {}
            for name in event.related_figures:
                agent_id = name_to_id.get(name)
                if agent_id is None:
                    logger.warning(
                        "historical_event_figure_unresolved",
                        extra={"event": event.event[:100], "figure_name": name},
                    )
                else:
                    resolved[name] = agent_id
            related_ids = list(resolved.values())
            if not related_ids:
                logger.warning(
                    "historical_event_dropped",
                    extra={"event": event.event[:100], "related_figures": event.related_figures},
                )
                continue
            created_step = -abs(event.step_offset) if event.step_offset != 0 else -1
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

    async def restore(self, world_id: str) -> World:
        """Restore a previously built world from persisted storage.

        Loads the world's persisted config asset and the step-0 manifest,
        reconstructs agents from their definitions, applies the latest
        AgentState from agent_store, and returns a runtime-ready World.
        The code template in the container is never consulted here.
        """
        config_data = await self._container.snapshot.load_world_config(world_id)
        if config_data is None:
            raise ValueError(
                f"No persisted world config found for world {world_id!r}; "
                "it must be rebuilt with the current build pipeline."
            )
        active_world_config = StoredWorldConfig(config_data)
        step_zero = await self._container.snapshot.load(world_id, 0)
        if step_zero is None:
            raise ValueError(f"No step-0 snapshot found for world {world_id!r}")

        manifest = step_zero.metadata.get("manifest")
        if manifest is None:
            raise ValueError(
                f"Step-0 snapshot for world {world_id!r} has no manifest; "
                "it was built before build/run separation was supported."
            )

        theme = str(manifest.get("theme", ""))
        analysis = ThemeAnalysis.from_dict(manifest["analysis"])
        agent_definitions = [
            AgentDefinition.from_dict(d) for d in manifest["agent_definitions"]
        ]

        environment = EnvironmentSystem(active_world_config, max_npcs=self._max_npcs)
        await self._instantiate_entity_seeds(analysis, environment, active_world_config)
        # Npcs aren't rebuilt here, deliberately unlike entity seeds: the npc_states payload is
        # self-contained (identity + position + condition + current errand) and _restore_environment_state
        # loads it in one go. Building them on both paths would collide on ids, and since ids are slugs of
        # names, a collision splits one person into two in the world.
        clock_config = self._clock_config_for(analysis, active_world_config)
        agents: dict[str, Agent] = {}

        # The step the world resumes at = the latest persisted snapshot. This is the
        # world's authoritative current step (not any per-agent counter, which lags for
        # idle agents); the runtime clock starts here and the next plan runs at
        # resume_step+1. reset deletes snapshots > 0 before restoring, so this is 0 there.
        resume_step = max(
            (s for s in await self._container.snapshot.list_steps(world_id) if s > 0),
            default=0,
        )

        for definition in agent_definitions:
            agent = self._instantiate_agent(
                world_id=world_id,
                definition=definition,
                clock_config=clock_config,
            )
            agents[agent.agent_id] = agent
            await agent.memory_system.ensure_collections()

            stored = await self._container.agent_store.load_agent_state(world_id, agent.agent_id)
            if stored is not None:
                _apply_stored_state(agent, stored)
                # Don't place the dead: death handling already removed them from the environment (current_location
                # isn't cleared, it only records where they fell). Placing them unconditionally would stand the
                # corpse back up in the room, visible to everyone there forever.
                # Check liveness with is_active (just derived by _apply_stored_state from vitality, the single source of truth), not a raw vitality comparison.
                if agent.is_active:
                    environment.place_agent(agent_id=agent.agent_id, location_id=stored.current_location)
                await agent.memory_system.warm_recent_memories(current_step=stored.updated_step)
            else:
                environment.place_agent(agent_id=agent.agent_id, location_id=definition.initial_location)
                await agent.memory_system.warm_recent_memories(current_step=0)

        await self._restore_environment_state(world_id, environment, resume_step)
        self._repatriate_transit_agents(agents, agent_definitions, environment)
        directory = LiveWorldDirectory.from_agents(agents, environment)

        logger.info(
            "world_restored",
            extra={
                "world_id": world_id,
                "world_name": analysis.world_name,
                "agent_count": len(agents),
            },
        )
        return World(
            world_id=world_id,
            theme=theme,
            analysis=analysis,
            agent_definitions=agent_definitions,
            agents=agents,
            world_config=active_world_config,
            clock_config=clock_config,
            environment=environment,
            step_zero_snapshot=step_zero,
            directory=directory,
            current_step=resume_step,
        )

    def _repatriate_transit_agents(
        self,
        agents: dict[str, Agent],
        agent_definitions: Sequence[AgentDefinition],
        environment: EnvironmentSystem,
    ) -> None:
        """Fix the location of agents caught mid-move in a snapshot.

        Execution state isn't persisted with snapshots (IN_PROGRESS is reset to IDLE on restore). If the
        snapshot lands mid multi-step MOVE, the agent's stored location is the IN_TRANSIT pseudo-location,
        which has no connections and no entities, so every move feasibility check fails and the agent is
        stuck forever. Matches movement's interrupt turn-back (<50%): return to the origin registered in
        the environment; if none is registered (older snapshots), fall back to the initial location.
        """
        initial_by_id = {d.agent_id: d.initial_location for d in agent_definitions}
        for agent_id, agent in agents.items():
            if environment.get_body_location(agent_id) != IN_TRANSIT:
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

    async def _restore_environment_state(
        self, world_id: str, environment: EnvironmentSystem, resume_step: int
    ) -> None:
        """Re-apply the persisted environment state onto a freshly seeded world.

        ``restore`` rebuilds the environment from the step-0 entity seeds, which leaves item
        state/location/ownership at their initial values. The resume step's snapshot records
        the live environment under ``metadata["environment"]``; without this the world's items
        silently revert to step 0 while agents carry their latest state.

        **STEP 0 IS A RESUME STEP LIKE ANY OTHER — do not skip it.** For entities the re-apply
        is a no-op, but mindless bodies have no seeds and ``restore`` deliberately doesn't
        rebuild them (ids would collide with this payload), so this call is the ONLY thing that
        puts them back. Skipping step 0 (every fresh world's first run, every reset) leaves the
        world with none: ERRAND silently out of reach.
        """
        snapshot = await self._container.snapshot.load(world_id, resume_step)
        if snapshot is None:
            return
        env_state = snapshot.metadata.get("environment")
        if isinstance(env_state, dict):
            environment.restore_state(env_state)

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
        than a perfectly frozen one, and a missing frozen asset falls back to the
        live template (exactly what every pre-freeze world already does).
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
        theme: str,
        analysis: ThemeAnalysis,
        agent_definitions: list[AgentDefinition],
        agents: dict[str, Agent],
        environment: EnvironmentSystem,
        directory: WorldDirectory,
        clock_config: WorldTimeConfig,
    ) -> WorldSnapshot:
        world_time = WorldTime.from_step(0, clock_config)
        # Mirror the runtime's observation enrichment: snapshot_agent_state is id-only
        # (agent/ can't touch the directory); the god-view resolves location_id → the
        # narrative name here so step 0 shows real homes, not the "某地" fallback.
        # (No transit at init — nobody is mid-move — so only location_name is added.)
        agent_states = {
            agent_id: snapshot_agent_state(agent) for agent_id, agent in agents.items()
        }
        for state in agent_states.values():
            state["location_name"] = directory.location_name(state["location_id"])
        snapshot = WorldSnapshot(
            world_id=world_id,
            step=0,
            timestamp=datetime.now(),
            world_time=world_time.clock_payload(),
            agent_states=agent_states,
            agent_relations=await relations_snapshot(self._container.agent_store, world_id, agents),
            pending_messages=await self._container.message_provider.peek_pending(world_id),
            events_this_step=[],
            actions_this_step=self._build_agent_summaries(agents, directory),
            metadata={
                "phase": "initialization",
                "theme": theme,
                "manifest": {
                    "theme": theme,
                    "analysis": analysis.as_dict(),
                    "agent_definitions": [defn.as_dict() for defn in agent_definitions],
                },
                "world_time": world_time.iso_label(),
                "time_label": world_time.time_label,
                "world": {
                    "world_name": analysis.world_name,
                    # Step duration: the review card shows it to the user before confirmation locks it in (once
                    # locked it holds for the whole story). It's in this narrative block rather than the manifest
                    # because the presentation layer reads it; the authoritative value stays in analysis. It's sent in
                    # seconds and the presentation layer renders it as words, so changing precision doesn't touch this chain.
                    "seconds_per_step": clock_config.seconds_per_step,
                    "era_description": analysis.era_description,
                    "core_tension": analysis.core_tension,
                    "narrative_theme": analysis.narrative_theme,
                    "narrative_pitch": analysis.narrative_pitch,
                    "figure_names": [figure.name for figure in analysis.key_figures],
                    "location_names": [location.name for location in analysis.key_locations],
                },
                "schedule": {
                    "step": 0,
                    "batches": [],
                },
                "messages": {
                    "step": 0,
                    "delivered": [],
                    "received": [],
                    "inboxes": {agent_id: [] for agent_id in agents},
                    "undelivered": [],
                },
                "broadcasts": [],
                "environment": environment.snapshot_state(),
                "character_profiles": {
                    agent_id: {
                        "name": agent.personality.soul.name,
                        "role": agent.personality.soul.role,
                        "age": agent.personality.soul.age,
                        "gender": agent.personality.soul.gender,
                        "background": agent.personality.soul.background,
                        "appearance": agent.personality.soul.appearance,
                        "color": agent.personality.soul.color,
                        "core_traits": list(agent.personality.soul.core_traits),
                        "core_values": list(agent.personality.soul.core_values),
                        "self_image": agent.personality.soul.self_image,
                        "life_goal": agent.personality.soul.life_goal,
                        "secret": agent.personality.soul.secret,
                        "is_main_character": agent.is_main_character,
                    }
                    for agent_id, agent in agents.items()
                },
            },
        )
        await self._container.snapshot.save(world_id, 0, snapshot)
        return snapshot

    def _resolve_initial_location(
        self, definition: AgentDefinition, world_config: WorldConfig
    ) -> str:
        resolved = world_config.resolve_location_id(definition.initial_location)
        if resolved is None:
            available = sorted(world_config.get_places())
            raise ValueError(
                "Unknown initial location "
                f"{definition.initial_location!r} for agent "
                f"{definition.agent_id!r} ({definition.name!r}). "
                f"Available locations: {available}"
            )
        return resolved

    def _instantiate_npc_seeds(
        self, analysis: ThemeAnalysis, environment: EnvironmentSystem, world_config: WorldConfig
    ) -> None:
        """Put the theme's mindless bodies into the world. Sibling of the entity seeds.

        If placement can't be resolved, fall back to the first location with a warning, the same fallback
        as entity seeds: someone standing in the wrong place is still usable, someone absent from the
        world isn't. Slots and quotas belong to ``spawn_npc`` (the single point where they come into being).
        """
        location_ids = sorted(world_config.get_places())
        default_location = location_ids[0] if location_ids else None

        for seed in analysis.npc_seeds:
            location_id = world_config.resolve_location_id(seed.location_name) or default_location
            if location_id is None:
                logger.warning("npc_seed_no_location", extra={"npc_name": seed.name})
                continue
            environment.spawn_npc(seed, location_id=location_id)

    async def _instantiate_entity_seeds(
        self, analysis: ThemeAnalysis, environment: EnvironmentSystem, world_config: WorldConfig
    ) -> None:
        location_ids = sorted(world_config.get_places())
        default_location = location_ids[0] if location_ids else None

        for seed in analysis.world_entity_seeds:
            location_id = world_config.resolve_location_id(seed.location_name)
            if location_id is None:
                location_id = default_location
                logger.warning(
                    "entity_seed_location_unresolvable",
                    extra={
                        "seed_name": seed.name,
                        "requested": seed.location_name,
                        "fallback": location_id,
                    },
                )
            if location_id is None:
                logger.warning(
                    "entity_seed_skipped_no_location",
                    extra={"seed_name": seed.name},
                )
                continue
            entity_id = seed.entity_id
            entity = WorldEntity(
                entity_id=entity_id,
                name=seed.name,
                entity_type=WorldEntityType(seed.entity_type),
                state=seed.initial_state,
                description=seed.description,
                presence_ref=location_id,   # presence defaults to AT_LOCATION
                is_takeable=seed.is_takeable,
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

    def _build_agent_summaries(
        self, agents: dict[str, Agent], directory: WorldDirectory
    ) -> list[dict[str, object]]:
        summaries: list[dict[str, object]] = []
        for agent in agents.values():
            state = agent.personality.state
            # Narrative text carries the location NAME, never the raw id (id doesn't
            # cross into the narrative layer). action_description is name-free — the
            # observer already shows who, so it isn't doubled.
            loc_name = directory.location_name(state.current_location)
            summaries.append(
                {
                    "agent_id": agent.agent_id,
                    "agent_name": agent.personality.soul.name,
                    "is_main_character": agent.is_main_character,
                    "location_id": state.current_location,
                    "emotion": {
                        "primary": state.emotion.primary,
                        "intensity": state.emotion.intensity,
                        "valence": state.emotion.valence,
                    },
                    "dominant_need": state.dominant_need,
                    "long_term_goals": list(state.long_term_goals),
                    "short_term_goals": list(state.short_term_goals),
                    "summary": f"{agent.personality.soul.name} 在 {loc_name} 就位。",
                    "action_description": f"在 {loc_name} 就位",
                    "outcome": "初始化完成",
                    "succeeded": True,
                    "phase": "initialization",
                }
            )
        return summaries

    def _clock_config_for(
        self,
        analysis: ThemeAnalysis,
        world_config: WorldConfig,
    ) -> WorldTimeConfig:
        runtime_context = world_config.to_runtime_context()
        time_config = dict(analysis.world_time_config)
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
            # frozen: analysis is persisted with the step-0 manifest and restore reads the same copy back, so
            # a world's time pricing never drifts with deployment config, and no second frozen copy is needed.
            seconds_per_step=resolve_step_seconds(time_config.get("seconds_per_step")),
        )


def _goal_entity_from_dict(ent: dict, *, fallback_id: str) -> GoalEntity:
    """Deserialize a persisted short-term goal dict into a full GoalEntity.

    Mirrors the fields written by ``Agent.persist_state`` so lifecycle metadata
    survives a restore. ``related_need`` is coerced silently: an unknown/empty
    value degrades to None rather than raising (field-level coercion).
    """
    raw_need = ent.get("related_need")
    try:
        related_need = NeedType(raw_need) if raw_need else None
    except ValueError:
        related_need = None
    last_evaluated = ent.get("last_evaluated_step")
    # origin is coerced silently too: missing (older saves without the field) or an unknown value → COGNITIVE.
    # COGNITIVE rather than RESIDUE is the conservative choice: misreading an old plan as unfinished business would give it eviction resistance it shouldn't have.
    try:
        origin = GoalOrigin(ent["origin"]) if ent.get("origin") else GoalOrigin.COGNITIVE
    except ValueError:
        origin = GoalOrigin.COGNITIVE
    return GoalEntity(
        id=ent.get("id", fallback_id),
        text=ent.get("text", ""),
        goal_type=ent.get("goal_type", "short_term"),
        status=GoalStatus(ent["status"]) if ent.get("status") else GoalStatus.ACTIVE,
        related_need=related_need,
        created_step=int(ent.get("created_step", 0) or 0),
        due_step=_coerce_due_step(ent.get("due_step")),
        last_evaluated_step=int(last_evaluated) if last_evaluated is not None else None,
        progress_summary=str(ent.get("progress_summary", "")),
        origin=origin,
    )


def _coerce_due_step(raw: object) -> int | None:
    """The agreed time: missing / null / unrecognized values all become None. No deadline is the normal case, not a parse failure."""
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _apply_stored_state(agent: Agent, stored: AgentState) -> None:
    """Sync a loaded AgentState into an Agent's in-memory StateLayer.

    Calls personality.restore_state() to write directly into _state, bypassing
    the defensive copy returned by personality.state.
    """
    # Build goal entities: prefer rich entity dicts (carry lifecycle metadata),
    # fall back to strings. The dict branch must pass through every persisted field
    # (related_need / created_step / last_evaluated_step / progress_summary) — dropping
    # them silently resets fail-retry counts, goal age, and need linkage on every restore.
    if stored.short_term_goal_entities:
        goal_entities = [
            _goal_entity_from_dict(ent, fallback_id=f"restored-stg-{i}")
            for i, ent in enumerate(stored.short_term_goal_entities)
        ]
    elif stored.short_term_goals:
        goal_entities = [
            GoalEntity(
                id=f"restored-stg-{i}",
                text=g,
                goal_type="short_term",
                status=GoalStatus.ACTIVE,
                related_need=None,
            )
            for i, g in enumerate(stored.short_term_goals)
        ]
    else:
        goal_entities = []

    restored = StateLayer(
        agent_id=agent.agent_id,
        step=stored.updated_step,
        emotion=EmotionState(
            primary=parse_emotion_type(stored.current_emotion),
            intensity=stored.emotion_intensity,
            valence=stored.emotion_valence,
            triggered_by=stored.emotion_triggered_by,
        ),
        active_needs=list(stored.active_needs),
        dominant_need=stored.dominant_need,
        long_term_goals=list(stored.long_term_goals),
        short_term_goals=list(stored.short_term_goals),
        current_location=stored.current_location,
        activity_status=(
            AgentActivityStatus.IDLE
            if stored.action_status == ActionStatus.IN_PROGRESS.value
            else AgentActivityStatus(stored.activity_status)
        ),
        activity_target=(
            None
            if stored.action_status == ActionStatus.IN_PROGRESS.value
            else stored.activity_target
        ),
        action_status=(
            ActionStatus.IDLE
            if stored.action_status == ActionStatus.IN_PROGRESS.value
            else ActionStatus(stored.action_status)
        ),
        current_action=(
            None
            if stored.action_status == ActionStatus.IN_PROGRESS.value
            else stored.current_action
        ),
        action_remaining_steps=(
            0
            if stored.action_status == ActionStatus.IN_PROGRESS.value
            else stored.action_remaining_steps
        ),
        last_action=stored.last_action,
        last_action_result=stored.last_action_result,
        last_action_succeeded=stored.last_action_succeeded,
        short_term_goal_entities=goal_entities,
        # Long-term goals persist as text; restore rebuilds the entities from text (related_need left None, as at build).
        long_term_goal_entities=[
            text_to_long_term_goal_entity(t) for t in stored.long_term_goals
        ],
        need_intensities=dict(stored.need_intensities),
        vitality=stored.vitality,
        # Stored as a dict (see the AgentState.condition comment) and rebuilt into a BodyCondition here.
        # This is the only rehydrate point, like _goal_entity_from_dict.
        condition=condition_from_dict(stored.condition),
        # Without restoring it, everyone's starvation resets to zero after restore and they all flood the decision loop next step (the cadence gate is bypassed for a beat).
        last_decision_step=stored.last_decision_step,
    )
    agent.personality.restore_state(restored)
    # vitality is the single source of truth for alive/dead: restore is_active from it, or the dead come
    # back to life on restart (default True) → decay kills them again next step → duplicate death notices. is_active isn't persisted separately.
    agent.set_active(stored.vitality > 0.0)
    if stored.action_status == ActionStatus.IN_PROGRESS.value:
        logger.info(
            "in_progress_action_cleared",
            extra={
                "agent_id": agent.agent_id,
                "cleared_action": stored.current_action,
            },
        )
