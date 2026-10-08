"""World-building pipeline for Asamana."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence
from uuid import uuid4

from core.container import Container
from core.context import get_log_context, set_log_context
from core.interfaces.trace import Stage
from core.logging import get_logger
from core.interfaces.world_config import WorldConfig

from world.initializer import WorldInitializer
from world.models import (
    AgentDefinition, HistoricalEventSeed, LocationSeed, ThemeAnalysis, World, WorldEntitySeed,
)
from world.builders.agent_generator import AgentGenerator
from world.builders.cast_designer import CastDesigner
from world.builders.template_selector import TemplateSelector
from world.builders.theme_analyzer import ThemeAnalyzer

logger = get_logger(__name__)


class WorldBuilder:
    """Facade that coordinates the full one-shot world build."""

    def __init__(self, container: Container) -> None:
        self._container = container
        self._template_selector = TemplateSelector(container.llm_router)
        self.theme_analyzer = ThemeAnalyzer(container.llm_router)
        self.agent_generator = AgentGenerator(container.llm_router)
        self._cast_designer = CastDesigner(container.llm_router)

    async def build(
        self,
        theme: str,
        *,
        world_id: str | None = None,
        world_configs: Sequence[WorldConfig],
        min_agents: int | None = None,
        max_agents: int | None = None,
        max_npcs: int = 0,
        on_phase: Callable[[str, Any], None] | None = None,
    ) -> World:
        """Build and initialize a runtime-ready world.

        ``world_configs`` is the set of maps this world MAY be built on; the
        builder picks the one the theme belongs on (see ``TemplateSelector``).
        Passing exactly one means "this map, no choice"; passing none is an error
        (there is no default map). Choosing comes first because ``ThemeAnalyzer``
        places the whole cast into that map's locations.

        ``on_phase`` is an optional tuning observation hook (see ``tuning/``),
        invoked with ``(phase_name, product)`` after each build sub-step. It must
        not mutate products, and any exception it raises is the caller's to
        contain; production leaves it ``None``.
        """

        candidates = list(world_configs)
        resolved_world_id = world_id or str(uuid4())
        previous_context = get_log_context()
        # WORLD_INIT covers the whole build: theme analysis, cast design, agent
        # generation, and initial goal generation all run within this context and
        # inherit the stage (and world_id) for observability.
        set_log_context(world_id=resolved_world_id, stage=Stage.WORLD_INIT.value)
        try:
            active_world_config = await self._template_selector.select(theme, candidates)
            if on_phase is not None:
                # The chosen map's runtime context, not the WorldConfig itself —
                # the hook's products get serialized, and a config object is not.
                on_phase(
                    "world_build.template_selection",
                    active_world_config.to_runtime_context(),
                )
            analysis = await self.theme_analyzer.analyze(
                theme,
                world_context=active_world_config.to_runtime_context(),
                min_agents=min_agents,
                max_agents=max_agents,
                max_npcs=max_npcs,
                available_locations=_world_config_locations(active_world_config),
            )
            _canonicalize_analysis_locations(analysis, active_world_config)
            _drop_seeds_shadowing_locations(analysis, active_world_config)
            _purge_orphan_references(analysis)
            if on_phase is not None:
                on_phase("world_build.theme_analysis", analysis)
            cast_design = await self._cast_designer.design(analysis)
            logger.info(
                "cast_design_complete",
                extra={
                    "world_id": resolved_world_id,
                    "figure_count": len(cast_design.roles),
                },
            )
            if on_phase is not None:
                on_phase("world_build.cast_design", cast_design)
            agent_definitions = await self.agent_generator.generate(analysis, cast_design)
            _canonicalize_agent_locations(agent_definitions, active_world_config)
            if on_phase is not None:
                on_phase("world_build.agent_generation", agent_definitions)
            initializer = WorldInitializer(self._container, max_npcs=max_npcs)
            world = await initializer.initialize(
                theme=theme,
                analysis=analysis,
                agent_definitions=agent_definitions,
                world_id=resolved_world_id,
                world_config=active_world_config,
            )
            if on_phase is not None:
                on_phase("world_build.initialization", world)
            logger.info(
                "world_build_completed",
                extra={
                    "world_id": resolved_world_id,
                    "world_name": world.analysis.world_name,
                    "agent_count": len(world.agents),
                },
            )
            return world
        finally:
            set_log_context(**previous_context)

def _purge_orphan_references(analysis: ThemeAnalysis) -> None:
    """Remove relations and event figure-refs that name truncated figures."""

    retained = {f.name for f in analysis.key_figures}

    before_relations = len(analysis.initial_relations)
    analysis.initial_relations = [
        r for r in analysis.initial_relations
        if r.source_name in retained and r.target_name in retained
    ]
    dropped_relations = before_relations - len(analysis.initial_relations)

    cleaned_events: list[HistoricalEventSeed] = []
    dropped_events = 0
    for ev in analysis.historical_events:
        if not ev.related_figures:
            cleaned_events.append(ev)
            continue
        kept = [n for n in ev.related_figures if n in retained]
        if kept:
            cleaned_events.append(dataclasses.replace(ev, related_figures=kept))
        else:
            dropped_events += 1
    analysis.historical_events = cleaned_events

    if dropped_relations or dropped_events:
        logger.debug(
            "orphan_references_purged",
            extra={
                "dropped_relations": dropped_relations,
                "dropped_events": dropped_events,
            },
        )



def _world_config_locations(world_config: WorldConfig) -> list[LocationSeed]:
    """Every place on this map, presented by its name (falling back to the id only if unnamed).

    This list goes into build-time prompts (entity placement, agents' starting locations), so it
    must carry names. An id is a code-layer coordinate, and feeding it to the LLM is a layer leak:
    the model picks a spot by `xuanwu_gate` and that id then lands verbatim in the world asset and
    in post-hoc audit text. Mapping a name back to an id is ``WorldConfig.resolve_location_id``'s
    job (it accepts ids, aliases and display names), so the upward side has no reason to use ids
    as names.
    """
    return [
        LocationSeed(
            name=str(getattr(place, "name", "") or place_id),
            description=str(getattr(place, "description", "")),
        )
        for place_id, place in world_config.get_places().items()
    ]


def _drop_seeds_shadowing_locations(
    analysis: ThemeAnalysis,
    world_config: WorldConfig,
) -> None:
    """Drop world_entity_seeds whose name is already a location on the map.

    The map is the single source of truth for locations. The LLM sometimes generates an entity
    seed for an existing location (usually a landmark, e.g. "玄武门"), so the same name appears
    twice in the directory (an enterable location and a non-enterable entity), ambiguous for both
    narrative and addressing.

    Match on exact identity only (location id / display name / alias, case- and
    whitespace-insensitive). Don't use ``resolve_location_id``: its substring fallback would also
    kill legitimate landmarks like "玄武门前的石狮". No type filter: any seed sharing a location's
    name is the same ambiguity, whether it calls itself an item or a landmark.
    """
    places = world_config.get_places()
    location_ids = set(places)
    taken: set[str] = set()
    for place_id, place in places.items():
        taken.add(str(place_id).strip().lower())
        place_name = str(getattr(place, "name", "")).strip()
        if place_name:
            taken.add(place_name.lower())
    for alias, target in world_config.get_location_aliases().items():
        if target in location_ids:
            taken.add(str(alias).strip().lower())

    kept: list[WorldEntitySeed] = []
    for seed in analysis.world_entity_seeds:
        if seed.name.strip().lower() in taken:
            logger.warning(
                "entity_seed_shadows_location",
                extra={
                    "seed_name": seed.name,
                    "entity_type": seed.entity_type,
                    "location_name": seed.location_name,
                },
            )
            continue
        kept.append(seed)
    analysis.world_entity_seeds = kept


def _canonicalize_analysis_locations(
    analysis: ThemeAnalysis,
    world_config: WorldConfig,
) -> None:
    locations = _world_config_locations(world_config)
    if locations:
        analysis.key_locations = locations


def _canonicalize_agent_locations(
    agent_definitions: Sequence[AgentDefinition],
    world_config: WorldConfig,
) -> None:
    available = sorted(world_config.get_places())
    default_location = available[0] if available else "origin"
    for definition in agent_definitions:
        resolved = world_config.resolve_location_id(definition.initial_location)
        if resolved is None:
            logger.warning(
                "agent_location_unresolvable",
                extra={
                    "agent_id": definition.agent_id,
                    "agent_name": definition.name,
                    "requested": definition.initial_location,
                    "fallback": default_location,
                },
            )
            definition.initial_location = default_location
        else:
            definition.initial_location = resolved
