from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from core.container import Container
from engine.application import NarrativeApplication
from core.interfaces.llm import LLMMessage, LLMProvider, LLMResponse, LLMRouter, LLMScene
from providers.llm.mock import MockLLMProvider
from world.builder import _canonicalize_agent_locations, _purge_orphan_references
from world.builders.agent_generator import AgentGenerator
from world.builders.cast_designer import CastDesigner
from world.builders.theme_analyzer import (
    ThemeAnalyzer,
    _distinct_figures,
    _distinct_npc_seeds,
    _normalize_gender,
)
from world.initializer import WorldInitializer
from world.models import AgentDefinition, AgentTier, HistoricalEventSeed, LocationSeed, NpcSeed, RelationSeed, ThemeAnalysis, ThemeFigure
from agent.personality import EmotionState, SoulLayer
from worlds.tiled import TiledWorldConfig


class FailingLLMProvider(LLMProvider):
    """Always raises RuntimeError — used to test fallback paths."""

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        raise RuntimeError("LLM unavailable")

    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        raise RuntimeError("LLM unavailable")
        yield  # make this an async generator


class TrackingLLMProvider(LLMProvider):
    def __init__(self) -> None:
        self.active_calls = 0
        self.max_active_calls = 0

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            await asyncio.sleep(0.01)
            return LLMResponse(content="{}", input_tokens=0, output_tokens=0, model="tracking")
        finally:
            self.active_calls -= 1

    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        response = await self.complete(messages, temperature=temperature)
        yield response.content


def _analysis() -> ThemeAnalysis:
    return ThemeAnalysis(
        theme_input="audit theme",
        world_name="Audit World",
        era_description="A compact audit world.",
        core_tension="A and B want different outcomes; what each will choose is open.",
        narrative_theme="The weight of unspoken positions.",
        narrative_pitch="A and B are about to choose; what each chooses is undecided.",
        world_time_config={"era_name": "Audit Era", "start_month": 1, "start_day": 1, "start_hour": 6},
        key_figures=[
            ThemeFigure("A", role="leader", importance="main", brief="Acts first."),
            ThemeFigure("B", role="rival", importance="main", brief="Answers pressure."),
            ThemeFigure("C", role="witness", importance="background", brief="Watches closely."),
        ],
        initial_relations=[
            RelationSeed("A", "B", trust=0.2, affection=-0.4, labels=["one-way suspicion"]),
        ],
        historical_events=[
            HistoricalEventSeed(
                event="A and B disagreed before the opening step.",
                step_offset=-24,
                related_figures=["A", "B"],
                importance=0.72,
            )
        ],
        key_locations=[LocationSeed("taiji_palace", "Opening location.")],
    )


@pytest.fixture()
def container(test_config):
    """Container with PERSONA_GENERATION returning valid JSON for world-builder tests."""
    c = Container.from_config(test_config)
    providers = {scene: c.llm_router.get(scene) for scene in LLMScene}
    providers[LLMScene.PERSONA_GENERATION] = MockLLMProvider(fixed_response="{}")
    c.llm_router = LLMRouter(providers)
    return c


@pytest.mark.asyncio
async def test_agent_generator_runs_independent_agents_concurrently() -> None:
    provider = TrackingLLMProvider()
    router = LLMRouter({scene: provider for scene in LLMScene})

    definitions = await AgentGenerator(router).generate(_analysis())

    assert len(definitions) == 3
    assert provider.max_active_calls > 1


@pytest.mark.asyncio
async def test_each_persona_call_is_traced_under_the_figure_it_builds(container) -> None:
    """Persona-generation traces carry who's being generated: agent_id and name. Concurrent
    generation must not cross-attribute calls."""
    from providers.trace.in_memory import InMemoryTraceSink

    sink = InMemoryTraceSink()
    providers = {scene: container.llm_router.get(scene) for scene in LLMScene}
    router = LLMRouter(providers, trace_sink=sink)

    definitions = await AgentGenerator(router).generate(_analysis())

    calls = [c for c in sink.llm_calls if c.scene == LLMScene.PERSONA_GENERATION.value]
    by_id = {d.agent_id: d.name for d in definitions}
    assert len(calls) == len(definitions)
    assert {c.agent_id for c in calls} == set(by_id)
    for c in calls:
        assert c.extra["agent_name"] == by_id[c.agent_id]


@pytest.mark.asyncio
async def test_world_initializer_creates_memory_streams_bidirectional_relations_and_factual_history(container) -> None:
    analysis = _analysis()
    definitions = await AgentGenerator(container.llm_router).generate(analysis)

    world = await WorldInitializer(container).initialize(
        theme=analysis.theme_input,
        analysis=analysis,
        agent_definitions=definitions,
        world_id="world-builder-audit",
        world_config=TiledWorldConfig(template="changan_iso"),
    )

    agent_a = world.agents["a"]
    agent_b = world.agents["b"]
    relation_a_b = await container.agent_store.load_relation(world.world_id, "a", "b")
    relation_b_a = await container.agent_store.load_relation(world.world_id, "b", "a")

    assert relation_a_b is not None
    assert relation_b_a is not None
    assert relation_b_a.trust_objective == relation_a_b.trust_objective
    assert relation_b_a.affection_objective == relation_a_b.affection_objective

    for agent in (agent_a, agent_b):
        factual_collection = f"{world.world_id}:{agent.agent_id}:memory:factual"
        experiential_collection = f"{world.world_id}:{agent.agent_id}:memory:experiential"

        factual_records = await container.vector_store.list_all(factual_collection)
        experiential_records = await container.vector_store.list_all(experiential_collection)

        assert factual_records
        assert experiential_collection in getattr(container.vector_store, "_records", {})
        assert experiential_records == []
        assert factual_records[0].payload["stored_content"] == analysis.historical_events[0].event
        assert factual_records[0].payload["triggered_by"] == "historical_init"
        assert factual_records[0].payload["decay_score"] < 1.0


def test_purge_orphan_references_removes_dropped_figures() -> None:
    analysis = ThemeAnalysis(
        theme_input="test",
        world_name="Test World",
        era_description="",
        core_tension="",
        narrative_theme="",
        narrative_pitch="",
        key_figures=[
            ThemeFigure("A", role="leader", importance="main", brief=""),
            ThemeFigure("B", role="rival", importance="main", brief=""),
        ],
        initial_relations=[
            RelationSeed("A", "B", trust=0.5, labels=["rivals"]),
            RelationSeed("A", "C", trust=0.3, labels=["orphan"]),   # C is dropped
            RelationSeed("C", "B", trust=0.4, labels=["orphan"]),   # C is dropped
        ],
        historical_events=[
            HistoricalEventSeed(event="A and B clashed.", step_offset=-3, related_figures=["A", "B"]),
            HistoricalEventSeed(event="C did something.", step_offset=-5, related_figures=["C"]),   # fully orphaned
            HistoricalEventSeed(event="A, B, C met.", step_offset=-6, related_figures=["A", "B", "C"]),  # C stripped
        ],
        key_locations=[LocationSeed("loc", "")],
    )
    _purge_orphan_references(analysis)

    assert len(analysis.initial_relations) == 1
    assert analysis.initial_relations[0].source_name == "A"
    assert analysis.initial_relations[0].target_name == "B"

    assert len(analysis.historical_events) == 2
    event_texts = [e.event for e in analysis.historical_events]
    assert "C did something." not in event_texts
    mixed_event = next(e for e in analysis.historical_events if "A, B, C met." in e.event)
    assert "C" not in mixed_event.related_figures
    assert set(mixed_event.related_figures) == {"A", "B"}


def test_theme_max_tokens_scales_quadratically_with_cast_size() -> None:
    """World-building's output budget must grow with max_agents, quadratically: the realistic upper
    bound for initial_relations is C(N,2) pairs. A fixed constant lets config's max_agents silently
    decide whether this run truncates (truncation → invalid JSON → Rule 2 raise, and world building
    fails)."""
    from world.builders.theme_analyzer import _theme_max_tokens

    assert _theme_max_tokens(7) >= 9000          # production max_agents (config.yaml)
    assert _theme_max_tokens(3) < _theme_max_tokens(7) < _theme_max_tokens(12)
    # The quadratic term dominates: doubling the cast must far more than double the budget.
    assert _theme_max_tokens(14) > _theme_max_tokens(7) * 2
    # With no cast-size constraint, don't fall to 0/tiny: the prompt gives no constraint either, and
    # the LLM still writes a full world.
    assert _theme_max_tokens(None) >= _theme_max_tokens(7)


def test_drop_seeds_shadowing_locations() -> None:
    """A location already on the map must not also get an entity seed (or the same "玄武门" is both an
    enterable location and a non-enterable item). Matching is exact identity only (id / display name
    / alias, ignoring case and whitespace); legitimate landmarks starting with a location name, like
    "玄武门前的石狮", must stay."""
    from world.builder import _drop_seeds_shadowing_locations
    from world.models import WorldEntitySeed
    from worlds.tiled import TiledWorldConfig

    world_config = TiledWorldConfig(template="changan_iso")
    analysis = ThemeAnalysis(
        theme_input="test",
        world_name="长安",
        era_description="",
        core_tension="",
        narrative_theme="",
        narrative_pitch="",
        key_figures=[],
        world_entity_seeds=[
            WorldEntitySeed(name="玄武门", entity_type="landmark", location_name="xuanwu_gate"),
            WorldEntitySeed(name="  Xuanwu_Gate ", entity_type="landmark", location_name="xuanwu_gate"),
            WorldEntitySeed(name="东宫", entity_type="item", location_name="donggong"),
            WorldEntitySeed(name="玄武门前的石狮", entity_type="landmark", location_name="xuanwu_gate"),
            WorldEntitySeed(name="高祖调兵虎符", entity_type="item", location_name="taiji_palace"),
        ],
    )

    _drop_seeds_shadowing_locations(analysis, world_config)

    assert [s.name for s in analysis.world_entity_seeds] == ["玄武门前的石狮", "高祖调兵虎符"]


@pytest.mark.asyncio
async def test_agent_generator_raises_when_llm_always_fails() -> None:
    """All agent LLM calls fail → generate() raises ValueError after retries."""
    router = LLMRouter({scene: FailingLLMProvider() for scene in LLMScene})
    with pytest.raises(ValueError, match="AgentGenerator failed"):
        await AgentGenerator(router).generate(_analysis())


def test_canonicalize_agent_locations_falls_back_on_unknown_location(container) -> None:
    definitions = [
        AgentDefinition(
            agent_id="x",
            name="X",
            tier=AgentTier.BACKGROUND,
            soul=SoulLayer(name="X", role="", agent_id="x"),
            initial_location="unknown-place-xyz",
            initial_emotion=EmotionState(),
        )
    ]
    _canonicalize_agent_locations(definitions, TiledWorldConfig(template="changan_iso"))

    available = sorted(TiledWorldConfig(template="changan_iso").get_places())
    assert definitions[0].initial_location == available[0]


def test_the_location_menu_handed_to_the_llm_carries_names_not_ids(container) -> None:
    """The location list fed to build-time prompts must be names: ids are code-layer
    coordinates; in a prompt the model picks them back verbatim, and they land in world assets and
    audit text. resolve_location_id maps names back to ids."""
    from world.builder import _world_config_locations

    world_config = TiledWorldConfig(template="changan_iso")
    seeds = _world_config_locations(world_config)
    names = {s.name for s in seeds}

    assert "玄武门" in names and "xuanwu_gate" not in names
    assert all(world_config.resolve_location_id(s.name) is not None for s in seeds)


def test_world_config_resolves_canonical_names_and_aliases(container) -> None:
    world_config = TiledWorldConfig(template="changan_iso")

    assert world_config.resolve_location_id("taiji_palace") == "taiji_palace"
    assert world_config.resolve_location_id("东宫") == "donggong"  # by name
    assert world_config.resolve_location_id("大雁塔") == "daci_en_temple"  # by alias
    assert world_config.resolve_location_id("小雁塔") == "jianfu_temple"  # by alias
    assert world_config.resolve_location_id("unknown-place") is None


@pytest.mark.asyncio
async def test_world_initializer_rejects_locations_outside_world_graph(container) -> None:
    analysis = _analysis()
    definitions = await AgentGenerator(container.llm_router).generate(analysis)
    definitions[0].initial_location = "unknown-place"

    with pytest.raises(ValueError, match="Unknown initial location"):
        await WorldInitializer(container).initialize(
            theme=analysis.theme_input,
            analysis=analysis,
            agent_definitions=definitions,
            world_id="world-builder-bad-location",
            world_config=TiledWorldConfig(template="changan_iso"),
        )


@pytest.mark.asyncio
async def test_theme_analyzer_raises_when_llm_always_fails() -> None:
    """ThemeAnalyzer must raise ValueError after retries, not silently degrade."""
    router = LLMRouter({scene: FailingLLMProvider() for scene in LLMScene})
    with pytest.raises(ValueError, match="ThemeAnalyzer failed for theme"):
        await ThemeAnalyzer(router).analyze("any theme")


# ---------------------------------------------------------------------------
# Build-time invariants: anchor injection, persistence, and runtime non-leakage.
# ---------------------------------------------------------------------------


def _spy_router_capturing_scene(scene: LLMScene, fixed_response: str) -> tuple[LLMRouter, MockLLMProvider]:
    """Build an LLMRouter that returns ``fixed_response`` and exposes the spy
    provider for the given scene so tests can inspect ``call_history``."""
    spy = MockLLMProvider(fixed_response=fixed_response)
    providers = {s: MockLLMProvider(fixed_response=fixed_response) for s in LLMScene}
    providers[scene] = spy
    return LLMRouter(providers), spy


def _prompt_text(spy: MockLLMProvider, call_index: int = 0) -> str:
    """Concatenate every message content from the call_history entry at ``call_index``."""
    return "\n".join(msg.content for msg in spy.call_history[call_index])


@pytest.mark.asyncio
async def test_cast_designer_prompt_contains_all_three_invariants() -> None:
    """CastDesigner must inject core_tension / narrative_theme / narrative_pitch."""
    import json as _json
    analysis = _analysis()
    cast_payload = _json.dumps({
        "roles": [
            {"index": i, "narrative_role": "role", "arc_summary": "arc",
             "key_relationships": []}
            for i, _ in enumerate(analysis.key_figures, 1)
        ]
    }, ensure_ascii=False)
    router, spy = _spy_router_capturing_scene(LLMScene.CAST_DESIGN, cast_payload)

    await CastDesigner(router).design(analysis)

    prompt = _prompt_text(spy)
    assert analysis.core_tension in prompt
    assert analysis.narrative_theme in prompt
    assert analysis.narrative_pitch in prompt


@pytest.mark.asyncio
async def test_cast_designer_reasons_first_and_the_reason_is_not_kept() -> None:
    """reason opens the schema so the roles follow from it; parsing never reads it."""
    import json as _json
    analysis = _analysis()
    cast_payload = _json.dumps({
        "reason": "A presses, B answers, C watches.",
        "roles": [
            {"index": i, "narrative_role": "role", "arc_summary": "arc", "key_relationships": []}
            for i, _ in enumerate(analysis.key_figures, 1)
        ],
    }, ensure_ascii=False)
    router, spy = _spy_router_capturing_scene(LLMScene.CAST_DESIGN, cast_payload)

    design = await CastDesigner(router).design(analysis)

    prompt = _prompt_text(spy)
    assert prompt.index('"reason"') < prompt.index('"roles"')
    assert len(design.roles) == len(analysis.key_figures)
    assert "A presses" not in _json.dumps(design.as_dict(), ensure_ascii=False)


@pytest.mark.asyncio
async def test_agent_generator_reasons_first_and_the_reason_is_not_kept() -> None:
    import json as _json
    analysis = _analysis()
    payload = _json.dumps({"reason": "Driven by fear of losing rank.", "core_traits": ["cautious"]})
    router, spy = _spy_router_capturing_scene(LLMScene.PERSONA_GENERATION, payload)

    definitions = await AgentGenerator(router).generate(analysis)

    prompt = _prompt_text(spy)
    assert prompt.index('"reason"') < prompt.index('"core_traits"')
    assert definitions[0].soul.core_traits == ("cautious",)
    assert "fear of losing rank" not in _json.dumps(definitions[0].as_dict(), ensure_ascii=False)


@pytest.mark.asyncio
async def test_agent_generator_prompt_contains_all_three_invariants() -> None:
    """Each per-figure AgentGenerator call must include all three invariants."""
    analysis = _analysis()
    router, spy = _spy_router_capturing_scene(LLMScene.PERSONA_GENERATION, "{}")

    await AgentGenerator(router).generate(analysis)

    assert len(spy.call_history) == len(analysis.key_figures)
    for i in range(len(analysis.key_figures)):
        prompt = _prompt_text(spy, i)
        assert analysis.core_tension in prompt
        assert analysis.narrative_theme in prompt
        assert analysis.narrative_pitch in prompt


@pytest.mark.asyncio
async def test_agent_generator_schema_does_not_ask_to_echo_identity() -> None:
    """name / role / agent_id are fixed by code: the schema doesn't ask for them back, and echoes
    are ignored."""
    analysis = _analysis()
    echo = '{"name": "改名", "role": "改写的定位", "agent_id": "agent-typo"}'
    router, spy = _spy_router_capturing_scene(LLMScene.PERSONA_GENERATION, echo)
    definitions = await AgentGenerator(router).generate(analysis)
    for i in range(len(analysis.key_figures)):
        output_block = _prompt_text(spy, i).split("【输出】", 1)[1]
        assert '"name":' not in output_block and '"agent_id":' not in output_block
        assert '"role":' not in output_block
    for figure, definition in zip(analysis.key_figures, definitions):
        assert (definition.soul.name, definition.soul.role) == (figure.name, figure.role)
        assert definition.agent_id != "agent-typo"


@pytest.mark.asyncio
async def test_agent_generator_prompt_splits_need_weight_from_intensity() -> None:
    """initial_needs guidance must separate weight (identity/lifetime) from intensity
    (opening/current), and keep the opening crisis out of weight, so a situational crisis isn't
    frozen into a permanent structural weight (the source of structural saturation)."""
    analysis = _analysis()
    router, spy = _spy_router_capturing_scene(LLMScene.PERSONA_GENERATION, "{}")
    await AgentGenerator(router).generate(analysis)
    prompt = _prompt_text(spy, 0)
    assert "weight 由核心张力 / 身份决定" in prompt
    assert "intensity 由" in prompt and "开局情境决定" in prompt
    assert "开局危机只抬高 intensity，绝不写进 weight" in prompt


@pytest.mark.asyncio
async def test_initial_snapshot_manifest_persists_invariants(container) -> None:
    """Step-0 snapshot must surface all three invariants both in
    manifest.analysis and as redundant top-level metadata.world fields."""
    analysis = _analysis()
    definitions = await AgentGenerator(container.llm_router).generate(analysis)

    world = await WorldInitializer(container).initialize(
        theme=analysis.theme_input,
        analysis=analysis,
        agent_definitions=definitions,
        world_id="world-invariants-persisted",
        world_config=TiledWorldConfig(template="changan_iso"),
    )

    snapshot = await container.snapshot.load(world.world_id, 0)
    assert snapshot is not None
    world_meta = snapshot.metadata["world"]
    assert world_meta["core_tension"] == analysis.core_tension
    assert world_meta["narrative_theme"] == analysis.narrative_theme
    assert world_meta["narrative_pitch"] == analysis.narrative_pitch

    manifest_analysis = snapshot.metadata["manifest"]["analysis"]
    assert manifest_analysis["core_tension"] == analysis.core_tension
    assert manifest_analysis["narrative_theme"] == analysis.narrative_theme
    assert manifest_analysis["narrative_pitch"] == analysis.narrative_pitch

    # Step-0 agent_states carry the god-view location_name (id→narrative name), so
    # the map/observer shows real homes at init, not the "某地" fallback.
    for state in snapshot.agent_states.values():
        assert state.get("location_id")
        assert state.get("location_name"), state.get("location_id")


@pytest.mark.asyncio
async def test_runtime_decision_prompt_does_not_leak_narrative_pitch(container, test_config) -> None:
    """Runtime agent decision prompts must remain transparent to build-time
    narrative invariants — emergence must be free of pitch / tension anchors."""
    import json as _json

    # Distinctive markers we can grep for in the decision LLM's call_history.
    marker_tension = "MARKER_TENSION_DO_NOT_LEAK_XQ8K"
    marker_theme = "MARKER_THEME_DO_NOT_LEAK_XQ8K"
    marker_pitch = "MARKER_PITCH_DO_NOT_LEAK_XQ8K"

    world_response = _json.dumps({
        "world_name": "Leak Test",
        "era_description": "Era.",
        "core_tension": marker_tension,
        "narrative_theme": marker_theme,
        "narrative_pitch": marker_pitch,
        "world_time_config": {"start_year": 1},
        "key_figures": [
            {"name": "Alpha", "role": "r1", "importance": "main", "brief": "A."},
            {"name": "Beta", "role": "r2", "importance": "main", "brief": "B."},
        ],
        "initial_relations": [
            {"from": "Alpha", "to": "Beta", "trust": 0.5, "affection": 0.0, "labels": ["相识"]},
        ],
    }, ensure_ascii=False)
    cast_response = _json.dumps({
        "roles": [
            {"index": 1, "narrative_role": "r", "arc_summary": "a",
             "key_relationships": [2]},
            {"index": 2, "narrative_role": "r", "arc_summary": "a",
             "key_relationships": [1]},
        ]
    }, ensure_ascii=False)

    decision_spy_main = MockLLMProvider(fixed_response='{"action_type":"rest"}')
    providers = {scene: MockLLMProvider(fixed_response="{}") for scene in LLMScene}
    providers[LLMScene.WORLD_BUILDING] = MockLLMProvider(fixed_response=world_response)
    providers[LLMScene.CAST_DESIGN] = MockLLMProvider(fixed_response=cast_response)
    providers[LLMScene.AGENT_DECISION_MAIN] = decision_spy_main
    container.llm_router = LLMRouter(providers)

    from engine.application import NarrativeApplication

    app = NarrativeApplication(container, test_config)
    world = await app.build_world("leak-test", template="changan_iso")
    await app.run_world(world.world_id, steps=1)

    for msgs in decision_spy_main.call_history:
        blob = "\n".join(m.content for m in msgs)
        assert marker_pitch not in blob, "narrative_pitch leaked into decision prompt"
        assert marker_tension not in blob, "core_tension leaked into decision prompt"
        assert marker_theme not in blob, "narrative_theme leaked into decision prompt"


# ---------------------------------------------------------------------------
# Scale definitions stay centralized.
# ---------------------------------------------------------------------------


def test_memory_importance_score_definition_references_thresholds() -> None:
    """SCORE_DEFINITION text must reference the same anchor thresholds used by
    agent.memory_types.importance_level() — protect against silent drift."""
    from core.prompts import MEMORY_IMPORTANCE_SCORE_DEFINITION

    for anchor in ("0.4", "0.65", "0.9"):
        assert anchor in MEMORY_IMPORTANCE_SCORE_DEFINITION, (
            f"threshold {anchor} missing from MEMORY_IMPORTANCE_SCORE_DEFINITION; "
            "if importance_level() thresholds changed the prompt must follow"
        )


def test_need_type_definition_covers_all_enum_members() -> None:
    """NEED_TYPE_DEFINITION must enumerate every NeedType value so the LLM is
    told about all five tiers — protect against silent omission when the enum
    is extended."""
    from agent.need import NeedType
    from core.prompts import NEED_TYPE_DEFINITION

    for need_type in NeedType:
        assert need_type.value in NEED_TYPE_DEFINITION, (
            f"NeedType.{need_type.name} ({need_type.value!r}) missing from "
            "NEED_TYPE_DEFINITION"
        )


def test_historical_event_importance_is_float_throughout() -> None:
    """LLM emits importance as float; build path parses, persists, and routes
    it to importance_level() correctly without string→enum conversion."""
    import json as _json
    from agent.memory_types import MemoryImportance, importance_level
    from world.builders.theme_analyzer import ThemeAnalyzer

    payload = _json.dumps({
        "world_name": "Float Importance World",
        "era_description": "An era.",
        "core_tension": "A and B want different outcomes.",
        "narrative_theme": "The weight of choice.",
        "narrative_pitch": "A and B are about to choose.",
        "world_time_config": {"start_year": 1},
        "key_figures": [
            {"name": "A", "role": "r1", "importance": "main", "brief": "b1"},
            {"name": "B", "role": "r2", "importance": "main", "brief": "b2"},
        ],
        "historical_events": [
            {
                "event": "An event happened.",
                "hours_before_start": 24,
                "related_figures": ["A", "B"],
                "importance": 0.72,
            }
        ],
    }, ensure_ascii=False)
    router = LLMRouter({scene: MockLLMProvider(fixed_response=payload) for scene in LLMScene})

    analysis = asyncio.run(ThemeAnalyzer(router).analyze("any theme"))

    assert len(analysis.historical_events) == 1
    seed = analysis.historical_events[0]
    assert isinstance(seed.importance, float)
    assert seed.importance == pytest.approx(0.72)
    # 0.72 sits in HIGH bucket (≥ 0.65, < 0.9)
    assert importance_level(seed.importance) == MemoryImportance.HIGH


def _theme_payload(**overrides) -> str:
    """A minimal but complete world_building payload; overrides merge at top level."""
    import json as _json

    return _json.dumps({
        "world_name": "Tempo World",
        "era_description": "An era.",
        "core_tension": "A and B want different outcomes.",
        "narrative_theme": "The weight of choice.",
        "narrative_pitch": "A and B are about to choose.",
        # A world has to be dated — see the start_year contract in _analysis_from_payload.
        "world_time_config": {"start_year": 9},
        "key_figures": [
            {"name": "A", "role": "r1", "importance": "main", "brief": "b1"},
            {"name": "B", "role": "r2", "importance": "main", "brief": "b2"},
        ],
        **overrides,
    }, ensure_ascii=False)


def _analyze(payload: str, **kwargs):
    from world.builders.theme_analyzer import ThemeAnalyzer

    router = LLMRouter({scene: MockLLMProvider(fixed_response=payload) for scene in LLMScene})
    return asyncio.run(ThemeAnalyzer(router).analyze("any theme", **kwargs))


@pytest.mark.parametrize("time_config", [{}, {"start_year": 0}, {"start_year": "近来"}])
def test_a_world_without_a_year_fails_the_build_rather_than_being_dated_by_default(
    time_config: dict,
) -> None:
    """The opening year has no harmless default, so if it's missing, fail and retry per Rule 2.

    Month/day/hour can fall back to neutral values (dawn on the first day of the first month) and
    the world still holds; the year can't. Maps carry no era (that's the theme's job), and an
    arbitrary fallback stamps the wrong year on every timestamp in the world; in a modern
    calendar it reads "1年9月1日". A world with the wrong era throughout is worse than a clean failure.
    """
    with pytest.raises(ValueError, match="ThemeAnalyzer failed"):
        _analyze(_theme_payload(world_time_config=time_config))


def test_two_seeds_that_would_share_an_id_are_reduced_to_one(caplog) -> None:
    """When two seeds map to the same entity_id only one can stay, and it must be reported.

    register_entity assigns unconditionally, so the later one silently replaces the earlier: the
    world is missing a catalyst item while the log says both were "built". On the content axis it's
    worse: two letters' contents become one.

    The criterion is the derived entity_id, not the name: "Letter" and "letter" differ in name
    but map to the same slug, and dedup by name misses exactly this case.
    """
    from world.builders.theme_analyzer import _distinct_entity_seeds
    from world.models import WorldEntitySeed

    def _seed(name: str) -> WorldEntitySeed:
        return WorldEntitySeed(name=name, entity_type="item")

    with caplog.at_level("WARNING", logger="world.builders.theme_analyzer"):
        kept = _distinct_entity_seeds([
            _seed("密信"), _seed("玉佩"), _seed("密信"),   # literal duplicate names
            _seed("Letter"), _seed("letter"),              # different names, same slug
        ])

    assert [x.name for x in kept] == ["密信", "玉佩", "Letter"]
    dropped = [r for r in caplog.records if r.message == "entity_seed_duplicate_id"]
    assert [r.seed_name for r in dropped] == ["密信", "letter"]


def test_the_analysis_itself_comes_back_deduped() -> None:
    """Dedup must live on the parse path: a correct deduplicator nobody calls is the same bug in
    another form.

    Closing it on the analysis side (not the initializer) keeps what the world builds and what the
    analysis says it has identical.
    """
    seeds = [
        {"name": "密信", "entity_type": "item", "location_name": ""},
        {"name": "密信", "entity_type": "item", "location_name": ""},
        {"name": "玉佩", "entity_type": "item", "location_name": ""},
    ]
    analysis = _analyze(_theme_payload(world_entity_seeds=seeds))

    assert [e.name for e in analysis.world_entity_seeds] == ["密信", "玉佩"]


def test_a_seed_knows_its_own_id() -> None:
    """The id derives from the name with a single owner: the initializer and the deduplicator must
    read the same one.

    Each building its own slugify makes two sources of truth; once they drift, dedup judges by one
    and placement builds by the other.
    """
    from world.models import WorldEntitySeed

    assert WorldEntitySeed(name="Letter", entity_type="item").entity_id == "seed_letter"
    assert (
        WorldEntitySeed(name="letter", entity_type="item").entity_id
        == WorldEntitySeed(name="Letter", entity_type="item").entity_id
    )


def test_a_missing_social_graph_or_prehistory_stays_empty_instead_of_being_invented(caplog) -> None:
    """If the relation graph or pre-history is missing, leave it empty; the world still holds, but
    nothing may be invented.

    A fallback is a different world, not a slightly worse one: neutral "acquaintance" relations
    erase structural labels like blood ties, which are never rewritten once set, and template
    pre-history would be seeded as persistent memory and recalled forever. Empty heals itself:
    relations grow at runtime.
    """
    with caplog.at_level("WARNING", logger="world.builders.theme_analyzer"):
        analysis = _analyze(_theme_payload())

    assert analysis.initial_relations == []
    assert analysis.historical_events == []
    # Everything else is produced as usual; missing these two doesn't stop the world from holding.
    assert len(analysis.key_figures) == 2
    assert analysis.core_tension
    # Silently empty is undetectable: the log must say exactly what was missing.
    missing = next(
        r for r in caplog.records if r.message == "theme_analysis_fields_missing"
    )
    assert set(missing.fields) == {"initial_relations", "historical_events"}


def test_the_analyzer_is_not_shown_the_map_s_own_name() -> None:
    """The map name stays out of ThemeAnalyzer's prompt; shown it, the model names every world
    after the map ("长安", "现代都市").

    Hiding it is more reliable than handing it over with "don't use this"; map identity belongs to
    template selection, not naming. (`template` likewise: a code-layer identifier.)
    """
    from world.builders.theme_analyzer import ThemeAnalyzer

    router = LLMRouter({scene: MockLLMProvider(fixed_response="{}") for scene in LLMScene})
    context = {
        "world_name": "长安", "era_name": "大唐", "calendar": "classical_cn",
        "start_month": 6, "start_day": 1, "start_hour": 6, "template": "changan_iso",
    }
    system, user = ThemeAnalyzer(router)._build_prompt("any theme", context)
    prompt = system + user

    assert "长安" not in prompt and "changan_iso" not in prompt
    # What's actually useful is still there: era range, calendar, opening-time anchor.
    assert "大唐" in prompt and "classical_cn" in prompt and "start_hour" in prompt


@pytest.mark.parametrize(
    ("authored_hours", "expected_hours"),
    [
        (6, 6),        # accept the given value
        (24, 24),      # the upper bound itself is valid
        (None, 2),     # missing field → default 2 hours
        (0, 2),        # below the lower bound → default (not clamped to 1)
        (48, 2),       # above the upper bound → default (not clamped to 24)
        ("两小时", 2),  # wrong type → default
    ],
)
def test_step_duration_is_authored_in_hours_and_stored_in_seconds(
    authored_hours, expected_hours
) -> None:
    """Step length is set in hours by theme analysis, then normalized and quantized to seconds at
    this LLM boundary.

    Hours are only the LLM's output unit; past the boundary world_time_config holds only
    seconds_per_step, and nothing downstream assumes "a step is a whole hour". Out-of-range /
    missing / invalid values all fall back to the default rather than clamp, and one integer
    never fails world building.
    """
    time_config = {"era_name": "E", "start_year": 1, "start_month": 1, "start_day": 1, "start_hour": 6}
    if authored_hours is not None:
        time_config["hours_per_step"] = authored_hours

    analysis = _analyze(_theme_payload(world_time_config=time_config))

    assert analysis.world_time_config["seconds_per_step"] == expected_hours * 3600
    assert "hours_per_step" not in analysis.world_time_config  # one source of truth


def test_historical_events_convert_hours_with_the_authored_step_duration() -> None:
    """Pre-history "hours ago" converts to steps using the step length just decided; both ends are
    narrative durations, the conversion is code-layer work.

    24 hours ago at 6 hours per step → 4 steps ago. Don't use an externally injected
    seconds_per_step: that would be a second source alongside the world's own step length.
    """
    analysis = _analyze(_theme_payload(
        world_time_config={"start_year": 1, "hours_per_step": 6},
        historical_events=[{
            "event": "An event happened.",
            "hours_before_start": 24,
            "related_figures": ["A", "B"],
            "importance": 0.5,
        }],
    ))

    assert [e.step_offset for e in analysis.historical_events] == [-4]


def test_world_entity_seeds_schema_excludes_narrative_meaning_hint() -> None:
    """world_entity_seeds.description must describe the entity itself, not its
    narrative significance (no "叙事意义" hint)."""
    from world.builders.theme_analyzer import _THEME_PROMPT_SCHEMA

    desc = _THEME_PROMPT_SCHEMA["world_entity_seeds"][0]["description"]
    assert "叙事意义" not in desc
    assert "entity 本身" in desc or "实体本身" in desc
    # Everyone can see the description; what's written may only go into content, or "readable only
    # once picked up" means nothing.
    assert "content" in desc


@pytest.mark.asyncio
async def test_world_initializer_persists_each_agent_state_exactly_once(container) -> None:
    analysis = _analysis()
    definitions = await AgentGenerator(container.llm_router).generate(analysis)

    agent_store = container.agent_store
    original_save = agent_store.save_agent_state
    save_call_count: dict[str, int] = {}

    async def counting_save(world_id: str, agent_id: str, state) -> None:
        save_call_count[agent_id] = save_call_count.get(agent_id, 0) + 1
        return await original_save(world_id, agent_id, state)

    agent_store.save_agent_state = counting_save  # type: ignore[method-assign]
    try:
        await WorldInitializer(container).initialize(
            theme=analysis.theme_input,
            analysis=analysis,
            agent_definitions=definitions,
            world_id="world-persist-once",
            world_config=TiledWorldConfig(template="changan_iso"),
        )
    finally:
        agent_store.save_agent_state = original_save  # type: ignore[method-assign]

    assert save_call_count, "save_agent_state was never called"
    for agent_id, count in save_call_count.items():
        assert count == 1, (
            f"agent {agent_id!r} state persisted {count} times during build; "
            "expected exactly 1 (Phase F removed the duplicate persist call)"
        )


# ---- TemplateSelector: which land a story belongs on -------------------------


class _ScriptedLLMProvider(LLMProvider):
    """Returns canned bodies in order, recording the prompts it was given."""

    def __init__(self, *bodies: str) -> None:
        self._bodies = list(bodies)
        self.calls: list[list[LLMMessage]] = []

    async def complete(self, messages, temperature=0.7, max_tokens=1000, **kwargs):
        self.calls.append(list(messages))
        body = self._bodies.pop(0) if self._bodies else "{}"
        return LLMResponse(
            content=body, model="scripted", input_tokens=0, output_tokens=0
        )

    async def stream(self, messages, temperature=0.7):
        raise NotImplementedError


def _selector(provider: LLMProvider):
    from world.builders.template_selector import TemplateSelector

    return TemplateSelector(LLMRouter({scene: provider for scene in LLMScene}))


def _maps(*templates: str):
    from worlds.tiled import TiledWorldConfig

    return [TiledWorldConfig(template=t) for t in templates]


@pytest.mark.asyncio
async def test_template_selector_picks_by_index_not_by_name() -> None:
    """The LLM returns a 1-based index into the menu it was shown, never an id.

    Opaque identifiers are what the model hallucinates; an integer is bounded and
    checkable (CLAUDE.md "LLM Indexed Reference Pattern"). The menu it sees must
    describe each map well enough to choose between — name, era and description —
    and must carry the theme it is choosing for.
    """
    provider = _ScriptedLLMProvider('{"reason": "现代都市题材", "index": 2}')
    candidates = _maps("changan_iso", "metro")

    chosen = await _selector(provider).select("都市白领的一天", candidates)

    assert chosen is candidates[1]
    user_prompt = provider.calls[0][1].content
    assert "#1" in user_prompt and "#2" in user_prompt
    assert "都市白领的一天" in user_prompt
    for candidate in candidates:
        assert candidate.get_world_description()[:12] in user_prompt


@pytest.mark.asyncio
async def test_template_selector_does_not_call_the_llm_when_there_is_no_choice() -> None:
    """One candidate is already an answer — spending a call to confirm it is waste."""
    provider = _ScriptedLLMProvider()
    only = _maps("metro")

    assert await _selector(provider).select("任意主题", only) is only[0]
    assert provider.calls == []


@pytest.mark.asyncio
async def test_template_selector_raises_rather_than_guessing_a_map() -> None:
    """A build that cannot choose fails; it does not quietly settle for a map.

    Rule 2: a mismatched map is the wrong substrate under every later step and cannot heal; the
    user retrying is cheaper.
    """
    selector = _selector(FailingLLMProvider())
    with pytest.raises(ValueError, match="TemplateSelector failed"):
        await selector.select("任意主题", _maps("changan_iso", "metro"))

    # An unusable answer is the same case — a number that indexes nothing cannot
    # be rounded into a choice.
    garbage = _ScriptedLLMProvider('{"reason": "…", "index": 99}', "not json at all")
    with pytest.raises(ValueError, match="TemplateSelector failed"):
        await _selector(garbage).select("任意主题", _maps("changan_iso", "metro"))
    assert len(garbage.calls) == 2, "a transient bad answer deserves one retry"


@pytest.mark.asyncio
async def test_template_selector_rejects_an_empty_menu() -> None:
    """No maps installed is a deployment error, not something to work around."""
    with pytest.raises(ValueError, match="No world templates"):
        await _selector(_ScriptedLLMProvider()).select("任意主题", [])


@pytest.mark.asyncio
async def test_a_build_with_no_map_fails_instead_of_falling_back(container) -> None:
    """No map = world building fails, not "just use the default one".

    A deployment-level default map would launder two configuration errors (no maps installed, the
    caller not listing candidates) into a world on a base nobody chose, which can't heal.
    """
    from world.builder import WorldBuilder

    with pytest.raises(ValueError, match="No world templates"):
        await WorldBuilder(container).build(theme="任意主题", world_configs=[])


def test_a_cast_below_the_floor_fails_the_build_rather_than_being_padded() -> None:
    """Too few characters = world building fails, not pad with a few extras.

    Narrative grows between characters: a one-person world has no relations, no information
    asymmetry, no one to talk to. Characters padded in by code have nothing to do with core_tension
    and only dilute it (Rule 2).
    """
    solo = _theme_payload(key_figures=[{"name": "A", "role": "r", "importance": "main", "brief": "b"}])
    with pytest.raises(ValueError, match="ThemeAnalyzer failed for theme"):
        _analyze(solo, min_agents=3)

    # Reaching the minimum returns normally; the minimum is a threshold, not a quota.
    assert len(_analyze(_theme_payload(), min_agents=2).key_figures) == 2


def test_the_cast_size_constraint_names_both_bounds_and_forbids_padding() -> None:
    """The minimum must appear in the prompt together with "don't invent people to fill the count".

    Saying only "at least N", the model invents a few to hit the number when short, and characters
    unrelated to the tension are worse than a smaller cast.
    """
    from world.builders.theme_analyzer import _cast_size_constraint

    both = _cast_size_constraint(3, 8)
    assert "3-8" in both and "凑数" in both
    assert "不超过 8" in _cast_size_constraint(None, 8)
    assert "不少于 3" in _cast_size_constraint(3, None)
    assert _cast_size_constraint(None, None) == ""


def test_a_cast_above_the_ceiling_fails_the_build_rather_than_being_trimmed() -> None:
    """Exceeding the maximum also means retry, not cutting the extras.

    A cast isn't a list to trim at will: relations, historical events and cast_design are all
    written around the whole group. Cut a few afterwards and the remaining world isn't the one the
    model wrote (dangling references get cleaned up, but the tension is missing a piece).
    """
    crowd = _theme_payload(key_figures=[
        {"name": n, "role": "r", "importance": "main", "brief": "b"} for n in "ABCD"
    ])
    with pytest.raises(ValueError, match="ThemeAnalyzer failed for theme"):
        _analyze(crowd, max_agents=3)

    assert len(_analyze(crowd, min_agents=2, max_agents=4).key_figures) == 4


def test_an_unsatisfiable_cast_range_raises_before_any_llm_call() -> None:
    """min > max is unsolvable: retrying can't change it, so fail on the spot without burning two
    build calls (Rule 3)."""
    from world.builders.theme_analyzer import ThemeAnalyzer

    provider = _ScriptedLLMProvider(_theme_payload())
    router = LLMRouter({scene: provider for scene in LLMScene})
    with pytest.raises(ValueError, match="Unsatisfiable cast bounds"):
        asyncio.run(ThemeAnalyzer(router).analyze("any theme", min_agents=5, max_agents=3))
    assert provider.calls == []


def test_one_body_per_appellation_and_none_wearing_a_key_figure_s_name():
    """One label refers to one group: drop it if it collides with a key_figure, and keep only one of
    its own duplicates."""
    seeds = [
        NpcSeed(name="李世民", gender="男", age=27, description="", location_name=""),
        NpcSeed(name="脚夫", gender="男", age=30, description="跑得快", location_name=""),
        NpcSeed(name="脚夫", gender="男", age=41, description="认得路", location_name=""),
    ]
    kept = _distinct_npc_seeds(seeds, {"李世民", "李建成"})
    assert [seed.name for seed in kept] == ["脚夫"]
    assert kept[0].description == "跑得快"  # first of a repeated appellation wins


def test_a_key_figure_named_twice_becomes_one_agent():
    """Two agents with the same name can't be told apart downstream: keep only the first."""
    figures = [
        ThemeFigure("李建成", role="太子", importance="main", brief="先写的"),
        ThemeFigure("李世民", role="秦王", importance="main", brief=""),
        ThemeFigure("李建成", role="太子", importance="main", brief="重写的"),
    ]
    kept = _distinct_figures(figures)
    assert [f.name for f in kept] == ["李建成", "李世民"]
    assert kept[0].brief == "先写的"


def test_the_mindless_tier_is_named_by_appellation_not_by_person():
    system, _user = ThemeAnalyzer(
        LLMRouter({scene: MockLLMProvider() for scene in LLMScene})
    )._build_prompt("任意主题", {}, max_npcs=3)
    # This asserts the prompt's exact wording: this tier's positioning lives only in that text,
    # so if the wording changes this should change with it, rather than the rule silently
    # disappearing.
    assert "key_figures" in system
    assert "称谓" in system              # name means "which kind of person"
    assert "不写具体的人名" in system


@pytest.mark.asyncio
async def test_agent_generator_warns_when_traits_or_values_missing(container, caplog) -> None:
    """Missing core_traits/core_values: no retry, left empty, one warning per character."""
    with caplog.at_level("WARNING", logger="world.builders.agent_generator"):
        definitions = await AgentGenerator(container.llm_router).generate(_analysis())

    assert all(not d.soul.core_traits and not d.soul.core_values for d in definitions)
    missing = [r for r in caplog.records if r.getMessage() == "agent_definition_fields_missing"]
    assert len(missing) == len(definitions)
    assert all(r.fields == ["core_traits", "core_values"] for r in missing)


def test_a_location_name_from_the_llm_reaches_the_resolver_untouched() -> None:
    """The LLM gets location names and returns names, passed through untouched to
    ``resolve_location_id``.

    "Text → location id" may have only one owner. A second id-style parser on the generation side
    (regex + slugify) grinds "东宫" into ``agent_13d5cce4f4``, which the real resolver can no longer
    recognize, so every character falls back to the same default location with just one warning
    each, even when the LLM named every location correctly.
    """
    import dataclasses

    from world.builders.agent_generator import AgentGenerator
    from world.models import LocationSeed, ThemeFigure

    analysis = dataclasses.replace(
        _analysis(), key_locations=[LocationSeed(name="玄武门", description="宫城北门")])
    figure = ThemeFigure(name="李建成", role="太子", importance="main", brief="", age=37, gender="男")
    definition = AgentGenerator.__new__(AgentGenerator)._definition_from_payload(  # noqa: SLF001
        figure, analysis, {"initial_location": "东宫"}, "agent-x")
    assert definition.initial_location == "东宫"

    # If the LLM gives none, fall back to the first of key_locations, also by name, not id.
    fallback = AgentGenerator.__new__(AgentGenerator)._definition_from_payload(  # noqa: SLF001
        figure, analysis, {}, "agent-x")
    assert fallback.initial_location == "玄武门"


def test_entity_seed_content_is_parsed_and_absent_content_stays_empty() -> None:
    analysis = _analyze(_theme_payload(world_entity_seeds=[
        {"name": "石碑", "entity_type": "landmark", "location_name": "x", "content": "敢有擅入者斩"},
        {"name": "佩刀", "entity_type": "item", "location_name": "x"},
    ]))
    assert [s.content for s in analysis.world_entity_seeds] == ["敢有擅入者斩", ""]


# --- Build-time write failures must propagate (Rule 2/6) -------------------------------------
#
# Relations and pre-history are written once by the LLM for this world and never come again: initial
# relation labels are never rewritten (rule 3), and pre-history has no second write. So these two
# rank with the initial state write; logging an error and letting build report ready would have the
# user confirm a world permanently missing content.


@pytest.mark.asyncio
async def test_build_raises_when_a_relation_write_fails(mock_build_container, test_config) -> None:
    store = mock_build_container.agent_store
    original = store.save_relation
    seen = {"n": 0}

    async def flaky_save_relation(relation):  # type: ignore[no-untyped-def]
        seen["n"] += 1
        if seen["n"] == 2:
            raise OSError("disk full")
        await original(relation)

    store.save_relation = flaky_save_relation  # type: ignore[method-assign]

    app = NarrativeApplication(mock_build_container, test_config)
    with pytest.raises(OSError):
        await app.build_world("宫廷权谋", template="changan_iso")


@pytest.mark.asyncio
async def test_build_raises_when_a_historical_memory_write_fails(
    mock_build_container, test_config,
) -> None:
    vectors = mock_build_container.vector_store
    original = vectors.upsert
    seen = {"n": 0}

    async def flaky_upsert(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen["n"] += 1
        if seen["n"] == 2:
            raise OSError("disk full")
        return await original(*args, **kwargs)

    vectors.upsert = flaky_upsert  # type: ignore[method-assign]

    # The underlying provider exception is logged as a warning and swallowed in _persist_vector per
    # Rule 1 (that path is shared with runtime, where one 429 mustn't kill the world loop), so the
    # seed path raises based on "did it persist".
    app = NarrativeApplication(mock_build_container, test_config)
    with pytest.raises(RuntimeError, match="could not be persisted"):
        await app.build_world("宫廷权谋", template="changan_iso")


def test_repeated_traits_and_values_are_kept_once_in_order() -> None:
    from world.builders.agent_generator import AgentGenerator
    from world.models import ThemeFigure

    figure = ThemeFigure(name="蒋一萍", role="老队员", importance="background")
    definition = AgentGenerator.__new__(AgentGenerator)._definition_from_payload(  # noqa: SLF001
        figure, _analysis(),
        {"core_traits": ["沉稳", "隐忍", "沉稳"], "core_values": ["体面", "体面", "忠诚"]}, "a1",
    )
    assert definition.soul.core_traits == ("沉稳", "隐忍")
    assert definition.soul.core_values == ("体面", "忠诚")


@pytest.mark.asyncio
async def test_cast_roles_bind_to_figures_by_index_not_by_echoed_name() -> None:
    """The model once wrote 「方老师」 for the figure 「辅导员方老师」 twice and the build failed: a
    role is bound by its number in the roster, never by the name it echoes back."""
    import json as _json
    analysis = _analysis()  # figures A, B, C
    cast_payload = _json.dumps({"reason": "r", "roles": [
        {"index": 3, "narrative_role": "c", "arc_summary": "c", "key_relationships": ["#1"]},
        {"index": "#1", "narrative_role": "a", "arc_summary": "a", "key_relationships": [2, 99]},
        {"index": 2, "narrative_role": "b", "arc_summary": "b", "key_relationships": []},
    ]}, ensure_ascii=False)
    router, spy = _spy_router_capturing_scene(LLMScene.CAST_DESIGN, cast_payload)

    design = await CastDesigner(router).design(analysis)

    assert " #1 名字：A" in _prompt_text(spy)
    assert {r.name: r.narrative_role for r in design.roles} == {"A": "a", "B": "b", "C": "c"}
    assert design.role_for("A").key_relationships[0] == "B"  # 99 is out of range and dropped
    assert design.role_for("C").key_relationships[0] == "A"


@pytest.mark.asyncio
async def test_a_figure_given_twice_does_not_cover_one_left_out() -> None:
    import json as _json
    cast_payload = _json.dumps({"roles": [
        {"index": i, "narrative_role": "r", "arc_summary": "a", "key_relationships": []}
        for i in (1, 1, 2)
    ]})
    router, _ = _spy_router_capturing_scene(LLMScene.CAST_DESIGN, cast_payload)

    with pytest.raises(ValueError, match="CastDesigner failed"):
        await CastDesigner(router).design(_analysis())


@pytest.mark.parametrize("raw, expected", [
    ("男", "男"), ("女", "女"), (" 女性 ", "女"), ("Female", "女"), ("m", "男"),
    ("男 / 女", ""), ("未知", ""), (None, ""), ("", ""),
])
def test_gender_is_normalized_to_the_two_canonical_values(raw, expected) -> None:
    """Consumers compare against 「男」/「女」 exactly; an unreadable label is unknown, not guessed."""
    assert _normalize_gender(raw) == expected
