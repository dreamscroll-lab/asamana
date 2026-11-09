"""Sub-places: generated at build, used to place people and things, carried into the runtime view."""

from __future__ import annotations

import json

import pytest

from core.interfaces.llm import LLMRouter, LLMScene
from core.interfaces.place import SubPlace
from core.prompts import render_location
from engine.space import SpaceManager
from core.interfaces.place import Place
from providers.llm.mock import MockLLMProvider
from world.builder import (
    _canonicalize_seed_locations,
    _drop_seeds_shadowing_locations,
    _world_config_locations,
)
from world.builders.sub_place_generator import SubPlaceGenerator
from world.models import SUB_PLACE_LIMIT, ThemeAnalysis, WorldEntitySeed
from worlds.tiled import TiledWorldConfig


def _router(reply: str) -> tuple[LLMRouter, MockLLMProvider]:
    spy = MockLLMProvider(fixed_response=reply)
    return LLMRouter({scene: spy for scene in LLMScene}), spy


def _index_of(world_config: TiledWorldConfig, place_id: str) -> int:
    return list(world_config.get_places()).index(place_id) + 1


def _reply(*items: dict) -> str:
    return json.dumps({"sub_places": list(items)}, ensure_ascii=False)


@pytest.mark.asyncio
async def test_each_sub_place_lands_under_the_place_its_index_names() -> None:
    config = TiledWorldConfig(template="changan_iso")
    router, spy = _router(_reply(
        {"reason": "秦王一党聚议之所，府在此坊", "name": "秦王府", "place_index": _index_of(config, "buzheng_fang")},
        {"reason": "太子与僚属议事之殿，在东宫之内", "name": "显德殿", "place_index": _index_of(config, "donggong")},
    ))

    sub_places = await SubPlaceGenerator(router).generate("玄武门之变", config)

    assert sub_places == (
        SubPlace(location_id="buzheng_fang", name="秦王府"),
        SubPlace(location_id="donggong", name="显德殿"),
    )
    # The map goes before the theme (stable prefix first); the LLM sees names, never ids.
    user = spy.call_history[0][-1].content
    assert user.index("【地图上的地点】") < user.index("玄武门之变")
    assert "东宫" in user and "donggong" not in user


@pytest.mark.asyncio
async def test_entries_the_world_cannot_hold_are_dropped() -> None:
    config = TiledWorldConfig(template="changan_iso")
    palace = _index_of(config, "taiji_palace")
    router, _ = _router(_reply(
        {"name": "", "place_index": palace},                  # unnamed
        {"name": "无处之所", "place_index": 999},              # no such place
        {"name": "玄武门", "place_index": palace},             # a place's own name
        *({"name": f"殿{i}", "place_index": palace} for i in range(SUB_PLACE_LIMIT + 1)),
        {"name": "殿0", "place_index": _index_of(config, "donggong")},  # a name already taken
    ))

    sub_places = await SubPlaceGenerator(router).generate("宫变", config)

    assert [s.name for s in sub_places] == [f"殿{i}" for i in range(SUB_PLACE_LIMIT + 1)]
    assert {s.location_id for s in sub_places} == {"taiji_palace"}


@pytest.mark.asyncio
async def test_a_place_past_the_limit_is_kept_whole_and_warned(caplog: pytest.LogCaptureFixture) -> None:
    config = TiledWorldConfig(template="changan_iso")
    palace = _index_of(config, "taiji_palace")
    router, _ = _router(_reply(*({"name": f"殿{i}", "place_index": palace} for i in range(SUB_PLACE_LIMIT + 1))))

    with caplog.at_level("WARNING"):
        sub_places = await SubPlaceGenerator(router).generate("宫变", config)

    assert len(sub_places) == SUB_PLACE_LIMIT + 1
    assert any(r.getMessage() == "sub_places_over_limit" for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ['{"sub_places": []}', "{}"])
async def test_a_theme_may_need_no_sub_places(reply: str) -> None:
    router, _ = _router(reply)
    assert await SubPlaceGenerator(router).generate("一场旅行", TiledWorldConfig(template="changan_iso")) == ()


@pytest.mark.asyncio
async def test_an_unreadable_reply_is_retried_once_then_raises() -> None:
    router, spy = _router("not json")
    with pytest.raises(ValueError, match="SubPlaceGenerator failed"):
        await SubPlaceGenerator(router).generate("宫变", TiledWorldConfig(template="changan_iso"))
    assert len(spy.call_history) == 2


def _analysis(*seeds: WorldEntitySeed) -> ThemeAnalysis:
    return ThemeAnalysis(
        theme_input="t", world_name="长安", era_description="", core_tension="", narrative_theme="",
        narrative_pitch="", key_figures=[], world_entity_seeds=list(seeds),
    )


def test_a_seed_placed_at_a_sub_place_stands_in_its_place() -> None:
    config = TiledWorldConfig(template="changan_iso")
    sub_places = (SubPlace(location_id="buzheng_fang", name="秦王府"),
                  SubPlace(location_id="huangcheng", name="承天门街"))
    analysis = _analysis(
        WorldEntitySeed(name="秦府密令", entity_type="item", location="秦王府"),
        # A sub-place's name wins over the map's substring match ("承天门" is a place of its own).
        WorldEntitySeed(name="石狮", entity_type="landmark", location="承天门街"),
    )

    _canonicalize_seed_locations(analysis, config, sub_places)

    assert [s.location for s in analysis.world_entity_seeds] == ["buzheng_fang", "huangcheng"]


def test_a_seed_named_like_a_sub_place_is_dropped() -> None:
    analysis = _analysis(
        WorldEntitySeed(name="秦王府", entity_type="landmark", location="buzheng_fang"),
        WorldEntitySeed(name="秦府密令", entity_type="item", location="buzheng_fang"),
    )
    _drop_seeds_shadowing_locations(
        analysis, TiledWorldConfig(template="changan_iso"),
        (SubPlace(location_id="buzheng_fang", name="秦王府"),),
    )
    assert [s.name for s in analysis.world_entity_seeds] == ["秦府密令"]


def test_build_prompts_list_each_places_sub_places() -> None:
    locations = _world_config_locations(
        TiledWorldConfig(template="changan_iso"), (SubPlace(location_id="donggong", name="显德殿"),),
    )
    by_name = {location.name: location.as_dict() for location in locations}
    assert by_name["东宫"]["sub_places"] == ["显德殿"]
    assert "sub_places" not in by_name["玄武门"]


def test_a_places_view_and_rendering_carry_its_sub_places() -> None:
    space = SpaceManager()
    space.register_place(Place(place_id="donggong", name="东宫", description="皇太子居所"))
    space.register_place(Place(place_id="xuanwu", name="玄武门"))
    space.register_sub_places((SubPlace(location_id="donggong", name="显德殿"),
                               SubPlace(location_id="donggong", name="崇教殿")))

    assert render_location(space.view_of("donggong")) == "东宫——皇太子居所；显德殿、崇教殿都在此地"
    assert render_location(space.view_of("xuanwu")) == "玄武门"


@pytest.mark.asyncio
async def test_a_confirmed_world_shows_its_sub_places_and_a_restored_one_keeps_them(
    mock_build_container, test_config,
) -> None:
    container = mock_build_container
    from dataclasses import replace

    from interaction.application import NarrativeApplication
    from world.initializer import WorldInitializer
    from world.setup import WorldSetup

    application = NarrativeApplication(container, test_config)
    world_id = await application.build_world("玄武门之变", template="changan_iso")
    setup = WorldSetup.from_dict(await container.snapshot.load_world_setup(world_id))
    revised = replace(setup, sub_places=(SubPlace(location_id="donggong", name="显德殿"),))
    await container.snapshot.save_world_setup(world_id, revised.as_dict())

    world = await application.confirm_world(world_id)
    restored = await WorldInitializer(container).restore(world_id)

    for env in (world.environment, restored.environment):
        assert env.space.view_of("donggong").sub_places == ("显德殿",)
        assert env.space.view_of("xuanwu_gate").sub_places == ()


@pytest.mark.asyncio
async def test_the_two_world_building_calls_are_traced_apart() -> None:
    """Sub-place generation and theme analysis share the world_building scene; each trace says which."""
    from providers.trace.in_memory import InMemoryTraceSink
    from tests.conftest import MOCK_WORLD_BUILDING_RESPONSE
    from world.builders.theme_analyzer import ThemeAnalyzer

    sink = InMemoryTraceSink()
    router = LLMRouter({scene: MockLLMProvider(fixed_response=_reply()) for scene in LLMScene}, trace_sink=sink)
    await SubPlaceGenerator(router).generate("宫变", TiledWorldConfig(template="changan_iso"))
    router = LLMRouter(
        {scene: MockLLMProvider(fixed_response=MOCK_WORLD_BUILDING_RESPONSE) for scene in LLMScene}, trace_sink=sink,
    )
    await ThemeAnalyzer(router).analyze("宫变")

    assert [(c.scene, c.extra.get("builder")) for c in sink.llm_calls] == [
        ("world_building", "sub_places"), ("world_building", "theme_analysis"),
    ]

