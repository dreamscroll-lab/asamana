"""How a world NAMES its dates belongs to the world, not to the engine.

The clock is theme-neutral — every setting has months, days and hours — but the
name of a date travels into prompts, memory and the feed as narrative text. With
「正月初一」 hard-coded, a near-future city would talk about its own time like a
Tang capital. The style is therefore authored on the map and frozen with the
world's config, exactly like era_name and the start date.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.clock import CalendarStyle, WorldTime, WorldTimeConfig, parse_calendar_style
from world.initializer import WorldInitializer
from worlds.tiled import TiledWorldConfig


def _at(step: int, **kwargs) -> str:
    defaults = dict(start_year=9, start_month=3, start_day=4, start_hour=9, seconds_per_step=3600)
    return WorldTime.from_step(step, WorldTimeConfig(**{**defaults, **kwargs})).time_label


def test_classical_style_names_dates_by_ordinal() -> None:
    assert _at(0, era_name="武德", calendar=CalendarStyle.CLASSICAL_CN) == "武德九年，三月初四，上午九点"


def test_modern_style_names_dates_numerically() -> None:
    """Same instant, same spoken clock — only the DATE changes. The hour stays
    「上午九点」 because naming an hour that way is a property of the language."""
    assert _at(0, start_year=2077, calendar=CalendarStyle.MODERN) == "2077年3月4日，上午九点"


def test_a_modern_date_carries_its_year_and_not_a_period_label() -> None:
    """A modern world must not read 「当代，6月1日」 — a period LABEL where the year belongs,
    leaving the date with no year at all. The year IS the modern date's era information;
    era_name is a bracket for the map picker and has no business in a timestamp."""
    label = _at(0, era_name="当代", start_year=2026, calendar=CalendarStyle.MODERN)
    assert label == "2026年3月4日，上午九点"
    assert "当代" not in label


def test_the_two_styles_differ_only_in_the_date() -> None:
    classical = _at(40, calendar=CalendarStyle.CLASSICAL_CN)
    modern = _at(40, calendar=CalendarStyle.MODERN)
    assert classical != modern
    assert classical.endswith("凌晨一点") and modern.endswith("凌晨一点")
    assert "初六" in classical and "3月6日" in modern


def test_classical_is_the_default_so_existing_worlds_are_unchanged() -> None:
    assert WorldTimeConfig().calendar is CalendarStyle.CLASSICAL_CN
    assert _at(0, era_name="武德") == _at(0, era_name="武德", calendar=CalendarStyle.CLASSICAL_CN)


def test_the_first_year_of_an_era_is_read_元年() -> None:
    """Reign years are ordinals and the first one has its own word — 「武德一年」 is not
    something anyone writes."""
    assert _at(0, era_name="武德", start_year=1).startswith("武德元年，")
    assert _at(0, era_name="武德", start_year=2).startswith("武德二年，")


def test_the_date_rolls_over_into_the_next_year() -> None:
    """The year exists so the MONTH has somewhere to carry into.

    Without one, `month_of_year` would be start_month + elapsed months with no upper bound,
    and a long world would announce 「16月」. A year is 12 months of 30 days, the same coarse
    grid as the day and month.
    """
    a_year_of_steps = 24 * 360  # seconds_per_step=3600 → 24 steps a day
    start = dict(start_year=9, start_month=3, start_day=4, seconds_per_step=3600)
    later = WorldTime.from_step(a_year_of_steps, WorldTimeConfig(**start, era_name="武德"))
    assert (later.year, later.month_of_year, later.day_of_month) == (10, 3, 4)
    assert later.time_label.startswith("武德十年，三月初四")

    # …and the month never leaves its own range on the way there.
    for step in range(0, a_year_of_steps, 24 * 17):
        moment = WorldTime.from_step(step, WorldTimeConfig(**start))
        assert 1 <= moment.month_of_year <= 12 and 1 <= moment.day_of_month <= 30


def test_a_year_is_a_number_that_advances_not_a_label_that_cannot() -> None:
    """The year is structural, so it may not be smuggled back into era_name: a world built
    with era_name=「武德九年」 would still say 「武德九年」 a decade in."""
    config = WorldTimeConfig(era_name="武德", start_year=9, seconds_per_step=86400)
    labels = {WorldTime.from_step(360 * n, config).time_label for n in range(3)}
    assert len(labels) == 3


@pytest.mark.parametrize("raw", ["", "  ", None, "gregorian", "宋", 7])
def test_an_unreadable_calendar_name_degrades_instead_of_failing(raw: object) -> None:
    """A map is authored content. An unrecognized value must leave a world that
    still tells the time, not one that refuses to build."""
    assert parse_calendar_style(raw) is CalendarStyle.CLASSICAL_CN


@pytest.mark.parametrize("raw,expected", [
    ("modern", CalendarStyle.MODERN),
    ("MODERN", CalendarStyle.MODERN),
    ("  Classical_CN  ", CalendarStyle.CLASSICAL_CN),
])
def test_authored_names_are_read_case_and_space_insensitively(raw: str, expected: CalendarStyle) -> None:
    assert parse_calendar_style(raw) is expected


def test_the_map_supplies_the_style_all_the_way_into_the_clock(container) -> None:
    """The whole chain: map property → runtime context → WorldTimeConfig."""
    config = TiledWorldConfig(template="changan_iso")
    assert config.to_runtime_context()["calendar"] == "classical_cn"

    clock = WorldInitializer(container)._clock_config_for(
        SimpleNamespace(world_name="x", world_time_config={}), config
    )
    assert clock.calendar is CalendarStyle.CLASSICAL_CN



def test_the_runtime_clock_keeps_the_world_s_whole_clock_config(container, test_config) -> None:
    """A running world keeps every clock setting it was built with.

    Copying `WorldTimeConfig` field by field drops whatever is left off the list: omit
    `calendar` and a world on a non-default calendar reverts to 「九月初一」 once running,
    while its own step-0 snapshot says otherwise.

    The assertion is over the dataclass's own fields, so a field added later is covered
    without extending this test.
    """
    import dataclasses
    from types import SimpleNamespace

    from engine.application import NarrativeApplication
    from engine.directory import LiveWorldDirectory
    from engine.environment import EnvironmentSystem

    built = WorldTimeConfig(
        era_name="2087年",
        calendar=CalendarStyle.MODERN,
        start_year=2087,
        start_month=9,
        start_day=2,
        start_hour=8,
        seconds_per_step=1800,
    )
    # Every field must differ from its default, or a dropped one would still match.
    for field in dataclasses.fields(WorldTimeConfig):
        assert getattr(built, field.name) != field.default, field.name

    environment = EnvironmentSystem()
    world = SimpleNamespace(
        world_id="clock-fidelity",
        clock_config=built,
        environment=environment,
        directory=LiveWorldDirectory.from_agents({}, environment),
        analysis=SimpleNamespace(core_tension="", narrative_theme=""),
    )
    runtime = NarrativeApplication(container, test_config)._build_runtime(world)

    assert runtime._clock.config == built
