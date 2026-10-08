"""The source of truth for step duration: the world's own theme analysis, not deployment config.

WorldTimeConfig.seconds_per_step produced by _clock_config_for:
  - comes only from analysis.world_time_config.seconds_per_step — how long a step lasts is a
    narrative judgment (how much time the tension needs to move one step), part of the same
    LLM-authored calendar as start_hour / era_name. Hours→seconds quantization happens further
    upstream at the LLM boundary (_analysis_from_payload), so what's read here is already seconds;
  - out-of-range/missing/invalid values all fall back to the 2-hour default (not clamped, see
    resolve_step_seconds) — silent field-level correction, no exception;
  - since analysis is saved with the step-0 manifest and restore reads back the same copy, it's
    frozen by construction, with no second copy needed.

So step duration is NOT a deployment-level knob on EngineConfig: that would need seconds_per_step
frozen into the manifest so a config change can't pollute existing worlds. Authored by the world
itself, it needs no such machinery.
"""
from __future__ import annotations

from types import SimpleNamespace

from engine.clock import DEFAULT_HOURS_PER_STEP, SECONDS_PER_HOUR
from world.initializer import WorldInitializer
from worlds.tiled import TiledWorldConfig


def _analysis_stub(world_time_config: dict) -> SimpleNamespace:
    return SimpleNamespace(world_name="chang_an", world_time_config=world_time_config)


def _clock_for(container, world_time_config: dict):
    initializer = WorldInitializer(container)
    return initializer._clock_config_for(_analysis_stub(world_time_config), TiledWorldConfig(template="changan_iso"))


def test_seconds_per_step_comes_from_the_analysis(container) -> None:
    """Step duration comes from the value written in analysis (already seconds)."""
    assert (
        _clock_for(container, {"seconds_per_step": 6 * SECONDS_PER_HOUR}).seconds_per_step
        == 6 * SECONDS_PER_HOUR
    )


def test_missing_value_falls_back_to_default(container) -> None:
    """analysis doesn't set it (old world / LLM missed the field) → default 2 hours per step, no
    exception."""
    assert (
        _clock_for(container, {}).seconds_per_step
        == DEFAULT_HOURS_PER_STEP * SECONDS_PER_HOUR
    )


def test_out_of_range_values_fall_back_to_default(container) -> None:
    """Out-of-range values fall back to the default rather than clamping — clamping would launder
    a runaway output into what looks like a deliberately chosen extreme."""
    assert (
        _clock_for(container, {"seconds_per_step": 60}).seconds_per_step
        == DEFAULT_HOURS_PER_STEP * SECONDS_PER_HOUR
    )
    assert (
        _clock_for(container, {"seconds_per_step": 999_999}).seconds_per_step
        == DEFAULT_HOURS_PER_STEP * SECONDS_PER_HOUR
    )


def test_non_integer_value_degrades_to_default(container) -> None:
    """Invalid types get silent field-level correction to the default instead of failing world
    building."""
    assert (
        _clock_for(container, {"seconds_per_step": "每步两小时"}).seconds_per_step
        == DEFAULT_HOURS_PER_STEP * SECONDS_PER_HOUR
    )


def test_clock_config_no_longer_reads_deployment_config(container) -> None:
    """Step duration is not a deployment-level knob — Container/EngineConfig must have no such
    value to read."""
    assert not hasattr(container, "seconds_per_step")
