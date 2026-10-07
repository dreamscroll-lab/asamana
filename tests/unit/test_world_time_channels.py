"""The world's clock reaches the observer on two layers, and they must not be swapped.

``world_time.hour`` / ``.minute`` are the CODE-layer clock — structured, theme-neutral, what a
renderer reads (the map's day/night wash). Nothing downstream parses a clock string for them.

``world_time.label`` is the WORLD's own name for the moment ("武德九年，六月初一，酉时，夕阳渐浓").
Its SHAPE belongs to whatever theme keeps this world's time, so nothing downstream may take it
apart.

Both the live payload and the persisted snapshot carry ``WorldTime.clock_payload()``: one shape,
one producer, so the two paths cannot drift.
"""

from __future__ import annotations

from datetime import datetime, timezone

from core.interfaces.snapshot import WorldSnapshot
from engine.clock import WorldTime, WorldTimeConfig
from interaction.models import StepEvent, WorldTimeView

_ERA = "武德九年"


def _world_time(step: int) -> WorldTime:
    return WorldTime.from_step(step, WorldTimeConfig(era_name=_ERA, start_month=6, start_hour=6))


def _live_payload(wt: WorldTime) -> dict:
    """Exactly the shape engine.runtime publishes to the event bus."""
    return {
        "type": "step",
        "world_id": "w",
        "step": wt.step,
        "world_time": wt.clock_payload(),
    }


def _persisted_snapshot(wt: WorldTime) -> WorldSnapshot:
    """Exactly the shape engine.runtime hands the snapshot provider."""
    return WorldSnapshot(
        world_id="w",
        step=wt.step,
        timestamp=datetime.now(timezone.utc),
        world_time=wt.clock_payload(),
        metadata={},  # the clock lives in ONE place — no second copy here
    )


def test_the_snapshot_keeps_the_two_layers_apart() -> None:
    wt = _world_time(13)
    snap = _persisted_snapshot(wt)

    assert (snap.world_time["hour"], snap.world_time["minute"]) == (wt.hour_of_day, wt.minute_of_hour)
    assert snap.time_label == wt.time_label  # narrative: passed through whole
    assert "step=" not in snap.time_label


def test_live_and_replay_agree_on_both_clocks() -> None:
    wt = _world_time(13)

    for event in (
        StepEvent.from_runtime_payload(_live_payload(wt)),
        StepEvent.from_snapshot(_persisted_snapshot(wt)),
    ):
        assert (event.world_time.hour, event.world_time.minute) == (wt.hour_of_day, wt.minute_of_hour)
        assert event.world_time.label == wt.time_label
        assert _ERA in event.world_time.label
        assert "step=" not in event.world_time.label  # no code-layer coordinate in narrative


def test_a_step_recorded_without_a_structured_clock_reports_no_hour() -> None:
    """The renderer keeps its current wash on None; it must not be handed a guessed hour."""
    snap = WorldSnapshot(
        world_id="w", step=1, timestamp=datetime.now(timezone.utc),
        world_time={"label": "某时"},
    )
    event = StepEvent.from_snapshot(snap)
    assert event.world_time.hour is None and event.world_time.minute is None


def test_a_clock_that_is_not_a_payload_reads_as_unknown() -> None:
    """A record that stored something other than the clock payload yields no time, not a guess."""
    assert WorldTimeView.from_payload("晨") == WorldTimeView(label="", hour=None, minute=None)
