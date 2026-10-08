"""A place's authoritative history: the only source of intelligence for covert adjudication, and
the track most likely to leak.

Everything in this world has two layers: the degraded text onlookers see, and the full outcome of
the same event. Ordinary perception gives only the former; what COVERT buys with a step is exactly
the difference. This track stores the latter.

It has exactly one legitimate reader (CovertExecutor). These tests check that the right things go
in, in the right order, and never leak out.
"""

from __future__ import annotations

from engine.clock import WorldTime
from engine.environment import (
    HAPPENINGS_MAX_ENTRIES, HAPPENINGS_WINDOW_STEPS, EnvironmentSystem,
)
from core.interfaces.place import Place


def _world_time(step: int) -> WorldTime:
    return WorldTime(step=step, elapsed_seconds=step * 3600)


def _env() -> EnvironmentSystem:
    env = EnvironmentSystem()
    env.space.register_place(Place(
        place_id="hall", name="正堂", description="",
        connections={}, is_public=True, capacity=50,
    ))
    return env


def test_history_returns_content_in_order_within_window() -> None:
    env = _env()
    for step in (3, 4, 5):
        env.record_happening(
            location_id="hall", outcome=f"第{step}件事", step=step, actor_ids=("a",),
        )

    # Returns (step, full text). The caller turns the step into a time prefix; the raw step never
    # goes into a prompt.
    assert env.recent_happenings("hall", since_step=4) == [(4, "第4件事"), (5, "第5件事")]
    assert env.recent_happenings("hall", since_step=3) == [
        (3, "第3件事"), (4, "第4件事"), (5, "第5件事"),
    ]


def test_history_orders_by_step_not_by_write_order() -> None:
    """The judge reconstructs events from this; out of order, it reads a world running backwards.

    Write order isn't guaranteed to be occurrence order (scene setup can insert in reverse), so
    only the step decides precedence.
    """
    env = _env()
    for step, text in ((5, "后来的事"), (3, "更早的事"), (4, "中间的事")):
        env.record_happening(
            location_id="hall", outcome=text, step=step, actor_ids=(),
        )
    assert [c for _, c in env.recent_happenings("hall", since_step=0)] == [
        "更早的事", "中间的事", "后来的事",
    ]


def test_history_excludes_those_who_already_hold_it() -> None:
    """The exclusion set is "whoever already holds this": the actor and anyone who overheard it.

    It isn't news to them; handing it over would let adjudication rule success on something they
    already knew, which is exactly the empty gain this mechanism exists to prevent.
    """
    env = _env()
    env.record_happening(
        location_id="hall", outcome="甲与乙谈定了动手的日子", step=1, actor_ids=("甲", "乙"),
    )
    env.record_happening(
        location_id="hall", outcome="丙在案前写了一封信", step=1, actor_ids=("丙",),
    )

    assert [c for _, c in env.recent_happenings("hall", since_step=0, exclude_ids=("丁",))] == [
        "甲与乙谈定了动手的日子", "丙在案前写了一封信",
    ]
    # a participant doesn't get their own event back
    assert [c for _, c in env.recent_happenings("hall", since_step=0, exclude_ids=("乙",))] == [
        "丙在案前写了一封信",
    ]


def test_history_is_bounded_by_both_steps_and_entries() -> None:
    """Both bounds are needed: with only a step window, a busy place can still feed the judge an
    unbounded history."""
    env = _env()
    # fill a single step past the entry cap
    for i in range(HAPPENINGS_MAX_ENTRIES + 10):
        env.record_happening(
            location_id="hall", outcome=f"事{i}", step=1, actor_ids=(),
        )
    kept = [c for _, c in env.recent_happenings("hall", since_step=0)]
    assert len(kept) == HAPPENINGS_MAX_ENTRIES
    assert kept[-1] == f"事{HAPPENINGS_MAX_ENTRIES + 9}"  # the most recent survive

    env2 = _env()
    env2.record_happening(
        location_id="hall", outcome="很久以前的事", step=1, actor_ids=(),
    )
    env2.record_happening(
        location_id="hall", outcome="刚刚的事", step=1 + HAPPENINGS_WINDOW_STEPS + 1,
        actor_ids=(),
    )
    assert [c for _, c in env2.recent_happenings("hall", since_step=0)] == ["刚刚的事"]


def test_history_never_reaches_ordinary_perception() -> None:
    """Never exposed through spatial_for. That would hand everyone the full layer for free and break
    information asymmetry."""
    env = _env()
    env.place_agent(agent_id="watcher", location_id="hall")
    env.record_happening(
        location_id="hall", outcome="密谈的内容是三日后动手", step=1, actor_ids=("甲",),
    )
    env.begin_step(step=2, world_time=_world_time(2))

    spatial = env.spatial_for(agent_id="watcher")
    assert not any("三日后动手" in ev.content for ev in spatial.ambient_events)


def test_history_never_reaches_the_snapshot() -> None:
    """Never included in snapshot_state. That is the real leak path: metadata["environment"] flows
    to the event editor's brief (the step_annotations reader in engine/event.py) and to every web/replay
    consumer.
    """
    env = _env()
    env.record_happening(
        location_id="hall", outcome="密谈的内容是三日后动手", step=1, actor_ids=("甲",),
    )
    env.begin_step(step=2, world_time=_world_time(2))

    assert "三日后动手" not in str(env.snapshot_state())


def test_blank_and_placeless_writes_are_dropped() -> None:
    env = _env()
    env.record_happening(location_id="hall", outcome="   ", step=1)
    env.record_happening(location_id="", outcome="无处安放", step=1)
    assert env.recent_happenings("hall", since_step=0) == []
    assert env.recent_happenings("", since_step=0) == []
