"""When someone present arrived, as the viewer knows it.

The world records one fact per body (the step it reached where it stands); perception turns it into
a viewer-relative mark: an arrival the viewer witnessed is timed, someone who was here before the
viewer came is only "already here" (the viewer never saw how long).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.decision import ActionType, AgentAction
from core.interfaces.action import ActionTarget, Ref
from core.interfaces.place import Place
from core.interfaces.perception import LocationView, Situation
from core.prompts import (
    ALREADY_HERE_MARK, SituationVoice, arrival_mark, own_arrival_mark,
    person_referent, render_situation_header,
)
from engine.clock import WorldTimeConfig
from engine.directory import LiveWorldDirectory
from engine.environment import IN_TRANSIT, EnvironmentSystem
from engine.executors.movement import MovementExecutor
from engine.presence import attach_presence
from engine.scene import SceneVisibility, assemble_scene_context, situation_for
from tests.unit.env_state import restore, saved


def _env() -> EnvironmentSystem:
    env = EnvironmentSystem(clock_config=WorldTimeConfig(start_hour=8, seconds_per_step=3600))
    env.space.register_place(Place(place_id="hall", name="大殿", connections={"gate": 1}))
    env.space.register_place(Place(place_id="gate", name="宫门", connections={"hall": 1}))
    return env


def _at(env: EnvironmentSystem, step: int) -> None:
    env.begin_step(step=step, world_time=SimpleNamespace(time_label="", hour_of_day=-1))


def _agent() -> SimpleNamespace:
    return SimpleNamespace(personality=SimpleNamespace(state=SimpleNamespace(condition=None)))


# ── the mark ────────────────────────────────────────────────────────────────


def test_someone_here_before_me_is_only_already_here() -> None:
    """No duration: the viewer didn't see them arrive."""
    for other in (0, 4):
        assert arrival_mark(
            now_step=40, viewer_arrived=5, other_arrived=other, seconds_per_step=3600,
        ) == ALREADY_HERE_MARK


def test_the_same_beat_is_timed_but_step_zero_is_no_arrival() -> None:
    """Equal stamps mean two things: one real beat both arrived in, which the viewer saw and so
    is timed like any later arrival (it ages; a day on it isn't news), or no record at all (world
    start, an old snapshot), where nobody arrived and "already here" is the truth."""
    kw = dict(seconds_per_step=3600, world_start_second_of_day=8 * 3600)
    assert arrival_mark(now_step=6, viewer_arrived=5, other_arrived=5, **kw) == "刚刚抵达"
    assert arrival_mark(now_step=30, viewer_arrived=5, other_arrived=5, **kw) == "昨日抵达"
    assert arrival_mark(now_step=40, viewer_arrived=0, other_arrived=0, **kw) == ALREADY_HERE_MARK


def test_a_timed_arrival_reads_the_same_as_the_viewers_own() -> None:
    """Arriving together, the viewer's header and the mark show one time, so the LLM can read
    that they came together without a mark of its own."""
    kw = dict(now_step=30, seconds_per_step=3600, world_start_second_of_day=8 * 3600)
    assert arrival_mark(viewer_arrived=5, other_arrived=5, **kw) == own_arrival_mark(arrived_step=5, **kw)


def test_an_arrival_i_saw_is_timed_from_when_it_was_perceived() -> None:
    """Arriving in beat t is perceived at t+1: "just now" then, matching the dated memory of it."""
    kw = dict(viewer_arrived=0, other_arrived=10, seconds_per_step=3600, world_start_second_of_day=8 * 3600)
    assert arrival_mark(now_step=11, **kw) == "刚刚抵达"
    assert arrival_mark(now_step=13, **kw) == "几小时前抵达"
    assert arrival_mark(now_step=35, **kw) == "昨日抵达"


# ── the world fact ──────────────────────────────────────────────────────────


def test_arrived_steps_follows_real_moves_only() -> None:
    env = _env()
    env.place_agent(agent_id="a1", location_id="hall")
    assert env.arrived_step("a1") == 0
    _at(env, 3)
    env.move_body(body_id="a1", location_id="hall")   # no change: keeps the old stamp
    assert env.arrived_step("a1") == 0
    env.move_body(body_id="a1", location_id="gate")
    assert env.arrived_step("a1") == 3
    env.remove_body("a1")
    assert env.arrived_step("a1") == 0


def test_arrived_steps_survives_a_snapshot() -> None:
    env = _env()
    env.place_agent(agent_id="a1", location_id="hall")
    _at(env, 7)
    env.move_body(body_id="a1", location_id="gate")
    payload = saved(env)

    back = _env()
    back.place_agent(agent_id="a1", location_id="gate")
    restore(back, payload)
    assert back.arrived_step("a1") == 7


def test_a_snapshot_without_the_table_restores_everyone_as_already_here() -> None:
    """Missing stamps stay 0, so nobody is rendered as a fresh arrival that never happened."""
    env = _env()
    env.place_agent(agent_id="a1", location_id="hall")
    payload = saved(env)
    payload["engine"].pop("arrived_steps")

    back = _env()
    back.place_agent(agent_id="a1", location_id="hall")
    restore(back, payload)
    assert back.arrived_step("a1") == 0


# ── perception ──────────────────────────────────────────────────────────────


def test_presence_marks_are_relative_to_the_viewer() -> None:
    env = _env()
    for aid in ("viewer", "early", "late"):
        env.place_agent(agent_id=aid, location_id="gate")
    _at(env, 2)
    env.move_body(body_id="viewer", location_id="hall")
    _at(env, 1)   # "early" got to the hall before the viewer
    env.move_body(body_id="early", location_id="hall")
    _at(env, 6)
    env.move_body(body_id="late", location_id="hall")
    _at(env, 7)

    agents = {aid: _agent() for aid in ("viewer", "early", "late")}
    directory = LiveWorldDirectory.from_agents({}, env)

    seen = env.spatial_for(agent_id="viewer")
    attach_presence(seen, viewer_id="viewer", directory=directory, agents=agents, environment=env)
    assert seen.visible_agents["early"].arrival == ALREADY_HERE_MARK
    assert seen.visible_agents["late"].arrival == "刚刚抵达"

    # The latecomer sees both of the others as already here.
    theirs = env.spatial_for(agent_id="late")
    attach_presence(theirs, viewer_id="late", directory=directory, agents=agents, environment=env)
    assert {p.arrival for p in theirs.visible_agents.values()} == {ALREADY_HERE_MARK}


def test_the_mark_shares_the_roster_bracket() -> None:
    assert person_referent("尉迟恭", "男", "在场", "刚刚抵达", "") == "尉迟恭（男，在场，刚刚抵达）"


# ── my own arrival, in the situation header ─────────────────────────────────


def test_my_own_arrival_is_timed_like_anyone_elses_but_step_zero_says_nothing() -> None:
    """Step 0 is no arrival (world start, or a snapshot without the record): say nothing rather
    than claim "here since the beginning"."""
    kw = dict(seconds_per_step=3600, world_start_second_of_day=8 * 3600)
    assert own_arrival_mark(now_step=11, arrived_step=10, **kw) == "刚刚抵达"
    assert own_arrival_mark(now_step=35, arrived_step=10, **kw) == "昨日抵达"
    assert own_arrival_mark(now_step=35, arrived_step=0, **kw) == ""


def test_my_arrival_reaches_every_situation_source() -> None:
    """Perception (and the Situation cached from it) and the god's-eye ``situation_for`` agree;
    on the road there is no place to have arrived at."""
    env = _env()
    env.place_agent(agent_id="a1", location_id="gate")
    _at(env, 4)
    env.move_body(body_id="a1", location_id="hall")
    _at(env, 5)

    spatial = env.spatial_for(agent_id="a1")
    assert spatial.arrival == "刚刚抵达"
    assert Situation.from_spatial(spatial).arrival == "刚刚抵达"
    assert situation_for(env, "a1").arrival == "刚刚抵达"

    env.move_body(body_id="a1", location_id=IN_TRANSIT)
    assert env.own_arrival("a1") == ""


def test_the_header_arrival_always_names_whose_it_is() -> None:
    """"我" in FIRST voice, the named person in THIRD; THIRD with no one named leaves it out (a
    judge's scene holds several people)."""
    hall = LocationView(name="大殿")
    here = Situation(location_view=hall, time_label="三月初四 午时", arrival="昨日抵达")
    assert render_situation_header(here, voice=SituationVoice.FIRST) == (
        "我此刻在大殿，当前时间为三月初四 午时；昨日抵达此处。"
    )
    assert render_situation_header(here, voice=SituationVoice.THIRD, who="尉迟恭") == (
        "当前时间：三月初四 午时；当前地点：大殿；尉迟恭昨日抵达此处。"
    )
    assert render_situation_header(here, voice=SituationVoice.THIRD) == (
        "当前时间：三月初四 午时；当前地点：大殿。"
    )
    unknown = Situation(location_view=hall, time_label="三月初四 午时")
    assert render_situation_header(unknown, voice=SituationVoice.FIRST) == (
        "我此刻在大殿，当前时间为三月初四 午时。"
    )


def test_a_judges_scene_names_whose_arrival_it_is() -> None:
    """The scene is built around one person's place, so its header names that person."""
    env = _env()
    env.place_agent(agent_id="doer", location_id="gate")
    env.place_agent(agent_id="other", location_id="hall")
    _at(env, 4)
    env.move_body(body_id="doer", location_id="hall")
    _at(env, 5)
    soul = SimpleNamespace(personality=SimpleNamespace(soul=SimpleNamespace(name="尉迟恭", gender="男")))
    directory = LiveWorldDirectory.from_agents({"doer": soul}, env)

    scene = assemble_scene_context(
        "doer", environment=env, directory=directory, visibility=SceneVisibility.GOD,
    )
    assert "；尉迟恭刚刚抵达此处。" in scene.text


# ── a trip over several steps ───────────────────────────────────────────────


def _route_env() -> EnvironmentSystem:
    """gate —1h— yard —2h— hall, under 1-hour steps: a trip from gate to hall stands on yard
    after its first step, is on the road for the next, and lands on the third."""
    env = EnvironmentSystem(clock_config=WorldTimeConfig(start_hour=8, seconds_per_step=3600))
    env.space.register_place(Place(place_id="gate", name="宫门", connections={"yard": 3600}))
    env.space.register_place(Place(place_id="yard", name="庭院", connections={"gate": 3600, "hall": 7200}))
    env.space.register_place(Place(place_id="hall", name="大殿", connections={"yard": 7200}))
    for aid, where in (("m1", "gate"), ("m2", "gate"), ("keeper", "yard"), ("host", "hall")):
        env.place_agent(agent_id=aid, location_id=where)
    return env


def _marks(env: EnvironmentSystem, viewer_id: str) -> dict[str, str]:
    """What ``viewer_id`` sees of each person present, as perception assembles it."""
    spatial = env.spatial_for(agent_id=viewer_id)
    attach_presence(
        spatial, viewer_id=viewer_id, directory=LiveWorldDirectory.from_agents({}, env),
        agents={aid: _agent() for aid in spatial.visible_agent_ids}, environment=env,
    )
    return {aid: p.arrival for aid, p in spatial.visible_agents.items()}


def _trip(mover_id: str) -> AgentAction:
    return AgentAction(
        agent_id=mover_id, step=1, action_type=ActionType.MOVE, action_description="去大殿",
        target=ActionTarget(acts_on=[Ref.place("hall")]),
    )


async def _tick(executor: MovementExecutor, trips: list, step: int, env: EnvironmentSystem) -> None:
    """One step of the runtime's execution sweep: count down, then tick or complete."""
    for state in trips:
        state.remaining_steps -= 1
        if state.remaining_steps <= 0:
            await executor.complete(state, step, agents={}, environment=env, message_system=None)
        else:
            await executor.tick(state, step, agents={}, environment=env, message_system=None)


@pytest.mark.asyncio
async def test_a_trip_over_several_steps_is_marked_at_every_stop() -> None:
    """Each place the trip stands on is an arrival; the road between is none. Two who travel
    together are each other's timed arrival at every stop, and the marks age with the clock."""
    env = _route_env()
    executor = MovementExecutor(LiveWorldDirectory.from_agents({}, env), seconds_per_step=3600)

    _at(env, 1)
    trips = [
        await executor.start(_trip(mid), 1, agents={}, environment=env, message_system=None)
        for mid in ("m1", "m2")
    ]
    assert env.get_body_location("m1") == "yard" and env.arrived_step("m1") == 1

    # Step 2 perceives the waypoint: the keeper saw them come, they find him already there.
    _at(env, 2)
    assert _marks(env, "keeper") == {"m1": "刚刚抵达", "m2": "刚刚抵达"}
    assert _marks(env, "m1") == {"keeper": ALREADY_HERE_MARK, "m2": "刚刚抵达"}
    assert env.own_arrival("m1") == "刚刚抵达"

    # On the road: nowhere arrived at, nobody in view, the origin kept for a turn-back.
    await _tick(executor, trips, 2, env)
    _at(env, 3)
    assert env.get_body_location("m1") == IN_TRANSIT
    assert env.own_arrival("m1") == ""
    assert env.spatial_for(agent_id="m1").arrival == ""
    assert _marks(env, "m1") == {}
    assert env.transit_origin("m1") == "yard"
    assert "m1" not in _marks(env, "keeper")

    await _tick(executor, trips, 3, env)
    _at(env, 4)
    assert env.get_body_location("m1") == "hall" and env.transit_origin("m1") is None
    assert _marks(env, "host") == {"m1": "刚刚抵达", "m2": "刚刚抵达"}
    assert _marks(env, "m1") == {"host": ALREADY_HERE_MARK, "m2": "刚刚抵达"}
    situation = Situation.from_spatial(env.spatial_for(agent_id="m1"))
    assert render_situation_header(situation, voice=SituationVoice.FIRST).endswith("；刚刚抵达此处。")

    _at(env, 28)   # the next day: still timed, now aged
    assert _marks(env, "m1") == {"host": ALREADY_HERE_MARK, "m2": "昨日抵达"}
    assert env.own_arrival("m1") == "昨日抵达"


@pytest.mark.asyncio
async def test_a_trip_saved_on_the_road_resumes_and_lands_with_its_arrival() -> None:
    """A snapshot taken mid-road keeps the stamp and the origin; the resumed trip lands as
    though it had never stopped."""
    env = _route_env()
    executor = MovementExecutor(LiveWorldDirectory.from_agents({}, env), seconds_per_step=3600)
    _at(env, 1)
    trip = await executor.start(_trip("m1"), 1, agents={}, environment=env, message_system=None)
    _at(env, 2)
    await _tick(executor, [trip], 2, env)
    payload = saved(env)

    back = _route_env()
    back.place_agent(agent_id="m1", location_id=IN_TRANSIT)   # the initializer places first
    restore(back, payload)
    assert back.arrived_step("m1") == 2
    assert back.transit_origin("m1") == "yard"
    assert back.own_arrival("m1") == ""

    _at(back, 3)
    await _tick(executor, [trip], 3, back)
    _at(back, 4)
    assert back.get_body_location("m1") == "hall"
    assert _marks(back, "host") == {"m1": "刚刚抵达"}
