"""Run-control: RunController semantics + NarrativeApplication background runs."""

from __future__ import annotations

import asyncio

import pytest

from engine.application import NarrativeApplication
from engine.run_control import RunController, RunState


# --- RunController (deterministic, no runtime) ---------------------------------


def test_controller_state_transitions() -> None:
    c = RunController()
    assert not c.is_paused and not c.stop_requested
    c.pause()
    assert c.is_paused
    c.resume()
    assert not c.is_paused
    c.stop()
    assert c.stop_requested and not c.is_paused


def test_controller_stop_wins_over_pause() -> None:
    c = RunController()
    c.stop()
    c.pause()  # no-op once stopping
    assert not c.is_paused
    c.resume()  # no-op once stopping
    assert c.stop_requested


@pytest.mark.asyncio
async def test_wait_if_paused_blocks_until_resume() -> None:
    c = RunController()
    c.pause()
    waiter = asyncio.create_task(c.wait_if_paused())
    await asyncio.sleep(0)
    assert not waiter.done()  # parked while paused
    c.resume()
    await asyncio.wait_for(waiter, timeout=1.0)  # released


@pytest.mark.asyncio
async def test_stop_releases_a_paused_waiter() -> None:
    c = RunController()
    c.pause()
    waiter = asyncio.create_task(c.wait_if_paused())
    await asyncio.sleep(0)
    assert not waiter.done()
    c.stop()
    await asyncio.wait_for(waiter, timeout=1.0)  # stop unblocks the loop to exit


@pytest.mark.asyncio
async def test_step_once_releases_exactly_one_step_then_parks_again() -> None:
    """The director's working rhythm: pause → inject → advance one step → look.

    "One step" is the whole of this assertion: the first waiter is released, and the second right
    after must be blocked again. Releasing two steps means there is no single-stepping.
    """
    c = RunController()
    c.pause()

    c.step_once()
    first = asyncio.create_task(c.wait_if_paused())
    await asyncio.wait_for(first, timeout=1.0)      # this step is released

    second = asyncio.create_task(c.wait_if_paused())
    await asyncio.sleep(0)
    assert not second.done()                        # the next step is blocked again
    assert c.is_paused

    c.step_once()                                   # press again, one more step
    await asyncio.wait_for(second, timeout=1.0)


@pytest.mark.asyncio
async def test_step_once_is_a_noop_once_stopping() -> None:
    """After stop there is no "next step": single-stepping must not drag back a loop that's
    exiting."""
    c = RunController()
    c.stop()
    c.step_once()
    assert c.stop_requested
    assert not c.is_paused
    await asyncio.wait_for(asyncio.create_task(c.wait_if_paused()), timeout=1.0)


# --- NarrativeApplication background run --------------------------------------


@pytest.mark.asyncio
async def test_background_run_completes_and_persists(mock_build_container, test_config) -> None:
    app = NarrativeApplication(mock_build_container, test_config)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    wid = world.world_id

    statuses: list[str] = []
    queue = mock_build_container.event_bus.subscribe()

    app.start_run(wid, steps=2)
    assert app.run_status(wid) == RunState.RUNNING
    task = app.get_session(wid).task
    assert task is not None
    await asyncio.wait_for(task, timeout=30.0)

    assert app.run_status(wid) == RunState.COMPLETED
    # Steps were actually advanced and persisted.
    steps = await mock_build_container.snapshot.list_steps(wid)
    assert max(steps) >= 2

    # Status transitions were announced on the bus (running → completed).
    while not queue.empty():
        ev = queue.get_nowait()
        if ev and ev.get("type") == "status" and ev.get("world_id") == wid:
            statuses.append(ev["status"])
    assert "running" in statuses and "completed" in statuses


@pytest.mark.asyncio
async def test_step_world_advances_one_step_from_idle(mock_build_container, test_config) -> None:
    """With no live loop, a single step is a run of exactly one step: it finishes on its own and the
    world goes idle."""
    app = NarrativeApplication(mock_build_container, test_config)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    wid = world.world_id
    assert app.run_status(wid) == RunState.IDLE

    app.step_world(wid)
    task = app.get_session(wid).task
    assert task is not None
    await asyncio.wait_for(task, timeout=30.0)

    assert max(await mock_build_container.snapshot.list_steps(wid)) == 1


@pytest.mark.asyncio
async def test_step_world_on_a_paused_run_advances_one_and_stays_parked(
    mock_build_container, test_config,
) -> None:
    """A paused world advanced one step must still be paused; otherwise "advance one step" becomes
    "resume"."""
    app = NarrativeApplication(mock_build_container, test_config)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    wid = world.world_id

    app.start_run(wid, steps=20)   # long enough for the controller to steer it midway
    app.pause_run(wid)
    assert app.run_status(wid) == RunState.PAUSED
    before = max(await mock_build_container.snapshot.list_steps(wid), default=0)

    app.step_world(wid)
    # Release one step → parks again. Poll until that step is persisted, to avoid racing step
    # duration.
    for _ in range(300):
        await asyncio.sleep(0.05)
        if max(await mock_build_container.snapshot.list_steps(wid), default=0) > before:
            break
    after = max(await mock_build_container.snapshot.list_steps(wid), default=0)

    assert after == before + 1
    assert app.run_status(wid) == RunState.PAUSED   # still paused, not running
    assert app.get_session(wid).controller.is_paused

    app.stop_run(wid)
    task = app.get_session(wid).task
    if task is not None:
        await asyncio.wait_for(task, timeout=30.0)


@pytest.mark.asyncio
async def test_control_methods_reject_when_not_running(mock_build_container, test_config) -> None:
    app = NarrativeApplication(mock_build_container, test_config)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    wid = world.world_id
    assert app.run_status(wid) == RunState.IDLE
    for method in (app.pause_run, app.resume_run, app.stop_run):
        with pytest.raises(ValueError):
            method(wid)


@pytest.mark.asyncio
async def test_a_run_of_no_steps_is_refused(mock_build_container, test_config) -> None:
    """Steps must be a positive integer: a 0-step run reports "running" while doing nothing, looking
    like it ran."""
    app = NarrativeApplication(mock_build_container, test_config)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    with pytest.raises(ValueError):
        await app.run_world(world.world_id, steps=0)


@pytest.mark.asyncio
async def test_a_long_run_stops_on_request(mock_build_container, test_config) -> None:
    app = NarrativeApplication(mock_build_container, test_config)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    wid = world.world_id

    app.start_run(wid, steps=10_000)  # can't finish, so only stop_run can stop it
    task = app.get_session(wid).task
    assert task is not None
    await asyncio.sleep(0)  # let at least one step begin
    app.stop_run(wid)
    assert app.run_status(wid) == RunState.STOPPING
    await asyncio.wait_for(task, timeout=30.0)
    assert app.run_status(wid) == RunState.COMPLETED


@pytest.mark.asyncio
async def test_delete_world_purges_and_stops(mock_build_container, test_config) -> None:
    app = NarrativeApplication(mock_build_container, test_config)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    wid = world.world_id
    app.start_run(wid, steps=3)
    await app.delete_world(wid)  # stops the run, then purges
    assert not app.has_session(wid)
    assert await mock_build_container.snapshot.list_steps(wid) == []


# --- lifecycle mutations vs. a concurrent run ---------------------------------
#
# InMemory stores never yield, so a reset/delete runs start-to-finish inside one
# scheduling slot and no interleaving is possible. Production stores DO yield
# (`asyncio.to_thread`) dozens of times, so these tests make one store call yield to
# recreate that — the window is otherwise unreachable from a test.


def _make_store_yield(container) -> None:
    """Make the first store call inside reset_session yield, like a file-backed store."""
    store = container.agent_store
    original = store.clear_relations

    async def slow_clear_relations(world_id: str) -> None:
        await asyncio.sleep(0.05)
        await original(world_id)

    store.clear_relations = slow_clear_relations  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_run_refused_while_a_reset_is_in_flight(mock_build_container, test_config) -> None:
    """Reset spans dozens of awaits; starting a run in that window lets the new session take that
    loop, so it must be blocked.

    Letting it in doesn't just mean "one extra step": the world reports stopped at step 0 and idle,
    while another loop keeps advancing the old runtime and writing snapshots to the same world_id,
    out of stop's reach.
    """
    app = NarrativeApplication(mock_build_container, test_config)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    wid = world.world_id
    _make_store_yield(mock_build_container)

    reset = asyncio.create_task(app.reset_session(wid))
    await asyncio.sleep(0.01)                      # reset is in progress

    with pytest.raises(ValueError):
        app.start_run(wid, steps=1)
    with pytest.raises(ValueError):
        await app.run_world(wid, steps=1)

    await reset
    # After reset only step 0 remains, with no loop writing other steps alongside.
    assert await mock_build_container.snapshot.list_steps(wid) == [0]
    assert app.get_session(wid).task is None
    assert app.run_status(wid) == RunState.IDLE


@pytest.mark.asyncio
async def test_reset_refused_while_a_run_is_live(mock_build_container, test_config) -> None:
    """No reset while running; the check is redone inside the lock, not based on the run state the
    caller read before entering."""
    app = NarrativeApplication(mock_build_container, test_config)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    wid = world.world_id

    app.start_run(wid, steps=10_000)   # reset while running, so it mustn't finish on its own first
    with pytest.raises(ValueError):
        await app.reset_session(wid)

    app.stop_run(wid)
    task = app.get_session(wid).task
    assert task is not None
    await asyncio.wait_for(task, timeout=30.0)
    await app.reset_session(wid)                   # once stopped, reset works
    assert app.get_session(wid).runtime.clock.current_step == 0


@pytest.mark.asyncio
async def test_run_refused_while_a_delete_is_in_flight(mock_build_container, test_config) -> None:
    """Starting a run during delete keeps writing to a world being deleted; it must be blocked.

    Nothing "already running" guards this moment: the previous run finished, ``session.task`` is
    cleared to None, and the session isn't removed from the table until after ``aclose``. In real
    runs ``aclose`` awaits in-flight event generation, and that's the window; here it's slowed down
    to pin it. A loop that slips in grows new snapshots on disk after the world has vanished from
    the catalog.
    """
    app = NarrativeApplication(mock_build_container, test_config)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    wid = world.world_id

    app.start_run(wid, steps=1)
    task = app.get_session(wid).task
    assert task is not None
    await asyncio.wait_for(task, timeout=30.0)      # done: task None, session kept
    assert app.get_session(wid).task is None

    runtime = app.get_session(wid).runtime
    original_aclose = runtime.aclose

    async def slow_aclose() -> None:
        await asyncio.sleep(0.05)
        await original_aclose()

    runtime.aclose = slow_aclose  # type: ignore[method-assign]

    async def _start_during_delete() -> None:
        await asyncio.sleep(0.01)                   # delete parked in aclose
        assert app.has_session(wid)
        with pytest.raises(ValueError):
            app.start_run(wid, steps=1)

    intruder = asyncio.create_task(_start_during_delete())
    await app.delete_world(wid)
    await intruder

    assert not app.has_session(wid)
    assert await mock_build_container.snapshot.list_steps(wid) == []


@pytest.mark.asyncio
async def test_a_directive_accepted_during_the_last_step_still_lands(
    mock_build_container, test_config, monkeypatch,
) -> None:
    """The director queue drains only at a step's start, so a directive accepted after the last
    step's drain needs one more step; otherwise the run completes with it still queued."""
    import json

    from core.interfaces.llm import LLMScene

    app = NarrativeApplication(mock_build_container, test_config)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    wid = world.world_id
    mock_build_container.llm_router.get(LLMScene.DIRECTIVE).fixed_response = json.dumps({
        "reason": "r", "feasible": True, "refusal": "",
        "broadcast": {"content": "钟声大作。", "severity": "medium",
                      "location_scope": None, "phenomenon": "none"},
        "message": None, "mutations": [], "narrative_desc": "钟声大作",
    }, ensure_ascii=False)
    director = app.get_session(wid).runtime.director
    drain = director.drain
    results = []

    async def drain_then_submit(**kwargs):
        drained = await drain(**kwargs)
        if not results:
            results.append(await app.submit_directive(wid, "敲响全城的钟"))
        return drained

    monkeypatch.setattr(director, "drain", drain_then_submit)

    app.start_run(wid, steps=1)
    while (task := app.get_session(wid).task) is not None:
        await asyncio.wait_for(task, timeout=30.0)

    assert results[0].accepted
    assert director.pending_count() == 0
    assert max(await mock_build_container.snapshot.list_steps(wid)) == 2


@pytest.mark.asyncio
async def test_restore_sees_changes_no_act_of_the_agent_made(mock_build_container, test_config) -> None:
    """A wound or a move made to an agent by someone else (a target effect, the director) must
    survive a restore even if the agent itself did nothing since: agent state is saved at every
    step's end, not only when the agent's own action lands."""
    app = NarrativeApplication(mock_build_container, test_config)
    world = await app.build_world("玄武门之变", template="changan_iso")
    wid = world.world_id
    await app.run_world(wid, steps=1)
    env = world.environment
    for agent in world.agent_list():
        agent.personality.apply_vitality_damage(0.4)
        here = env.get_body_location(agent.agent_id)
        there = next(p for p in env.space.all_place_ids() if p != here)
        env.move_body(body_id=agent.agent_id, location_id=there)
        agent.personality.update_location(step=1, location=there)
    await app.run_world(wid, steps=2)
    live = {a.agent_id: (a.personality.state.vitality, a.personality.state.current_location)
            for a in world.agent_list()}

    restored = await NarrativeApplication(mock_build_container, test_config).restore_session(wid)

    assert {a.agent_id: (a.personality.state.vitality, a.personality.state.current_location)
            for a in restored.agent_list()} == live


@pytest.mark.asyncio
async def test_shutdown_stops_runs_at_a_step_boundary_and_starts_no_more(
    mock_build_container, test_config,
) -> None:
    """A step writes memories as it goes and its snapshot at the end; shutting down mid-step
    would replay it on restore. Shutdown waits for the boundary, and refuses new runs."""
    app = NarrativeApplication(mock_build_container, test_config)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    wid = world.world_id
    app.start_run(wid, steps=50)
    task = app.get_session(wid).task

    await asyncio.wait_for(app.shutdown(), timeout=30.0)

    assert task.done() and not task.cancelled()
    assert app.run_status(wid) == RunState.COMPLETED
    assert max(await mock_build_container.snapshot.list_steps(wid)) < 50
    with pytest.raises(ValueError):
        app.start_run(wid, steps=1)


@pytest.mark.asyncio
async def test_a_stopping_world_refuses_pause_resume_and_step(mock_build_container, test_config) -> None:
    """After stop the controller ignores them, so accepting them would report a state the loop
    never enters and hide STOPPING."""
    app = NarrativeApplication(mock_build_container, test_config)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    wid = world.world_id
    app.start_run(wid, steps=20)
    app.stop_run(wid)

    for control in (app.pause_run, app.resume_run, app.step_world):
        with pytest.raises(ValueError):
            control(wid)
        assert app.run_status(wid) == RunState.STOPPING

    await asyncio.wait_for(app.get_session(wid).task, timeout=30.0)
