"""Casting the actor: chosen while the world is set up, kept across restores, and given up once
for good. A stop while the actor's turn is open discards the half-run step."""

from __future__ import annotations

import asyncio

import pytest

from core.container import Container
from engine.application import NarrativeApplication
from world import ActorCasting, WorldCatalog


@pytest.fixture()
def catalog() -> WorldCatalog:
    return WorldCatalog()


async def _built(container: Container, test_config, catalog: WorldCatalog):
    application = NarrativeApplication(container, test_config, catalog=catalog)
    world = await application.build_world("玄武门之变", template="changan_iso")
    character_id = next(a.agent_id for a in world.agents.values() if not a.is_main_character)
    return application, world.world_id, character_id


@pytest.mark.asyncio
async def test_the_cast_character_is_performed_by_the_actor_and_main_tier(
    mock_build_container, test_config, catalog,
) -> None:
    application, world_id, character_id = await _built(mock_build_container, test_config, catalog)
    assert not application.get_session(world_id).world.agents[character_id].is_main_character

    world = await application.cast_actor(world_id, character_id)

    assert application.get_session(world_id).world is world
    assert [a.agent_id for a in world.agents.values() if a.actor is not None] == [character_id]
    assert world.agents[character_id].is_main_character
    assert catalog.actor_casting(world_id) == ActorCasting(character_id=character_id)


@pytest.mark.asyncio
async def test_the_casting_survives_a_restore_in_a_new_process(
    mock_build_container, test_config, catalog,
) -> None:
    application, world_id, character_id = await _built(mock_build_container, test_config, catalog)
    await application.cast_actor(world_id, character_id)

    restored = await NarrativeApplication(
        mock_build_container, test_config, catalog=catalog,
    ).restore_session(world_id)

    assert restored.agents[character_id].actor is not None
    assert restored.agents[character_id].is_main_character


@pytest.mark.asyncio
async def test_a_confirmed_world_takes_no_actor(mock_build_container, test_config, catalog) -> None:
    application, world_id, character_id = await _built(mock_build_container, test_config, catalog)
    catalog.set_confirmed(world_id, True)

    with pytest.raises(ValueError, match="confirmed"):
        await application.cast_actor(world_id, character_id)
    assert catalog.actor_casting(world_id) is None


@pytest.mark.asyncio
async def test_a_world_that_has_run_takes_no_actor(mock_build_container, test_config, catalog) -> None:
    application, world_id, character_id = await _built(mock_build_container, test_config, catalog)
    await application.run_world(world_id, steps=1)

    with pytest.raises(ValueError, match="already run"):
        await application.cast_actor(world_id, character_id)


@pytest.mark.asyncio
async def test_an_unknown_character_changes_nothing(mock_build_container, test_config, catalog) -> None:
    application, world_id, _character_id = await _built(mock_build_container, test_config, catalog)
    session = application.get_session(world_id)

    with pytest.raises(ValueError, match="no character"):
        await application.cast_actor(world_id, "nobody")

    assert application.get_session(world_id) is session
    assert catalog.actor_casting(world_id) is None


@pytest.mark.asyncio
async def test_the_choice_can_change_before_the_world_is_confirmed(
    mock_build_container, test_config, catalog,
) -> None:
    application, world_id, character_id = await _built(mock_build_container, test_config, catalog)
    other_id = next(a for a in application.get_session(world_id).world.agents if a != character_id)
    await application.cast_actor(world_id, character_id)

    world = await application.cast_actor(world_id, other_id)

    assert [a.agent_id for a in world.agents.values() if a.actor is not None] == [other_id]


@pytest.mark.asyncio
async def test_an_exit_is_for_good(mock_build_container, test_config, catalog) -> None:
    application, world_id, character_id = await _built(mock_build_container, test_config, catalog)
    await application.cast_actor(world_id, character_id)
    catalog.set_confirmed(world_id, True)

    world = await application.release_actor(world_id)

    assert not any(a.actor is not None for a in world.agents.values())
    assert world.agents[character_id].is_main_character
    assert catalog.actor_casting(world_id) == ActorCasting(character_id=character_id, exited=True)
    with pytest.raises(ValueError, match="No actor"):
        await application.release_actor(world_id)
    catalog.set_confirmed(world_id, False)
    with pytest.raises(ValueError, match="can't return"):
        await application.cast_actor(world_id, character_id)


@pytest.mark.asyncio
async def test_a_reset_keeps_the_casting(mock_build_container, test_config, catalog) -> None:
    application, world_id, character_id = await _built(mock_build_container, test_config, catalog)
    await application.cast_actor(world_id, character_id)

    world = await application.reset_session(world_id)

    assert world.agents[character_id].actor is not None


@pytest.mark.asyncio
async def test_stopping_at_the_actors_turn_discards_the_half_run_step(
    mock_build_container, test_config, catalog,
) -> None:
    application, world_id, character_id = await _built(mock_build_container, test_config, catalog)
    await application.cast_actor(world_id, character_id)
    application.start_run(world_id, steps=3)
    session = application.get_session(world_id)
    task = session.task
    assert task is not None
    for _ in range(5000):
        if session.runtime.actor.pending_turn is not None:
            break
        await asyncio.sleep(0)
    assert session.runtime.actor.pending_turn is not None

    application.stop_run(world_id)
    await asyncio.wait_for(task, timeout=30.0)

    assert not application.has_session(world_id)
    assert max(await mock_build_container.snapshot.list_steps(world_id)) == 0
    restored = await application.restore_session(world_id)
    assert restored.current_step == 0
    assert restored.agents[character_id].actor is not None
