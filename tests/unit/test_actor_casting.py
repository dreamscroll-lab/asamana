"""Casting the actor: chosen when the world is confirmed, kept across restores, and given up once
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
    """A world built and not yet confirmed, and a background character to cast."""
    application = NarrativeApplication(container, test_config, catalog=catalog)
    world_id = await application.build_world("玄武门之变", template="changan_iso")
    setup = await application.world_setup(world_id)
    character_id = next(d.agent_id for d in setup.characters if not d.is_main_character)
    return application, world_id, character_id


@pytest.mark.asyncio
async def test_the_cast_character_is_performed_by_the_actor_and_main_tier(
    mock_build_container, test_config, catalog,
) -> None:
    application, world_id, character_id = await _built(mock_build_container, test_config, catalog)

    world = await application.confirm_world(world_id, character_id=character_id)

    assert application.get_session(world_id).world is world
    assert [a.agent_id for a in world.agents.values() if a.actor is not None] == [character_id]
    assert world.agents[character_id].is_main_character
    assert catalog.actor_casting(world_id) == ActorCasting(character_id=character_id)
    assert catalog.get(world_id)["confirmed"]


@pytest.mark.asyncio
async def test_the_casting_survives_a_restore_in_a_new_process(
    mock_build_container, test_config, catalog,
) -> None:
    application, world_id, character_id = await _built(mock_build_container, test_config, catalog)
    await application.confirm_world(world_id, character_id=character_id)

    restored = await NarrativeApplication(
        mock_build_container, test_config, catalog=catalog,
    ).restore_session(world_id)

    assert restored.agents[character_id].actor is not None
    assert restored.agents[character_id].is_main_character


@pytest.mark.asyncio
async def test_a_confirmed_world_takes_no_actor(mock_build_container, test_config, catalog) -> None:
    application, world_id, character_id = await _built(mock_build_container, test_config, catalog)
    await application.confirm_world(world_id)

    with pytest.raises(ValueError, match="confirmed"):
        await application.confirm_world(world_id, character_id=character_id)
    assert catalog.actor_casting(world_id) is None


@pytest.mark.asyncio
async def test_a_world_the_director_reviewed_takes_no_actor(mock_build_container, test_config, catalog) -> None:
    application, world_id, character_id = await _built(mock_build_container, test_config, catalog)
    await application.open_director_review(world_id)

    with pytest.raises(ValueError, match="director"):
        await application.confirm_world(world_id, character_id=character_id)
    assert not catalog.get(world_id).get("confirmed")


@pytest.mark.asyncio
async def test_an_unknown_character_changes_nothing(mock_build_container, test_config, catalog) -> None:
    application, world_id, _character_id = await _built(mock_build_container, test_config, catalog)

    with pytest.raises(ValueError, match="no character"):
        await application.confirm_world(world_id, character_id="nobody")

    assert not application.has_session(world_id)
    assert await mock_build_container.snapshot.list_steps(world_id) == []
    assert catalog.actor_casting(world_id) is None
    assert not catalog.get(world_id).get("confirmed")


@pytest.mark.asyncio
async def test_an_exit_is_for_good(mock_build_container, test_config, catalog) -> None:
    application, world_id, character_id = await _built(mock_build_container, test_config, catalog)
    await application.confirm_world(world_id, character_id=character_id)

    world = await application.release_actor(world_id)

    assert not any(a.actor is not None for a in world.agents.values())
    assert world.agents[character_id].is_main_character
    assert catalog.actor_casting(world_id) == ActorCasting(character_id=character_id, exited=True)
    with pytest.raises(ValueError, match="No actor"):
        await application.release_actor(world_id)


@pytest.mark.asyncio
async def test_a_reset_keeps_the_casting(mock_build_container, test_config, catalog) -> None:
    application, world_id, character_id = await _built(mock_build_container, test_config, catalog)
    await application.confirm_world(world_id, character_id=character_id)

    world = await application.reset_session(world_id)

    assert world.agents[character_id].actor is not None


@pytest.mark.asyncio
async def test_stopping_at_the_actors_turn_discards_the_half_run_step(
    mock_build_container, test_config, catalog,
) -> None:
    application, world_id, character_id = await _built(mock_build_container, test_config, catalog)
    await application.confirm_world(world_id, character_id=character_id)
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
