"""FileAgentStore id validation covers every path, not just delete_world."""

from __future__ import annotations

import pytest

from core.interfaces.agent_store import AgentRelation, AgentState
from providers.agent_store.file import FileAgentStore


def _agent_state(world_id: str, agent_id: str) -> AgentState:
    return AgentState(
        world_id=world_id,
        agent_id=agent_id,
        updated_step=0,
        current_emotion="calm",
        emotion_intensity=0.0,
        emotion_valence=0.0,
        emotion_triggered_by=None,
        active_needs=[],
        dominant_need=None,
        long_term_goals=[],
        short_term_goals=[],
        current_location="palace",
        activity_status="idle",
        activity_target=None,
        action_status="idle",
        current_action=None,
        action_remaining_steps=0,
        last_action=None,
        last_action_result=None,
        last_action_succeeded=None,
    )


@pytest.fixture
def store(tmp_path) -> FileAgentStore:
    return FileAgentStore(path=str(tmp_path))


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["../escape", "a/b", ""])
async def test_bad_world_id_rejected_on_every_path(store: FileAgentStore, bad: str) -> None:
    for call in (
        store.save_agent_state(bad, "a1", _agent_state(bad, "a1")),
        store.load_agent_state(bad, "a1"),
        store.list_agent_ids(bad),
        store.load_all_relations(bad, "a1"),
        store.delete_world(bad),
    ):
        with pytest.raises(ValueError, match="Invalid world_id"):
            await call


@pytest.mark.asyncio
async def test_bad_agent_id_rejected(store: FileAgentStore) -> None:
    with pytest.raises(ValueError, match="Invalid agent_id"):
        await store.load_agent_state("w1", "../secret")


@pytest.mark.asyncio
async def test_bad_relation_id_rejected(store: FileAgentStore) -> None:
    relation = AgentRelation(
        world_id="w1",
        from_id="a1",
        to_id="../b",
        trust_objective=0.5,
        affection_objective=0.0,
        updated_step=0,
    )
    with pytest.raises(ValueError, match="Invalid to_id"):
        await store.save_relation(relation)


@pytest.mark.asyncio
async def test_valid_ids_round_trip(store: FileAgentStore) -> None:
    await store.save_agent_state("world_1", "li-shimin", _agent_state("world_1", "li-shimin"))
    assert await store.list_agent_ids("world_1") == ["li-shimin"]
    loaded = await store.load_agent_state("world_1", "li-shimin")
    assert loaded is not None and loaded.current_location == "palace"
