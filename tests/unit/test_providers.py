from __future__ import annotations

from datetime import datetime

import pytest

from core.interfaces.agent_store import AgentRelation, AgentState
from core.interfaces.message import Message
from core.interfaces.snapshot import WorldSnapshot


@pytest.mark.asyncio
async def test_in_memory_embedding_provider(container) -> None:
    result = await container.embedding.embed("abc")
    other = await container.embedding.embed("xyz")

    assert len(result.dense) == 4
    assert result.sparse is None
    assert result.dense != other.dense  # different texts produce distinct vectors
    assert await container.embedding.embed("abc") == result  # deterministic per text
    assert container.embedding.dimension == 4


@pytest.mark.asyncio
async def test_in_memory_vector_store(container) -> None:
    await container.vector_store.create_collection("memories", dimension=4)
    await container.vector_store.upsert(
        "memories",
        "1",
        [1.0, 0.0, 0.0, 0.0],
        {0: 1.0},
        {"text": "alpha", "stream": "factual"},
    )
    await container.vector_store.upsert(
        "memories",
        "2",
        [0.0, 1.0, 0.0, 0.0],
        {1: 1.0},
        {"text": "beta", "stream": "experiential"},
    )

    results = await container.vector_store.search(
        "memories",
        [1.0, 0.0, 0.0, 0.0],
        {0: 1.0},
        top_k=2,
    )
    all_records = await container.vector_store.list_all("memories")
    await container.vector_store.delete("memories", "1")
    after_delete = await container.vector_store.list_all("memories")

    assert [result.id for result in results] == ["1", "2"]
    assert {result.id: result.payload["text"] for result in all_records}["2"] == "beta"
    assert {result.id for result in all_records} == {"1", "2"}
    assert [result.id for result in after_delete] == ["2"]


@pytest.mark.asyncio
async def test_in_memory_agent_store(container) -> None:
    state = AgentState(
        world_id="world-1",
        agent_id="agent-1",
        updated_step=1,
        current_emotion="calm",
        emotion_intensity=0.2,
        emotion_valence=0.0,
        emotion_triggered_by=None,
        active_needs=["safety"],
        dominant_need="safety",
        long_term_goals=["survive"],
        short_term_goals=["wait"],
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
    relation = AgentRelation(
        world_id="world-1",
        from_id="agent-1",
        to_id="agent-2",
        trust_objective=0.6,
        affection_objective=0.2,
        labels=["ally"],
        updated_step=1,
    )

    await container.agent_store.save_agent_state("world-1", "agent-1", state)
    await container.agent_store.save_relation(relation)

    assert await container.agent_store.load_agent_state("world-1", "agent-1") == state
    assert await container.agent_store.list_agent_ids("world-1") == ["agent-1"]
    assert await container.agent_store.load_relation("world-1", "agent-1", "agent-2") == relation
    assert await container.agent_store.load_all_relations("world-1", "agent-1") == [relation]
    assert await container.agent_store.load_agent_state("world-2", "agent-1") is None


@pytest.mark.asyncio
async def test_in_memory_message_provider(container) -> None:
    due_message = Message(
        id="msg-1",
        world_id="world-1",
        sender_id="agent-1",
        content="Meet me at dawn",
        recipients=["agent-2"],
        location_scope=None,
        created_step=1,
        deliver_step=2,
    )
    future_message = Message(
        id="msg-2",
        world_id="world-1",
        sender_id="agent-3",
        content="Wait until dusk",
        recipients=["agent-2"],
        location_scope=None,
        created_step=1,
        deliver_step=3,
    )
    other_world_message = Message(
        id="msg-3",
        world_id="world-2",
        sender_id="agent-5",
        content="Other world",
        recipients=["agent-7"],
        location_scope=None,
        created_step=1,
        deliver_step=2,
    )

    await container.message_provider.enqueue(due_message)
    await container.message_provider.enqueue(future_message)
    await container.message_provider.enqueue(other_world_message)

    assert await container.message_provider.dequeue_ready("world-1", 1) == []
    assert await container.message_provider.peek_pending("world-1") == [due_message, future_message]
    assert await container.message_provider.dequeue_ready("world-1", 2) == [due_message]
    assert await container.message_provider.dequeue_ready("world-1", 2) == []
    assert await container.message_provider.dequeue_ready("world-2", 2) == [other_world_message]
    assert await container.message_provider.dequeue_ready("world-1", 3) == [future_message]


@pytest.mark.asyncio
async def test_in_memory_snapshot_provider(container) -> None:
    later_snapshot = WorldSnapshot(
        world_id="world-1",
        step=3,
        timestamp=datetime(2026, 1, 1, 9, 0),
        actions_this_step=[{"agent_id": "a1"}],
        metadata={"world_time": "09:00"},
    )
    earlier_snapshot = WorldSnapshot(
        world_id="world-1",
        step=1,
        timestamp=datetime(2026, 1, 1, 7, 0),
        actions_this_step=[{"agent_id": "a1"}],
        metadata={"world_time": "07:00"},
    )
    other_world_snapshot = WorldSnapshot(
        world_id="world-2",
        step=2,
        timestamp=datetime(2026, 1, 1, 8, 0),
    )

    await container.snapshot.save("world-1", 3, later_snapshot)
    await container.snapshot.save("world-1", 1, earlier_snapshot)
    await container.snapshot.save("world-2", 2, other_world_snapshot)

    assert await container.snapshot.load("world-1", 3) == later_snapshot
    assert await container.snapshot.list_steps("world-1") == [1, 3]
    assert await container.snapshot.load_latest("world-1") == later_snapshot
