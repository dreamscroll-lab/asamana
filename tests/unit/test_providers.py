from __future__ import annotations

from datetime import datetime

import pytest

from core.interfaces.message import Message
from core.interfaces.snapshot import EngineState, WorldSnapshot, WorldState


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
async def test_snapshot_save_and_load_freeze_nested_records(container) -> None:
    snapshot = WorldSnapshot(
        "w", 0, datetime.now(),
        state=WorldState(agents={"a": {"goals": ["survive"]}}),
        engine=EngineState(active_executions=[{"members": ["a"]}]),
    )
    await container.snapshot.save("w", 0, snapshot)
    snapshot.state.agents["a"]["goals"].clear()
    snapshot.engine.active_executions[0]["members"].append("b")
    loaded = await container.snapshot.load("w", 0)
    assert loaded.state.agents["a"]["goals"] == ["survive"]
    assert loaded.engine.active_executions[0]["members"] == ["a"]
    loaded.state.agents.clear()
    assert (await container.snapshot.load_latest("w")).state.agents


@pytest.mark.asyncio
@pytest.mark.parametrize("artifact", ["config", "setup", "map"])
async def test_world_artifact_reads_and_writes_are_isolated(container, artifact) -> None:
    save = getattr(container.snapshot, f"save_world_{artifact}")
    load = getattr(container.snapshot, f"load_world_{artifact}")
    data = {"nested": {"items": ["seed"]}}
    await save("w", data)
    data["nested"]["items"].append("external write")
    loaded = await load("w")
    assert loaded == {"nested": {"items": ["seed"]}}
    loaded["nested"]["items"].clear()
    assert (await load("w"))["nested"]["items"] == ["seed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["in_memory", "file"])
async def test_vector_payload_input_and_results_are_isolated(tmp_path, provider) -> None:
    from providers.vector_store.file import FileVectorStore
    from providers.vector_store.in_memory import InMemoryVectorStore

    store = FileVectorStore(str(tmp_path)) if provider == "file" else InMemoryVectorStore()
    payload = {"nested": {"related_agents": ["a"]}}
    await store.upsert("c", "m", [1.0], None, payload)
    payload["nested"]["related_agents"].append("external write")
    listed = (await store.list_all("c"))[0]
    assert listed.payload == {"nested": {"related_agents": ["a"]}}
    listed.payload["nested"]["related_agents"].append("external list")
    hit = (await store.search("c", [1.0], None))[0]
    assert hit.payload == {"nested": {"related_agents": ["a"]}}
    hit.payload["nested"]["related_agents"].clear()
    assert (await store.list_all("c"))[0].payload == {"nested": {"related_agents": ["a"]}}
    await store.update_payload("c", "m", payload)
    payload["nested"]["related_agents"].clear()
    assert (await store.list_all("c"))[0].payload["nested"]["related_agents"] == ["a", "external write"]


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
        events=[{"kind": "action", "seq": 3, "initiator_id": "a1"}],
    )
    earlier_snapshot = WorldSnapshot(
        world_id="world-1",
        step=1,
        timestamp=datetime(2026, 1, 1, 7, 0),
        events=[{"kind": "action", "seq": 1, "initiator_id": "a1"}],
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
