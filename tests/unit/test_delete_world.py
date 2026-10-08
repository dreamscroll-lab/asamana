"""Whole-world delete contract tests: the in-memory delete_world must clear all of the target
world's state and leave other worlds untouched. delete_world is an atomic sweep across providers,
so a provider that clears another world, or misses part of its own, breaks
`NarrativeApplication.delete_world`."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from core.interfaces.agent_store import AgentRelation, AgentState
from core.interfaces.snapshot import WorldSnapshot
from core.interfaces.trace import LLMCallTrace, StepTrace
from providers.agent_store.in_memory import InMemoryAgentStore
from providers.snapshot.in_memory import InMemorySnapshotProvider
from providers.trace.in_memory import InMemoryTraceSink
from providers.trace.null import NullTraceSink
from providers.vector_store.in_memory import InMemoryVectorStore
from world.catalog import WorldCatalog


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


def _relation(world_id: str, from_id: str, to_id: str) -> AgentRelation:
    return AgentRelation(
        world_id=world_id,
        from_id=from_id,
        to_id=to_id,
        trust_objective=0.5,
        affection_objective=0.0,
        updated_step=0,
    )


@pytest.mark.asyncio
async def test_in_memory_snapshot_delete_world_purges_only_target() -> None:
    provider = InMemorySnapshotProvider()
    a = WorldSnapshot(world_id="wa", step=1, timestamp=datetime(2026, 1, 1))
    b = WorldSnapshot(world_id="wb", step=1, timestamp=datetime(2026, 1, 2))
    await provider.save("wa", 1, a)
    await provider.save("wb", 1, b)
    await provider.save_world_config("wa", {"theme": "alpha"})
    await provider.save_world_config("wb", {"theme": "beta"})

    await provider.delete_world("wa")

    assert await provider.list_steps("wa") == []
    assert await provider.load_world_config("wa") is None
    assert await provider.list_steps("wb") == [1]
    assert await provider.load_world_config("wb") == {"theme": "beta"}


@pytest.mark.asyncio
async def test_in_memory_agent_store_delete_world_purges_only_target() -> None:
    store = InMemoryAgentStore()
    await store.save_agent_state("wa", "a1", _agent_state("wa", "a1"))
    await store.save_initial_agent_state("wa", "a1", _agent_state("wa", "a1"))
    await store.save_relation(_relation("wa", "a1", "a2"))
    await store.save_initial_relation(_relation("wa", "a1", "a2"))
    await store.save_agent_state("wb", "b1", _agent_state("wb", "b1"))
    await store.save_initial_agent_state("wb", "b1", _agent_state("wb", "b1"))
    await store.save_relation(_relation("wb", "b1", "b2"))
    await store.save_initial_relation(_relation("wb", "b1", "b2"))

    await store.delete_world("wa")

    assert await store.list_agent_ids("wa") == []
    assert await store.load_agent_state("wa", "a1") is None
    assert await store.load_initial_agent_state("wa", "a1") is None
    assert await store.load_relation("wa", "a1", "a2") is None
    assert await store.load_all_initial_relations("wa", "a1") == []
    # Other world untouched.
    assert await store.list_agent_ids("wb") == ["b1"]
    assert await store.load_agent_state("wb", "b1") is not None
    assert await store.load_relation("wb", "b1", "b2") is not None
    assert len(await store.load_all_initial_relations("wb", "b1")) == 1


@pytest.mark.asyncio
async def test_in_memory_vector_store_delete_world_purges_only_target() -> None:
    store = InMemoryVectorStore()
    await store.create_collection("wa:a1:memory:factual", dimension=2)
    await store.create_collection("wa:a2:memory:experiential", dimension=2)
    await store.create_collection("wb:b1:memory:factual", dimension=2)
    await store.upsert("wa:a1:memory:factual", "m1", [1.0, 0.0], None, {"t": "x"})
    await store.upsert("wb:b1:memory:factual", "m2", [0.0, 1.0], None, {"t": "y"})

    await store.delete_world("wa")

    assert len(await store.list_all("wa:a1:memory:factual")) == 0
    assert len(await store.list_all("wa:a2:memory:experiential")) == 0
    # dimension entry gone too — a fresh create_collection with a different dim
    # must succeed after delete_world.
    await store.create_collection("wa:a1:memory:factual", dimension=3)
    await store.upsert("wa:a1:memory:factual", "m3", [1.0, 0.0, 0.0], None, {"t": "z"})
    assert len(await store.list_all("wa:a1:memory:factual")) == 1
    # Other world untouched.
    assert len(await store.list_all("wb:b1:memory:factual")) == 1


def test_in_memory_trace_sink_delete_world_purges_only_target() -> None:
    sink = InMemoryTraceSink()
    prompt = [{"role": "user", "content": ""}]
    sink.record_llm_call(
        LLMCallTrace(
            world_id="wa", stage="decision", scene="AGENT_DECISION_MAIN",
            prompt_messages=prompt, response_content="", temperature=0.0, max_tokens=0,
            input_tokens=0, output_tokens=0, model="mock", latency_ms=1.0,
            timestamp="2026-01-01T00:00:00", agent_id="a1", step=0,
        )
    )
    sink.record_llm_call(
        LLMCallTrace(
            world_id="wb", stage="decision", scene="AGENT_DECISION_MAIN",
            prompt_messages=prompt, response_content="", temperature=0.0, max_tokens=0,
            input_tokens=0, output_tokens=0, model="mock", latency_ms=1.0,
            timestamp="2026-01-01T00:00:00", agent_id="b1", step=0,
        )
    )
    sink.record_step(StepTrace(world_id="wa", step=0, world_time={}, wall_ms=1.0, timestamp=""))
    sink.record_step(StepTrace(world_id="wb", step=0, world_time={}, wall_ms=1.0, timestamp=""))

    sink.delete_world("wa")

    assert sink.read_calls("wa") == []
    assert sink.read_step_summaries("wa") == []
    assert len(sink.read_calls("wb")) == 1
    assert len(sink.read_step_summaries("wb")) == 1


def test_null_trace_sink_delete_world_is_noop() -> None:
    NullTraceSink().delete_world("wa")  # must not raise


def test_world_catalog_remove_purges_entry_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "worlds.json"
    cat = WorldCatalog(str(path))
    cat.register("wa", theme="alpha")
    cat.register("wb", theme="beta")

    cat.remove("wa")

    assert cat.get("wa") is None
    assert cat.get("wb") is not None
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert set(on_disk) == {"wb"}


def test_world_catalog_remove_missing_is_noop(tmp_path: Path) -> None:
    cat = WorldCatalog(str(tmp_path / "worlds.json"))
    cat.register("wa", theme="alpha")
    cat.remove("does-not-exist")  # must not raise
    assert cat.get("wa") is not None


def test_world_catalog_remove_merges_disk_changes(tmp_path: Path) -> None:
    """remove shares a source with register: a world another instance registers later must not be
    wiped by this instance's remove flush."""
    path = tmp_path / "worlds.json"
    cat_a = WorldCatalog(str(path))
    cat_b = WorldCatalog(str(path))
    cat_a.register("wa")
    cat_b.register("wb")
    cat_a.remove("wa")
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert set(on_disk) == {"wb"}
