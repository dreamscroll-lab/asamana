"""The dry-run vector overlay keeps baseline reads and all writes isolated."""

from __future__ import annotations

from providers.vector_store.in_memory import InMemoryVectorStore
from tuning.dry_run import DryRunVectorStore


async def test_vector_shadow_preserves_vectors_and_filters_deleted_candidates() -> None:
    baseline = InMemoryVectorStore()
    collection = "world:agent:memory:factual"
    await baseline.upsert(collection, "best", [1.0, 0.0], None, {"kind": "event"})
    await baseline.upsert(collection, "next", [0.8, 0.2], None, {"kind": "event"})
    shadow = DryRunVectorStore(baseline)
    await shadow.delete(collection, "best")
    assert [r.id for r in await shadow.search(collection, [1.0, 0.0], None, top_k=1)] == ["next"]
    await shadow.upsert(collection, "next", [0.0, 1.0], None, {"kind": "insight"})
    await shadow.upsert(collection, "new", [1.0, 0.0], None, {"kind": "event"})
    assert [r.id for r in await shadow.search(
        collection, [1.0, 0.0], None, filters={"kind": "event"},
    )] == ["new"]
    assert {r.id for r in await baseline.list_all(collection)} == {"best", "next"}


async def test_payload_updates_and_reads_do_not_alias_baseline() -> None:
    baseline = InMemoryVectorStore()
    collection = "world:agent:memory:factual"
    await baseline.upsert(collection, "one", [1.0], None, {"nested": ["baseline"]})
    shadow = DryRunVectorStore(baseline)
    read = await shadow.list_all(collection)
    read[0].payload["nested"].append("local")
    assert (await baseline.list_all(collection))[0].payload == {"nested": ["baseline"]}
    assert await shadow.update_payload(collection, "one", {"nested": ["changed"]})
    assert (await shadow.search(collection, [1.0], None))[0].payload == {"nested": ["changed"]}
    assert (await shadow.search(
        collection, [1.0], None, filters={"nested": "changed"},
    ))[0].id == "one"
    assert await shadow.search(collection, [1.0], None, filters={"nested": "baseline"}) == []
    assert (await baseline.list_all(collection))[0].payload == {"nested": ["baseline"]}
    await shadow.delete_world("world")
    assert await shadow.list_all(collection) == []
    assert await baseline.list_all(collection)
