"""FileVectorStore: the InMemory similarity logic plus JSON persistence on disk."""

import pytest

from providers.vector_store.file import FileVectorStore


@pytest.mark.asyncio
async def test_persists_across_instances(tmp_path) -> None:
    """Records upserted before creating a new instance (simulating a restart) are still searchable."""
    path = str(tmp_path / "vec")
    store = FileVectorStore(path=path)
    await store.create_collection("w:a:memory:factual", dimension=3)
    await store.upsert(
        "w:a:memory:factual", "m1",
        dense_vector=[1.0, 0.0, 0.0], sparse_vector={5: 0.9}, payload={"text": "alpha"},
    )

    reloaded = FileVectorStore(path=path)
    results = await reloaded.search(
        "w:a:memory:factual", dense_vector=[1.0, 0.0, 0.0], sparse_vector=None, top_k=5
    )
    assert [r.id for r in results] == ["m1"]
    assert results[0].payload["text"] == "alpha"


@pytest.mark.asyncio
async def test_sparse_int_keys_survive_roundtrip(tmp_path) -> None:
    """Sparse int keys become strings in JSON and must be converted back on reload, or they never
    intersect the query's int keys."""
    path = str(tmp_path / "vec")
    store = FileVectorStore(path=path)
    await store.create_collection("c", dimension=2)
    await store.upsert("c", "m1", dense_vector=[0.0, 0.0], sparse_vector={7: 1.0}, payload={})

    reloaded = FileVectorStore(path=path)
    # Sparse-only hit (all-zero dense gives cosine 0): the score is > 0 only through the shared
    # sparse key 7.
    results = await reloaded.search(
        "c", dense_vector=[0.0, 0.0], sparse_vector={7: 1.0}, top_k=5,
        dense_weight=0.0, sparse_weight=1.0,
    )
    assert results and results[0].id == "m1"
    assert results[0].score > 0.0


@pytest.mark.asyncio
async def test_reupsert_last_wins_across_reload(tmp_path) -> None:
    """Repeated upserts of one id append several lines; reload keeps only the last."""
    path = str(tmp_path / "vec")
    store = FileVectorStore(path=path)
    await store.create_collection("c", dimension=2)
    await store.upsert("c", "m1", dense_vector=[1.0, 0.0], sparse_vector=None, payload={"v": 1})
    await store.upsert("c", "m1", dense_vector=[0.0, 1.0], sparse_vector=None, payload={"v": 2})

    reloaded = FileVectorStore(path=path)
    got = await reloaded.list_all("c")
    assert [(r.id, r.payload["v"]) for r in got] == [("m1", 2)]


@pytest.mark.asyncio
async def test_delete_persists_across_reload(tmp_path) -> None:
    """delete writes a tombstone line, so the id is gone after reload."""
    path = str(tmp_path / "vec")
    store = FileVectorStore(path=path)
    await store.create_collection("c", dimension=2)
    await store.upsert("c", "m1", dense_vector=[1.0, 0.0], sparse_vector=None, payload={})
    await store.delete("c", "m1")

    reloaded = FileVectorStore(path=path)
    assert len(await reloaded.list_all("c")) == 0


@pytest.mark.asyncio
async def test_dimension_mismatch_raises(tmp_path) -> None:
    """create_collection fixes the dimension; an upsert with the wrong dimension raises instead of
    silently corrupting data (as Chroma does)."""
    store = FileVectorStore(path=str(tmp_path / "vec"))
    await store.create_collection("c", dimension=3)
    with pytest.raises(ValueError):
        await store.upsert("c", "m1", dense_vector=[1.0, 2.0], sparse_vector=None, payload={})


@pytest.mark.asyncio
async def test_readonly_instance_never_rewrites_files(tmp_path) -> None:
    """A read-only process (list/observe/web) must not rewrite the .jsonl after loading. Compacting
    on startup would silently erase records that a concurrently running writer appended between the
    read and the atomic replace."""
    writer = FileVectorStore(path=str(tmp_path))
    await writer.create_collection("c", dimension=2)
    await writer.upsert("c", "m1", [1.0, 0.0], None, {"k": "v"})
    await writer.upsert("c", "m1", [0.0, 1.0], None, {"k": "v2"})  # append a redundant line

    fp = next(tmp_path.glob("*.jsonl"))
    before = fp.read_bytes()

    reader = FileVectorStore(path=str(tmp_path))  # read-only use: load only
    assert len(await reader.list_all("c")) == 1
    assert fp.read_bytes() == before  # file untouched


@pytest.mark.asyncio
async def test_first_write_compacts_stale_lines(tmp_path) -> None:
    """A writer compacts on its first write to the collection: redundant lines collapse to the live
    records, including the new one."""
    w1 = FileVectorStore(path=str(tmp_path))
    await w1.create_collection("c", dimension=2)
    await w1.upsert("c", "m1", [1.0, 0.0], None, {"k": "v"})
    await w1.upsert("c", "m1", [0.0, 1.0], None, {"k": "v2"})
    await w1.upsert("c", "m2", [1.0, 1.0], None, {})
    await w1.delete("c", "m2")

    w2 = FileVectorStore(path=str(tmp_path))
    await w2.upsert("c", "m3", [0.5, 0.5], None, {})  # first write triggers compaction

    fp = next(tmp_path.glob("*.jsonl"))
    lines = [l for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
    # meta + m1 (last write wins) + m3; m2's tombstone and the redundant lines are gone.
    assert len(lines) == 3
    w3 = FileVectorStore(path=str(tmp_path))
    records = {r.id: r.payload for r in await w3.list_all("c")}
    assert len(records) == 2
    assert records["m1"]["k"] == "v2"


@pytest.mark.asyncio
async def test_a_cut_short_line_costs_only_itself(tmp_path) -> None:
    """A write cut short by a crash leaves a partial line; reload keeps every other record, and the
    first write's compaction doesn't erase them."""
    w1 = FileVectorStore(path=str(tmp_path))
    await w1.create_collection("c", dimension=2)
    for i in range(3):
        await w1.upsert("c", f"m{i}", [1.0, 0.0], None, {"i": i})
    fp = next(tmp_path.glob("*.jsonl"))
    with fp.open("a", encoding="utf-8") as f:
        f.write('{"id": "m9", "dense": [0.1, ')

    w2 = FileVectorStore(path=str(tmp_path))
    assert len(await w2.list_all("c")) == 3
    await w2.upsert("c", "m3", [0.0, 1.0], None, {})

    w3 = FileVectorStore(path=str(tmp_path))
    assert len(await w3.list_all("c")) == 4


@pytest.mark.asyncio
async def test_search_filter_scalar_equality_and_list_membership(tmp_path) -> None:
    """A filter's predicate depends on the payload field type: scalar means equality, list means
    membership.

    Membership is what person-scoped retrieval ("what I know about someone") relies on:
    related_agents is a list, and the id restricts the candidate set.
    """
    store = FileVectorStore(path=str(tmp_path))
    await store.create_collection("c", dimension=2)
    await store.upsert("c", "m1", [1.0, 0.0], None,
                       {"kind": "event", "related_agents": ["a", "b"]})
    await store.upsert("c", "m2", [1.0, 0.0], None,
                       {"kind": "event", "related_agents": ["b"]})
    await store.upsert("c", "m3", [1.0, 0.0], None,
                       {"kind": "insight", "related_agents": []})

    membership = await store.search("c", [1.0, 0.0], None, top_k=10,
                                    filters={"related_agents": "a"})
    assert [r.id for r in membership] == ["m1"]

    both = await store.search("c", [1.0, 0.0], None, top_k=10,
                              filters={"related_agents": "b"})
    assert sorted(r.id for r in both) == ["m1", "m2"]

    # Scalar fields still use equality, and both kinds can be combined (AND).
    scalar = await store.search("c", [1.0, 0.0], None, top_k=10, filters={"kind": "insight"})
    assert [r.id for r in scalar] == ["m3"]
    conjunction = await store.search("c", [1.0, 0.0], None, top_k=10,
                                     filters={"kind": "event", "related_agents": "a"})
    assert [r.id for r in conjunction] == ["m1"]

    # Out-of-scope memories don't take top_k slots: m2/m3 have the same similarity but aren't returned.
    assert all(r.id != "m3" for r in membership)


@pytest.mark.asyncio
async def test_search_result_exposes_raw_dense_score(tmp_path) -> None:
    """SearchResult returns both the fused score and the raw dense score (plain cosine, independent
    of the weights)."""
    store = FileVectorStore(path=str(tmp_path))
    await store.create_collection("c", dimension=2)
    await store.upsert("c", "m1", [1.0, 0.0], None, {})

    [hit] = await store.search("c", [1.0, 0.0], None, top_k=5, dense_weight=0.4, sparse_weight=0.6)
    assert hit.dense_score == pytest.approx(1.0)   # perfect match, independent of weights
    assert hit.score == pytest.approx(0.4)         # fused is scaled by dense_weight


def test_sparse_similarity_is_bounded_cosine() -> None:
    """Sparse similarity is a cosine in [0,1], on dense's scale, not an unbounded dot product:
    otherwise fused = dense*w_d + sparse*w_s is unbounded and no absolute floor on it means anything.
    """
    from providers.vector_store.similarity import sparse_similarity

    heavy = {1: 3.0, 2: 4.0}          # norm 5; a raw dot product would give 25
    assert sparse_similarity(heavy, heavy) == pytest.approx(1.0)   # self-similarity is exactly 1, not 25
    assert 0.0 <= sparse_similarity(heavy, {1: 1.0, 3: 9.0}) <= 1.0
    assert sparse_similarity(heavy, {7: 1.0}) == 0.0               # no shared keys
    assert sparse_similarity({}, heavy) == 0.0


def test_sparse_exact_hit_outranks_semantically_close_noise(tmp_path) -> None:
    """An exact proper-noun hit (high sparse, middling dense) must beat semantically similar noise.

    In a single-theme corpus dense scores sit in a narrow band and can't tell which memory is about
    this person; sparse can, and with both streams on one scale the hit rises.
    """
    from providers.vector_store.similarity import hybrid_score

    # proper-noun hit: exact sparse overlap, middling dense
    exact = {"dense": [0.5, 0.5], "sparse": {42: 1.0}, "payload": {}}
    # semantically close but without the name: higher dense, no sparse overlap
    noise = {"dense": [1.0, 0.0], "sparse": {7: 1.0}, "payload": {}}
    q_dense, q_sparse = [1.0, 0.0], {42: 1.0}

    exact_fused, _, _ = hybrid_score(exact, q_dense, q_sparse, 0.4, 0.6)
    noise_fused, _, _ = hybrid_score(noise, q_dense, q_sparse, 0.4, 0.6)
    assert exact_fused > noise_fused
    assert 0.0 <= exact_fused <= 1.0 and 0.0 <= noise_fused <= 1.0


@pytest.mark.asyncio
async def test_update_payload_keeps_vectors_and_persists(tmp_path) -> None:
    """update_payload replaces only the payload, keeps the stored vectors, and survives a restart.
    Write-backs that don't change the text (retrieve touch / decay) use it to avoid re-embedding."""
    path = str(tmp_path / "vec")
    store = FileVectorStore(path=path)
    await store.create_collection("c", dimension=2)
    await store.upsert("c", "m1", dense_vector=[1.0, 0.0], sparse_vector={7: 1.0},
                       payload={"text": "alpha", "last_accessed_step": 1})

    assert await store.update_payload("c", "m1", {"text": "alpha", "last_accessed_step": 9})

    reloaded = FileVectorStore(path=path)
    (m1,) = [r for r in await reloaded.list_all("c") if r.id == "m1"]
    assert m1.payload["last_accessed_step"] == 9
    # The vectors survive: a sparse-only hit shows the payload write-back didn't drop the sparse vector.
    results = await reloaded.search("c", dense_vector=[0.0, 0.0], sparse_vector={7: 1.0},
                                    top_k=5, dense_weight=0.0, sparse_weight=1.0)
    assert [r.id for r in results] == ["m1"] and results[0].score > 0.0


@pytest.mark.asyncio
async def test_update_payload_missing_record_returns_false(tmp_path) -> None:
    """A missing record returns False, and the caller falls back to a full embed + upsert."""
    store = FileVectorStore(path=str(tmp_path / "vec"))
    await store.create_collection("c", dimension=2)
    assert await store.update_payload("c", "nope", {"v": 1}) is False


@pytest.mark.asyncio
async def test_orphan_payload_line_does_not_resurrect_deleted_record(tmp_path) -> None:
    """A payload line after a tombstone is an orphan with no vectors and is dropped on reload. It must
    not bring a deleted record back as a fragment."""
    path = tmp_path / "vec"
    store = FileVectorStore(path=str(path))
    await store.create_collection("c", dimension=2)
    await store.upsert("c", "m1", dense_vector=[1.0, 0.0], sparse_vector=None, payload={"v": 1})
    await store.delete("c", "m1")
    # Append a payload line after the tombstone by writing the file directly (cross-process write
    # ordering can produce this).
    log = next(path.glob("*.jsonl"))
    with log.open("a", encoding="utf-8") as f:
        f.write('{"id": "m1", "_payload": {"v": 2}}\n')

    reloaded = FileVectorStore(path=str(path))
    assert len(await reloaded.list_all("c")) == 0


@pytest.mark.asyncio
async def test_writes_racing_the_first_compaction_all_survive_reload(tmp_path, monkeypatch, caplog) -> None:
    """A process's first write compacts the collection in a worker thread; writes landing on the
    loop meanwhile must neither break that rewrite (a dict growing under iteration) nor be erased by
    it (an append to the old file that the rewrite then replaces)."""
    import asyncio
    import time

    path = str(tmp_path / "vec")
    earlier = FileVectorStore(path=path)
    await earlier.create_collection("c", dimension=1)
    for i in range(50):
        await earlier.upsert("c", f"old{i}", dense_vector=[1.0], sparse_vector=None, payload={})

    store = FileVectorStore(path=path)  # a new process: its first write will compact
    plain_dumps = FileVectorStore._dumps

    def _slow(obj):  # stretch the rewrite so the later writes land inside it
        time.sleep(0.002)
        return plain_dumps(obj)

    monkeypatch.setattr(FileVectorStore, "_dumps", staticmethod(_slow))
    first = asyncio.create_task(
        store.upsert("c", "first", dense_vector=[1.0], sparse_vector=None, payload={})
    )
    await asyncio.sleep(0.03)  # the rewrite is now part-way through the records
    await asyncio.gather(*(
        store.upsert("c", f"new{i}", dense_vector=[1.0], sparse_vector=None, payload={})
        for i in range(10)
    ))
    await first

    assert "vector_store_rewrite_failed" not in caplog.text
    assert len(await FileVectorStore(path=path).list_all("c")) == 61
