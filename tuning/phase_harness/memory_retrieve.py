"""Retrieval / recall (retrieve_both) dry-run"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agent.memory_types import Memory
from config.models import Config
from core.container import Container
from core.context import clear_log_context, set_log_context
from core.logging import get_logger
from core.interfaces.perception import PerceivedIdentity
from providers.vector_store.in_memory import InMemoryVectorStore

from tuning.phase_harness.common import restore_traced
from tuning.phase_harness.scenario import resolve_ref, seed_memories, seed_view
from tuning.trace import InMemoryTraceSink, PhaseTrace

logger = get_logger(__name__)


def _query_view(q: Any) -> dict[str, Any]:
    """JSON-safe view: a dict (the four structured RetrievalQuery parts) or a str (plain text)."""
    if isinstance(q, dict):
        return {"kind": "structured", "spatial": q.get("spatial", ""), "message": q.get("message", ""),
                "broadcast": q.get("broadcast", ""), "goal": q.get("goal", "")}
    return {"kind": "text", "text": str(q or "")}


def _build_query(q: Any):
    """scene.query → a RetrievalQuery (structured, same shape the real _build_retrieval_query builds) or a plain str."""
    from agent.perception import RetrievalQuery
    if isinstance(q, dict):
        return RetrievalQuery(
            spatial_context=str(q.get("spatial", "")), message_context=str(q.get("message", "")),
            broadcast_context=str(q.get("broadcast", "")))
    return str(q or "")


def _recalled_mem_view(m: Memory) -> dict[str, Any]:
    return {"stream": m.stream.value, "kind": m.kind, "content": m.stored_content,
            "importance": round(float(m.importance), 3), "decay": round(float(m.decay_score), 3),
            "created_step": m.created_step, "valence": round(float(m.emotion_valence), 3)}


async def _retrieve_view(ms, query, step: int, top_k: int) -> dict[str, Any]:
    """Drive the production retrieve_both and return a structured recall view."""
    result = await ms.retrieve_both(query, current_step=step, top_k_each=top_k)
    return {
        "events": [_recalled_mem_view(m) for m in result.events],
        "insights": [_recalled_mem_view(m) for m in result.insights],
        "summaries": [_recalled_mem_view(m) for m in result.period_summaries],
        "stream_links": len(result.stream_links),
        "insight_sources": len(result.insight_sources),
    }


async def run_memory_retrieve(
    container: Container,
    config: Config,
    world_id: str,
    *,
    scenario: dict | None = None,
    sink: InMemoryTraceSink | None = None,
    trace_dir: str = "./data/tuning_traces",
) -> int:
    """Isolated dry-run of memory **retrieval** for a restored world.

    Rebuilds an isolated memory_system on the real text-embedding-v3 hybrid (DashScope native:
    dense + learned sparse), since the baseline in_memory dim-8 cosine is noise and relevance needs
    real embeddings. Seeds a narrative-faithful candidate pool, then drives the production
    retrieve_both (per-query min-max normalized). scenario.query is either plain text or structured
    (the four RetrievalQuery parts, the same shape the real _build_retrieval_query builds).
    kind=contrast supplies two queries to test query-sensitivity.
    Zero production intrusion: isolated embedding + InMemory store; baseline ./data is never written.
    """
    trace_sink = sink if sink is not None else InMemoryTraceSink()
    _, world, run_step, _ = await restore_traced(container, config, world_id, trace_sink)
    name_by_id = world.directory.all_agent_names()
    id_by_name = {v: k for k, v in name_by_id.items()}
    valid_ids = list(world.agents.keys())
    scenario = scenario or {}
    kind = scenario.get("kind", "single")
    step = int(scenario.get("step", 60))
    top_k = int(scenario.get("top_k", 5))

    actor_id = resolve_ref(scenario["actor"], id_by_name, valid_ids)
    agent = world.agents[actor_id]
    from providers.embedding.dashscope import DashScopeEmbeddingProvider
    ms = agent.memory_system
    ms._embedding = DashScopeEmbeddingProvider(model="text-embedding-v3", dimension=1024)  # noqa: SLF001
    ms._vector_store = InMemoryVectorStore()  # noqa: SLF001
    await ms.ensure_collections()
    for aid, nm in name_by_id.items():
        agent.remember_agent(aid, PerceivedIdentity(name=nm))
    seeded = await seed_memories(agent, world_id, scenario, step, id_by_name, valid_ids)
    pool = [{**seed_view(m), "content": m.stored_content} for m in ms._entries.values()]  # noqa: SLF001

    set_log_context(world_id=world_id, agent_id=actor_id, step=str(step))
    try:
        recalled = await _retrieve_view(ms, _build_query(scenario.get("query")), step, top_k)
        recalled_b = None
        if kind == "contrast":
            recalled_b = await _retrieve_view(ms, _build_query(scenario.get("query_b")), step, top_k)
        error = None
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory_retrieve_failed", extra={"world_id": world_id, "actor": actor_id, "error": str(exc)})
        recalled, recalled_b, error = {}, None, str(exc)
    clear_log_context()

    inputs = {"kind": kind, "actor_name": name_by_id.get(actor_id, actor_id), "step": step,
              "top_k": top_k, "seeded": seeded, "query": _query_view(scenario.get("query")), "pool": pool}
    outputs = {"kind": kind, "recalled": recalled, "error": error}
    if kind == "contrast":
        inputs["query_b"] = _query_view(scenario.get("query_b"))
        outputs["recalled_b"] = recalled_b
    trace_sink.record_phase(PhaseTrace(
        world_id=world_id, phase="stage.memory_retrieve", step=step, agent_id=actor_id,
        inputs=inputs, outputs=outputs, timestamp=datetime.now().isoformat()))
    logger.info("tuning_memory_retrieve_dry_run_complete",
                extra={"world_id": world_id, "step": step, "kind": kind, "seeded": seeded})
    return step
