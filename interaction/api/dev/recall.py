"""Recall inspector: a query through one agent's production retrieval, returned as the whole funnel."""

from __future__ import annotations

from typing import Any

from interaction.api.app import ApiServices
from interaction.api.dev.names import agent_names


def build_recall_router(services: ApiServices) -> Any:
    from fastapi import APIRouter, Body, HTTPException

    router = APIRouter(tags=["dev"])
    manager = services.manager
    container = services.container
    llm_router = container.llm_router

    @router.post("/api/worlds/{world_id}/memory/recall")
    async def memory_recall(world_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        """Recall inspector: run any query against an agent's memory and return the whole funnel.

        Per-candidate dense / sparse / normalized / final scores and drop reasons show whether the
        floor, the identity scope, MMR dedup or the candidate set is at fault, and let
        RETRIEVAL_SCORE_FLOOR be calibrated from data.

        - Use the same MemorySystem.retrieve as production; a shadow implementation would drift.
        - ``touch=False``: retrieve normally touch()es recalled memories (raising
          last_accessed_step), so a diagnostic read would change what it observes.
        """
        from agent.memory import MemorySystem
        from agent.memory_ranking import RETRIEVAL_SCORE_FLOOR
        from agent.memory_types import MemoryStream

        services.require_model_keys()
        agent_id = str(payload.get("agent_id") or "").strip()
        query = str(payload.get("query") or "").strip()
        if not agent_id:
            raise HTTPException(status_code=400, detail="agent_id is required")
        if not query:
            raise HTTPException(status_code=400, detail="query is required")

        stream_val = payload.get("stream") or None
        try:
            stream = MemoryStream(str(stream_val)) if stream_val else None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Unknown stream: {stream_val}") from exc

        top_k = max(1, min(int(payload.get("top_k") or 5), 50))
        # Identity scope (optional): when set, recall only memories that agent actually took part in;
        # this is the path "what I know about someone" uses.
        related_agent_id = (str(payload.get("related_agent_id") or "").strip()) or None
        # The floor is overridable so it can be swept in the UI and set from data, not back-fitted to
        # a single scenario.
        floor = payload.get("floor")
        floor = float(floor) if floor is not None else RETRIEVAL_SCORE_FLOOR
        current_step = int(payload.get("current_step") or 0)
        if not current_step:
            steps = await manager.list_steps(world_id)
            current_step = max(steps) if steps else 0

        memory = MemorySystem(
            llm_router,
            container.embedding,
            container.vector_store,
            world_id=world_id,
            agent_id=agent_id,
            retrieval_score_floor=floor,
        )

        names = await agent_names(manager, world_id)
        trace: list = []
        try:
            selected = await memory.retrieve(
                query,
                current_step=current_step,
                top_k=top_k,
                stream=stream,
                related_agent_id=related_agent_id,
                trace=trace,
                touch=False,
            )
        except Exception as exc:  # noqa: BLE001 — embed / store failure is upstream: 502, not 500
            raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}") from exc

        selected_ids = {m.id for m in selected}

        def _row(item: Any) -> dict[str, Any]:
            mem = item.memory
            return {
                "id": mem.id,
                "stream": mem.stream.value,
                "kind": mem.kind.value,
                "content": mem.stored_content,
                "created_step": mem.created_step,
                "importance_raw": round(float(mem.importance), 3),
                "decay": round(float(mem.decay_score), 3),
                "related_agents": [
                    {"id": aid, "name": names.get(aid, aid)} for aid in mem.related_agents
                ],
                # Funnel: raw relevance -> normalized three factors -> final score -> outcome
                "dense": round(item.dense, 4),
                "sparse": round(item.sparse, 4),
                "fused": round(item.fused, 4),
                "relevance_n": round(item.relevance, 3),
                "recency_n": round(item.recency, 3),
                "importance_n": round(item.importance, 3),
                "score": round(item.score, 4),
                "dropped": item.dropped,
                "selected": mem.id in selected_ids and item.dropped is None,
            }

        rows = [_row(item) for item in trace]
        rows.sort(key=lambda r: (r["dropped"] is not None, -r["score"], -r["dense"]))
        dense_scores = [r["dense"] for r in rows]
        return {
            "query": query,
            "agent_id": agent_id,
            "agent_name": names.get(agent_id, agent_id),
            "stream": stream.value if stream else "both",
            "related_agent_id": related_agent_id,
            "current_step": current_step,
            "floor": floor,
            "top_k": top_k,
            "candidates": rows,
            "stats": {
                "candidates": len(rows),
                "selected": len(selected_ids),
                "dropped_by_floor": sum(1 for r in rows if r["dropped"] == "floor"),
                "dropped_by_mmr": sum(1 for r in rows if r["dropped"] == "mmr_duplicate"),
                "dropped_by_top_k": sum(1 for r in rows if r["dropped"] == "top_k"),
                # The dense distribution is what the floor is calibrated against: in a
                # single-theme world even unrelated memories score a fair cosine, so only this
                # distribution shows whether an absolute threshold discriminates at all.
                "dense_min": round(min(dense_scores), 4) if dense_scores else None,
                "dense_max": round(max(dense_scores), 4) if dense_scores else None,
            },
        }

    @router.get("/api/worlds/{world_id}/memory/agents")
    async def memory_agents(world_id: str) -> dict[str, Any]:
        """Data for the recall inspector's agent picker (id + name)."""
        names = await agent_names(manager, world_id)
        return {"agents": [{"id": aid, "name": name} for aid, name in sorted(names.items(), key=lambda kv: kv[1])]}

    return router
