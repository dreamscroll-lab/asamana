"""LLM trace explorer: per-step dimensions, calls grouped by stage/step/agent, build calls, one call."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from core.interfaces.trace import (
    build_scene_sort_key,
    stage_sort_key,
    to_jsonable as trace_to_jsonable,
)

from interaction.api.app import ApiServices
from interaction.api.dev.names import agent_names
from interaction.models import WorldTimeView


def _aggregate(calls: list) -> dict[str, Any]:
    n = len(calls)
    failures = sum(1 for c in calls if not getattr(c, "ok", True))
    parsed = [c for c in calls if getattr(c, "parse_ok", None) is not None]
    parse_failures = sum(1 for c in parsed if c.parse_ok is False)
    # Adoption: of the calls whose consumer filed a verdict, how many were discarded.
    # Rated against `judged` (not `calls`) so uninstrumented paths — which file no
    # verdict — cannot dilute the rate into looking healthier than it is.
    judged = [c for c in calls if getattr(c, "adopted", None) is not None]
    discarded = sum(1 for c in judged if c.adopted is False)
    return {
        "calls": n,
        "failures": failures,
        "failure_rate": round(failures / n, 4) if n else 0.0,
        "parse_attempts": len(parsed),
        "parse_failures": parse_failures,
        "parse_failure_rate": round(parse_failures / len(parsed), 4) if parsed else 0.0,
        "adoption_verdicts": len(judged),
        "discarded": discarded,
        "discard_rate": round(discarded / len(judged), 4) if judged else 0.0,
        "input_tokens": sum(c.input_tokens for c in calls),
        "output_tokens": sum(c.output_tokens for c in calls),
        "latency_ms": round(sum(c.latency_ms for c in calls), 1),
    }


def _call_id_filter(calls: list, prefix: str | None) -> list:
    """Filter by call id prefix: list chips and logs show the first 8 chars, the clipboard holds
    the full id, and both must match."""
    if not prefix:
        return calls
    head = prefix.strip().lower()
    return [c for c in calls if c.call_id.lower().startswith(head)]


def _keyword_filter(calls: list, query: str | None) -> list:
    """Filter by keywords: every whitespace-separated term must appear (case-insensitive) in the
    prompt, response, thinking, error or extra."""
    terms = (query or "").lower().split()
    if not terms:
        return calls

    def _haystack(c: Any) -> str:
        parts = [m.get("content", "") for m in c.prompt_messages]
        parts += [c.response_content, c.thinking, c.error, c.reject_reason]
        if c.extra:
            parts.append(json.dumps(c.extra, ensure_ascii=False, default=str))
        return "\n".join(parts).lower()

    return [c for c in calls if all(t in _haystack(c) for t in terms)]


def _serialise_call(call: Any, names: dict[str, str]) -> dict[str, Any]:
    data = trace_to_jsonable(call)
    data["agent_name"] = names.get(call.agent_id, call.agent_id) if call.agent_id else None
    data["total_tokens"] = call.total_tokens
    return data


def build_trace_router(services: ApiServices) -> Any:
    from fastapi import APIRouter, HTTPException

    router = APIRouter(tags=["dev"])
    manager = services.manager
    trace_sink = services.container.trace_sink

    @router.get("/api/worlds/{world_id}/trace/dimensions")
    async def trace_dimensions(world_id: str) -> dict[str, Any]:
        calls = trace_sink.read_calls(world_id)
        step_summaries = {s.step: s for s in trace_sink.read_step_summaries(world_id)}
        by_step: dict[int, list] = {}
        for call in calls:
            if call.step is not None:
                by_step.setdefault(call.step, []).append(call)
        steps = [
            {
                "step": step,
                "world_time": (
                    asdict(WorldTimeView.from_payload(step_summaries[step].world_time))
                    if step in step_summaries else None
                ),
                "wall_ms": step_summaries[step].wall_ms if step in step_summaries else None,
                **_aggregate(step_calls),
            }
            for step, step_calls in sorted(by_step.items())
        ]
        names = await agent_names(manager, world_id)
        totals = {
            **_aggregate(calls),
            "wall_ms": round(sum(s.wall_ms for s in step_summaries.values()), 1),
        }
        return {
            "steps": steps,
            "stages": trace_sink.list_stages(world_id),
            "agents": [
                {"id": aid, "name": names.get(aid, aid)} for aid in trace_sink.list_agents(world_id)
            ],
            "has_build": any(c.step is None for c in calls),
            "totals": totals,
        }

    @router.get("/api/worlds/{world_id}/trace/calls")
    async def trace_calls(
        world_id: str,
        step: int | None = None,
        stage: str | None = None,
        agent_id: str | None = None,
        call_id: str | None = None,
        q: str | None = None,
        group_by: str = "stage",
    ) -> dict[str, Any]:
        calls = trace_sink.read_calls(world_id, step=step, stage=stage, agent_id=agent_id)
        calls = _keyword_filter(_call_id_filter(calls, call_id), q)
        names = await agent_names(manager, world_id)

        def _group_key(rec: Any) -> str:
            if group_by == "step":
                return str(rec.step) if rec.step is not None else "build"
            if group_by == "agent":
                return rec.agent_id or "—"
            return rec.stage

        groups: dict[str, list] = {}
        for call in calls:
            groups.setdefault(_group_key(call), []).append(call)

        def _group_sort(key: str) -> Any:
            if group_by == "stage":
                return (stage_sort_key(key), key)
            if group_by == "step":
                return (0 if key == "build" else 1, int(key) if key.isdigit() else -1)
            return key

        group_list = [
            {
                "key": key,
                "label": names.get(key, key) if group_by == "agent" else key,
                **_aggregate(group_calls),
                "items": [_serialise_call(c, names) for c in group_calls],
            }
            for key, group_calls in sorted(groups.items(), key=lambda kv: _group_sort(kv[0]))
        ]
        return {
            "group_by": group_by,
            "filters": {"step": step, "stage": stage, "agent_id": agent_id, "call_id": call_id, "q": q},
            "total": _aggregate(calls),
            "groups": group_list,
        }

    @router.get("/api/worlds/{world_id}/trace/build")
    async def trace_build(
        world_id: str, call_id: str | None = None, q: str | None = None
    ) -> dict[str, Any]:
        build_calls = _keyword_filter(
            _call_id_filter([c for c in trace_sink.read_calls(world_id) if c.step is None], call_id),
            q,
        )
        names = await agent_names(manager, world_id)
        groups: dict[str, list] = {}
        for call in build_calls:
            groups.setdefault(call.scene, []).append(call)
        group_list = [
            {
                "key": scene,
                "label": scene,
                **_aggregate(scene_calls),
                "items": [_serialise_call(c, names) for c in scene_calls],
            }
            for scene, scene_calls in sorted(
                groups.items(), key=lambda kv: (build_scene_sort_key(kv[0]), kv[0])
            )
        ]
        return {"total": _aggregate(build_calls), "groups": group_list}

    @router.get("/api/worlds/{world_id}/trace/calls/{call_id}")
    async def trace_call_detail(world_id: str, call_id: str) -> dict[str, Any]:
        """One LLM call's full record by call_id — feeds the prompt playground."""
        call = next((c for c in trace_sink.read_calls(world_id) if c.call_id == call_id), None)
        if call is None:
            raise HTTPException(status_code=404, detail=f"Call {call_id} not found")
        names = await agent_names(manager, world_id)
        return _serialise_call(call, names)

    return router
