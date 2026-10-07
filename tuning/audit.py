"""Post-hoc world-quality audit orchestrator (5 scopes, uniform judge).

Reads a finished world's ``data/traces/<world_id>/`` traces, reconstructs them, runs the judges
for five scopes, then aggregates in two levels (each scope collapses to one number first, then
scopes are weighted by evidence share; see ``audit_metrics``) into a world score + evidence
coverage + fidelity score + per-scope totals, written to ``data/traces/<world_id>/audit/``.

The five scopes (see ``audit_metrics.SCOPES``) and their call counts:
- ``single_agent_single_step``  agent × step calls (scored per stage)
- ``single_agent_multi_step``   one per agent (whole span at once, scored per metric)
- ``multi_agent_single_step``   one per step (scored per metric)
- ``multi_agent_multi_step``    1 call (scored per metric)
- ``initialization``            1 call (scored per metric, with a relation-reciprocity pre-check)

All calls run concurrently in one ``asyncio.gather(return_exceptions=True)``, each slot with its
own fallback (Rule 4); a judge failure returns an empty result per Rule 1. The judge uses a
dedicated provider (``llm.judge``), wrapped in an ``LLMRouter`` with a trace sink so its own
calls are recorded. The production web only reads the JSON on disk (it doesn't import this module).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.interfaces.llm import LLMProvider, LLMRouter, LLMScene
from core.logging import get_logger

from tuning.audit_checks import check_bindings, check_relations
from tuning.audit_metrics import SCOPES, aggregate_dimensions, scope_digest, scope_totals
from tuning.audit_reconstruct import reconstruct_world
from tuning.judge_audit import (
    agent_step_block,
    sass_units,
    agent_trajectory_block,
    cross_agent_block,
    init_block,
    judge_scope,
    world_block,
)
from tuning.judge_llm import create_judge
from tuning.trace import InMemoryTraceSink, write_json

logger = get_logger(__name__)


def _scopes_meta() -> dict[str, Any]:
    """scope_key -> {label, purpose, unit, metrics:[{id,name,inspect,category,weight}]} for the web."""
    return {
        key: {
            "label": sc.label, "purpose": sc.purpose, "unit": sc.unit,
            "metrics": [{"id": m.id, "name": m.name, "inspect": m.inspect,
                         "category": m.category, "weight": m.weight}
                        for m in sc.metrics],
        }
        for key, sc in SCOPES.items()
    }


def _flatten_scores(scores: Any, unit: str):
    """Flatten one call's scores (none = flat / stage|step = nested) into a (dim_id, value) stream."""
    if not isinstance(scores, dict):
        return
    cells = [scores] if unit == "none" else [c for c in scores.values() if isinstance(c, dict)]
    for cell in cells:
        for did, v in cell.items():
            if isinstance(v, (int, float)):
                yield did, float(v)


def _collect_dim_scores(out: Path) -> tuple[dict[str, dict[str, list[float]]], set[str]]:
    """Collect ({dimension: {scope: [scores]}}, scopes with data) from the merged scope files on disk.

    It reads disk rather than this run's results: with incremental merging, scopes not run this time
    keep contributing their previous scores (their freshness is identified by the fingerprints in
    scope_meta.json). The second return value lets downstream report "ran but didn't answer this
    dimension" separately from "didn't run". Both show up as missing scores, but the former means the
    judge abstained or dropped a field, which is worth investigating.
    """
    from collections import defaultdict
    bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    audited: set[str] = set()
    for key, (fname, is_dict) in _SCOPE_FILE.items():
        data = _load_json(out / fname)
        if not data:
            continue
        audited.add(key)
        unit = SCOPES[key].unit
        entries = list(data.values()) if is_dict else [data]
        for entry in entries:
            if isinstance(entry, dict):
                for did, v in _flatten_scores(entry.get("scores"), unit):
                    bucket[did][key].append(v)
    return {did: dict(by_scope) for did, by_scope in bucket.items()}, audited


# scope_key -> output file name (dict-of-entries vs single object). Dict files support agent/step-level incremental merging.
_SCOPE_FILE = {
    "single_agent_single_step": ("single_agent_single_step.json", True),
    "single_agent_multi_step": ("single_agent_multi_step.json", True),
    "multi_agent_single_step": ("multi_agent_single_step.json", True),
    "multi_agent_multi_step": ("multi_agent_multi_step.json", False),
    "initialization": ("initialization.json", False),
}


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_agents(view: Any, agents: list[str] | None) -> list[str]:
    """Tokens in agents may be agent_ids or names; None → all. Only applies to single_agent_* scopes."""
    if not agents:
        return list(view.per_agent)
    want = set(agents)
    return [aid for aid in view.per_agent if aid in want or view.agent_names.get(aid) in want]


def _merge_dict_file(path: Path, new: dict) -> dict:
    """Shallow-merge new entries into the existing file by key, so a targeted rerun updates only the agents/steps it hit and leaves the rest."""
    existing = _load_json(path)
    merged = dict(existing) if isinstance(existing, dict) else {}
    merged.update(new)
    write_json(path, merged)
    return merged


async def run_audit(
    container: Any,
    config: Any,
    world_id: str,
    *,
    steps: list[int] | None = None,
    scopes: list[str] | None = None,
    agents: list[str] | None = None,
    judge_model: str | None = None,
    judge_provider: LLMProvider | None = None,
    judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
    trace_dir: str = "data/traces",
) -> dict[str, Any]:
    """Audit one world's traces and persist the report. Returns the summary.

    ``scopes`` picks which scopes run (all 5 by default); ``agents`` / ``steps`` narrow the data.
    ``agents`` only applies to single_agent_* scopes (multi_agent_* naturally spans every agent);
    ``steps`` applies to all per-step work. A partial run merges incrementally at scope / agent /
    step level (parts not run are kept) and recomputes world_score.
    """
    selected = set(scopes) if scopes else set(SCOPES)
    unknown = selected - set(SCOPES)
    if unknown:
        raise ValueError(f"unknown scope(s): {sorted(unknown)}; valid: {sorted(SCOPES)}")

    view = reconstruct_world(world_id, trace_dir=trace_dir, steps=steps)
    if not view.per_agent and not view.init.get("key_figures"):
        raise ValueError(f"no usable traces for world {world_id!r} under {trace_dir}")

    agent_ids = _resolve_agents(view, agents)
    if agents and not agent_ids:
        logger.warning("audit_no_matching_agents", extra={"world_id": world_id, "agents": agents})
    step_ids = sorted(view.per_step)

    judge_llm = judge_provider or create_judge(config, judge_model)
    sink = InMemoryTraceSink()
    router = LLMRouter(
        {scene: judge_llm for scene in LLMScene},
        trace_sink=sink, max_concurrent=getattr(getattr(config, "engine", None), "max_concurrent_llm", 4))

    # ---- assemble judge calls for the selected scopes only ----
    labels: list[tuple[str, Any]] = []
    coros: list[Any] = []

    def _add(kind: str, key: Any, coro: Any) -> None:
        labels.append((kind, key)); coros.append(coro)

    # The header is derived from scope.purpose (matches the dimensions that scope actually scores; single source of truth).
    sc = SCOPES
    if "initialization" in selected:
        _add("init", None, judge_scope(
            router, sc["initialization"], header=f"审查对象：{sc['initialization'].purpose}。",
            context_block=init_block(view), det_findings=check_relations(view), judge_scene=judge_scene))
    if "multi_agent_multi_step" in selected:
        _add("mams", None, judge_scope(
            router, sc["multi_agent_multi_step"], header=f"审查对象：{sc['multi_agent_multi_step'].purpose}。",
            context_block=world_block(view), judge_scene=judge_scene))
    if "multi_agent_single_step" in selected:
        for step in step_ids:
            _add("mass", step, judge_scope(
                router, sc["multi_agent_single_step"],
                header=f"审查对象：第 {step} 步 —— {sc['multi_agent_single_step'].purpose}。",
                context_block=cross_agent_block(view, step), judge_scene=judge_scene))
    for aid in agent_ids:
        traj = view.per_agent[aid]
        if "single_agent_multi_step" in selected:
            _add("sams", aid, judge_scope(
                router, sc["single_agent_multi_step"],
                header=f"审查对象：角色 {traj.name} —— {sc['single_agent_multi_step'].purpose}。",
                context_block=agent_trajectory_block(view, aid),
                n_units=len(traj.steps), judge_scene=judge_scene))
        if "single_agent_single_step" in selected:
            for sv in traj.steps:
                _add("sass", (aid, sv.step), judge_scope(
                    router, sc["single_agent_single_step"],
                    header=f"审查对象：角色 {traj.name} 第 {sv.step} 步 —— {sc['single_agent_single_step'].purpose}。",
                    context_block=agent_step_block(view, aid, sv),
                    det_findings=check_bindings(sv),
                    # One cell spans several beats (a beat can land more than once); score per beat, skipping the ones that can't be judged.
                    n_units=sum(1 for u in sass_units(sv) if u.label),
                    judge_scene=judge_scene))

    results = await asyncio.gather(*coros, return_exceptions=True)

    # ---- re-assemble per scope; split each scored entry from its LLM call (prompt+response) ----
    init_res: dict[str, Any] = {}
    mams_res: dict[str, Any] = {}
    mass_res: dict[str, Any] = {}
    sams_res: dict[str, Any] = {}
    sass_res: dict[str, Any] = {}
    # scope_key -> entry_key -> {prompt, response}  (single-object scopes use entry_key "_")
    calls: dict[str, dict[str, Any]] = {sk: {} for sk in SCOPES}

    def _split_call(scope_key: str, entry_key: str, res: dict) -> dict:
        calls[scope_key][entry_key] = {"prompt": res.pop("_prompt", None),
                                       "response": res.pop("_response", None)}
        return res

    for (kind, key), res in zip(labels, results):
        if isinstance(res, BaseException):
            logger.warning("audit_slot_failed", extra={"kind": kind, "key": key, "error": str(res)})
            res = {"rationale": f"审查异常: {res}", "scores": {}, "total": None}
        if kind == "init":
            init_res = _split_call("initialization", "_", res)
        elif kind == "mams":
            mams_res = _split_call("multi_agent_multi_step", "_", res)
        elif kind == "mass":
            mass_res[str(key)] = _split_call("multi_agent_single_step", str(key), res)
        elif kind == "sams":
            sams_res[key] = {"name": view.per_agent[key].name, **_split_call("single_agent_multi_step", key, res)}
        else:  # sass
            aid, step = key
            ek = f"{aid}#{step}"
            sass_res[ek] = {"agent": aid, "name": view.per_agent[aid].name, "step": step,
                            **_split_call("single_agent_single_step", ek, res)}

    # ---- write ran scopes (incremental merge) ----
    out = Path(trace_dir) / world_id / "audit"
    out.mkdir(parents=True, exist_ok=True)
    if "initialization" in selected:
        write_json(out / "initialization.json", {**init_res, "det": check_relations(view)})
    if "multi_agent_multi_step" in selected:
        write_json(out / "multi_agent_multi_step.json", mams_res)
    if "multi_agent_single_step" in selected:
        _merge_dict_file(out / "multi_agent_single_step.json", mass_res)
    if "single_agent_multi_step" in selected:
        _merge_dict_file(out / "single_agent_multi_step.json", sams_res)
    if "single_agent_single_step" in selected:
        _merge_dict_file(out / "single_agent_single_step.json", sass_res)
    # audit_calls.json: each audit call's prompt + raw response, keyed scope → entry (same keys as the
    # score files), so the web can expand the matching LLM call next to each result. Partial runs merge per scope.
    prev_calls = _load_json(out / "audit_calls.json")
    merged_calls = dict(prev_calls) if isinstance(prev_calls, dict) else {}
    for key in selected:
        merged_calls[key] = {**(merged_calls.get(key) or {}), **calls.get(key, {})}
    write_json(out / "audit_calls.json", merged_calls)

    # ---- Provenance: whose scores each scope holds, and under which scoring version ----
    # Incremental merging makes "ran only sams, but the world score includes last run's mams" the
    # norm. Once the scoring criteria or the judge change, the world score stitches two versions of
    # evidence together. That isn't cleared (incremental is intended), but it has to be reported.
    model_name = getattr(judge_llm, "model", judge_model)
    stamp = datetime.now().isoformat()
    provenance = _load_json(out / "scope_meta.json")
    provenance = dict(provenance) if isinstance(provenance, dict) else {}
    for key in selected:
        provenance[key] = {"judge_model": model_name, "digest": scope_digest(key), "generated_at": stamp}
    write_json(out / "scope_meta.json", provenance)

    # ---- summary: the two-level world score (primary) + per-scope totals derived from the same ladder ----
    per_scope, audited = _collect_dim_scores(out)
    stale = {k for k in audited if provenance.get(k, {}).get("digest") != scope_digest(k)}
    dimensions, world_score, coverage = aggregate_dimensions(per_scope, audited, stale)
    totals = scope_totals(per_scope)
    # Scopes whose scoring version changed still count toward the world score (incremental is intended) but can't be trusted; report it so someone reruns them.
    contributing = {k for did in dimensions for k in (dimensions[did]["sources"] or {})}
    models = {provenance.get(k, {}).get("judge_model") for k in contributing}

    summary = {
        "world_id": world_id,
        "world_name": view.init.get("world_name"),
        "generated_at": stamp,
        "judge_model": model_name,
        "step_count": len(step_ids),
        "agent_count": len(agent_ids),
        "ran_scopes": sorted(selected),
        "filters": {"agents": agents, "steps": steps},
        "world_score": world_score,          # two-level aggregate (primary)
        "coverage": coverage,                # evidence coverage; audits with different coverage aren't comparable
        "dimensions": dimensions,            # 7-dimension scorecard + each dimension's evidence source (headline)
        "fidelity_score": totals["single_agent_single_step"],  # fidelity (engineering), reported alongside the world score
        "scope_totals": totals,              # per-scope totals (not comparable across scopes)
        "scope_provenance": provenance,
        "stale_scopes": sorted(stale),        # scoring criteria changed; the scores on disk weren't produced under the current version
        "mixed_provenance": len(models) > 1,  # the scopes in the world score come from different judge models
        "scopes_meta": _scopes_meta(),
    }
    write_json(out / "summary.json", summary)

    logger.info("audit_complete",
                extra={"world_id": world_id, "world_score": world_score, "coverage": coverage,
                       "ran": sorted(selected)})
    return summary
