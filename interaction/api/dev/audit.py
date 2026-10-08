"""Post-hoc world audit: read the report ``python -m tuning audit`` wrote, or start that audit."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from core.logging import get_logger
from interaction.api.app import ApiServices
from interaction.api.dev.jobs import JobRegistry

logger = get_logger(__name__)

# CLI scope aliases accepted by `python -m tuning audit --scope` (tuning/cli.py:_SCOPE_ALIAS).
_AUDIT_SCOPES = {"sass", "sams", "mass", "mams", "init"}


def build_audit_router(services: ApiServices, jobs: JobRegistry) -> Any:
    from fastapi import APIRouter, Body, HTTPException

    router = APIRouter(tags=["dev"])
    config = services.config

    # The audit subprocess writes under the trace provider's base dir, read from config
    # (not the provider's internals) so the two never drift.
    _audit_base = Path(config.observability.params.get("base_dir", "./data/traces"))

    @router.get("/api/dev/audit/judge")
    async def audit_judge() -> dict[str, Any]:
        """The judge this deployment scores with by default.

        ``provider`` is a deployment fact (endpoint and credential variable live on its
        declaration), so it is reported, not accepted; the model is what a run may change, via
        the run's ``judge_model``.
        """
        spec = config.llm.judge_llm()
        return {"provider": spec.provider, "model": spec.model}

    @router.get("/api/worlds/{world_id}/audit")
    async def get_audit(world_id: str) -> dict[str, Any]:
        base = _audit_base / world_id / "audit"
        if not (base / "summary.json").exists():
            raise HTTPException(
                status_code=404,
                detail=f"No audit report for this world yet. Run: python -m tuning audit {world_id}",
            )

        def _load(name: str) -> Any:
            p = base / name
            if not p.exists():
                return None
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None

        return {
            "summary": _load("summary.json"),
            "initialization": _load("initialization.json"),
            "single_agent_single_step": _load("single_agent_single_step.json"),
            "single_agent_multi_step": _load("single_agent_multi_step.json"),
            "multi_agent_single_step": _load("multi_agent_single_step.json"),
            "multi_agent_multi_step": _load("multi_agent_multi_step.json"),
            "calls": _load("audit_calls.json"),
        }

    @router.delete("/api/worlds/{world_id}/audit")
    async def clear_audit(world_id: str) -> dict[str, Any]:
        """Delete the whole report directory. Clear it before a rerun: sass/sams/mass merge
        incrementally by agent#step, and the world score re-aggregates every scope file on disk,
        so stale entries from the last run would count toward this one."""
        base = (_audit_base / world_id / "audit").resolve()
        # world_id comes straight from the URL and a `..` would point rmtree elsewhere: accept only
        # the <base>/<world>/audit shape.
        if base.name != "audit" or base.parent.parent != _audit_base.resolve():
            raise HTTPException(status_code=400, detail=f"Bad world id {world_id!r}")
        existed = base.is_dir()
        if existed:
            shutil.rmtree(base)
            logger.info("dev_audit_cleared", extra={"world_id": world_id})
        return {"world_id": world_id, "cleared": existed}

    @router.post("/api/worlds/{world_id}/audit/run")
    async def run_audit_job(world_id: str, payload: dict = Body(default={})) -> dict[str, Any]:
        services.require_model_keys()
        if not (_audit_base / world_id).exists():
            raise HTTPException(status_code=404, detail=f"World {world_id} has no trace to audit")
        scopes = payload.get("scopes") or None
        if scopes:
            bad = [s for s in scopes if s not in _AUDIT_SCOPES]
            if bad:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown scope {bad}; valid: {sorted(_AUDIT_SCOPES)}",
                )
        cmd = [sys.executable, "-m", "tuning", "audit", world_id]
        if scopes:
            cmd += ["--scope", ",".join(scopes)]
        if payload.get("agents"):
            cmd += ["--agent", ",".join(payload["agents"])]
        if payload.get("steps"):
            cmd += ["--steps", str(payload["steps"]).strip()]
        if payload.get("judge_model"):
            cmd += ["--judge-model", str(payload["judge_model"])]
        return jobs.spawn("audit", world_id, cmd)

    return router
