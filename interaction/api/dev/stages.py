"""Stage suites: the stage registry, the scenario files, one run's report, and starting a run."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core.logging import get_logger

from interaction.api.app import ApiServices
from interaction.api.dev.jobs import PROJECT_ROOT, JobRegistry

logger = get_logger(__name__)

# Where `python -m tuning validate` writes its per-stage reports (tuning/cli.py:_TRACE_DIR).
_TUNING_TRACE_DIR = Path("./data/tuning_traces")


@contextmanager
def _scenario_errors() -> Any:
    """Scenario-library data errors are always 400: the request is invalid, the server is fine."""
    from fastapi import HTTPException

    from tuning.scenario_store import ScenarioError

    try:
        yield
    except ScenarioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def build_stages_router(services: ApiServices, jobs: JobRegistry) -> Any:
    from fastapi import APIRouter, Body, HTTPException

    router = APIRouter(tags=["dev"])
    manager = services.manager

    # Stage suites: `python -m tuning validate` restores a world, injects a scenario, runs the
    # production cognition path and writes the freshly assembled prompt and response for this
    # router to read back. Unlike Prompt Replay (which resends the prompt text recorded in the
    # trace), this re-runs the current prompt builder, so it can verify a code change.

    _STAGE_META_CACHE: dict[str, Any] = {}

    async def _stage_meta() -> list[dict[str, Any]]:
        """The stage registry, read from `python -m tuning stages` rather than importing tuning."""
        if "stages" not in _STAGE_META_CACHE:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "tuning", "stages",
                cwd=str(PROJECT_ROOT), env=os.environ.copy(),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await proc.communicate()
            try:
                _STAGE_META_CACHE["stages"] = json.loads(out.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise HTTPException(status_code=500, detail=f"Could not read the stage registry: {exc}") from exc
        return _STAGE_META_CACHE["stages"]

    def _stage_dir(stage_key: str, world_id: str) -> Path:
        return _TUNING_TRACE_DIR / world_id / "validation" / stage_key

    @router.get("/api/dev/stages")
    async def list_stages() -> dict[str, Any]:
        stages = await _stage_meta()
        # Which worlds each stage has run on; the directories themselves are the index.
        runs: dict[str, list[str]] = {}
        for s in stages:
            base = _TUNING_TRACE_DIR
            found = [
                d.name for d in sorted(base.iterdir())
                if d.is_dir() and (d / "validation" / s["key"] / "summary.json").exists()
            ] if base.exists() else []
            runs[s["key"]] = found
        return {"stages": stages, "runs": runs}

    # Scenario library CRUD. Scenarios are reviewed test corpus in git, so edits land directly
    # in `tuning/scenarios/<stage>.json`. Don't add a draft store: the CLI, CI and the next
    # person read the file, so UI-only drafts would split the truth. These are the only dev
    # routes that write to disk, and only under tuning/scenarios/.

    async def _scenario_ctx(stage_key: str) -> tuple[Path, list[str]]:
        """(scenario file path, the stage's scoring criteria).

        Import only `tuning.scenario_store` (stdlib only), never `tuning.stages`: its registry
        pulls hundreds of agent / engine / world / providers modules into the web process.
        Criteria come from the cached `_stage_meta()`, the registry the subprocess reported.
        """
        stages = await _stage_meta()
        spec = next((st for st in stages if st["key"] == stage_key), None)
        if spec is None:
            raise HTTPException(status_code=400, detail=f"Unknown stage {stage_key}")
        return PROJECT_ROOT / "tuning" / "scenarios" / f"{stage_key}.json", spec["criteria"]

    @router.get("/api/dev/stages/{stage_key}/scenarios")
    async def stage_scenarios(stage_key: str) -> dict[str, Any]:
        """The stage's scenario file as-is, including non-scenario keys such as `_comment` /
        `knob_sets`: the payload shape differs per stage, and the page only displays and edits it.
        """
        from tuning import scenario_store

        path, _ = await _scenario_ctx(stage_key)
        with _scenario_errors():
            data = scenario_store.load(path)
        return {
            "stage": stage_key,
            "path": f"tuning/scenarios/{stage_key}.json",
            "file": data,
        }

    @router.post("/api/dev/stages/{stage_key}/scenarios")
    async def create_scenario(stage_key: str, payload: dict = Body(...)) -> dict[str, Any]:
        """Add a scenario (duplicate name -> 400), appended at the end; existing order is kept."""
        from tuning import scenario_store

        path, criteria = await _scenario_ctx(stage_key)
        with _scenario_errors():
            name = scenario_store.add(path, payload, criteria=criteria, stage=stage_key)
        logger.info("dev_scenario_added", extra={"stage": stage_key, "scenario": name})
        return {"stage": stage_key, "name": name}

    @router.put("/api/dev/stages/{stage_key}/scenarios/{name}")
    async def update_scenario(stage_key: str, name: str, payload: dict = Body(...)) -> dict[str, Any]:
        """Replace the *name* entry. A different name in the payload renames it, in place."""
        from tuning import scenario_store

        path, criteria = await _scenario_ctx(stage_key)
        with _scenario_errors():
            saved = scenario_store.replace(path, name, payload, criteria=criteria, stage=stage_key)
        logger.info(
            "dev_scenario_saved",
            extra={"stage": stage_key, "scenario": saved, "renamed_from": name if saved != name else ""},
        )
        return {"stage": stage_key, "name": saved}

    @router.delete("/api/dev/stages/{stage_key}/scenarios/{name}")
    async def remove_scenario(stage_key: str, name: str) -> dict[str, Any]:
        """Delete one entry. The file is tracked by git, so `git checkout` undoes a mistake."""
        from tuning import scenario_store

        path, _ = await _scenario_ctx(stage_key)
        with _scenario_errors():
            scenario_store.delete(path, name, stage=stage_key)
        logger.info("dev_scenario_deleted", extra={"stage": stage_key, "scenario": name})
        return {"stage": stage_key, "deleted": name}

    @router.get("/api/dev/stages/{stage_key}/runs/{world_id}")
    async def stage_report(stage_key: str, world_id: str) -> dict[str, Any]:
        """Everything one stage-suite run produced: the summary plus each scenario's
        input/prompt/checks/judge...

        Every `*.json` in a scenario directory is read back, keyed by file name, since each stage
        writes its own payload (event.json / interrupt.json ...) besides the common artifacts.
        """
        base = _stage_dir(stage_key, world_id)
        summary_path = base / "summary.json"
        if not summary_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No {stage_key} run for this world yet. Run: python -m tuning validate {stage_key} {world_id}",
            )

        def _load(path: Path) -> Any:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None

        scenarios: dict[str, dict[str, Any]] = {}
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            scenarios[d.name] = {f.stem: _load(f) for f in sorted(d.glob("*.json"))}
        return {
            "stage": stage_key,
            "world_id": world_id,
            "summary": _load(summary_path),
            "scenarios": scenarios,
        }

    @router.post("/api/dev/stages/{stage_key}/run")
    async def run_stage_job(stage_key: str, payload: dict = Body(default={})) -> dict[str, Any]:
        services.require_model_keys()
        stages = await _stage_meta()
        spec = next((s for s in stages if s["key"] == stage_key), None)
        if spec is None:
            raise HTTPException(status_code=400, detail=f"Unknown stage {stage_key}")
        world_id = str(payload.get("world_id") or "").strip()
        if not world_id:
            raise HTTPException(status_code=400, detail="world_id is required")
        # Which scenarios to run; empty means all. Names are checked against the file on disk
        # here so a typo fails now rather than after the subprocess starts. Order is kept: they
        # run in the order selected.
        picked = payload.get("scenario_names") or []
        if not isinstance(picked, list) or not all(isinstance(n, str) for n in picked):
            raise HTTPException(status_code=400, detail="scenario_names must be a list of strings")
        if picked:
            from tuning import scenario_store

            path, _ = await _scenario_ctx(stage_key)
            with _scenario_errors():
                known = set(scenario_store.names(path))
            unknown = [n for n in picked if n not in known]
            if unknown:
                raise HTTPException(
                    status_code=400, detail=f"{stage_key} has no scenario named {unknown[0]!r}",
                )
        if await manager.get_world(world_id) is None:
            raise HTTPException(status_code=404, detail=f"World not found: {world_id}")
        cmd = [sys.executable, "-m", "tuning", "validate", stage_key, world_id]
        if picked:
            cmd += ["--scenario", ",".join(picked)]
        if payload.get("judge_model") and spec["judged"]:
            cmd += ["--judge-model", str(payload["judge_model"])]
        return jobs.spawn(f"stage:{stage_key}", world_id, cmd)

    return router
