"""World lifecycle + run-control routes.

Create/build (async job), delete, and run control (run/pause/resume/stop/reset).
Build is a one-shot LLM operation that can take minutes, so it runs as a
background job the client polls — the request never blocks on it. Run control
maps 1:1 onto ``NarrativeApplication``'s background-run surface.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from config.content import theme_presets
from core.logging import get_logger
from engine.run_control import RunState
from world.catalog import MAX_WORLD_NAME_LEN

from worlds.tiled import TiledWorldConfig, list_templates

from interaction.api.app import ApiServices
from interaction.api.serialization import to_jsonable

logger = get_logger(__name__)


@dataclass
class _BuildJob:
    """Tracks one in-flight or finished background world build."""

    job_id: str
    theme: str
    template: str | None = None
    status: str = "building"  # building | ready | failed
    world_id: str | None = None
    world_name: str | None = None
    error: str | None = None
    # The event loop only holds weak references to tasks, so the job keeps a strong one to
    # stop it being garbage-collected mid-run (same as RuntimeSession.task).
    task: "asyncio.Task[None] | None" = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "theme": self.theme,
            "template": self.template,
            "status": self.status,
            "world_id": self.world_id,
            "world_name": self.world_name,
            "error": self.error,
        }


def build_lifecycle_router(services: ApiServices) -> Any:
    from fastapi import APIRouter, Body, HTTPException

    # Two scopes on one router: everything about a PARTICULAR world hangs off
    # /api/worlds, while what this DEPLOYMENT offers before any world exists —
    # its map catalog, its sample themes — sits beside it. `parent` carries the
    # latter and mounts the former.
    parent = APIRouter(tags=["lifecycle"])

    @parent.get("/api/content")
    async def get_interface_content() -> dict[str, Any]:
        """Editable interface text (sample themes), read from disk on every call so an edit
        shows up without a restart."""
        return {"presets": theme_presets()}

    router = APIRouter(prefix="/api/worlds", tags=["lifecycle"])
    application = services.application
    manager = services.manager

    build_jobs: dict[str, _BuildJob] = {}

    @parent.get("/api/templates")
    async def list_map_templates() -> list[dict[str, Any]]:
        """The maps a world can be built on, one per template directory on disk (the
        directory IS the registration). An unreadable template is omitted rather than
        failing the list; ``tests/unit/test_world_templates`` is where it should surface.
        """
        entries: list[dict[str, Any]] = []
        for name in list_templates():
            try:
                config = TiledWorldConfig(template=name)
                context = config.to_runtime_context()
                entries.append(
                    {
                        "template": name,
                        "world_name": context["world_name"],
                        "era_name": context["era_name"],
                        "description": config.get_world_description(),
                        "location_count": len(config.get_places()),
                    }
                )
            except (OSError, ValueError, KeyError) as exc:
                logger.warning(
                    "map_template_unreadable", extra={"template": name, "error": str(exc)}
                )
        return entries

    async def _run_build(job: _BuildJob) -> None:
        try:
            # No template named → the build reads the theme and picks one from
            # what is installed. WHICH maps exist is this layer's answer to give
            # (same list GET /api/templates serves), not the engine's to discover.
            world = await application.build_world(
                job.theme, template=job.template, available_templates=list_templates()
            )
            job.world_id = world.world_id
            job.world_name = world.analysis.world_name
            # Report the map actually built on, whether it was named or chosen.
            job.template = str(
                world.world_config.to_runtime_context().get("template") or job.template or ""
            ) or None
            job.status = "ready"
            logger.info("build_job_ready", extra={"job_id": job.job_id, "world_id": world.world_id})
        except Exception as exc:  # noqa: BLE001 — build failures are reported, not fatal to the server
            job.status = "failed"
            job.error = str(exc)
            logger.error("build_job_failed", extra={"job_id": job.job_id, "error": str(exc)})

    # ----- create / list / get / delete -----------------------------------

    @router.post("")
    async def create_world(payload: dict = Body(...)) -> dict[str, Any]:
        services.require_model_keys()
        theme = str(payload.get("theme", "")).strip()
        if not theme:
            raise HTTPException(status_code=400, detail="theme is required")
        # Which map to build on; omitted → the configured default. Validated here so an
        # unknown template is a request error, not a background job dying minutes later.
        template = payload.get("template")
        template = str(template).strip() if template else None
        if template is not None and template not in list_templates():
            raise HTTPException(status_code=400, detail=f"Unknown map template: {template}")
        job = _BuildJob(job_id=uuid.uuid4().hex, theme=theme, template=template)
        build_jobs[job.job_id] = job
        job.task = asyncio.create_task(_run_build(job), name=f"build:{job.job_id}")
        return job.as_dict()

    @router.get("/jobs/{job_id}")
    async def get_build_job(job_id: str) -> dict[str, Any]:
        job = build_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Build job not found: {job_id}")
        return job.as_dict()

    @router.get("")
    async def list_worlds() -> list[dict[str, Any]]:
        worlds = await manager.list_worlds()
        result = []
        for meta in worlds:
            result.append(_with_run_state(meta))
        return result

    @router.get("/{world_id}")
    async def get_world(world_id: str) -> dict[str, Any]:
        meta = await manager.get_world(world_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"World not found: {world_id}")
        return _with_run_state(meta)

    @router.patch("/{world_id}")
    async def rename_world(world_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        """Rename a world's display name in the catalog. The step-0 snapshot and anything the
        simulation reads are untouched, and a confirmed world stays confirmed, so a rename
        never reopens the run gate."""
        name = str(payload.get("world_name", "")).strip()
        if not name:
            raise HTTPException(status_code=400, detail="world_name is required")
        if len(name) > MAX_WORLD_NAME_LEN:
            raise HTTPException(
                status_code=400, detail=f"world_name must be at most {MAX_WORLD_NAME_LEN} characters"
            )
        try:
            meta = await manager.rename_world(world_id, name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _with_run_state(meta)

    @router.post("/{world_id}/confirm")
    async def confirm_world(world_id: str) -> dict[str, Any]:
        try:
            meta = await manager.confirm_world(world_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _with_run_state(meta)

    @router.delete("/{world_id}")
    async def delete_world(world_id: str) -> dict[str, Any]:
        meta = await manager.get_world(world_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"World not found: {world_id}")
        await application.delete_world(world_id)
        return {"world_id": world_id, "deleted": True}

    # ----- run control -----------------------------------------------------

    @router.post("/{world_id}/run")
    async def run_world(world_id: str, payload: dict = Body(default={})) -> dict[str, Any]:
        # ensure_advanceable checks too, but only after the step-count check below, which would
        # then answer a keyless deployment with the wrong reason.
        services.require_model_keys()
        # A missing step count means one step. Don't default to running indefinitely: each step
        # makes several LLM calls per character, so a forgotten field could run up an unbounded
        # bill unnoticed. Callers wanting a long run pass a large number.
        steps = payload.get("steps")
        steps = 1 if steps is None else steps
        # Accept only a JSON integer. ``int(2.7)`` would silently run 2 steps, and since ``bool``
        # subclasses ``int``, ``true`` would run one. Rejecting the request is better than
        # running something the caller didn't ask for.
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise HTTPException(status_code=400, detail="steps must be a positive integer")
        await services.ensure_advanceable(world_id)
        try:
            application.start_run(world_id, steps=steps)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _status_payload(world_id)

    @router.post("/{world_id}/pause")
    async def pause_world(world_id: str) -> dict[str, Any]:
        return _control(world_id, application.pause_run)

    @router.post("/{world_id}/resume")
    async def resume_world(world_id: str) -> dict[str, Any]:
        return _control(world_id, application.resume_run)

    @router.post("/{world_id}/stop")
    async def stop_world(world_id: str) -> dict[str, Any]:
        return _control(world_id, application.stop_run)

    @router.post("/{world_id}/reset")
    async def reset_world(world_id: str) -> dict[str, Any]:
        # Up front: the reset rewrites stored state before it restores the session, and the
        # restore is what needs the keys.
        services.require_model_keys()
        # Early check for a friendly error only; the state can change during the reset's many
        # awaits. reset_session repeats the check under its lock and raises ValueError.
        state = application.run_status(world_id)
        if state in (RunState.RUNNING, RunState.PAUSED, RunState.STOPPING):
            raise HTTPException(status_code=409, detail="Stop the run before resetting.")
        try:
            await application.reset_session(world_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        services.directive_history.pop(world_id, None)
        return _status_payload(world_id)

    def _control(world_id: str, action: Any) -> dict[str, Any]:
        try:
            action(world_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _status_payload(world_id)

    def _with_run_state(meta: Any) -> dict[str, Any]:
        """A world's catalog entry plus its run state — what every per-world route answers with."""
        return {**to_jsonable(meta), "run_state": services.run_state(meta.world_id)}

    def _status_payload(world_id: str) -> dict[str, Any]:
        return {"world_id": world_id, "run_state": services.run_state(world_id)}

    parent.include_router(router)
    return parent
