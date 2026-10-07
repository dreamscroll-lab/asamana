"""Director routes: how a human author reaches into the world.

The only routes in ``interaction/`` that change the world (besides lifecycle). Not gated by
``dev_tools_enabled``: a product feature, not a developer tool.

- ``POST /direct``: submit one line of free text. Parsing runs synchronously so a directive
  that can't be carried out is refused on the spot with a reason and nothing is injected;
  otherwise a vague injection would be written into character memory and recalled
  indefinitely. Once accepted, an idle world is advanced one step, because a queued directive
  nothing picks up has no effect (see ``NarrativeApplication.submit_directive``).
- ``POST /step``: advance the world exactly one step, with no directive.
- ``GET /directives``: the directives given to this world and whom each one reached. It lives
  here rather than with observe because it reports what the author did, not world state.
"""

from __future__ import annotations

from typing import Any

from core.logging import get_logger

from interaction.api.app import ApiServices
from interaction.api.serialization import to_jsonable

logger = get_logger(__name__)

# Max directive length. This is a quality limit, not a security one (the API has no auth):
# a few thousand characters of prose is not a directive, and the parser would only improvise
# on it.
_MAX_DIRECTIVE_CHARS = 500


def build_direct_router(services: ApiServices) -> Any:
    from fastapi import APIRouter, Body, HTTPException

    router = APIRouter(prefix="/api/worlds", tags=["direct"])
    application = services.application
    manager = services.manager

    history = services.directive_history

    @router.post("/{world_id}/direct")
    async def submit_directive(world_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        """Turn one line of free text into an intervention in this world.

        Always 200: a refusal is a normal answer (the directive needs rewording), carried by
        ``accepted`` and ``reason``. An accepted directive always takes effect (advancing a
        step if needed), so the reply carries no run state.
        """
        text = str(payload.get("text", "")).strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        if len(text) > _MAX_DIRECTIVE_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"指令过长（{len(text)} 字），请压缩到 {_MAX_DIRECTIVE_CHARS} 字以内。",
            )

        await services.ensure_advanceable(world_id)
        try:
            result = await application.submit_directive(world_id, text)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        logger.info(
            "directive_submitted",
            extra={"world_id": world_id, "accepted": result.accepted},
        )
        return {
            "accepted": result.accepted,
            "reason": result.reason,
            "preview": result.preview,
            "queued": result.queued,
        }

    @router.post("/{world_id}/step")
    async def step_world(world_id: str) -> dict[str, Any]:
        """Advance exactly one step and leave the world paused."""
        await services.ensure_advanceable(world_id)
        try:
            application.step_world(world_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"world_id": world_id, "run_state": services.run_state(world_id)}

    @router.get("/{world_id}/directives")
    async def list_directives(world_id: str) -> list[dict[str, Any]]:
        """Directives given to this world, oldest first; rejected ones never happened and are excluded.

        Read from snapshots: the narrative feed starts empty on reload and replay doesn't backfill
        unwatched steps, so it can't answer this.
        """
        replayer = manager.create_replayer(world_id)
        steps = await replayer.list_steps()
        latest = steps[-1] if steps else -1
        scanned, records = history.get(world_id, (-1, []))
        # Watermark with the `latest` read before scanning: list_interventions re-reads the step
        # list, so a step written in between may be scanned but not covered. Rescanning is
        # harmless (duplicates are dropped per step, read together); skipping a step loses it.
        recorded_steps = {r["step"] for r in records}
        fresh = await replayer.list_interventions(after=scanned)
        records = records + [
            to_jsonable(r) for r in fresh if r.step not in recorded_steps
        ]
        history[world_id] = (latest, records)
        return records

    return router
