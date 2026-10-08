"""Director console: the production directive path, assembled and interpreted outside the world."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from interaction.api.app import ApiServices
from interaction.models import WorldTimeView


def build_director_console_router(services: ApiServices) -> Any:
    from fastapi import APIRouter, Body, HTTPException

    router = APIRouter(tags=["dev"])
    application = services.application

    # Director console: the production director path in three steps, the middle one being the
    # playground (so the prompt stays editable and the backend needs no second LLM path):
    #
    #   /director/prompt     assemble the prompt production would send (not sent)
    #   /api/llm/replay      send it straight to the provider (no trace written)
    #   /director/interpret  run the raw response through production validation
    #
    # - Use the same build_prompt / interpret as production; a console-built prompt would drift.
    # - Never inject, enqueue or write a trace. To act on the world, use POST /api/worlds/{id}/direct.

    @router.post("/api/worlds/{world_id}/director/prompt")
    async def director_prompt(world_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        """The prompt production would use to parse this directive, built for inspection only."""
        text = str(payload.get("text", "")).strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        await services.ensure_session(world_id)
        try:
            prompt = application.directive_prompt(world_id, text)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        now = application.world_time(world_id)
        return {
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "scene": prompt.scene.value,
            "temperature": prompt.temperature,
            "max_tokens": prompt.max_tokens,
            "json_mode": True,
            # Index -> name. The LLM only emits indices (IndexedRef), so this table is the only way
            # to check who it actually pointed at.
            "menus": prompt.menus,
            "step": now.step,
            "world_time": asdict(WorldTimeView.from_payload(now.clock_payload())),
        }

    @router.post("/api/worlds/{world_id}/director/interpret")
    async def director_interpret(world_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        """Run raw LLM output through production validation: accepted or not, and what it becomes.

        Nothing is injected.
        """
        raw = str(payload.get("response", ""))
        if not raw.strip():
            raise HTTPException(status_code=400, detail="response is required")
        await services.ensure_session(world_id)
        try:
            result, plan = application.interpret_directive(world_id, raw)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "accepted": result.accepted,
            "reason": result.reason,
            "preview": result.preview,
            "plan": plan,
        }

    return router
