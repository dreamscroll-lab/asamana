"""Prompt playground: one LLM call with an edited prompt, outside any world's trace."""

from __future__ import annotations

import time
from typing import Any

from interaction.api.app import ApiServices


def build_playground_router(services: ApiServices) -> Any:
    from fastapi import APIRouter, Body, HTTPException

    router = APIRouter(tags=["dev"])
    config = services.config

    @router.post("/api/llm/replay")
    async def llm_replay(payload: dict = Body(...)) -> dict[str, Any]:
        """Fire a one-off LLM call with an edited prompt — the playground's engine.

        Deliberately **untraced**: it goes straight to a provider (not the router's
        traced complete()), so experimentation never pollutes a world's trace. Dev
        tool only — arbitrary prompts hit the paid LLM.
        """
        services.require_model_keys()
        from core.interfaces.llm import LLMMessage, LLMScene

        raw = payload.get("messages") or []
        if not raw:
            raise HTTPException(status_code=400, detail="messages is required")
        try:
            messages = [LLMMessage(role=str(m["role"]), content=str(m["content"])) for m in raw]
        except (KeyError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=f"Malformed messages: {exc}") from exc
        temperature = float(payload.get("temperature", 0.7))
        max_tokens = int(payload.get("max_tokens", 1000))
        # Faithfulness knob: a scene that runs in JSON mode in production behaves
        # differently without it, so replaying such a call must be able to say so.
        json_mode = bool(payload.get("json_mode", False))
        # Overrides: a model swaps the model, a provider swaps the endpoint; with neither, run the
        # scene's configuration.
        scene_val = payload.get("scene") or None
        scene: LLMScene | None = None
        if scene_val:
            try:
                scene = LLMScene(scene_val)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Unknown scene: {scene_val}") from exc
        chosen = config.llm.scene_llm(scene) if scene is not None else None
        # No scene means the default provider: trying a model name shouldn't require picking a scene.
        provider_name = str(
            payload.get("provider")
            or (chosen.provider if chosen else config.llm.default_provider)
        )
        model = str(payload.get("model") or (chosen.model if chosen else ""))
        # Call params (the endpoint dialect) must match production: without
        # enable_thinking: false a thinking endpoint spends all of max_tokens on reasoning and
        # returns empty content, so the replay would not be the same call.
        call_params = (
            config.llm.params_for(scene) if scene is not None else config.llm.provider_params(provider_name)
        )
        # A different model often needs a different dialect: glm-5.3 on Bailian only accepts
        # enable_thinking: true, so sending the scene's false gets a 400. Overrides are merged
        # key by key over the scene params and passed through as-is, uninterpreted.
        override = payload.get("params")
        if override is not None:
            if not isinstance(override, dict):
                raise HTTPException(status_code=400, detail="params must be a JSON object")
            call_params = {**call_params, **override}
        try:
            provider = services.container.create_llm(
                provider_name, model, timeout=chosen.timeout if chosen else None, **call_params,
            )
        except Exception as exc:  # noqa: BLE001 — bad spec → 400, not a 500
            raise HTTPException(status_code=400, detail=f"Could not create provider: {exc}") from exc
        started = time.monotonic()
        try:
            resp = await provider.complete(
                messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
            )
        except Exception as exc:  # noqa: BLE001 — provider failure is an upstream error
            raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc
        return {
            "content": resp.content,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "model": resp.model,
            "provider_spec": f"{provider_name}/{model}" if model else provider_name,
            # The params actually sent, after overrides, so the UI can show how the call was made.
            "params": call_params,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
        }

    return router
