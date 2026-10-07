"""FastAPI application factory for the Asamana backend.

Single-process topology: the same process serves the API/WebSocket and runs
worlds as background asyncio tasks, so the in-process event bus reaches live
observers directly. The factory wires one set of shared services (a
``NarrativeApplication`` that owns run sessions, a read-only ``WorldManager``,
and the container) and composes the responsibility-split routers onto them.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.models import Config
from core.container import Container
from core.logging import get_logger
from core.model_keys import ModelKeysMissing
from engine.application import NarrativeApplication
from world import DEFAULT_CATALOG_PATH, WorldCatalog

from interaction.models import WorldMeta
from interaction.world_manager import WorldManager

logger = get_logger(__name__)

# Built frontend assets (Vite `npm run build` output). Served at `/` in
# production; in dev the Vite server proxies to this backend instead.
_FRONTEND_DIST = Path(__file__).parent.parent.parent / "frontend" / "dist"


@dataclass
class ApiServices:
    """Shared services handed to every router."""

    container: Container
    config: Config
    application: NarrativeApplication
    manager: WorldManager
    # world_id -> (last step scanned, director interventions up to that step). The history costs
    # one snapshot read per step and the director panel polls every step, so it scans forward
    # only. A reset renumbers steps from 1, so the reset route drops the entry; left behind on
    # world delete, since world_ids are never reused and an entry is a few KB.
    directive_history: dict[str, tuple[int, list[dict[str, Any]]]] = field(default_factory=dict)

    async def ensure_session(self, world_id: str) -> None:
        """A live session for this world, restored if this process has none; 404 if the world
        does not exist. A session runs on the models, so it requires the model keys."""
        await self._existing_world(world_id)
        if not self.application.has_session(world_id):
            await self.application.restore_session(world_id)

    async def ensure_advanceable(self, world_id: str) -> None:
        """``ensure_session`` for a request that will advance the world (run, step, directive):
        409 until the world is confirmed. One gate for every entry, or the one left out would
        start the narrative of a world still under review."""
        from fastapi import HTTPException

        meta = await self._existing_world(world_id)
        if not meta.confirmed:
            raise HTTPException(
                status_code=409, detail="Confirm the world before starting the narrative."
            )
        if not self.application.has_session(world_id):
            await self.application.restore_session(world_id)

    async def _existing_world(self, world_id: str) -> WorldMeta:
        """The world's metadata, read once; model keys required, 404 if it doesn't exist."""
        from fastapi import HTTPException

        self.require_model_keys()
        meta = await self.manager.get_world(world_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"World not found: {world_id}")
        return meta

    def require_model_keys(self) -> None:
        """Refuse a request that would call a model when the deployment has no keys. Called before
        any side effect: a build job or run loop would otherwise start and fail inside."""
        if not self.container.model_keys:
            raise ModelKeysMissing()

    def run_state(self, world_id: str) -> str | None:
        state = self.application.run_status(world_id)
        return state.value if state is not None else None


def create_app(
    container: Container,
    config: Config,
    *,
    catalog: WorldCatalog | None = None,
) -> Any:
    """Build and return the FastAPI application.

    FastAPI is imported lazily so the module imports even where the package is
    absent; the error then surfaces only when the app is actually constructed.
    ``catalog`` is injectable so tests can pass an in-memory catalog instead of
    touching the on-disk registry.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles

    # One catalog instance shared by the build path (application) and the read
    # path (manager), so a world built via the API is immediately enumerable.
    if catalog is None:
        catalog = WorldCatalog(DEFAULT_CATALOG_PATH)
    services = ApiServices(
        container=container,
        config=config,
        application=NarrativeApplication(container, config, catalog=catalog),
        manager=WorldManager(
            snapshot_provider=container.snapshot,
            catalog=catalog,
        ),
    )

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncIterator[None]:
        yield
        # Runs after uvicorn has stopped taking requests: let every run finish its step.
        await services.application.shutdown()

    app = FastAPI(title="Asamana API", lifespan=lifespan)

    @app.exception_handler(ModelKeysMissing)
    async def model_keys_missing(request: Any, exc: ModelKeysMissing) -> Any:
        logger.info(
            "model_call_refused",
            extra={"method": request.method, "path": request.url.path},
        )
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    # CORS: the frontend may be deployed on its own origin. Origins come from
    # config.web.cors_origins, overridden by ASAMANA_CORS_ORIGINS (comma-separated) so a
    # deployment can restrict them without editing the baked config. "*" requires
    # credentials off, which holds because the API is token-free. WebSockets ignore CORS.
    env_origins = [o.strip() for o in os.environ.get("ASAMANA_CORS_ORIGINS", "").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=env_origins or config.web.cors_origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from interaction.api.direct import build_direct_router
    from interaction.api.lifecycle import build_lifecycle_router
    from interaction.api.maps import build_maps_router
    from interaction.api.observe import build_observe_router

    app.include_router(build_lifecycle_router(services))
    app.include_router(build_observe_router(services))
    app.include_router(build_maps_router(services))
    # Director routes are a product feature, not a developer instrument — they ship in
    # every deployment, unlike the dev_tools block below.
    app.include_router(build_direct_router(services))

    # What this deployment offers, fixed for the process's lifetime. Each flag is reported from
    # the same value its gate reads, so a client never infers a gate from 404s or 503s.
    # Developer instruments spawn subprocesses and send arbitrary prompts to the paid LLM, so a
    # public deployment must not expose them: they sit behind config.web.dev_tools_enabled.
    @app.get("/api/deployment")
    async def get_deployment() -> dict[str, Any]:
        return {
            "dev_tools": config.web.dev_tools_enabled,
            "model_keys": container.model_keys,
        }

    if config.web.dev_tools_enabled:
        from interaction.api.dev import build_dev_router
        from interaction.api.template_import import build_template_import_router
        from interaction.api.template_preview import build_template_preview_router

        app.include_router(build_dev_router(services))
        # The map workbench's data source: a template's map/art as it is on disk,
        # with no world in front of it. A workbench for whoever draws the maps.
        app.include_router(build_template_preview_router())
        # The same workbench's write side — it installs a template into the source
        # tree, so it belongs behind this gate rather than beside the preview alone.
        app.include_router(build_template_import_router())

    # Co-located frontend is a LOCAL convenience only. In production the frontend
    # ships as its own image (nginx) and this backend has no dist → API-only, with
    # a small info page at "/". Mounted last so it never shadows the API/WS routes.
    if _FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
    else:

        @app.get("/")
        async def root() -> Any:
            return JSONResponse({"service": "asamana-api", "docs": "/docs"})

    logger.info(
        "api_app_created",
        extra={
            "frontend_built": _FRONTEND_DIST.exists(),
            "dev_tools": config.web.dev_tools_enabled,
            "model_keys": container.model_keys,
        },
    )
    return app
