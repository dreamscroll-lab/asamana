"""Serve a TEMPLATE's map artifact directly — the map workbench's data source.

A world's map is frozen at build: right for replay, wrong for working on the art. These
routes serve the same four payloads the renderer needs (map, asset index, ground, cast)
from ``worlds/templates/<name>/`` as it is on disk now, with no world needed. Gated with
the other developer instruments (``config.web.dev_tools_enabled``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Must be importable at module scope so FastAPI can resolve `request: Request` (see maps.py).
from fastapi import Request

from core.logging import get_logger

from interaction.api import map_artifact

# Same two helpers the world-scoped asset route serves art with — one answer to
# "what may be handed out, and with what caching", so a template's art and a
# world's copy of that same art are never served on different terms.
from interaction.api.maps import ASSET_SUFFIXES, image_response

logger = get_logger(__name__)


def build_template_preview_router() -> Any:
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import JSONResponse

    from worlds.tiled import (
        CHARACTERS_FILENAME, MAP_FILENAME, read_template_asset, template_asset_path, template_dir,
    )

    router = APIRouter(prefix="/api/templates/{template}", tags=["template-preview"])

    def _fresh(payload: Any) -> JSONResponse:
        """A payload that must never be served from a browser cache.

        These routes exist to show the template as it is on disk now, and a response with no
        cache directives may be held heuristically, so a reload would show the previous
        manifest. The art route can say `no-cache` because it has an ETag; these have none.
        """
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})

    # Deriving ground decodes every tileset, too much to repeat per page load, but a template
    # is edited between loads. Keyed on the map file's mtime: reloads are free, a save invalidates.
    _ground_cache: dict[tuple[str, int], dict[str, Any]] = {}

    def _root(template: str) -> Path:
        """The template's directory, or a 404. ``template_dir`` confines the name
        to one path segment, so this cannot be walked out of."""
        root = template_dir(template)
        if not (root / MAP_FILENAME).is_file():
            raise HTTPException(status_code=404, detail=f"No map template: {template}")
        return root

    def _doc(template: str) -> dict[str, Any]:
        """The template's map document. Read per request — it is being edited."""
        path = _root(template) / MAP_FILENAME
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=f"Map unreadable: {exc}") from exc

    @router.get("/map")
    async def get_map(template: str) -> Any:
        return _fresh(_doc(template))

    @router.get("/map/assets")
    async def get_map_assets(template: str) -> Any:
        """``{path as written in the .tmj: URL to fetch it from}`` — the client
        never composes an asset path, exactly as on the world route."""
        return _fresh(
            map_artifact.asset_index(_doc(template), f"/api/templates/{template}/map/assets")
        )

    @router.get("/map/ground")
    async def get_map_ground(template: str) -> Any:
        """Standable/walkable cells, derived from the template's art as it is now.

        Cached server-side on the map file's mtime (see ``_ground_cache``); the browser gets
        none (see ``_fresh``).
        """
        root = _root(template)
        key = (template, (root / MAP_FILENAME).stat().st_mtime_ns)
        if key in _ground_cache:
            return _fresh(_ground_cache[key])

        doc = _doc(template)

        async def read_asset(image: str) -> bytes | None:
            return read_template_asset(root, image)

        payload = await map_artifact.ground_payload(doc, read_asset)
        _ground_cache.clear()  # only the current revision is worth holding
        _ground_cache[key] = payload
        return _fresh(payload)

    @router.get("/characters")
    async def get_characters(template: str) -> Any:
        path = _root(template) / CHARACTERS_FILENAME
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"No cast art for: {template}")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=f"Cast manifest unreadable: {exc}") from exc
        return _fresh(
            map_artifact.with_asset_urls(manifest, f"/api/templates/{template}/map/assets")
        )

    @router.get("/map/assets/{asset_path:path}")
    async def get_map_asset(template: str, asset_path: str, request: Request) -> Any:
        """One image (or atlas descriptor) the map or the cast manifest names.

        Never ``immutable``: a template is the copy being edited, so the answer at
        this URL genuinely changes. ``no-cache`` still costs nothing on an
        unchanged file (one 304, zero bytes) and picks up a redraw immediately.
        """
        if Path(asset_path).suffix.lower() not in ASSET_SUFFIXES:
            raise HTTPException(status_code=404, detail="No such map asset")
        root = _root(template)
        target = template_asset_path(root, asset_path)
        if target is None:
            raise HTTPException(status_code=404, detail=f"No such map asset: {asset_path}")
        stat = target.stat()
        return image_response(request, target, f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"')

    return router
