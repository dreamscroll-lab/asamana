"""A built world's map artifact: the .tmj, the ground derived from its art, the art, and the cast's art.

Each route serves the world's copy frozen at build, and the live template only for a world built
before the freeze. ``ASSET_SUFFIXES`` and ``image_response`` are the terms art is served on, shared
with ``template_preview``.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any, Mapping

from core.logging import get_logger

# Request must be importable at module scope: with `from __future__ import annotations`, FastAPI
# resolves the `request: Request` hint against module globals, so a function-local import leaves it
# unresolved and the parameter is mis-read as a required query field.
from fastapi import Request, Response
from fastapi.responses import FileResponse

from interaction.api import map_artifact
from interaction.api.app import ApiServices

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
# What the asset route will hand out. Images, plus the sprite-sheet descriptor a
# character atlas is indexed by — a sheet without it is an undivided rectangle.
# `.tmj` and `.json` stay OUT: the map document and the character manifest are
# served by their own routes, which resolve asset paths into URLs first.
ASSET_SUFFIXES = _IMAGE_SUFFIXES | {".xml"}


def _weak_etag(data: bytes) -> str:
    return f'"{hashlib.md5(data, usedforsecurity=False).hexdigest()}"'


def image_response(
    request: Request,
    body: bytes | Path,
    etag: str,
    media_type: str = "image/png",
    *,
    immutable: bool = False,
) -> Response:
    """Serve an image with the strongest caching its mutability allows.

    Starlette's FileResponse emits an ETag but never answers a conditional request; this does.

    * A world's frozen art never changes (written once at build, URL carries the world id):
      `max-age` + `immutable`, so the client stops asking — one round trip saved per tileset.
    * Live template art (pre-freeze worlds) is edited by hand between runs: `no-cache`
      ("revalidate first"), so an unchanged file costs a 304 and an edit shows up at once.
    """
    cache_control = "public, max-age=31536000, immutable" if immutable else "no-cache"
    headers = {"etag": etag, "cache-control": cache_control}
    if etag in [tag.strip() for tag in request.headers.get("if-none-match", "").split(",")]:
        return Response(status_code=304, headers=headers)
    if isinstance(body, Path):
        return FileResponse(body, headers=headers)
    return Response(content=body, media_type=media_type, headers=headers)


logger = get_logger(__name__)

# The map a world renders on when its config names none.
_DEFAULT_TEMPLATE = "changan_iso"


def _template_name(config: Mapping[str, Any] | None) -> str:
    return str((config or {}).get("runtime_context", {}).get("template") or _DEFAULT_TEMPLATE)


def build_maps_router(services: ApiServices) -> Any:
    from fastapi import APIRouter, HTTPException

    router = APIRouter(tags=["maps"])
    manager = services.manager

    # Derived ground per world (see /map/ground). Both its inputs — the map and the
    # art — are frozen at build, so an entry can never go stale; a deleted world's
    # entry is dead weight of a few KB, which is cheaper than a cross-router hook.
    _ground_cache: dict[str, dict[str, Any]] = {}

    @router.get("/api/worlds/{world_id}/map")
    async def get_map(world_id: str) -> dict[str, Any]:
        """The world's Tiled map (.tmj), served from the backend so the 2D renderer and the
        engine share one map artifact.

        Served from the per-world copy frozen at build (``save_world_map``), so template edits
        never drift a built world's map away from what its engine simulates. Worlds built before
        the freeze fall back to the live template named in their config's runtime_context.
        """
        from worlds.tiled import MAP_FILENAME, template_dir

        frozen = await manager.snapshot_provider.load_world_map(world_id)
        if frozen is not None:
            return frozen

        config = await manager.snapshot_provider.load_world_config(world_id)
        if not config:
            raise HTTPException(status_code=404, detail=f"No world config for: {world_id}")
        template = _template_name(config)
        logger.debug("world_map_fallback_to_template", extra={"world_id": world_id, "template": template})
        tmj_path = template_dir(template) / MAP_FILENAME
        if not tmj_path.exists():
            raise HTTPException(status_code=404, detail=f"No map template: {template}")
        try:
            return json.loads(tmj_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=f"Map unreadable: {exc}") from exc

    @router.get("/api/worlds/{world_id}/map/ground")
    async def get_map_ground(world_id: str) -> dict[str, Any]:
        """Which cells of this world's map a body may stand on, and which carry a route,
        derived from the map's own art (server-side: see ``map_artifact.ground_payload``).

        Read through the world's own copy of the art, the same precedence ``/map`` uses, and
        cached per world since those inputs are frozen at build.
        """
        from worlds.tiled import read_template_asset, template_dir

        if world_id in _ground_cache:
            return _ground_cache[world_id]

        doc = await get_map(world_id)
        config = await manager.snapshot_provider.load_world_config(world_id)
        template = _template_name(config)
        root = template_dir(template)

        async def read_asset(image: str) -> bytes | None:
            frozen = await manager.snapshot_provider.load_world_asset(world_id, image)
            if frozen is not None:
                return frozen
            return read_template_asset(root, image)

        payload = await map_artifact.ground_payload(doc, read_asset)
        _ground_cache[world_id] = payload
        return payload

    @router.get("/api/worlds/{world_id}/map/assets")
    async def get_map_assets(world_id: str) -> dict[str, str]:
        """``{path as written in the .tmj: URL to fetch it from}`` for each image the map names.

        Where art is reachable from is the backend's call (the route below today, a CDN or a
        signed URL tomorrow); the client never composes a path.
        """
        return map_artifact.asset_index(
            await get_map(world_id), f"/api/worlds/{world_id}/map/assets"
        )

    @router.get("/api/worlds/{world_id}/characters")
    async def get_characters(world_id: str) -> dict[str, Any]:
        """The cast's art manifest: which body serves each (gender, age bracket), what this art
        calls its poses, and where each sheet is reachable.

        Character art belongs to the map (a body is dressed for the map's period), so it is
        authored beside the map, frozen at build, and served with URLs resolved here, as
        ``/map/assets`` does. The frozen copy wins over the live template, so cast and ground
        never come from two points in time.
        """
        from worlds.tiled import CHARACTERS_FILENAME, template_dir

        manifest: dict[str, Any] | None = None
        frozen = await manager.snapshot_provider.load_world_asset(
            world_id, CHARACTERS_FILENAME
        )
        if frozen is not None:
            try:
                manifest = json.loads(frozen.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                logger.warning(
                    "frozen_character_manifest_unreadable",
                    extra={"world_id": world_id, "error": str(exc)},
                )
        if manifest is None:
            config = await manager.snapshot_provider.load_world_config(world_id)
            if not config:
                raise HTTPException(status_code=404, detail=f"No world config for: {world_id}")
            template = _template_name(config)
            path = template_dir(template) / CHARACTERS_FILENAME
            if not path.is_file():
                raise HTTPException(status_code=404, detail=f"No cast art for: {template}")
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except ValueError as exc:
                raise HTTPException(
                    status_code=500, detail=f"Cast manifest unreadable: {exc}"
                ) from exc

        return map_artifact.with_asset_urls(manifest, f"/api/worlds/{world_id}/map/assets")

    @router.get("/api/worlds/{world_id}/map/assets/{asset_path:path}")
    async def get_map_asset(world_id: str, asset_path: str, request: Request) -> Response:
        """A tileset image the world's .tmj references, by the RELATIVE path written
        in the map itself (``tilesets/base/ground/ground.png``).

        Map and art are one artifact, so the backend serves the pixels too and a template stays
        one self-contained drop-in directory. Also serves the cast manifest's sheets, which sit
        in the same directory. The frozen copy wins over the live template, as for the map.

        The path comes off the URL, so it is confined to that directory and to known asset
        suffixes before anything touches the filesystem.
        """
        from worlds.tiled import template_asset_path, template_dir

        if Path(asset_path).suffix.lower() not in ASSET_SUFFIXES:
            raise HTTPException(status_code=404, detail="No such map asset")

        frozen = await manager.snapshot_provider.load_world_asset(world_id, asset_path)
        if frozen is not None:
            media_type = mimetypes.guess_type(asset_path)[0] or "application/octet-stream"
            # Frozen art can never change under this address, so the client should stop asking.
            return image_response(
                request, frozen, _weak_etag(frozen), media_type, immutable=True
            )

        config = await manager.snapshot_provider.load_world_config(world_id)
        if not config:
            raise HTTPException(status_code=404, detail=f"No world config for: {world_id}")
        template = _template_name(config)

        root = template_dir(template)
        # The suffix allowlist keeps the route from reading a template's own map.tmj.
        target = template_asset_path(root, asset_path)
        if target is None:
            raise HTTPException(status_code=404, detail=f"No such map asset: {asset_path}")
        stat = target.stat()
        return image_response(
            request, target, f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
        )

    return router
