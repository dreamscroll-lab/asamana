"""The shaping every map artifact needs, wherever it is read FROM.

A built world serves its copy frozen at build (so a replay looks as the world looked); a
template serves the live files on disk (so a map author sees what they just drew). Source
resolution and cache policy differ and stay with each router. What is done to the bytes does
not: resolving the paths a .tmj and a cast manifest name into URLs, and deriving the ground
from the art. That depends on the artifact format, so it lives here once; per-router copies
drift silently into a map that fails to draw on one page.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from worlds.tiled import CAST_ART_FIELDS

# How a caller hands over one asset's bytes, by the path AS WRITTEN IN THE MAP.
# Async because one of the two sources is a storage provider; the other simply
# wraps its file read, which is cheaper than making this module care.
ReadAsset = Callable[[str], Awaitable[bytes | None]]


def tileset_images(doc: dict[str, Any]) -> list[str]:
    """Every image path the map's tilesets name, in document order, without blanks.

    A collection-of-images tileset has no sheet of its own and simply contributes
    nothing — the same reading the renderer takes when it skips one (see
    frontend `mapSource.prepareMapSource`).
    """
    return [
        image
        for ts in doc.get("tilesets", [])
        if (image := str(ts.get("image") or ""))
    ]


def asset_index(doc: dict[str, Any], base: str) -> dict[str, str]:
    """``{path as written in the .tmj: URL to fetch it from}``.

    The client never composes an asset path: where the art is reachable from is a
    deployment fact (a route, a CDN, a signed URL). A path with no entry is absent,
    not guessed.
    """
    return {image: f"{base}/{image}" for image in tileset_images(doc)}


def with_asset_urls(manifest: dict[str, Any], base: str) -> dict[str, Any]:
    """A cast manifest with its atlases' relative paths resolved into URLs.

    Same rule as ``asset_index``: the sheets sit beside the map and are frozen and
    addressed the same way.
    """
    atlases = manifest.get("atlases")
    if isinstance(atlases, dict):
        manifest["atlases"] = {
            key: {
                **entry,
                **{
                    field: f"{base}/{value}"
                    for field in CAST_ART_FIELDS
                    if (value := str(entry.get(field) or ""))
                },
            }
            for key, entry in atlases.items()
            if isinstance(entry, dict)
        }
    return manifest


async def ground_payload(doc: dict[str, Any], read_asset: ReadAsset) -> dict[str, Any]:
    """Which cells of this map a body may stand on, and which carry a route.

    Derived on the server because it has three consumers (the renderer's staging, its A*
    network, ``template_check``); a client-side copy would leave the validator
    re-implementing the geometry or checking nothing. Sheets are resolved up front, each
    fetched once, so ``derive_ground`` stays synchronous and free of the storage layer.
    """
    from worlds.ground import derive_ground

    sheets: dict[str, bytes | None] = {}
    for image in tileset_images(doc):
        if image not in sheets:
            sheets[image] = await read_asset(image)
    # Decoding and per-tile sampling is synchronous and takes tens of milliseconds per map;
    # running it on the event loop would stall LLM callbacks of running worlds, so use a thread.
    return await asyncio.to_thread(lambda: derive_ground(doc, sheets.get).to_payload())
