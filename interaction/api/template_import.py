"""Install or remove a map from the browser — the write half of the map workbench.

Mounted with ``template_preview`` behind the developer gate, since it writes into the
source tree. What a template is stays in ``worlds/template_import``; this layer reads the
request and maps the module's refusals to status codes.
"""

from __future__ import annotations

import asyncio
from typing import Any

# Must be importable at module scope so FastAPI can resolve `request: Request` (see maps.py).
from fastapi import Request

from worlds.template_import import (
    MAX_ARCHIVE_BYTES,
    TemplateExistsError,
    TemplateImportError,
    TemplateMissingError,
    WorldNameTakenError,
    delete_template,
    install_template,
)


def build_template_import_router() -> Any:
    from fastapi import APIRouter, HTTPException

    router = APIRouter(prefix="/api/templates/{template}", tags=["template-import"])

    @router.put("/archive")
    async def put_archive(template: str, request: Request) -> Any:
        """Install the uploaded zip as template ``{template}``.

        The archive is the raw body; a multipart envelope would cost a dependency for nothing.
        Read with a running cap rather than ``await request.body()``: nothing above imposes a
        limit, and this process also runs the live worlds, so an oversized body would hurt every
        simulation in flight.
        """
        chunks: list[bytes] = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_ARCHIVE_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"archive exceeds {MAX_ARCHIVE_BYTES >> 20} MB",
                )
            chunks.append(chunk)

        try:
            # Unpacking and checking decode every tileset in the map — the same block
            # of synchronous work that took the ground derivation off the loop.
            installed = await asyncio.to_thread(
                install_template, template, b"".join(chunks)
            )
        except (TemplateExistsError, WorldNameTakenError) as exc:
            # Both are a clash with what is already installed, and neither is fixable
            # by sending the same archive again — 409, not 400.
            raise HTTPException(status_code=409, detail=exc.message) from exc
        except TemplateImportError as exc:
            if exc.problems:
                # An object, where FastAPI's own 422 carries a list — so a client can
                # tell a failed contract from a malformed request at the same status.
                raise HTTPException(
                    status_code=422,
                    detail={"message": exc.message, "problems": exc.problems},
                ) from exc
            raise HTTPException(status_code=400, detail=exc.message) from exc

        # Nothing to invalidate: the template list is scanned per request, its entries
        # are re-parsed per request, and the preview's ground cache is keyed on the
        # map file's mtime. The map is live on the next page load.
        return {
            "template": template,
            "files": installed.files,
            # Said out loud because the archive is not byte-for-byte what landed: the
            # map's own ground decides its connections (see worlds.template_import).
            "connections": installed.connections,
        }

    @router.delete("")
    async def delete_map(template: str) -> Any:
        """Remove the installed map ``{template}``.

        Worlds built on it keep working — each carries its own frozen copy of the map
        (see ``worlds.template_import.delete_template``). Removing 20 MB of art is real
        disk work, so it goes off the loop like the install does.
        """
        try:
            await asyncio.to_thread(delete_template, template)
        except TemplateMissingError as exc:
            raise HTTPException(status_code=404, detail=exc.message) from exc
        except TemplateImportError as exc:
            raise HTTPException(status_code=400, detail=exc.message) from exc
        return {"template": template, "deleted": True}

    return router
