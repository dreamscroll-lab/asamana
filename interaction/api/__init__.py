"""HTTP + WebSocket API for the Asamana backend.

Split by responsibility:

- ``app`` — the FastAPI factory that wires shared services and mounts routers.
- ``lifecycle`` — world create/build/confirm/rename/delete + run control (run/pause/resume/stop/reset).
- ``setup`` — a world's setup, read and revised until the world is confirmed.
- ``observe`` — read-only observation: steps, the relationship graph, the world's identity
  tables (characters, NPCs, places) and the live WebSocket feed. Render-neutral: emits
  semantic world state only.
- ``maps`` — a built world's map artifact: the .tmj, its ground, its art, the cast's art.
- ``template_preview`` / ``template_import`` — the map workbench: a template's artifact as it is
  on disk, and installing or removing one. Developer-gated like ``dev``.
- ``vocabulary`` — the closed vocabularies a client names cognition in (emotions, needs).
- ``direct`` — the director: submits a free-text intervention and advances the world one step
  at a time.
- ``actor`` — the actor performing one character: its turns, answers, mind and exit.
- ``dev`` — developer tooling, one module per tool (LLM traces, audit, stage suites,
  playground, director console, recall inspector). Gated behind
  ``config.web.dev_tools_enabled``; never mounted in a public deployment.

Besides lifecycle, only ``setup``, ``direct`` and ``actor`` change a world; everything else
observes it.
"""

from __future__ import annotations

from interaction.api.app import create_app

__all__ = ["create_app"]
