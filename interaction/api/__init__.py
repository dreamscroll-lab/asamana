"""HTTP + WebSocket API for the Asamana backend.

Split by responsibility:

- ``app`` — the FastAPI factory that wires shared services and mounts routers.
- ``lifecycle`` — world create/build/delete + run control (run/pause/resume/stop/reset).
- ``observe`` — read-only observation: steps, the relationship graph,
  and the live WebSocket feed. Render-neutral: emits semantic world state only.
- ``maps`` — a built world's map artifact: the .tmj, its ground, its art, the cast's art.
- ``direct`` — the human author. Submits a free-text intervention and advances the
  world one step at a time. This is the only surface that reaches INTO a running
  world; everything else here either observes it or governs its lifecycle.
- ``dev`` — developer tooling, one module per tool (LLM traces, audit, stage suites,
  playground, director console, recall inspector). Gated behind
  ``config.web.dev_tools_enabled``; never mounted in a public deployment.
"""

from __future__ import annotations

from interaction.api.app import create_app

__all__ = ["create_app"]
