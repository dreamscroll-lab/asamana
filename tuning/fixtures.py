"""Frozen stage-input snapshots for isolated per-stage tuning.

A fixture holds the upstream-derived inputs a downstream stage consumes
(e.g. ``external_goals`` produced by world_pressure feeds perception_emotion /
need). Capturing them once and re-injecting on later runs is what makes
single-stage tuning deterministic: only the stage's own prompt varies.

Stored as ``{trace_dir}/{world_id}/fixtures/{name}.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.logging import get_logger
from tuning.trace import write_json

logger = get_logger(__name__)


def _fixture_path(trace_dir: str, world_id: str, name: str) -> Path:
    return Path(trace_dir) / world_id / "fixtures" / f"{name}.json"


def save_fixture(trace_dir: str, world_id: str, name: str, data: dict[str, Any]) -> None:
    path = _fixture_path(trace_dir, world_id, name)
    write_json(path, data)
    logger.info("fixture_saved", extra={"world_id": world_id, "fixture": name})


def load_fixture(trace_dir: str, world_id: str, name: str) -> dict[str, Any] | None:
    path = _fixture_path(trace_dir, world_id, name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

