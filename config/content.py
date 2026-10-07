"""Editable interface content, read fresh on every request.

Kept apart from ``config.yaml`` (why: see ``config/content.yaml``). Deliberately
uncached: editing the file and reloading the page is the whole workflow.

Missing file → no content. A file that exists but can't be read is logged at
``error`` yet still degrades to empty: a typo in sample prompts must not take
the home screen down.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CONTENT_PATH = Path("config/content.yaml")
CONTENT_ENV_VAR = "CONTENT_CONFIG"


def content_path() -> Path:
    """Override with ``CONTENT_CONFIG``."""
    return Path(os.getenv(CONTENT_ENV_VAR) or DEFAULT_CONTENT_PATH)


def _load() -> dict[str, Any]:
    path = content_path()
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.error("content_config_unreadable", extra={"path": str(path), "error": str(exc)})
        return {}
    if data is None:
        return {}
    if not isinstance(data, dict):
        logger.error("content_config_not_a_mapping", extra={"path": str(path)})
        return {}
    return data


def theme_presets() -> list[dict[str, str]]:
    """Sample themes offered on the home screen, in authored order.

    Entries missing either field are dropped rather than rendered half-blank.
    """
    raw = _load().get("presets")
    if raw is None:
        return []
    if not isinstance(raw, list):
        logger.error("content_presets_not_a_list", extra={"path": str(content_path())})
        return []

    presets: list[dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "").strip()
        theme = str(entry.get("theme") or "").strip()
        if title and theme:
            presets.append({"title": title, "theme": theme})
        else:
            logger.error(
                "content_preset_incomplete",
                extra={"path": str(content_path()), "entry": repr(entry)[:120]},
            )
    return presets
