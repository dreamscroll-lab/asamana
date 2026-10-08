"""Canonical JSON serialization for persisted artifacts; writers route through here.

- ``ensure_ascii=False`` — keep CJK readable (``长安``, not ``\\uXXXX``): stored data is read by
  humans and round-tripped back through prompts.
- ``default=str`` — coerce non-JSON-native values (``datetime``, ``Path``, …) rather than raising.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def dump_json(obj: Any) -> str:
    """Serialize *obj* to a JSON string for persistence (CJK-readable)."""

    return json.dumps(obj, ensure_ascii=False, default=str)


def read_json_file(path: Path) -> Any | None:
    """Read a persisted JSON file, returning None when it doesn't exist.

    The existence check stays inside: the whole function runs via ``asyncio.to_thread``, and
    splitting check from read would leave a gap in which the file could vanish.
    """
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* via tmp + rename so a crash never leaves a truncated file.

    The tmp file sits beside the target, so ``os.replace`` is an atomic rename. The pid alone
    is not a unique tmp name: callers run in ``asyncio.to_thread``, so two threads of one process
    can write the same target, and one would replace away the other's tmp (FileNotFoundError).
    The random suffix gives each writer its own tmp; the last replace wins.
    """

    tmp = path.with_name(f"{path.name}.{os.getpid()}.{os.urandom(4).hex()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
