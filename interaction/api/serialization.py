"""JSON serialization helpers shared across API routers.

The API boundary is render-neutral: it emits semantic world state (dataclasses,
enums, datetimes) as plain JSON. No presentation concepts (coordinates, pixels,
layout) cross this boundary — a client decides how to render.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from enum import Enum
from typing import Any


def _default(obj: Any) -> Any:
    # Reached for a dataclass anywhere in the value, not only at its top: a list of them too.
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def to_json(obj: Any) -> str:
    """Serialize *obj* to a JSON string, handling dataclasses and special types."""
    return json.dumps(obj, default=_default, ensure_ascii=False)


def to_wire_json(obj: Any) -> Any:
    """Convert *obj* into plain JSON-safe primitives (dict/list/str/...); raises TypeError on a
    value with no JSON form. Distinct from ``core.interfaces.trace.to_jsonable``, which never raises."""
    return json.loads(to_json(obj))
