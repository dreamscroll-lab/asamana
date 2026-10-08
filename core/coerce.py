"""Pure type-coercion primitives for loosely-typed external data (LLM payloads,
snapshots, config), shared across layers.

Two families:

- **Total** (`coerce_int/float/str/...`): always return a usable value, taking a
  ``default`` and optional ``[minimum, maximum]`` clamp. Use when the caller must
  have a value (world-building field coercion, config defaults).
- **Optional** (`coerce_optional_*`, `coerce_datetime`): return ``None`` when the
  value is absent/unparsable. Use when ``None`` is meaningful (a snapshot field
  that may be missing).

Domain coercion that returns project enums (e.g. ``NeedType``/``EmotionType``)
does NOT belong here — it stays in the owning subsystem, which picks the default
and delegates the mechanics to ``coerce_enum``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

_E = TypeVar("_E", bound=Enum)

# ---------------------------------------------------------------------------
# Total coercion — always returns a value
# ---------------------------------------------------------------------------


def coerce_int(
    value: Any,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Coerce ``value`` to ``int``, falling back to ``default``, then clamp."""

    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def coerce_float(
    value: Any,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Coerce ``value`` to ``float``, falling back to ``default``, then clamp."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def coerce_str(value: Any, fallback: str = "") -> str:
    """Stringify and strip; ``None``/empty becomes ``fallback``."""

    if value is None:
        return fallback
    return str(value).strip() or fallback


def coerce_enum(enum_cls: type[_E], value: Any, *, default: _E) -> _E:
    """Read a lowercase-valued enum case-insensitively; anything unknown becomes ``default``."""

    if isinstance(value, enum_cls):
        return value
    if value is None:
        return default
    try:
        return enum_cls(str(value).strip().lower())
    except ValueError:
        return default


def coerce_str_list(value: Any) -> list[str]:
    """Coerce a sequence (or scalar) into a list of non-empty stripped strings."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [text for item in value if (text := coerce_str(item))]
    text = coerce_str(value)
    return [text] if text else []


def coerce_mapping_list(value: Any) -> list[Mapping[str, Any]]:
    """Return only the ``Mapping`` items of a sequence (empty for non-sequence)."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def coerce_dict(value: Any) -> dict[str, Any]:
    """Return ``dict(value)`` when ``value`` is a mapping, else ``{}``."""

    return dict(value) if isinstance(value, Mapping) else {}


def coerce_list(value: Any) -> list[Any]:
    """Return ``list(value)`` when ``value`` is a list/tuple, else ``[]``."""

    return list(value) if isinstance(value, (list, tuple)) else []


# ---------------------------------------------------------------------------
# Optional coercion — returns None when absent/unparsable
# ---------------------------------------------------------------------------


def coerce_optional_int(value: Any) -> int | None:
    """Coerce to ``int`` or ``None``. Rejects ``bool``: a bool where an int is expected is a
    type error, not ``1``."""

    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_optional_float(
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    """Coerce to ``float`` or ``None``; clamp into ``[minimum, maximum]`` when given."""

    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def coerce_optional_str(value: Any) -> str | None:
    """Stringify and strip; ``None``/empty/whitespace becomes ``None``."""

    text = coerce_str(value)
    return text or None


def coerce_optional_bool(value: Any) -> bool | None:
    """Pass through real ``bool`` values; everything else becomes ``None``."""

    if isinstance(value, bool):
        return value
    return None


def coerce_datetime(value: Any) -> datetime | None:
    """Pass through ``datetime``; parse ISO-8601 strings; else ``None``."""

    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None
