"""Strict validation primitives for contracts, config and external input: a value that breaks
its rule raises ``ValueError`` (Rule 3) and is never corrected.

The lenient sibling is ``core.coerce``, which repairs loosely-typed data (LLM payloads,
snapshots) instead. Which range a value may take is owned by its domain (``TRUST_RANGE``,
``EMOTION_INTENSITY_RANGE``...); this module only says how it is checked and how a failure reads.
Checks that depend on context (what a turn listed, whether a world has run) are not value checks
and stay with their owner.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# Comparisons are written as ``not <holds>`` so that NaN, which fails every comparison, is refused.


def require_in_range(value: float, bounds: tuple[float, float], name: str) -> None:
    low, high = bounds
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value!r}")


def require_at_least(value: float, minimum: float, name: str) -> None:
    if not value >= minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")


def require_positive(value: float, name: str) -> None:
    if not value > 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")


def require_text(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def require_member(value: Any, choices: Iterable[Any], name: str) -> None:
    allowed = tuple(choices)
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")
