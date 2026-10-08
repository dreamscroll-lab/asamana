"""Pure numeric primitives shared across layers. Only stateless numeric utilities belong
here, not a catch-all bucket."""

from __future__ import annotations


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
