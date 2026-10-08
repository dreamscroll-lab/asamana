"""The result shape every stage judge returns."""

from __future__ import annotations

from typing import Any


def empty_judgment(criteria: tuple[str, ...], note: str) -> dict[str, Any]:
    """Zero on every criterion, with *note* as the reason — when the judge could not run or
    answered with something unusable."""
    base: dict[str, Any] = {c: {"score": 0, "rationale": note, "issues": []} for c in criteria}
    base["overall"] = note
    return base
