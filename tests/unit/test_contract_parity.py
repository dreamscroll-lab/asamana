"""The frontend's copy of the contract must match the closed sets in ``core.interfaces.action``
value for value.

``ActionType`` / ``Deed`` / ``RefKind`` are closed sets, and the frontend keeps a hand-copied
duplicate (the renderer doesn't import Python). A value missing from the copy raises nothing:
TypeScript checks exhaustiveness against the copy, so a "full table" missing a member is still
complete as far as types go, and at runtime the lookup misses and silently draws nothing — e.g. an
errand shows as a placeholder `•` next to a motionless figure.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.interfaces import action as A
from core.interfaces.action import ActionType, Deed

_CONTRACT = Path("frontend/src/lib/contract.ts")


def _ts_members(name: str) -> set[str]:
    src = _CONTRACT.read_text()
    body = src.split(f"export const {name} = {{", 1)[1].split("} as const;", 1)[0]
    # Strip trailing comments before taking keys, so colons in comments aren't read as members
    return set(re.findall(r"^\s*([a-z_]+)\s*:", re.sub(r"//.*", "", body), re.M))


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("ActionType", {t.value for t in ActionType}),
        ("Deed", {d.value for d in Deed}),
        # The backend has no RefKind enum; categories are a set of KIND_* constants — that set is
        # what the copy must match.
        ("RefKind", {
            v for k, v in vars(A).items() if k.startswith("KIND_") and isinstance(v, str)
        }),
    ],
)
def test_the_frontend_copy_of_a_closed_set_has_every_value(name: str, values: set[str]) -> None:
    assert _ts_members(name) == values
