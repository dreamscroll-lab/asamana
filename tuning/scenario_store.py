"""Scenario library: reads and writes `tuning/scenarios/<stage>.json`.

Scenarios are reviewed test corpus (see the tuning methodology: no cheating, able to genuinely
fail, anchored on the failure side) and live in git like tests. So every add/edit/delete goes to
that file itself. Don't keep a separate draft store: it would split into two sources of truth
immediately, with the CLI, CI and the next person reading the file while the UI's edits stay
invisible.

This module depends only on the standard library, and that is why it exists. The developer
tools' backend routes need add/edit/delete, while touching the `tuning.stages` registry pulls in
every validation module → phase_harness → agent / engine / world / providers, hundreds of
modules in all. That stack belongs to the CLI process and shouldn't be loaded into the web process just
because someone saved a scenario on a page. So functions here take (file path, criteria names)
rather than ``StageSpec``: ``tuning.stages`` (used by the CLI) needs the registry, this layer doesn't.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class ScenarioError(ValueError):
    """Invalid scenario data (name, type, criteria). Callers turn it into a 400."""


# A scenario name is also a report directory name (`validation/<stage>/<name>/`), so it must be
# a safe directory name: a name with `/` or `..` would write the report outside. This is a
# path-traversal guard, not a style rule.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def load(path: Path) -> dict[str, Any]:
    """The whole scenario file (including non-scenario keys like `_comment` / `knob_sets`)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenarioError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(data.get("scenarios"), list):
        raise ScenarioError(f"{path.name} has no scenarios list")
    return data


def save(path: Path, data: dict[str, Any]) -> None:
    """Write to disk, atomically and in a stable format.

    Atomic (temp file + `os.replace`): a half-written file corrupts a whole stage's corpus, which is
    a reviewed asset.

    The format is pinned (`indent=2`, non-ASCII unescaped, trailing newline) and the files on disk
    are normalized to it, so a UI edit diffs as just that scenario instead of reflowing the whole
    file. Hand edits still work: the next save normalizes them.
    """
    body = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)


def names(path: Path) -> list[str]:
    """Scenario names in the file. Missing/corrupt file → empty list (the stage still lists, there's just nothing to pick)."""
    try:
        data = load(path)
    except ScenarioError:
        return []
    return [str(s.get("name", "")) for s in data["scenarios"] if s.get("name")]


def validate(entry: Any, *, criteria: Sequence[str], stage: str) -> str:
    """Validate one scenario and return its name; raise ScenarioError if invalid.

    Checks only the three things that can be checked: the name (it's a directory name), the
    top-level types, and that `criteria_focus` is among the stage's criteria (a typo would silently
    focus on nothing). Keys of the `scenario` payload differ per stage, and this layer doesn't
    define them for phase_harness — that would mean one schema per stage to keep in sync.
    """
    if not isinstance(entry, dict):
        raise ScenarioError("a scenario must be a JSON object")
    name = str(entry.get("name") or "").strip()
    if not _NAME_RE.match(name):
        raise ScenarioError(
            f"invalid scenario name {name!r}: it doubles as the report directory name, so it must "
            "be 1-64 chars of letters, digits, _ . - and start with a letter or digit"
        )
    if "scenario" in entry and not isinstance(entry["scenario"], dict):
        raise ScenarioError("the `scenario` payload must be a JSON object")
    focus = entry.get("criteria_focus")
    if focus is not None:
        if not isinstance(focus, list) or not all(isinstance(c, str) for c in focus):
            raise ScenarioError("`criteria_focus` must be a list of strings")
        # Check membership only when the stage really has judge criteria. Judge-less suites (perception)
        # use this field for their deterministic focus (threshold / caps / dedup …), names the registry doesn't and shouldn't have.
        unknown = [c for c in focus if c not in criteria] if criteria else []
        if unknown:
            raise ScenarioError(
                f"unknown criteria_focus {unknown}; {stage} scores {', '.join(criteria)}"
            )
    return name


def add(path: Path, entry: dict[str, Any], *, criteria: Sequence[str], stage: str) -> str:
    """Add one (name already exists → ScenarioError). Appended at the end; existing order unchanged."""
    name = validate(entry, criteria=criteria, stage=stage)
    data = load(path)
    if any(s.get("name") == name for s in data["scenarios"]):
        raise ScenarioError(f"{stage} already has a scenario named {name!r}")
    data["scenarios"].append(entry)
    save(path, data)
    return name


def replace(
    path: Path, at: str, entry: dict[str, Any], *, criteria: Sequence[str], stage: str,
) -> str:
    """Replace the scenario named *at*. A different entry name means a rename (replaced in place, position kept)."""
    name = validate(entry, criteria=criteria, stage=stage)
    data = load(path)
    index = next((i for i, s in enumerate(data["scenarios"]) if s.get("name") == at), None)
    if index is None:
        raise ScenarioError(f"{stage} has no scenario named {at!r}")
    if name != at and any(s.get("name") == name for s in data["scenarios"]):
        raise ScenarioError(f"{stage} already has a scenario named {name!r}")
    data["scenarios"][index] = entry
    save(path, data)
    return name


def delete(path: Path, name: str, *, stage: str) -> None:
    """Delete one. The file is tracked by git, so an accidental delete is undone with `git checkout`."""
    data = load(path)
    remaining = [s for s in data["scenarios"] if s.get("name") != name]
    if len(remaining) == len(data["scenarios"]):
        raise ScenarioError(f"{stage} has no scenario named {name!r}")
    data["scenarios"] = remaining
    save(path, data)


def write_subset(path: Path, picked: Sequence[str], dest: Path, *, stage: str) -> None:
    """Trim the scenario file to just the *picked* ones and write it to dest; this is how "run a few" works.

    Trimming the file rather than adding a `scenario_names` parameter to every validate_*: all
    suites already share a signature that accepts `scenarios_path`, and a new parameter would be the
    same change in every one of them. Top-level keys other than `scenarios` (e.g. perception's
    `knob_sets`) are kept as-is. Scenarios are taken in the requested order, so they run in the
    order they were ticked in the UI.
    """
    data = load(path)
    by_name = {s.get("name"): s for s in data["scenarios"]}
    missing = [n for n in picked if n not in by_name]
    if missing:
        raise ScenarioError(f"stage {stage!r} has no scenario named {missing[0]!r}")
    save(dest, {**data, "scenarios": [by_name[n] for n in picked]})
