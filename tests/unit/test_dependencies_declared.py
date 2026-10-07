"""Every third-party module the shipped source imports must be a declared dependency.

A direct import is a direct dependency even when another package happens to install it today:
once that package drops it, the container crash-loops on boot with nothing in this repo changed.

Scope is the source the image carries (deploy/backend.Dockerfile's COPY list), not tests/.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Mirrors deploy/backend.Dockerfile. tuning/ is in here because it ships too: the
# observability UI runs `python -m tuning audit` as a subprocess.
SHIPPED = ["agent", "core", "engine", "interaction", "providers", "world", "worlds", "tuning"]

FIRST_PARTY = {
    "agent", "config", "core", "engine", "interaction",
    "main", "providers", "tests", "tuning", "world", "worlds",
}

# Distribution name → the name you actually import, where the two differ.
IMPORT_NAME = {
    "pyyaml": "yaml",
    "python-dotenv": "dotenv",
    "pillow": "PIL",
}


def _declared() -> set[str]:
    mods = set()
    for line in (REPO / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        dist = re.split(r"[><=!\[]", line)[0].strip().lower()
        mods.add(IMPORT_NAME.get(dist, dist.replace("-", "_")))
    return mods


def _imported() -> dict[str, set[str]]:
    """Top-level third-party module → the files that import it."""
    out: dict[str, set[str]] = {}
    for top in SHIPPED:
        root = REPO / top
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                # level > 0 is a relative import — first-party by construction.
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module.split(".")[0]]
                for name in names:
                    if name in FIRST_PARTY or name in sys.stdlib_module_names:
                        continue
                    out.setdefault(name, set()).add(str(path.relative_to(REPO)))
    return out


def test_every_imported_package_is_declared() -> None:
    declared = _declared()
    undeclared = {mod: files for mod, files in _imported().items() if mod not in declared}
    assert not undeclared, "third-party imports missing from requirements.txt: " + "; ".join(
        f"{mod} ({', '.join(sorted(files))})" for mod, files in sorted(undeclared.items())
    )


def test_every_requirement_has_an_upper_bound() -> None:
    """An open-ended range means the same commit builds different software on different days."""
    unbounded = []
    lines = [
        *(REPO / "requirements.txt").read_text().splitlines(),
        *(REPO / "requirements-dev.txt").read_text().splitlines(),
    ]
    for line in lines:
        line = line.strip()
        if not line or line.startswith(("#", "-r")):
            continue
        if "<" not in line and "==" not in line and "~=" not in line:
            unbounded.append(line)
    assert not unbounded, f"requirements without an upper bound: {unbounded}"
