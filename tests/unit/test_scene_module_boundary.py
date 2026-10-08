"""Scene assembly and outcome wording live outside the executor package.

Importing anything under ``engine.executors`` runs its ``__init__``, which loads every executor. A
module that only renders a scene or words an outcome must not load them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _executors_loaded_by(module: str) -> list[str]:
    code = (
        f"import sys, {module}; "
        "print(' '.join(sorted(m for m in sys.modules if m.startswith('engine.executors'))))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
    return out.stdout.split()


def test_scene_and_narration_do_not_load_the_executors() -> None:
    assert _executors_loaded_by("engine.scene") == []
    assert _executors_loaded_by("engine.narration") == []


def test_npc_runner_does_not_load_the_executors() -> None:
    assert _executors_loaded_by("engine.npc_runner") == []
