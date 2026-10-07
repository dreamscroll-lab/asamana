"""Guard: the `.get(x, x)` id-fallback antipattern must not appear on narrative paths.

Names resolved for narrative text fall back to a descriptive referent ("某人"/"某地"), never the raw
id; `mapping.get(aid, aid)` silently puts an id into a prompt or memory prose (see the CLAUDE.md
narrative/code-layer boundary and the directory contract).
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

# Narrative-producing paths: text built here flows into prompts / memory / ambient.
_SCANNED = [
    "agent/decision.py",
    "agent/perception_layer.py",
    "agent/agent.py",
    "agent/perception_emotion.py",
    "agent/memory.py",
    "engine/world_pressure.py",
]

# `.get(x, x)` — same identifier as key and default ⇒ id-as-name fallback. This is
# the precise, low-false-positive shape. Don't broaden it to an `or *_id` scan:
# that also flags legitimate code-layer id *routing* (target-id resolution, id
# fields), which is not a narrative leak.
_GET_SELF_DEFAULT = re.compile(r"\.get\(\s*([A-Za-z_]\w*)\s*,\s*\1\s*\)")


def test_no_get_self_default_id_fallback() -> None:
    offenders: list[str] = []
    for rel in _SCANNED:
        text = (_REPO / rel).read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if _GET_SELF_DEFAULT.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "id-as-name fallback `.get(x, x)` reintroduced:\n" + "\n".join(offenders)
