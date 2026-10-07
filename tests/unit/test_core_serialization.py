"""Serialization: persisted JSON must stay CJK-readable."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from core.serialization import atomic_write_text, dump_json


def test_dump_json_keeps_cjk_readable() -> None:
    out = dump_json({"world_name": "长安"})
    assert "长安" in out
    assert "\\u" not in out  # no \uXXXX escapes


def test_dump_json_round_trips() -> None:
    payload = {"name": "李建成", "nested": {"地点": "东宫"}, "items": ["密议", "伏杀"]}
    assert json.loads(dump_json(payload)) == payload


def test_dump_json_coerces_non_native() -> None:
    out = dump_json({"ts": datetime(2026, 6, 22, 1, 0, 0)})
    # datetime coerced via str(), not raised
    assert "2026-06-22" in out


# --- atomic_write_text: two writers to the same target must not share a tmp ---------------
#
# Every caller invokes it inside `asyncio.to_thread` (see the file providers), so two threads in
# one process can write the same target at once. With a pid-only tmp name they collide: A writes
# half, B truncates and rewrites, A replaces first and moves it away, and B's replace then raises
# FileNotFoundError on a tmp that no longer exists.


def test_two_writers_of_one_target_get_their_own_tmp(tmp_path, monkeypatch) -> None:
    """The tmp name must differ per writer, not per process."""
    seen: list[str] = []
    real_write_text = Path.write_text

    def spy(self, *args, **kwargs):          # type: ignore[no-untyped-def]
        if self.name.endswith(".tmp"):
            seen.append(self.name)
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", spy)
    target = tmp_path / "state.json"
    atomic_write_text(target, '{"a": 1}')
    atomic_write_text(target, '{"a": 2}')

    assert len(seen) == 2
    assert seen[0] != seen[1], f"两次写用了同一个 tmp 名: {seen[0]}"
    assert target.read_text(encoding="utf-8") == '{"a": 2}'


def test_the_target_is_written_whole(tmp_path) -> None:
    """The whole point of tmp + rename: the target is either the old or the new content, never a
    truncated in-between."""
    target = tmp_path / "state.json"
    atomic_write_text(target, "first")
    atomic_write_text(target, "second")
    assert target.read_text(encoding="utf-8") == "second"
    assert not list(tmp_path.glob("*.tmp"))   # no leftovers
