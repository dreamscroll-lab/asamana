"""A failed step-0 snapshot write surfaces as-is and is not retried by the initializer.

Retries belong inside providers (CLAUDE.md Provider Rules); a world build that cannot persist its
starting state fails outright.
"""

from __future__ import annotations

import pytest

from engine.application import NarrativeApplication


@pytest.mark.asyncio
async def test_initial_snapshot_write_failure_raises_without_retry(
    mock_build_container, test_config, monkeypatch
) -> None:
    calls: list[int] = []

    async def _failing_save(world_id: str, step: int, snapshot: object) -> None:
        calls.append(step)
        raise OSError("disk full")

    monkeypatch.setattr(mock_build_container.snapshot, "save", _failing_save)
    app = NarrativeApplication(mock_build_container, test_config)

    with pytest.raises(OSError, match="disk full"):
        await app.build_world("宫廷权谋", template="changan_iso")
    assert calls == [0]
