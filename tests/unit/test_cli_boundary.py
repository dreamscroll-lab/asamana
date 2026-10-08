from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from interaction.cli import NarrativeCLI
from interaction.models import WorldMeta, WorldStatus


@dataclass
class _StubWorld:
    world_id: str
    theme: str

    @property
    def analysis(self):
        return type("Analysis", (), {"world_name": "Stub World"})()

    @property
    def agents(self):
        return {"agent-1": object()}


class _StubApplication:
    def __init__(self) -> None:
        self.built: list[str] = []

    async def build_world(
        self, theme: str, *, template: str | None = None, available_templates=()
    ) -> _StubWorld:
        self.built.append(theme)
        # The CLI offers every installed map and lets the build read the theme.
        self.offered = list(available_templates)
        return _StubWorld(world_id="world-1", theme=theme)


@pytest.mark.asyncio
async def test_cli_delegates_build_to_application(container, test_config) -> None:
    cli = NarrativeCLI(container, test_config)
    stub_application = _StubApplication()
    cli._application = stub_application

    async def get_world(world_id: str) -> WorldMeta:
        # build_world registers the world itself; the CLI reads it back via get_world.
        return WorldMeta(
            world_id=world_id,
            theme="theme",
            world_name="Stub World",
            status=WorldStatus.IN_PROGRESS,
            created_at=datetime.now(timezone.utc),
            current_step=0,
            main_agent_names=[],
        )

    cli._manager.get_world = get_world

    await cli.cmd_build("theme")

    assert stub_application.built == ["theme"]
    # …and hands the build every installed map to choose from.
    assert stub_application.offered, "CLI offered no maps to pick between"


def test_world_status_never_claims_the_world_is_live() -> None:
    """A snapshot can't tell "a process is driving it" — any such value is bound to contradict
    reality while the world is actually running.

    Alive/stopped is answered by ``run_state`` (RunController), given alongside WorldMeta by the
    API.
    """
    from engine.run_control import RunState
    from interaction.models import WorldStatus

    liveness = {RunState.IDLE, RunState.RUNNING, RunState.PAUSED, RunState.STOPPING, RunState.FAILED}
    assert {s.value for s in WorldStatus}.isdisjoint({r.value for r in liveness})


def test_world_meta_status_reports_progress_from_snapshots() -> None:
    from datetime import datetime as _dt

    from core.interfaces.snapshot import WorldSnapshot
    from interaction.models import build_world_meta

    def meta(step: int | None) -> WorldStatus:
        snaps = [] if step is None else [WorldSnapshot(world_id="w", step=step, timestamp=_dt.now())]
        return build_world_meta(world_id="w", snapshots=snaps, catalog_entry={}).status

    assert meta(None) is WorldStatus.UNKNOWN
    assert meta(3) is WorldStatus.IN_PROGRESS   # worlds run open-ended: no step count ends the story


def test_main_rejects_an_unknown_command(container, test_config, monkeypatch) -> None:
    """Words not on the list must reach the dispatcher and exit non-zero.

    If the entry point kept its own command whitelist, a misspelled command would fall onto the
    entry's own path and silently return 0 — a failed operation would look identical to a
    successful one.
    """
    import main as entry

    monkeypatch.setattr(entry.sys, "argv", ["main.py", "frobnicate"])
    monkeypatch.setattr(entry, "bootstrap_application", lambda *a, **k: (test_config, container))

    assert entry.main() == 1


def test_web_does_not_build_a_second_world_catalog(container, test_config, monkeypatch) -> None:
    """``web`` is a process launcher, not a world operation: it must not go through
    ``NarrativeCLI``.

    ``create_app`` builds its own application / manager / catalog; if the CLI brought another set
    along, the same file would have two catalog instances in one process.
    """
    import asyncio

    from interaction import cli as cli_module

    served: dict[str, object] = {}

    async def fake_serve(c, cfg, *, host=None, port=None) -> None:
        served["host"], served["port"] = host, port

    monkeypatch.setattr(cli_module, "serve_web", fake_serve)
    monkeypatch.setattr(
        cli_module, "NarrativeCLI", lambda *a, **k: pytest.fail("web built a NarrativeCLI")
    )

    code = asyncio.run(
        cli_module.run_cli_async(container, test_config, ["web", "--port", "9123"])
    )
    assert code == 0
    assert served["port"] == 9123


def test_a_non_integer_port_is_refused(container, test_config) -> None:
    """``--port abc`` must not silently fall back to the config port — the service would bind
    somewhere nobody asked for."""
    import asyncio

    from interaction.cli import run_cli_async

    assert asyncio.run(run_cli_async(container, test_config, ["web", "--port", "abc"])) == 1
