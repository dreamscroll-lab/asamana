"""Command-line interface for Asamana.

Operational commands only: build a world, list worlds, serve the backend. Run
control, live observation and replay are the web app's (the React frontend over
the API/WebSocket).

Don't add a reset command here, even though it would be one ``reset_session`` call. A reset
deletes every snapshot after step 0, and the "is it running" guard checks this process's
``_sessions``, which is always empty in a fresh CLI process. The backend's in-memory session
would then rewrite the deleted steps. Resets go through ``POST /api/worlds/{id}/reset``, which
runs in the same process as the run and holds the lifecycle lock.
"""

from __future__ import annotations

import sys
import asyncio

from config.models import Config
from core.container import Container
from core.logging import get_logger
from core.model_keys import ModelKeysMissing
from engine.application import NarrativeApplication
from world import DEFAULT_CATALOG_PATH, WorldCatalog
from interaction.models import WorldMeta
from interaction.world_manager import WorldManager

logger = get_logger(__name__)


class NarrativeCLI:
    """Build and list worlds from the command line."""

    def __init__(self, container: Container, config: Config) -> None:
        # One shared catalog: the application writes entries at build time, the
        # manager reads them to list worlds — same instance keeps them in sync.
        catalog = WorldCatalog(DEFAULT_CATALOG_PATH)
        self._application = NarrativeApplication(container, config, catalog=catalog)
        self._manager = WorldManager(
            snapshot_provider=container.snapshot,
            catalog=catalog,
        )

    # ------------------------------------------------------------------
    # Public command handlers
    # ------------------------------------------------------------------

    async def cmd_list(self) -> None:
        """List known worlds from WorldManager."""

        worlds = await self._manager.list_worlds()
        if not worlds:
            print("No worlds found.")
            return
        print(f"{'ID':<38}  {'Name':<30}  {'Theme':<25}  {'Step':>5}  {'Status'}")
        print("-" * 110)
        for meta in worlds:
            print(
                f"{meta.world_id:<38}  {meta.world_name:<30}  {meta.theme:<25}  "
                f"{meta.current_step:>5}  {meta.status.value}"
            )
            if meta.main_agent_names:
                print(f"  {'':38}  main characters: {', '.join(meta.main_agent_names)}")

    async def cmd_build(self, theme: str) -> None:
        """Build a world from *theme* and print a summary."""

        print(f"Building world with theme: {theme!r}")
        print("(This calls the LLM and may take a few minutes.)\n")

        meta = await self._build_and_register(theme)
        _print_world_meta(meta)
        # New worlds are unconfirmed, and only the web app can confirm one: confirming means a
        # person reviews the cast and relations, so the CLI deliberately has no command for it.
        # Without this line the world just sits in the list unable to run.
        print("\nConfirm it in the web app before starting the narrative.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _build_and_register(self, theme: str) -> WorldMeta:
        """Build a world (which registers it in the shared catalog) and read its meta."""

        # Same as the web path: offer every installed map and let the build read
        # the theme to pick one. Enumerating them is this layer's job.
        from worlds.tiled import list_templates

        world = await self._application.build_world(
            theme, available_templates=list_templates()
        )
        meta = await self._manager.get_world(world.world_id)
        logger.info(
            "world_built",
            extra={
                "world_id": world.world_id,
                "world_name": world.analysis.world_name,
                "agent_count": len(world.agents),
            },
        )
        return meta


# ---------------------------------------------------------------------------
# Serving the backend
# ---------------------------------------------------------------------------

async def serve_web(
    container: Container,
    config: Config,
    *,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Start the backend (API + WebSocket + engine).

    Host/port default to ``config.web``; the CLI flags override them.

    Bypasses ``NarrativeCLI`` because this starts a process rather than operating on a world.
    ``create_app`` builds its own application / manager / catalog; passing the CLI's in would
    put two catalog instances over the same file.
    """
    import uvicorn
    from interaction.api import create_app

    host = host or config.web.host
    port = port or config.web.port
    # Logs which provider implementations are wired up (storage, retrieval, etc.);
    # config_loaded only covers endpoints and models.
    logger.info(
        "startup_completed",
        extra={"config_world": config.world.config, **container.describe()},
    )
    print(f"Starting Asamana backend at http://{host}:{port}")
    print("Press Ctrl+C to stop.\n")
    app = create_app(container, config)
    cfg = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    await server.serve()


# ---------------------------------------------------------------------------
# Argument parsing and dispatch
# ---------------------------------------------------------------------------

async def run_cli_async(container: Container, config: Config, args: list[str]) -> int:
    """Parse *args* and dispatch. Returns 0 on success, 1 on any error.

    This is the only place commands are parsed (see ``main.main``).
    """

    if not args:
        _print_usage()
        return 1

    command = args[0]
    rest = args[1:]

    try:
        if command == "web":
            await serve_web(
                container,
                config,
                host=_parse_flag_str(rest, "--host", default=None),
                port=_parse_flag_int(rest, "--port", default=None),
            )
            return 0

        cli = NarrativeCLI(container, config)

        if command == "list":
            await cli.cmd_list()

        elif command == "build":
            if not rest:
                print("Usage: build <theme>", file=sys.stderr)
                return 1
            if not container.model_keys:
                raise ModelKeysMissing()
            await cli.cmd_build(rest[0])

        else:
            print(f"Unknown command: {command!r}", file=sys.stderr)
            _print_usage()
            return 1

    except Exception as exc:  # noqa: BLE001
        logger.error("cli_command_failed", extra={"command": command, "error": str(exc)}, exc_info=True)
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


def run_cli(container: Container, config: Config, args: list[str]) -> int:
    """Synchronous wrapper for the async CLI."""

    return asyncio.run(run_cli_async(container, config, args))


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _print_world_meta(meta: WorldMeta) -> None:
    """Render WorldMeta as human-readable text."""

    print(f"World: {meta.world_name}")
    print(f"  ID:     {meta.world_id}")
    print(f"  Theme:  {meta.theme}")
    print(f"  Status: {meta.status.value}")
    print(f"  Step:   {meta.current_step}")
    if meta.main_agent_names:
        print(f"  Main characters: {', '.join(meta.main_agent_names)}")
    if meta.created_at:
        print(f"  Created: {meta.created_at.strftime('%Y-%m-%d %H:%M')}")


def _print_usage() -> None:
    print(
        "Usage: asamana <command> [args]\n"
        "\n"
        "Commands:\n"
        "  list                             List known worlds\n"
        "  build <theme>                    Build a world from <theme>\n"
        "  web [--host HOST] [--port PORT]  Start the backend (default: config.web)\n"
        "\n"
        "Confirming a world, run control, live observation, replay and reset\n"
        "live in the web app.\n"
    )


# ---------------------------------------------------------------------------
# Argument parsing utilities
# ---------------------------------------------------------------------------

def _parse_flag_int(
    args: list[str],
    flag: str,
    *,
    default: int | None,
) -> int | None:
    """Return the integer value following *flag* in *args*, or *default*.

    Raises on a non-integer value (CLAUDE.md Rule 3): silently falling back to the config port
    would bind the server somewhere nobody asked for.
    """

    for index, token in enumerate(args):
        if token == flag and index + 1 < len(args):
            value = args[index + 1]
            try:
                return int(value)
            except ValueError:
                raise ValueError(f"{flag} expects an integer, got {value!r}") from None
    return default


def _parse_flag_str(
    args: list[str],
    flag: str,
    *,
    default: str | None,
) -> str | None:
    """Return the string value following *flag* in *args*, or *default*."""

    for index, token in enumerate(args):
        if token == flag and index + 1 < len(args):
            return args[index + 1]
    return default
