"""Asamana application entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys

from config import Config, load_config, resolve_config_path
from core.container import Container
from core.logging import configure_logging, get_logger
from interaction.cli import run_cli
from interaction.examples import seed_examples
from world import DEFAULT_CATALOG_PATH

logger = get_logger(__name__)


def bootstrap_application(config_path: str | Path | None = None) -> tuple[Config, Container]:
    """Load config, initialize logging, and build the shared container."""

    resolved_path = resolve_config_path(config_path)
    config = load_config(resolved_path)
    configure_logging(**config.logging.model_dump())
    # Log which endpoints this run uses as soon as logging is up. Otherwise the only way to tell
    # whether a config edit took effect is the model name in traces, and two endpoints can serve
    # identically named models.
    logger.info(
        "config_loaded",
        extra={
            "config_path": str(resolved_path),
            "llm_default_provider": config.llm.default_provider,
            "llm_providers": {
                name: {"model": declared.model, "base_url": declared.base_url}
                for name, declared in config.llm.providers.items()
            },
            "embedding_provider": config.embedding.provider,
        },
    )
    # Before the container: the file vector store reads its directory once, at construction.
    seed_examples(config, DEFAULT_CATALOG_PATH)
    return config, Container.from_config(config)


def main() -> int:
    """Boot the Asamana application and hand the command line to the CLI.

    Don't parse commands here. A command allowlist in the entry point would send unknown words
    down its own path and return 0, so a typo would look like a successful run. Only
    ``run_cli`` parses commands; it prints usage and exits non-zero on anything unknown.
    """

    config, container = bootstrap_application()
    return run_cli(container, config, sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
