"""Configuration loading entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

import yaml

from config.models import Config

DEFAULT_CONFIG_PATH = Path("config/config.yaml")
CONFIG_ENV_VAR = "CONFIG"
# ``${VAR}`` raises when unset; ``${VAR:-default}`` uses default when unset or empty (as in shell).
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def resolve_config_path(path: str | Path | None = None) -> Path:
    """Which config this run actually reads: explicit path > ``CONFIG`` env var > default.

    Exposed separately so the startup log can report the path: it is decided before
    ``configure_logging``, when no logging stack is available yet.
    """
    return Path(path or os.getenv(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH))


def load_config(path: str | Path | None = None) -> Config:
    config_path = resolve_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return Config.model_validate(_resolve_env_placeholders(data, config_path=config_path))


def _resolve_env_placeholders(value: Any, *, config_path: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _resolve_env_placeholders(item, config_path=config_path)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _resolve_env_placeholders(item, config_path=config_path)
            for item in value
        ]

    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(
            lambda match: _read_env_var(
                match.group(1), default=match.group(2), config_path=config_path
            ),
            value,
        )

    return value


def _read_env_var(name: str, *, default: str | None, config_path: Path) -> str:
    value = os.getenv(name)
    if default is not None and not value:
        return default
    if value is None:
        raise ValueError(
            f"Missing environment variable '{name}' referenced by {config_path}"
        )
    return value
