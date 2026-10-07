from __future__ import annotations

import io
import json
import logging
from pathlib import Path

from core.context import clear_log_context, set_log_context
from core.logging import configure_logging


def teardown_function() -> None:
    clear_log_context()
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)


def test_json_logging_injects_context() -> None:
    configure_logging(fmt="json")
    json_formatter = logging.getLogger().handlers[0].formatter

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(json_formatter)

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    set_log_context(world_id="world-1", agent_id="agent-7", step="12")
    logging.getLogger("asamana.test").info("decision completed", extra={"provider": "mock"})

    payload = json.loads(stream.getvalue())
    assert payload["message"] == "decision completed"
    assert payload["world_id"] == "world-1"
    assert payload["agent_id"] == "agent-7"
    assert payload["step"] == "12"
    assert payload["provider"] == "mock"


def test_file_logging_requires_path() -> None:
    try:
        configure_logging(output="file")
    except ValueError as exc:
        assert "file_path" in str(exc)
    else:
        raise AssertionError("configure_logging should require file_path for file output")


def test_file_logging_writes_rotating_file(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "asamana.log"
    configure_logging(output="file", file_path=str(log_path), fmt="console")

    logging.getLogger("asamana.test").warning("world advanced")

    assert log_path.exists()
    assert "world advanced" in log_path.read_text(encoding="utf-8")


def test_noisy_loggers_are_suppressed() -> None:
    configure_logging()

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
    assert logging.getLogger("urllib3").level == logging.WARNING


def test_configure_logging_rejects_an_unknown_level() -> None:
    import pytest

    with pytest.raises(ValueError):
        configure_logging(level="VERBOSE")
