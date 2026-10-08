"""Centralized logging configuration for Asamana."""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path

from core.context import get_log_context

DEFAULT_FORMAT = "json"
DEFAULT_LEVEL = "INFO"
NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "openai",  # SDK DEBUG dumps full request/response bodies (whole prompts)
)


class ContextJsonFormatter(logging.Formatter):
    """Render log records as structured JSON with request-scoped context."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(get_log_context())

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key.startswith("_") or key in _RESERVED_RECORD_KEYS:
                continue
            payload[key] = value

        return json.dumps(payload, ensure_ascii=False)


class ContextConsoleFormatter(logging.Formatter):
    """Render human-readable log lines with compact context fields."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        context = get_log_context()
        context_bits = [f"{key}={value}" for key, value in context.items()]
        context_str = f" [{' '.join(context_bits)}]" if context_bits else ""
        message = f"{timestamp} {record.levelname} {record.name}{context_str} {record.getMessage()}"
        if record.exc_info:
            return f"{message}\n{self.formatException(record.exc_info)}"
        return message


def configure_logging(
    *,
    level: str = DEFAULT_LEVEL,
    fmt: str = DEFAULT_FORMAT,
    output: str = "stdout",
    file_path: str | None = None,
    max_size_mb: int = 20,
    backup_count: int = 5,
) -> logging.Logger:
    root_logger = logging.getLogger()
    # A name logging doesn't know raises: a typo must not quietly log at another level (Rule 3).
    root_logger.setLevel(level.upper())
    root_logger.handlers.clear()

    formatter = _build_formatter(fmt)

    if output in ("stdout", "stdout+file"):
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        root_logger.addHandler(stdout_handler)

    if output in ("file", "stdout+file"):
        if not file_path:
            raise ValueError("file_path is required when logging output includes file")
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=file_path,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if output not in ("stdout", "file", "stdout+file"):
        raise ValueError(f"Unsupported logging output: {output}")

    _suppress_noisy_loggers()
    return root_logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _build_formatter(fmt: str) -> logging.Formatter:
    if fmt == "json":
        return ContextJsonFormatter()
    if fmt == "console":
        return ContextConsoleFormatter()
    raise ValueError(f"Unsupported logging format: {fmt}")


def _suppress_noisy_loggers() -> None:
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


_RESERVED_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}

