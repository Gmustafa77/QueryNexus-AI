"""Consistent, structured logging for every module.

``get_logger(__name__)`` keeps its original signature. Output format is
controlled by LOG_FORMAT (text|json) and LOG_LEVEL. A per-request
correlation id can be bound with ``request_context`` so every log line from
one question/ingestion run can be traced together.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Iterator

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "request_id"}


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line; includes any ``extra={...}`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _build_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    if os.getenv("LOG_FORMAT", "text").strip().lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    handler.addFilter(_RequestIdFilter())
    return handler


def get_logger(name: str) -> logging.Logger:
    """Create (or reuse) a configured logger. Safe under Streamlit re-runs."""
    logger = logging.getLogger(name)
    if not getattr(logger, "_rag_configured", False):
        logger.handlers.clear()
        logger.addHandler(_build_handler())
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO")
        logger.propagate = False
        logger._rag_configured = True  # type: ignore[attr-defined]
    return logger


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def current_request_id() -> str:
    return _request_id.get()


@contextlib.contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """Bind a correlation id to every log line emitted inside the block."""
    token = _request_id.set(request_id or new_request_id())
    try:
        yield _request_id.get()
    finally:
        _request_id.reset(token)
