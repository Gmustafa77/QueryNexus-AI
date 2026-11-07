"""Exponential-backoff retry for transient provider failures (rate limits, timeouts, 5xx)."""
from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

_NON_RETRYABLE_MARKERS = (
    "invalid api key",
    "incorrect api key",
    "authentication",
    "unauthorized",
    "permission denied",
    "invalid_request_error",
    "content_filter",
    "context_length_exceeded",
    "not found",
)


def is_retryable(exc: BaseException) -> bool:
    """Retry network/rate-limit/server errors; never retry auth or bad-request errors."""
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return False
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status in (408, 409, 429) or status >= 500
    message = str(exc).lower()
    return not any(marker in message for marker in _NON_RETRYABLE_MARKERS)


def retry_call(
    func: Callable[[], T],
    *,
    operation: str,
    max_retries: int,
    base_delay: float,
    max_delay: float = 20.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``func`` with up to ``max_retries`` retries using jittered exponential backoff."""
    attempt = 0
    while True:
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - re-raised below when not retryable
            if attempt >= max_retries or not is_retryable(exc):
                logger.error("%s failed after %d attempt(s): %s", operation, attempt + 1, type(exc).__name__)
                raise
            delay = min(max_delay, base_delay * (2**attempt)) * (0.5 + random.random() / 2)
            attempt += 1
            logger.warning(
                "%s failed (%s); retry %d/%d in %.2fs", operation, type(exc).__name__, attempt, max_retries, delay
            )
            sleep(delay)
