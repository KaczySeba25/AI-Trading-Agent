"""Retry-capable REST helper placeholder for later Binance API phases."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar


T = TypeVar("T")


async def retry_with_backoff(
    operation: Callable[[], Awaitable[T]],
    attempts: int = 3,
    initial_delay_seconds: float = 0.5,
    max_delay_seconds: float = 5.0,
) -> T:
    delay = initial_delay_seconds
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return await operation()
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay_seconds)
    if last_error is None:
        raise RuntimeError("Retry operation failed without an exception")
    raise last_error
