"""Public Binance historical klines client.

Only public market-data endpoints are used, so no API credentials are required.
Candles are converted to :class:`MarketTick` so history flows through exactly the
same feature pipeline as the live websocket stream.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Sequence

import requests

from agent_system.core.config import Settings, settings
from agent_system.core.exceptions import MarketDataError
from agent_system.core.logger import get_logger
from agent_system.data.data_buffer import MarketTick

logger = get_logger(__name__)

_INTERVAL_UNITS_MS = {
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
}

#: Hard stop on pagination so a bad cursor can never loop forever.
MAX_PAGES = 500


def interval_to_milliseconds(interval: str) -> int:
    """Convert a Binance interval string such as ``1m`` or ``4h`` to ms."""
    if not interval or len(interval) < 2:
        raise MarketDataError(f"Invalid interval: {interval!r}")
    unit = interval[-1].lower()
    if unit not in _INTERVAL_UNITS_MS:
        raise MarketDataError(f"Unsupported interval unit: {interval!r}")
    try:
        amount = int(interval[:-1])
    except ValueError as exc:
        raise MarketDataError(f"Invalid interval: {interval!r}") from exc
    if amount <= 0:
        raise MarketDataError(f"Interval must be positive: {interval!r}")
    return amount * _INTERVAL_UNITS_MS[unit]


def klines_to_ticks(klines: Sequence[Sequence[Any]]) -> list[MarketTick]:
    """Convert raw klines to ticks using the close price.

    The synthetic bid/ask is derived from the candle's own high/low spread, which
    keeps a realistic (non-zero) spread feature instead of pretending execution
    is free.
    """
    ticks: list[MarketTick] = []
    for kline in klines:
        try:
            open_time = int(kline[0])
            high = float(kline[2])
            low = float(kline[3])
            close = float(kline[4])
            volume = float(kline[5])
        except (IndexError, TypeError, ValueError):
            logger.debug("Skipping malformed kline: %s", kline)
            continue

        half_spread = max((high - low) * 0.005, close * 1e-5)
        ticks.append(
            MarketTick(
                timestamp=open_time,
                price=close,
                volume=volume,
                bid=close - half_spread,
                ask=close + half_spread,
            )
        )
    return ticks


class HistoricalClient:
    """Downloads public klines with pagination, retries and a page cap."""

    def __init__(self, config: Settings = settings, session: Any | None = None) -> None:
        self.config = config
        self.session = session or requests.Session()

    def fetch_klines(
        self,
        symbol: str | None = None,
        interval: str = "1m",
        limit: int = 1000,
        days: int | None = None,
        max_rows: int | None = None,
    ) -> list[MarketTick]:
        """Fetch recent klines and return them as ticks (oldest -> newest)."""
        symbol = (symbol or self.config.symbol).upper()
        url = f"{self.config.public_market_data_base_url}/api/v3/klines"
        interval_ms = interval_to_milliseconds(interval)

        end_time = int(time.time() * 1000)
        if days is not None:
            start_time = end_time - days * 86_400_000
        else:
            start_time = end_time - limit * interval_ms

        collected: list[list[Any]] = []
        cursor = start_time

        for page in range(MAX_PAGES):
            payload = self._request(
                url,
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_time,
                    "limit": 1000,
                },
            )
            if not payload:
                break

            collected.extend(payload)
            last_open_time = int(payload[-1][0])

            # Guard against a stalled cursor returning the same page forever.
            next_cursor = last_open_time + interval_ms
            if next_cursor <= cursor:
                logger.warning("Kline cursor stalled at %s, stopping pagination", cursor)
                break
            cursor = next_cursor

            if cursor >= end_time or len(payload) < 1000:
                break
            if max_rows is not None and len(collected) >= max_rows:
                break
            if page == MAX_PAGES - 1:
                logger.warning("Reached MAX_PAGES=%d for %s", MAX_PAGES, symbol)

        if max_rows is not None:
            collected = collected[-max_rows:]

        ticks = klines_to_ticks(collected)
        logger.info("Fetched %d %s %s candles", len(ticks), symbol, interval)
        return ticks

    def _request(self, url: str, params: dict[str, Any]) -> list[list[Any]]:
        """GET with bounded exponential backoff on transient failures."""
        delay = 1.0
        last_error: Exception | None = None

        for attempt in range(4):
            try:
                response = self.session.get(url, params=params, timeout=15)
                if response.status_code == 429:
                    logger.warning("Rate limited by Binance, backing off %.1fs", delay)
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise MarketDataError(f"Unexpected klines payload: {type(payload)}")
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                logger.warning("Kline request failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

        raise MarketDataError(f"Failed to fetch klines after retries: {last_error}")


def load_ticks_from_csv(path: str | Path) -> list[MarketTick]:
    """Load OHLCV candles from CSV -- the offline path when the API is blocked.

    Expected header: ``timestamp,open,high,low,close,volume``.
    """
    path = Path(path)
    if not path.exists():
        raise MarketDataError(f"CSV not found: {path}")

    rows: list[list[Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for index, record in enumerate(csv.DictReader(handle)):
            try:
                rows.append(
                    [
                        index * 60_000,
                        record["open"],
                        record["high"],
                        record["low"],
                        record["close"],
                        record["volume"],
                    ]
                )
            except KeyError as exc:
                raise MarketDataError(f"CSV missing column: {exc}") from exc

    return klines_to_ticks(rows)
