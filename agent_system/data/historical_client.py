"""Historical klines download and CSV loading.

The public ``/api/v3/klines`` endpoint needs no API key, which matters because
the agent must be able to train from history before any credentials exist.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests

from agent_system.core.config import Settings, settings
from agent_system.core.exceptions import MarketDataError
from agent_system.core.logger import get_logger
from agent_system.data.data_buffer import MarketTick

logger = get_logger(__name__)

#: Binance caps a single klines response at 1000 rows.
KLINES_PAGE_LIMIT = 1000

#: Hard ceiling on pagination. Without it a server that keeps returning the
#: same timestamp turns the download loop into an infinite one.
MAX_PAGES = 500

_INTERVAL_UNITS = {
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
}


def interval_to_milliseconds(interval: str) -> int:
    """``"5m"`` -> ``300000``. Raises ``MarketDataError`` on garbage input."""
    if not interval or len(interval) < 2:
        raise MarketDataError(f"Invalid interval: {interval!r}")
    unit = interval[-1].lower()
    if unit not in _INTERVAL_UNITS:
        raise MarketDataError(f"Unsupported interval unit: {interval!r}")
    try:
        amount = int(interval[:-1])
    except ValueError as exc:
        raise MarketDataError(f"Invalid interval: {interval!r}") from exc
    if amount <= 0:
        raise MarketDataError(f"Interval must be positive: {interval!r}")
    return amount * _INTERVAL_UNITS[unit]


def klines_to_ticks(klines: Iterable[Sequence[Any]]) -> list[MarketTick]:
    """Convert raw Binance klines into ticks.

    A kline is a bar, not a tick, so bid/ask are synthesised from the close.
    Downstream code only uses the spread as a cost hint, and on 1m bars the
    real spread is far smaller than the bar range anyway.
    """
    ticks: list[MarketTick] = []
    for row in klines:
        try:
            timestamp = int(row[0])
            close = float(row[4])
            volume = float(row[5])
        except (IndexError, TypeError, ValueError) as exc:
            raise MarketDataError(f"Malformed kline row: {row!r}") from exc

        half_spread = close * 0.00005
        ticks.append(
            MarketTick(
                timestamp=timestamp,
                price=close,
                volume=volume,
                bid=close - half_spread,
                ask=close + half_spread,
            )
        )
    return ticks


class HistoricalClient:
    """Downloads klines from the public Binance REST API."""

    def __init__(self, config: Settings = settings, session: Any | None = None) -> None:
        self.config = config
        self.session = session or requests.Session()

    def fetch_klines(
        self,
        interval: str = "1m",
        days: int = 7,
        max_rows: int = 10_000,
        symbol: str | None = None,
    ) -> list[MarketTick]:
        symbol = (symbol or self.config.symbol).upper()
        step_ms = interval_to_milliseconds(interval)
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 86_400_000

        url = f"{self.config.public_market_data_base_url}/api/v3/klines"
        collected: list[Sequence[Any]] = []
        cursor = start_ms
        pages = 0

        while cursor < end_ms and len(collected) < max_rows and pages < MAX_PAGES:
            pages += 1
            limit = min(KLINES_PAGE_LIMIT, max_rows - len(collected))
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": limit,
            }
            try:
                response = self.session.get(url, params=params, timeout=15)
                response.raise_for_status()
                page = response.json()
            except requests.RequestException as exc:
                raise MarketDataError(f"Failed to download klines: {exc}") from exc

            if not page:
                break

            collected.extend(page)
            last_open = int(page[-1][0])
            next_cursor = last_open + step_ms
            # Guard against a server that never advances: without this the
            # loop would spin forever on the same page.
            if next_cursor <= cursor:
                break
            cursor = next_cursor

            if len(page) < limit:
                break

        if pages >= MAX_PAGES:
            logger.warning("Stopped kline download after %d pages", MAX_PAGES)

        logger.info("Downloaded %d klines for %s %s", len(collected), symbol, interval)
        return klines_to_ticks(collected[:max_rows])


def load_ticks_from_csv(path: str | Path, max_rows: int | None = None) -> list[MarketTick]:
    """Load ticks from an OHLCV CSV file.

    Accepts either a header row (``timestamp,open,high,low,close,volume``) or a
    bare numeric file. Rows that cannot be parsed are skipped rather than
    aborting the load, since a single corrupt line should not cost a whole
    training run.
    """
    path = Path(path)
    if not path.exists():
        raise MarketDataError(f"CSV not found: {path}")

    rows: list[Sequence[Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            if not row:
                continue
            if index == 0 and not row[0].strip().lstrip("-").isdigit():
                continue  # header
            if len(row) < 6:
                continue
            try:
                rows.append(
                    [int(float(row[0])), row[1], row[2], row[3], row[4], row[5]]
                )
            except (TypeError, ValueError):
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break

    if not rows:
        raise MarketDataError(f"No usable rows in {path}")

    logger.info("Loaded %d rows from %s", len(rows), path)
    return klines_to_ticks(rows)
