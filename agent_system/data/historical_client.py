"""Public historical market data client."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from agent_system.core.config import Settings, settings
from agent_system.data.data_buffer import MarketTick


class HistoricalMarketDataClient:
    def __init__(
        self,
        config: Settings = settings,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()

    def fetch_klines(
        self,
        symbol: str | None = None,
        interval: str = "1m",
        limit: int = 500,
    ) -> list[MarketTick]:
        selected_symbol = (symbol or self.config.symbol).upper()
        response = self.session.get(
            f"{self.config.public_market_data_base_url}/api/v3/klines",
            params={
                "symbol": selected_symbol,
                "interval": interval,
                "limit": min(limit, 1000),
            },
            timeout=15,
        )
        response.raise_for_status()
        return klines_to_ticks(response.json())

    def fetch_klines_range(
        self,
        symbol: str | None = None,
        interval: str = "1m",
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        max_rows: int | None = None,
    ) -> list[MarketTick]:
        selected_symbol = (symbol or self.config.symbol).upper()
        current_start = start_time_ms or int(
            (datetime.now(UTC) - timedelta(days=365)).timestamp() * 1000
        )
        final_end = end_time_ms or int(datetime.now(UTC).timestamp() * 1000)
        ticks: list[MarketTick] = []

        while current_start < final_end:
            params = {
                "symbol": selected_symbol,
                "interval": interval,
                "limit": 1000,
                "startTime": current_start,
                "endTime": final_end,
            }
            response = self.session.get(
                f"{self.config.public_market_data_base_url}/api/v3/klines",
                params=params,
                timeout=20,
            )
            response.raise_for_status()
            batch = klines_to_ticks(response.json())
            if not batch or (len(batch) == 1 and batch[0].price == 0.0):
                break
            ticks.extend(batch)
            if max_rows is not None and len(ticks) >= max_rows:
                return ticks[:max_rows]
            next_start = batch[-1].timestamp + interval_to_milliseconds(interval)
            if next_start <= current_start:
                break
            current_start = next_start
            time.sleep(0.05)

        return ticks

    def fetch_last_days(
        self,
        days: int = 365,
        symbol: str | None = None,
        interval: str = "1m",
        max_rows: int | None = None,
    ) -> list[MarketTick]:
        end = datetime.now(UTC)
        start = end - timedelta(days=days)
        return self.fetch_klines_range(
            symbol=symbol,
            interval=interval,
            start_time_ms=int(start.timestamp() * 1000),
            end_time_ms=int(end.timestamp() * 1000),
            max_rows=max_rows,
        )


def klines_to_ticks(payload: list[list[Any]]) -> list[MarketTick]:
    ticks: list[MarketTick] = []
    for item in payload:
        timestamp = int(item[0])
        close_price = float(item[4])
        volume = float(item[5])
        ticks.append(
            MarketTick(
                timestamp=timestamp,
                price=close_price,
                volume=volume,
                bid=close_price,
                ask=close_price,
            )
        )
    if not ticks:
        now = int(time.time() * 1000)
        return [MarketTick(now, 0.0, 0.0, 0.0, 0.0)]
    return ticks


def interval_to_milliseconds(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    multipliers = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }
    if unit not in multipliers:
        raise ValueError(f"Unsupported interval: {interval}")
    return value * multipliers[unit]
