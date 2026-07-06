"""Public historical market data client."""

from __future__ import annotations

import time
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
