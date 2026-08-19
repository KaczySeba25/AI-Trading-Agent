"""Shared fixtures.

Every test runs against a temporary state directory so nothing touches the real
paper account, models or journal.
"""

from __future__ import annotations

import math
import random

import pytest

from agent_system.core.config import Settings
from agent_system.data.data_buffer import MarketTick


def make_ticks(
    count: int = 500,
    start_price: float = 40_000.0,
    trend: float = 0.0,
    volatility: float = 0.002,
    seed: int = 7,
) -> list[MarketTick]:
    """Generate a deterministic synthetic price series."""
    rng = random.Random(seed)
    ticks: list[MarketTick] = []
    price = start_price

    for index in range(count):
        drift = trend + rng.gauss(0.0, volatility)
        price = max(price * (1.0 + drift), 1.0)
        half_spread = price * 0.0001
        ticks.append(
            MarketTick(
                timestamp=1_700_000_000_000 + index * 60_000,
                price=price,
                volume=10.0 + rng.random() * 5.0,
                bid=price - half_spread,
                ask=price + half_spread,
            )
        )
    return ticks


def make_wave_ticks(count: int = 500, start_price: float = 40_000.0) -> list[MarketTick]:
    """Oscillating series -- gives a mean-reversion policy something to trade."""
    ticks: list[MarketTick] = []
    for index in range(count):
        price = start_price * (1.0 + 0.02 * math.sin(index / 12.0))
        half_spread = price * 0.0001
        ticks.append(
            MarketTick(
                timestamp=1_700_000_000_000 + index * 60_000,
                price=price,
                volume=10.0,
                bid=price - half_spread,
                ask=price + half_spread,
            )
        )
    return ticks


@pytest.fixture
def ticks() -> list[MarketTick]:
    return make_ticks()


@pytest.fixture
def wave_ticks() -> list[MarketTick]:
    return make_wave_ticks()


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    """Settings isolated to a temp dir, with no credentials."""
    return Settings(
        symbol="BTCUSDT",
        initial_capital=1000.0,
        state_dir=tmp_path / "state",
        model_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        log_dir=tmp_path / "logs",
        observation_window=16,
        binance_api_key=None,
        binance_api_secret=None,
        enable_testnet_trading=False,
        enable_live_trading=False,
    )
