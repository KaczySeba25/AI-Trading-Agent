"""Shared test fixtures and helpers."""

from __future__ import annotations

import math

import pytest

from agent_system.core.config import Settings
from agent_system.data.data_buffer import MarketTick


def make_ticks(
    count: int,
    start_price: float = 100.0,
    volatility: float = 0.004,
    seed: int = 11,
    trend: float = 0.0,
) -> list[MarketTick]:
    """Deterministic pseudo-random tick series.

    Uses a plain LCG rather than numpy so the sequence is identical on every
    platform and test failures are reproducible.
    """
    ticks: list[MarketTick] = []
    price = start_price
    state = seed
    timestamp = 1_704_067_200_000

    for index in range(count):
        state = (state * 1_103_515_245 + 12_345) % (2**31)
        noise = (state / (2**31)) - 0.5
        price *= math.exp(trend + volatility * noise * 2.0)
        spread = price * 0.0001
        ticks.append(
            MarketTick(
                timestamp=timestamp + index * 60_000,
                price=round(price, 2),
                volume=round(5.0 + abs(noise) * 20.0, 4),
                bid=round(price - spread, 4),
                ask=round(price + spread, 4),
            )
        )
    return ticks


@pytest.fixture
def ticks() -> list[MarketTick]:
    return make_ticks(600)


@pytest.fixture
def config(tmp_path) -> Settings:
    """Settings pointed at a temp directory so tests never touch real state."""
    return Settings(
        state_dir=tmp_path / "state",
        model_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        log_dir=tmp_path / "logs",
        initial_capital=500.0,
        observation_window=16,
    )
