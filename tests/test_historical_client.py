"""Historical data client tests."""

from __future__ import annotations

import pytest

from agent_system.core.exceptions import MarketDataError
from agent_system.data.historical_client import (
    interval_to_milliseconds,
    klines_to_ticks,
    load_ticks_from_csv,
)


def test_klines_to_ticks_uses_close_price() -> None:
    ticks = klines_to_ticks(
        [[1000, "10.0", "11.0", "9.0", "10.5", "123.4", 0, "0", 0, "0", "0", "0"]]
    )

    assert len(ticks) == 1
    assert ticks[0].timestamp == 1000
    assert ticks[0].price == 10.5
    assert ticks[0].volume == 123.4
    # A synthetic but non-zero spread keeps execution costs honest.
    assert ticks[0].bid < ticks[0].price < ticks[0].ask


def test_klines_to_ticks_skips_malformed_rows() -> None:
    ticks = klines_to_ticks(
        [
            [1000, "10", "11", "9", "10.5", "1"],
            ["bad"],
            [2000, "10", "11", "9", "10.6", "1"],
        ]
    )

    assert len(ticks) == 2


def test_interval_to_milliseconds() -> None:
    assert interval_to_milliseconds("1m") == 60_000
    assert interval_to_milliseconds("5m") == 300_000
    assert interval_to_milliseconds("1h") == 3_600_000
    assert interval_to_milliseconds("1d") == 86_400_000


@pytest.mark.parametrize("value", ["", "m", "0m", "5x", "abc"])
def test_invalid_intervals_are_rejected(value: str) -> None:
    with pytest.raises(MarketDataError):
        interval_to_milliseconds(value)


def test_load_ticks_from_csv(tmp_path) -> None:
    path = tmp_path / "prices.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01 00:00,42000,42100,41900,42050,10\n"
        "2024-01-01 00:01,42050,42200,42000,42150,12\n",
        encoding="utf-8",
    )

    ticks = load_ticks_from_csv(path)

    assert len(ticks) == 2
    assert ticks[0].price == 42050.0
    assert ticks[1].price == 42150.0


def test_missing_csv_raises() -> None:
    with pytest.raises(MarketDataError):
        load_ticks_from_csv("does_not_exist.csv")
