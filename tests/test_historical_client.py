from agent_system.data.historical_client import interval_to_milliseconds, klines_to_ticks


def test_klines_to_ticks_uses_close_price() -> None:
    ticks = klines_to_ticks(
        [
            [
                1000,
                "10.0",
                "11.0",
                "9.0",
                "10.5",
                "123.4",
                0,
                "0",
                0,
                "0",
                "0",
                "0",
            ]
        ]
    )

    assert len(ticks) == 1
    assert ticks[0].timestamp == 1000
    assert ticks[0].price == 10.5
    assert ticks[0].volume == 123.4


def test_interval_to_milliseconds() -> None:
    assert interval_to_milliseconds("1m") == 60_000
    assert interval_to_milliseconds("5m") == 300_000
    assert interval_to_milliseconds("1h") == 3_600_000
