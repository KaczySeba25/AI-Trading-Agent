from agent_system.data.historical_client import klines_to_ticks


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
