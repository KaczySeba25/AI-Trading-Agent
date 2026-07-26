"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_market_ticks():
    """Provide sample market ticks for tests."""
    import time
    from agent_system.data.data_buffer import MarketTick

    now = int(time.time() * 1000)
    ticks = []
    for i in range(100):
        price = 100.0 + i * 0.5
        ticks.append(
            MarketTick(
                timestamp=now + i * 1000,
                price=price,
                volume=1.0 + (i % 5) * 0.1,
                bid=price - 0.01,
                ask=price + 0.01,
            )
        )
    return ticks
