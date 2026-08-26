"""Tests for the trading environment."""

import time

from agent_system.data.data_buffer import MarketTick
from agent_system.environment.trading_env import TradingAction, TradingEnvironment


def synthetic_ticks(count: int = 100) -> list[MarketTick]:
    """Generate synthetic market ticks for testing."""
    now = int(time.time() * 1000)
    ticks: list[MarketTick] = []
    for i in range(count):
        price = 100.0 + (i * 0.5)  # Trending upward
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


def test_environment_initializes_with_ticks() -> None:
    ticks = synthetic_ticks()
    env = TradingEnvironment(ticks)
    state = env.reset()

    assert len(state) == 17  # FEATURE_COUNT
    assert env.balance == 500.0
    assert env.equity == 500.0


def test_environment_step_returns_valid_output() -> None:
    ticks = synthetic_ticks()
    env = TradingEnvironment(ticks)
    env.reset()

    state, reward, done, info = env.step(int(TradingAction.HOLD))

    assert isinstance(state, list)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert "equity" in info
    assert "balance" in info


def test_environment_long_position() -> None:
    ticks = synthetic_ticks()
    env = TradingEnvironment(ticks)
    env.reset()

    # Open long
    _, _, _, _ = env.step(int(TradingAction.LONG))
    assert env.position.is_open
    assert env.position.side == 1

    # Close long
    _, _, _, _ = env.step(int(TradingAction.CLOSE))
    assert not env.position.is_open


def test_environment_respects_terminal_position_close() -> None:
    ticks = synthetic_ticks(count=10)
    env = TradingEnvironment(ticks)
    env.reset()

    # Open position
    env.step(int(TradingAction.LONG))

    # Step to terminal without explicit close
    done = False
    while not done:
        _, _, done, info = env.step(int(TradingAction.HOLD))

    # Position should be closed
    assert not env.position.is_open


def test_environment_reward_positive_on_profit() -> None:
    ticks = synthetic_ticks(count=50)
    env = TradingEnvironment(ticks)
    env.reset()

    env.step(int(TradingAction.LONG))
    total_reward = 0.0
    done = False
    while not done:
        _, reward, done, _ = env.step(int(TradingAction.HOLD))
        total_reward += reward

    # Trending market + long position = positive reward expected
    assert total_reward > 0.0
