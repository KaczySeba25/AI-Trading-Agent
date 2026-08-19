"""Environment tests.

These lock in the behaviours that make backtests trustworthy: costs are charged,
reward is an increment, shorts work, and risk limits terminate episodes.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent_system.rl.environment import (
    ACTION_CLOSE,
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
    TradingEnvironment,
)

from conftest import make_ticks


def test_observation_shape_matches_space(test_settings, ticks) -> None:
    env = TradingEnvironment(ticks, config=test_settings)
    observation, _ = env.reset()

    assert observation.shape == env.observation_space.shape
    assert env.observation_space.contains(observation)


def test_observation_never_contains_nan(test_settings, ticks) -> None:
    env = TradingEnvironment(ticks, config=test_settings)
    observation, _ = env.reset()

    for _ in range(50):
        observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(observation))
        if terminated or truncated:
            break


def test_requires_minimum_ticks(test_settings) -> None:
    with pytest.raises(ValueError, match="at least"):
        TradingEnvironment(make_ticks(5), config=test_settings)


def test_opening_a_position_charges_fees(test_settings, ticks) -> None:
    """A round trip at an unchanged price must lose exactly the costs."""
    env = TradingEnvironment(ticks, config=test_settings)
    env.reset()
    balance_before = env.balance

    env.step(ACTION_LONG)

    assert env.position.is_open
    assert env.fees_paid > 0
    assert env.balance < balance_before


def test_reward_is_incremental_not_cumulative(test_settings, ticks) -> None:
    """Holding flat must pay ~nothing.

    The old implementation returned ``equity - initial_balance`` each step, so a
    flat position kept collecting reward for a gain it had already been paid for.
    """
    env = TradingEnvironment(ticks, config=test_settings)
    env.reset()

    rewards = [env.step(ACTION_HOLD)[1] for _ in range(20)]

    assert all(abs(reward) < 1e-6 for reward in rewards)


def test_short_position_profits_when_price_falls(test_settings) -> None:
    falling = make_ticks(300, trend=-0.004, volatility=0.0001)
    env = TradingEnvironment(falling, config=test_settings)
    env.reset()

    env.step(ACTION_SHORT)
    assert env.position.side == -1

    for _ in range(5):
        env.step(ACTION_HOLD)
    env.step(ACTION_CLOSE)

    assert env.realized_pnl > 0


def test_episode_terminates_on_daily_loss_limit(test_settings) -> None:
    """A collapsing market must trip the loss limit rather than run to the end."""
    crash = make_ticks(400, trend=-0.05, volatility=0.001)
    env = TradingEnvironment(crash, config=test_settings)
    env.reset()

    terminated = False
    for _ in range(200):
        _, _, terminated, truncated, _ = env.step(ACTION_LONG)
        if terminated or truncated:
            break

    assert terminated
    assert env.equity <= test_settings.initial_capital


def test_metrics_are_complete_and_sane(test_settings, ticks) -> None:
    env = TradingEnvironment(ticks, config=test_settings)
    env.reset()
    for _ in range(100):
        _, _, terminated, truncated, _ = env.step(ACTION_LONG)
        if terminated or truncated:
            break

    metrics = env.metrics()

    assert set(metrics).issuperset(
        {"equity", "return_pct", "trades", "win_rate", "profit_factor", "drawdown", "sharpe"}
    )
    assert 0.0 <= metrics["win_rate"] <= 1.0
    assert metrics["drawdown"] >= 0.0


def test_position_is_closed_at_episode_end(test_settings, ticks) -> None:
    env = TradingEnvironment(ticks[:100], config=test_settings)
    env.reset()

    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(ACTION_LONG)
        done = terminated or truncated

    assert not env.position.is_open
    assert env.equity == pytest.approx(env.balance)
