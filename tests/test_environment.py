import numpy as np

from agent_system.data.feature_engineering import FEATURE_DIM
from agent_system.rl.environment import (
    ACCOUNT_FEATURES,
    CLOSE,
    HOLD,
    LONG,
    SHORT,
    TradingEnvironment,
)
from tests.conftest import make_ticks


def test_observation_shape_matches_space(config) -> None:
    env = TradingEnvironment(make_ticks(400), config=config)
    observation, _ = env.reset()

    assert observation.shape == (config.observation_window * FEATURE_DIM + ACCOUNT_FEATURES,)
    assert np.isfinite(observation).all()


def test_reward_is_incremental_not_cumulative(config) -> None:
    """Holding a winning position must not be paid for the same gain twice."""
    env = TradingEnvironment(make_ticks(400), config=config)
    env.reset()
    env.step(LONG)

    rewards = [env.step(HOLD)[1] for _ in range(20)]

    # If reward were `equity - initial_balance` a profitable position would
    # emit a large, persistently positive reward every step.
    assert all(abs(reward) < 5.0 for reward in rewards)


def test_closing_a_position_charges_fees(config) -> None:
    env = TradingEnvironment(make_ticks(400), config=config)
    env.reset()
    env.step(LONG)
    env.step(CLOSE)

    assert env.fees_paid > 0
    assert len(env.closed_trades) == 1
    assert env.closed_trades[0].fees > 0


def test_short_positions_profit_when_price_falls(config) -> None:
    """A short in a falling market must make money, open or already banked."""
    ticks = make_ticks(400, trend=-0.002, volatility=0.0005)
    env = TradingEnvironment(ticks, config=config)
    env.reset()
    env.step(SHORT)
    for _ in range(5):
        env.step(HOLD)

    if env.position is not None:
        assert env.position.unrealized(env.prices[env.index]) > 0
    else:
        # The move was big enough to hit the take-profit on the way down.
        assert env.closed_trades[-1].pnl > 0


def test_reversal_closes_and_reopens(config) -> None:
    env = TradingEnvironment(make_ticks(400), config=config)
    env.reset()
    env.step(LONG)
    env.step(SHORT)

    assert env.position is not None
    assert env.position.side == -1
    assert any(trade.reason == "reverse" for trade in env.closed_trades)


def test_idle_agent_is_penalised(config) -> None:
    """Doing nothing must score worse than zero, or holding is a safe optimum."""
    env = TradingEnvironment(make_ticks(400), config=config)
    env.reset()

    rewards = [env.step(HOLD)[1] for _ in range(40)]

    assert sum(rewards) < 0


def test_position_is_force_closed_after_max_holding_bars(config) -> None:
    config.max_holding_bars = 5
    config.trailing_stop_enabled = False
    env = TradingEnvironment(make_ticks(400, volatility=0.0001), config=config)
    env.reset()
    env.step(LONG)
    for _ in range(10):
        env.step(HOLD)

    assert any(trade.reason == "max_hold" for trade in env.closed_trades)


def test_terminates_on_daily_loss_limit(config) -> None:
    env = TradingEnvironment(make_ticks(400), config=config)
    env.reset()
    env.equity = config.initial_capital * 0.5
    env.balance = env.equity
    _, _, terminated, _, info = env.step(HOLD)

    assert terminated
    assert info["terminated_reason"] in {"bankrupt", "daily_loss_limit"}


def test_metrics_report_activity(config) -> None:
    env = TradingEnvironment(make_ticks(500), config=config)
    env.reset()
    for step in range(100):
        env.step(LONG if step % 10 == 0 else HOLD)

    metrics = env.metrics()
    assert metrics["trades"] > 0
    assert metrics["trades_per_100_steps"] > 0
    assert set(metrics["action_counts"]) == {"hold", "long", "short", "close"}
