"""Trading environment.

Replaces the old ``environment_features.py``. Key differences, all of which
matter for whether learned behaviour survives contact with a real exchange:

* **Costs are real.** Every fill pays taker fee + slippage. Without this the
  optimal policy degenerates into churning trades for imaginary profit.
* **Reward is an increment, not a level.** The old version returned
  ``equity - initial_balance`` every step, i.e. it re-paid the agent for the same
  unrealised gain on every tick. Here reward is the change in equity since the
  previous step.
* **Long and short are both supported**, with a single position at a time.
* **Episodes terminate** on bankruptcy or the daily loss limit.
* **Observation is a flattened window** of stationary features plus normalised
  account state, so the policy sees trend, not just a snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agent_system.core.config import Settings, settings
from agent_system.data.data_buffer import MarketTick
from agent_system.data.feature_engineering import FEATURE_DIM, FeatureEngine

# Discrete action space.
ACTION_HOLD = 0
ACTION_LONG = 1
ACTION_SHORT = 2
ACTION_CLOSE = 3
ACTION_NAMES = {ACTION_HOLD: "hold", ACTION_LONG: "long", ACTION_SHORT: "short", ACTION_CLOSE: "close"}

#: Account-state values appended to the flattened feature window.
ACCOUNT_STATE_DIM = 4


@dataclass
class Position:
    """An open position. ``side`` is +1 long, -1 short, 0 flat."""

    side: int = 0
    entry_price: float = 0.0
    quantity: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.side != 0

    def unrealized_pnl(self, price: float) -> float:
        if not self.is_open:
            return 0.0
        return self.side * (price - self.entry_price) * self.quantity


class TradingEnvironment(gym.Env):
    """Gymnasium environment replaying a fixed sequence of ticks."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        ticks: Sequence[MarketTick],
        config: Settings = settings,
        window: int | None = None,
        feature_engine: FeatureEngine | None = None,
        precomputed_features: np.ndarray | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.ticks = list(ticks)
        self.window = window or config.observation_window
        self.feature_engine = feature_engine or FeatureEngine()

        if len(self.ticks) < self.window + 2:
            raise ValueError(
                f"Need at least {self.window + 2} ticks, got {len(self.ticks)}"
            )

        self.features = (
            precomputed_features
            if precomputed_features is not None
            else self._precompute_features()
        )

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window * FEATURE_DIM + ACCOUNT_STATE_DIM,),
            dtype=np.float32,
        )

        self.initial_balance = float(config.initial_capital)
        self._reset_state()

    def _precompute_features(self) -> np.ndarray:
        """Compute features once per tick up-front.

        Recomputing a rolling window inside ``step`` is O(n^2) over an episode and
        was a real bottleneck in the original loop.
        """
        engine = self.feature_engine
        lookback = 60
        rows = []
        for index in range(len(self.ticks)):
            start = max(0, index - lookback + 1)
            rows.append(engine.compute(self.ticks[start : index + 1]).as_vector())
        return np.asarray(rows, dtype=np.float32)

    def _reset_state(self) -> None:
        self.step_index = self.window
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.position = Position()
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.max_drawdown = 0.0
        self.equity_curve: list[float] = [self.initial_balance]
        self.trades: list[dict[str, Any]] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._observation(), self._info()

    @property
    def current_price(self) -> float:
        return self.ticks[min(self.step_index, len(self.ticks) - 1)].price

    def _observation(self) -> np.ndarray:
        """Flattened feature window + normalised account state."""
        start = self.step_index - self.window + 1
        window = self.features[start : self.step_index + 1]

        if len(window) < self.window:  # defensive pad, should not trigger
            pad = np.zeros((self.window - len(window), FEATURE_DIM), dtype=np.float32)
            window = np.vstack([pad, window])

        price = self.current_price or 1.0
        account_state = np.array(
            [
                float(self.position.side),
                self.position.unrealized_pnl(price) / self.initial_balance,
                self.equity / self.initial_balance - 1.0,
                (self.equity - self.peak_equity) / self.peak_equity if self.peak_equity else 0.0,
            ],
            dtype=np.float32,
        )

        observation = np.concatenate([window.flatten(), account_state]).astype(np.float32)
        return np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)

    def _position_size(self, price: float) -> float:
        """Notional sized as a fraction of equity, scaled by leverage."""
        if price <= 0:
            return 0.0
        notional = self.equity * self.config.max_position_fraction * self.config.default_leverage
        return max(notional / price, 0.0)

    def _cost(self, price: float, quantity: float) -> float:
        """Taker fee plus slippage for a fill of ``quantity`` at ``price``."""
        return price * quantity * (self.config.taker_fee_fraction + self.config.slippage_fraction)

    def _open(self, side: int, price: float) -> None:
        quantity = self._position_size(price)
        if quantity <= 0:
            return
        cost = self._cost(price, quantity)
        self.balance -= cost
        self.fees_paid += cost
        self.position = Position(side=side, entry_price=price, quantity=quantity)
        self.trade_count += 1

    def _close(self, price: float) -> None:
        if not self.position.is_open:
            return

        pnl = self.position.unrealized_pnl(price)
        cost = self._cost(price, self.position.quantity)
        net = pnl - cost

        self.balance += net
        self.realized_pnl += net
        self.fees_paid += cost

        if net > 0:
            self.wins += 1
            self.gross_profit += net
        else:
            self.losses += 1
            self.gross_loss += abs(net)

        self.trades.append(
            {
                "side": self.position.side,
                "entry": self.position.entry_price,
                "exit": price,
                "pnl": net,
            }
        )
        self.position = Position()

    def _check_protective_exits(self, price: float) -> None:
        """Apply stop-loss / take-profit before the agent acts.

        Modelling these as automatic exits keeps risk bounded even if the policy
        never learns to close a losing position.
        """
        if not self.position.is_open:
            return

        move = self.position.side * (price - self.position.entry_price) / self.position.entry_price
        if move <= -self.config.stop_loss_fraction or move >= self.config.take_profit_fraction:
            self._close(price)

    def step(self, action: int):
        action = int(action)
        price = self.current_price
        equity_before = self.equity

        self._check_protective_exits(price)

        if action == ACTION_LONG:
            if self.position.side == -1:
                self._close(price)
            if not self.position.is_open:
                self._open(1, price)
        elif action == ACTION_SHORT:
            if self.position.side == 1:
                self._close(price)
            if not self.position.is_open:
                self._open(-1, price)
        elif action == ACTION_CLOSE:
            self._close(price)

        self.step_index += 1
        terminated = False
        truncated = self.step_index >= len(self.ticks) - 1

        new_price = self.current_price
        self.equity = self.balance + self.position.unrealized_pnl(new_price)
        self.equity_curve.append(self.equity)
        self.peak_equity = max(self.peak_equity, self.equity)

        drawdown = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity else 0.0
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Hard stops: bankruptcy or breaching the daily loss limit.
        if self.equity <= 0:
            terminated = True
        elif self.equity <= self.initial_balance * (1.0 - self.config.daily_loss_limit_fraction):
            terminated = True

        if (terminated or truncated) and self.position.is_open:
            self._close(new_price)
            self.equity = self.balance
            self.equity_curve[-1] = self.equity

        reward = self._reward(equity_before, drawdown)
        return self._observation(), reward, terminated, truncated, self._info()

    def _reward(self, equity_before: float, drawdown: float) -> float:
        """Equity increment, normalised, with a mild drawdown penalty.

        Scaling by initial balance keeps the reward magnitude comparable across
        account sizes; the drawdown term biases the policy toward smoother curves
        rather than lucky high-variance runs.
        """
        increment = (self.equity - equity_before) / self.initial_balance
        return float(np.clip(increment * 100.0 - drawdown * 0.1, -10.0, 10.0))

    def _info(self) -> dict[str, Any]:
        return {
            "balance": self.balance,
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.position.unrealized_pnl(self.current_price),
            "position_side": float(self.position.side),
            "trades": self.trade_count,
            "fees_paid": self.fees_paid,
            "max_drawdown": self.max_drawdown,
        }

    def metrics(self) -> dict[str, float]:
        """Summary statistics for evaluation and model promotion."""
        curve = np.asarray(self.equity_curve, dtype=np.float64)
        returns = np.diff(curve) / np.maximum(curve[:-1], 1e-9)

        sharpe = 0.0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(len(returns)))

        closed = self.wins + self.losses
        profit_factor = (
            self.gross_profit / self.gross_loss if self.gross_loss > 0
            else (float("inf") if self.gross_profit > 0 else 0.0)
        )

        return {
            "equity": float(self.equity),
            "return_pct": float((self.equity / self.initial_balance - 1.0) * 100.0),
            "trades": float(self.trade_count),
            "closed_trades": float(closed),
            "win_rate": float(self.wins / closed) if closed else 0.0,
            "profit_factor": float(min(profit_factor, 1e6)),
            "drawdown": float(self.max_drawdown),
            "sharpe": sharpe,
            "fees_paid": float(self.fees_paid),
        }

    def render(self) -> None:
        print(
            f"step={self.step_index} price={self.current_price:.2f} "
            f"equity={self.equity:.2f} side={self.position.side} trades={self.trade_count}"
        )
