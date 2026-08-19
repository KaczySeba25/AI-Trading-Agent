"""Gymnasium trading environment.

This module is where the two goals in the brief collide: the agent should be
in and out of the market constantly, *and* it should make money after costs.
Those pull in opposite directions, and the reward function is the only place
that can honestly arbitrate between them. The comments below explain each
term rather than just stating it, because the exact shape of this function is
what the learned policy optimises -- get it wrong and the agent will happily
trade itself to death.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

try:  # pragma: no cover - exercised implicitly by the import path taken
    import gymnasium as gym
    from gymnasium import spaces

    GYM_AVAILABLE = True
except ImportError:  # pragma: no cover
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    GYM_AVAILABLE = False

from agent_system.core.config import Settings, settings
from agent_system.core.logger import get_logger
from agent_system.data.data_buffer import MarketTick
from agent_system.data.feature_engineering import FEATURE_DIM, FeatureEngine

logger = get_logger(__name__)

HOLD, LONG, SHORT, CLOSE = 0, 1, 2, 3
ACTION_NAMES = {HOLD: "hold", LONG: "long", SHORT: "short", CLOSE: "close"}

#: Extra observation slots appended after the flattened feature window:
#: position side, unrealised pnl, equity change, drawdown, bars held,
#: and the cost hurdle the agent must clear.
ACCOUNT_FEATURES = 6

_BaseEnv = gym.Env if GYM_AVAILABLE else object


@dataclass
class ClosedTrade:
    side: int
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    fees: float
    bars_held: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "side": "long" if self.side > 0 else "short",
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "quantity": round(self.quantity, 6),
            "pnl": round(self.pnl, 4),
            "fees": round(self.fees, 4),
            "bars_held": self.bars_held,
            "reason": self.reason,
        }


@dataclass
class _Position:
    side: int
    entry_price: float
    quantity: float
    entry_index: int
    stop_loss: float
    take_profit: float
    peak_price: float = field(default=0.0)

    def unrealized(self, price: float) -> float:
        return (price - self.entry_price) * self.quantity * self.side


class TradingEnvironment(_BaseEnv):
    """Long/short trading environment with realistic costs.

    Action space is ``Discrete(4)``: hold, open long, open short, close.
    Observation is the flattened feature window plus account state.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        ticks: Sequence[MarketTick],
        config: Settings = settings,
        window: int | None = None,
        feature_engine: FeatureEngine | None = None,
        precomputed_features: np.ndarray | None = None,
    ) -> None:
        if len(ticks) < 64:
            raise ValueError(f"Need at least 64 ticks to build an environment, got {len(ticks)}")

        self.config = config
        self.ticks = list(ticks)
        self.window = int(window or config.observation_window)
        self.feature_engine = feature_engine or FeatureEngine()

        self.prices = np.array([tick.price for tick in self.ticks], dtype=np.float64)
        if precomputed_features is not None:
            self.features = precomputed_features.astype(np.float32)
        else:
            self.features = self.feature_engine.compute_many(self.ticks, self.window)

        self.observation_size = self.window * FEATURE_DIM + ACCOUNT_FEATURES

        if GYM_AVAILABLE:
            self.action_space = spaces.Discrete(4)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.observation_size,),
                dtype=np.float32,
            )

        # Cost model. Every open and every close pays fee + slippage, so a
        # round trip costs twice this. It is the hurdle for every trade.
        self.fee_fraction = config.entry_fee_fraction
        self.slippage_fraction = config.slippage_fraction
        self.round_trip_cost = config.round_trip_cost_fraction

        self.initial_balance = float(config.initial_capital)
        self._rng = np.random.default_rng(config.seed)
        self.reset()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.index = self.window
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.position: _Position | None = None
        self.closed_trades: list[ClosedTrade] = []
        self.fees_paid = 0.0
        self.trades_opened = 0
        self.action_counts = {name: 0 for name in ACTION_NAMES.values()}
        self.equity_curve: list[float] = [self.initial_balance]
        self.returns: list[float] = []
        self.bars_since_trade = 0
        self.terminated_reason: str | None = None

        return self._observation(), {}

    def step(self, action: int):  # type: ignore[override]
        action = int(action)
        price = float(self.prices[self.index])
        equity_before = self.equity

        self.action_counts[ACTION_NAMES.get(action, "hold")] += 1

        trade_event: str | None = None
        realized_pnl = 0.0

        # --- 1. protective exits fire before any new decision --------------
        # A stop that only triggers when the agent happens to ask is not a
        # stop. Checking first also stops the agent from flipping out of a
        # position that the market had already closed for it.
        if self.position is not None:
            exit_reason = self._protective_exit_reason(price)
            if exit_reason is not None:
                realized_pnl += self._close_position(price, exit_reason)
                trade_event = exit_reason

        # --- 2. the agent's action ------------------------------------------
        if action == CLOSE and self.position is not None:
            realized_pnl += self._close_position(price, "signal")
            trade_event = "close"
        elif action in (LONG, SHORT):
            desired = 1 if action == LONG else -1
            if self.position is None:
                self._open_position(desired, price)
                trade_event = "open"
            elif self.position.side != desired:
                # Reversal: close and immediately open the other way. Costs
                # are charged for both legs, which is what makes flip-flopping
                # expensive rather than free.
                realized_pnl += self._close_position(price, "reverse")
                self._open_position(desired, price)
                trade_event = "reverse"

        # --- 3. advance time --------------------------------------------------
        self.index += 1
        terminated = False
        truncated = False

        if self.index >= len(self.prices) - 1:
            truncated = True
            if self.position is not None:
                realized_pnl += self._close_position(float(self.prices[self.index]), "end_of_data")

        next_price = float(self.prices[min(self.index, len(self.prices) - 1)])
        self._mark_to_market(next_price)

        if self.position is not None:
            self.position.peak_price = (
                max(self.position.peak_price, next_price)
                if self.position.side > 0
                else min(self.position.peak_price, next_price)
            )
            self.bars_since_trade = 0
        else:
            self.bars_since_trade += 1

        # --- 4. termination on ruin or daily loss limit ------------------------
        if self.equity <= self.initial_balance * 0.2:
            terminated = True
            self.terminated_reason = "bankrupt"
        elif self.equity <= self.initial_balance * (1.0 - self.config.daily_loss_limit_fraction):
            terminated = True
            self.terminated_reason = "daily_loss_limit"

        reward = self._reward(equity_before, trade_event)

        self.equity_curve.append(self.equity)
        if equity_before > 0:
            self.returns.append((self.equity - equity_before) / equity_before)

        info = {
            "equity": self.equity,
            "price": next_price,
            "position": 0 if self.position is None else self.position.side,
            "realized_pnl": realized_pnl,
            "trade_event": trade_event,
            "terminated_reason": self.terminated_reason,
        }
        return self._observation(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _reward(self, equity_before: float, trade_event: str | None) -> float:
        """Shaped reward, in units of percent-of-initial-capital.

        The base term is the *change* in equity this step, never the level.
        Rewarding the level (``equity - initial_balance``) pays the agent
        again every single step for a profit it already banked, which teaches
        it to open one good position and then freeze -- exactly the passive
        behaviour we are trying to eliminate.
        """
        pnl_reward = (self.equity - equity_before) / self.initial_balance * 100.0

        # Drawdown penalty: proportional to how far below the high-water mark
        # we are, so recovering is worth more than drifting sideways.
        drawdown = self._drawdown()
        drawdown_cost = drawdown * self.config.drawdown_penalty * 100.0

        # Inactivity penalty -- the term that makes the agent trade.
        # A flat market with no position earns exactly zero, and zero is a
        # perfectly comfortable score, so a purely pnl-based agent learns that
        # never trading is a safe local optimum. Charging rent for sitting
        # flat makes "do nothing" strictly worse than "do something with
        # positive expectancy", while still being far cheaper than a losing
        # trade. It grows with the length of the idle streak so brief
        # patience is free but permanent hibernation is not.
        idle_cost = 0.0
        if self.position is None:
            idle_cost = self.config.inactivity_penalty * min(self.bars_since_trade, 50)

        # Churn penalty -- the counterweight. Without it the inactivity term
        # alone would be gamed by opening and closing every bar to reset the
        # idle counter, which pays the spread forever. This is charged per
        # trade and sized as a fraction of the real round-trip cost, so the
        # agent sees the cost of a trade slightly *before* the market charges
        # it and only trades when it expects to clear the hurdle.
        churn_cost = 0.0
        if trade_event in {"open", "reverse"}:
            churn_cost = self.config.churn_penalty
        elif trade_event in {"close", "stop_loss", "take_profit", "trailing_stop", "max_hold"}:
            churn_cost = self.config.churn_penalty * 0.5

        # Holding penalty: a small carry charge on open positions. Real
        # futures positions pay funding, and it discourages the agent from
        # parking in a position for hours, which is the other way to be
        # inactive while technically holding a trade.
        holding_cost = 0.0
        if self.position is not None:
            holding_cost = self.config.holding_penalty

        reward = pnl_reward - drawdown_cost - idle_cost - churn_cost - holding_cost
        clip = self.config.reward_clip
        return float(np.clip(reward, -clip, clip))

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------
    def _position_quantity(self, price: float) -> float:
        exposure = self.equity * self.config.sim_position_fraction
        exposure *= max(1, self.config.default_leverage)
        return max(exposure / max(price, 1e-9), 0.0)

    def _current_atr_fraction(self) -> float:
        """ATR at the current bar, as a fraction of price.

        Taken from the precomputed feature matrix (index 8, stored in
        percent) so the environment never recomputes indicators mid-episode.
        """
        row = self.features[min(self.index, len(self.features) - 1)]
        atr_pct = float(row[8])
        if atr_pct <= 0.0 or not math.isfinite(atr_pct):
            return self.config.stop_loss_fraction
        return atr_pct / 100.0

    def _exit_levels(self, side: int, price: float) -> tuple[float, float]:
        """Stop-loss and take-profit for a new position.

        Volatility-scaled by default. A fixed 0.4% stop is meaningless when
        the market is moving 0.05% a minute and suicidal when it is moving
        0.5%, so the distance is pinned to recent ATR and then clamped so a
        freak reading cannot produce an absurd level.
        """
        if self.config.use_atr_exits:
            atr = self._current_atr_fraction()
            stop_fraction = atr * self.config.atr_stop_multiple
            target_fraction = atr * self.config.atr_target_multiple
        else:
            stop_fraction = self.config.stop_loss_fraction
            target_fraction = self.config.take_profit_fraction

        stop_fraction = float(
            np.clip(stop_fraction, self.config.min_stop_fraction, self.config.max_stop_fraction)
        )
        # The target must at minimum cover the round trip, otherwise a
        # "winning" trade still loses money once fees are paid.
        target_fraction = max(target_fraction, stop_fraction * 1.2, self.round_trip_cost * 2.0)

        if side > 0:
            return price * (1.0 - stop_fraction), price * (1.0 + target_fraction)
        return price * (1.0 + stop_fraction), price * (1.0 - target_fraction)

    def _open_position(self, side: int, price: float) -> None:
        quantity = self._position_quantity(price)
        if quantity <= 0:
            return

        # Slippage always works against us: buy a touch high, sell a touch low.
        fill_price = price * (1.0 + self.slippage_fraction * side)
        fee = fill_price * quantity * self.fee_fraction
        self.balance -= fee
        self.fees_paid += fee
        self.trades_opened += 1

        stop_loss, take_profit = self._exit_levels(side, fill_price)
        self.position = _Position(
            side=side,
            entry_price=fill_price,
            quantity=quantity,
            entry_index=self.index,
            stop_loss=stop_loss,
            take_profit=take_profit,
            peak_price=fill_price,
        )

    def _close_position(self, price: float, reason: str) -> float:
        position = self.position
        if position is None:
            return 0.0

        fill_price = price * (1.0 - self.slippage_fraction * position.side)
        gross = (fill_price - position.entry_price) * position.quantity * position.side
        fee = fill_price * position.quantity * self.fee_fraction
        entry_fee = position.entry_price * position.quantity * self.fee_fraction

        self.balance += gross - fee
        self.fees_paid += fee

        self.closed_trades.append(
            ClosedTrade(
                side=position.side,
                entry_price=position.entry_price,
                exit_price=fill_price,
                quantity=position.quantity,
                pnl=gross - fee - entry_fee,
                fees=fee + entry_fee,
                bars_held=self.index - position.entry_index,
                reason=reason,
            )
        )
        self.position = None
        self.bars_since_trade = 0
        return gross - fee

    def _protective_exit_reason(self, price: float) -> str | None:
        position = self.position
        if position is None:
            return None

        if position.side > 0:
            if price <= position.stop_loss:
                return "stop_loss"
            if price >= position.take_profit:
                return "take_profit"
        else:
            if price >= position.stop_loss:
                return "stop_loss"
            if price <= position.take_profit:
                return "take_profit"

        # Trailing stop: locks in a move that already happened. Without it a
        # position that ran most of the way to target and reversed gives the
        # whole gain back and then some.
        if self.config.trailing_stop_enabled:
            atr = self._current_atr_fraction() * self.config.trailing_stop_atr_multiple
            atr = float(np.clip(atr, self.config.min_stop_fraction, self.config.max_stop_fraction))
            if position.side > 0:
                trail = position.peak_price * (1.0 - atr)
                if price <= trail and position.peak_price > position.entry_price:
                    return "trailing_stop"
            else:
                trail = position.peak_price * (1.0 + atr)
                if price >= trail and position.peak_price < position.entry_price:
                    return "trailing_stop"

        # Time stop: a thesis that has not worked within N bars is not
        # working. Capital tied up in a stale position cannot take the next
        # setup, which matters a great deal when trading this often.
        if self.index - position.entry_index >= self.config.max_holding_bars:
            return "max_hold"

        return None

    def _mark_to_market(self, price: float) -> None:
        unrealized = self.position.unrealized(price) if self.position else 0.0
        self.equity = self.balance + unrealized
        self.peak_equity = max(self.peak_equity, self.equity)

    def _drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _observation(self) -> np.ndarray:
        end = min(self.index, len(self.features))
        start = max(0, end - self.window)
        window = self.features[start:end]
        if window.shape[0] < self.window:
            padding = np.zeros((self.window - window.shape[0], FEATURE_DIM), dtype=np.float32)
            window = np.vstack([padding, window])

        price = float(self.prices[min(self.index, len(self.prices) - 1)])
        unrealized = self.position.unrealized(price) if self.position else 0.0
        bars_held = (self.index - self.position.entry_index) if self.position else 0

        account = np.array(
            [
                0.0 if self.position is None else float(self.position.side),
                unrealized / self.initial_balance,
                self.equity / self.initial_balance - 1.0,
                -self._drawdown(),
                min(bars_held / max(self.config.max_holding_bars, 1), 1.0),
                self.round_trip_cost * 100.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([window.flatten(), account]).astype(np.float32)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        pnls = [trade.pnl for trade in self.closed_trades]
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
            float(len(wins)) if wins else 0.0
        )

        returns = np.array(self.returns, dtype=np.float64)
        if returns.size > 1 and float(np.std(returns)) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * math.sqrt(len(returns)))
        else:
            sharpe = 0.0

        curve = np.array(self.equity_curve, dtype=np.float64)
        running_peak = np.maximum.accumulate(curve)
        max_drawdown = float(np.max((running_peak - curve) / np.maximum(running_peak, 1e-9)))

        steps = max(self.index - self.window, 1)

        return {
            "equity": round(self.equity, 4),
            "return_pct": round((self.equity / self.initial_balance - 1.0) * 100.0, 4),
            "trades": self.trades_opened,
            "closed_trades": len(self.closed_trades),
            "win_rate": round(len(wins) / len(pnls), 4) if pnls else 0.0,
            "profit_factor": round(profit_factor, 4),
            "drawdown": round(max_drawdown, 4),
            "sharpe": round(sharpe, 4),
            "fees_paid": round(self.fees_paid, 4),
            "avg_pnl_per_trade": round(float(np.mean(pnls)), 4) if pnls else 0.0,
            "avg_bars_held": round(
                float(np.mean([t.bars_held for t in self.closed_trades])), 2
            )
            if self.closed_trades
            else 0.0,
            "trades_per_100_steps": round(self.trades_opened / steps * 100.0, 2),
            "steps": steps,
            "action_counts": dict(self.action_counts),
            "exit_reasons": self._exit_reason_counts(),
            "terminated_reason": self.terminated_reason,
        }

    def _exit_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for trade in self.closed_trades:
            counts[trade.reason] = counts.get(trade.reason, 0) + 1
        return counts
