"""Minimal Gym-style trading environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from agent_system.data.data_buffer import MarketTick
from agent_system.data.feature_engineering import FeatureEngine
from agent_system.environment.reward import calculate_reward


class TradingAction(IntEnum):
    HOLD = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3


@dataclass
class Position:
    side: int = 0
    entry_price: float = 0.0
    quantity: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.side != 0 and self.quantity > 0


class TradingEnvironment:
    """Deterministic market replay environment with a Gym-like API."""

    def __init__(
        self,
        ticks: list[MarketTick],
        initial_balance: float = 10_000.0,
        fee_rate: float = 0.0004,
        slippage_rate: float = 0.0002,
        position_fraction: float = 0.02,
        time_penalty: float = 0.0001,
    ) -> None:
        if len(ticks) < 2:
            raise ValueError("TradingEnvironment requires at least two ticks")
        self.ticks = ticks
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.position_fraction = position_fraction
        self.time_penalty = time_penalty
        self.feature_engine = FeatureEngine()
        self.reset()

    def reset(self) -> list[float]:
        self.index = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = Position()
        self.realized_pnl = 0.0
        self.previous_unrealized_pnl = 0.0
        return self._state()

    def step(self, action: int) -> tuple[list[float], float, bool, dict[str, float]]:
        trading_action = TradingAction(action)
        tick = self.ticks[self.index]
        realized = self._apply_action(trading_action, tick)
        terminal_step = self.index >= len(self.ticks) - 1
        if terminal_step and self.position.is_open:
            realized += self._close_position(tick)
        unrealized = self._unrealized_pnl(tick)
        unrealized_delta = unrealized - self.previous_unrealized_pnl
        self.previous_unrealized_pnl = unrealized
        self.equity = self.balance + unrealized

        reward = calculate_reward(
            realized_pnl=realized,
            unrealized_pnl_delta=unrealized_delta,
            fees=abs(realized) * self.fee_rate,
            time_penalty=self.time_penalty,
        )

        self.index += 1
        done = self.index >= len(self.ticks)
        state = self._state() if not done else self.feature_engine.empty().as_vector()
        info = {
            "balance": self.balance,
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealized,
            "position_side": float(self.position.side),
        }
        return state, reward, done, info

    def _state(self) -> list[float]:
        window = self.ticks[max(0, self.index - 200) : self.index + 1]
        return self.feature_engine.compute(window).as_vector()

    def _apply_action(self, action: TradingAction, tick: MarketTick) -> float:
        if action == TradingAction.LONG and not self.position.is_open:
            self._open_position(side=1, tick=tick)
            return 0.0
        if action == TradingAction.SHORT and not self.position.is_open:
            self._open_position(side=-1, tick=tick)
            return 0.0
        if action == TradingAction.CLOSE and self.position.is_open:
            return self._close_position(tick)
        return 0.0

    def _open_position(self, side: int, tick: MarketTick) -> None:
        notional = self.equity * self.position_fraction
        execution_price = self._execution_price(tick.price, side)
        quantity = notional / execution_price
        fee = notional * self.fee_rate
        self.balance -= fee
        self.position = Position(
            side=side,
            entry_price=execution_price,
            quantity=quantity,
        )

    def _close_position(self, tick: MarketTick) -> float:
        side = self.position.side
        execution_price = self._execution_price(tick.price, -side)
        pnl = (execution_price - self.position.entry_price) * self.position.quantity * side
        fee = execution_price * self.position.quantity * self.fee_rate
        realized = pnl - fee
        self.balance += realized
        self.realized_pnl += realized
        self.position = Position()
        self.previous_unrealized_pnl = 0.0
        return realized

    def _unrealized_pnl(self, tick: MarketTick) -> float:
        if not self.position.is_open:
            return 0.0
        return (
            (tick.price - self.position.entry_price)
            * self.position.quantity
            * self.position.side
        )

    def _execution_price(self, price: float, side: int) -> float:
        return price * (1.0 + self.slippage_rate * side)
