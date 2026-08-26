"""Safety guard for model actions before simulated or live execution."""

from __future__ import annotations

from dataclasses import dataclass

from agent_system.core.config import Settings, settings
from agent_system.environment.trading_env import TradingAction


@dataclass
class GuardState:
    open_side: int = 0
    entry_price: float = 0.0
    opened_at_step: int = 0

    @property
    def has_position(self) -> bool:
        return self.open_side != 0 and self.entry_price > 0


class PolicyGuard:
    def __init__(
        self,
        config: Settings = settings,
        take_profit_fraction: float = 0.004,
        max_hold_steps: int = 120,
    ) -> None:
        self.config = config
        self.take_profit_fraction = take_profit_fraction
        self.max_hold_steps = max_hold_steps
        self.state = GuardState()

    def filter_action(self, requested_action: int, price: float, step: int) -> int:
        action = TradingAction(requested_action)
        if self.state.has_position:
            pnl_fraction = self._pnl_fraction(price)
            held_steps = step - self.state.opened_at_step
            if pnl_fraction <= -self.config.stop_loss_fraction:
                return int(TradingAction.CLOSE)
            if pnl_fraction >= self.take_profit_fraction:
                return int(TradingAction.CLOSE)
            if held_steps >= self.max_hold_steps:
                return int(TradingAction.CLOSE)
            if action in {TradingAction.LONG, TradingAction.SHORT}:
                return int(TradingAction.HOLD)
            return int(action)

        if action == TradingAction.LONG:
            self.state = GuardState(open_side=1, entry_price=price, opened_at_step=step)
            return int(action)
        if action == TradingAction.SHORT:
            self.state = GuardState(open_side=-1, entry_price=price, opened_at_step=step)
            return int(action)
        return int(action)

    def observe_executed_action(self, action: int) -> None:
        if TradingAction(action) == TradingAction.CLOSE:
            self.state = GuardState()

    def _pnl_fraction(self, price: float) -> float:
        if not self.state.has_position:
            return 0.0
        if self.state.open_side == 1:
            return (price - self.state.entry_price) / self.state.entry_price
        return (self.state.entry_price - price) / self.state.entry_price
