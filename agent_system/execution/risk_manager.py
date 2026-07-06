"""Execution risk controls."""

from __future__ import annotations

from dataclasses import dataclass

from agent_system.core.config import Settings, settings
from agent_system.execution.position_tracker import LivePosition


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str
    quantity: float = 0.0
    leverage: int = 1


class RiskManager:
    def __init__(
        self,
        config: Settings = settings,
        starting_equity: float | None = None,
    ) -> None:
        self.config = config
        self.starting_equity = starting_equity or config.initial_capital
        self.current_equity = starting_equity
        if self.current_equity is None:
            self.current_equity = config.initial_capital
        self.kill_switch_enabled = False

    def enable_kill_switch(self) -> None:
        self.kill_switch_enabled = True

    def disable_kill_switch(self) -> None:
        self.kill_switch_enabled = False

    def update_equity(self, equity: float) -> None:
        self.current_equity = equity

    def assess_entry(
        self,
        price: float,
        requested_leverage: int | None = None,
    ) -> RiskDecision:
        if self.kill_switch_enabled:
            return RiskDecision(False, "kill_switch_enabled")
        if self._daily_loss_limit_hit():
            return RiskDecision(False, "daily_loss_limit_hit")
        if price <= 0:
            return RiskDecision(False, "invalid_price")

        leverage = min(
            requested_leverage or self.config.default_leverage,
            self.config.max_leverage,
        )
        if leverage < 1:
            return RiskDecision(False, "invalid_leverage")

        notional = self.current_equity * self.config.max_position_fraction
        quantity = notional / price
        return RiskDecision(True, "allowed", quantity=quantity, leverage=leverage)

    def assess_position_capacity(self, open_positions_count: int) -> RiskDecision:
        if open_positions_count >= self.config.max_open_positions:
            return RiskDecision(False, "max_open_positions_reached")
        return RiskDecision(True, "allowed")

    def should_stop_loss(self, position: LivePosition, mark_price: float) -> bool:
        if not position.is_open or position.entry_price <= 0:
            return False
        if position.side == "LONG":
            pnl_fraction = (mark_price - position.entry_price) / position.entry_price
        else:
            pnl_fraction = (position.entry_price - mark_price) / position.entry_price
        return pnl_fraction <= -self.config.stop_loss_fraction

    def _daily_loss_limit_hit(self) -> bool:
        drawdown = (self.starting_equity - self.current_equity) / self.starting_equity
        return drawdown >= self.config.daily_loss_limit_fraction
