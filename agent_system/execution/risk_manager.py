"""Risk limits and position sizing.

These are circuit breakers, not a strategy. The agent is *meant* to trade
constantly, so the limits here are set wide enough not to pace it and tight
enough to stop a malfunctioning loop from emptying an account.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent_system.core.config import Settings, settings
from agent_system.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskState:
    equity: float
    day_start_equity: float
    day: str
    trades_today: int = 0
    consecutive_losses: int = 0
    last_trade_time: float = field(default=0.0)
    halted: bool = False
    halt_reason: str = ""


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class RiskManager:
    """Enforces per-trade and per-day risk limits."""

    def __init__(
        self,
        starting_equity: float | None = None,
        config: Settings = settings,
    ) -> None:
        self.config = config
        equity = float(starting_equity if starting_equity is not None else config.initial_capital)
        self.state = RiskState(equity=equity, day_start_equity=equity, day=_utc_day())

    # ------------------------------------------------------------------
    def _roll_day_if_needed(self) -> None:
        today = _utc_day()
        if today != self.state.day:
            # A new UTC day resets the daily loss budget and the trade count,
            # and clears a halt that was caused by either of them.
            self.state.day = today
            self.state.day_start_equity = self.state.equity
            self.state.trades_today = 0
            self.state.consecutive_losses = 0
            if self.state.halt_reason in {"daily_loss_limit", "max_trades_per_day"}:
                self.state.halted = False
                self.state.halt_reason = ""

    def can_trade(self) -> tuple[bool, str]:
        """Whether a new position may be opened right now."""
        self._roll_day_if_needed()

        if self.state.halted:
            return False, self.state.halt_reason or "halted"

        if self.config.enable_live_trading and not self.config.has_credentials:
            return False, "missing_credentials"

        loss_fraction = 0.0
        if self.state.day_start_equity > 0:
            loss_fraction = (
                self.state.day_start_equity - self.state.equity
            ) / self.state.day_start_equity
        if loss_fraction >= self.config.daily_loss_limit_fraction:
            self.halt("daily_loss_limit")
            return False, "daily_loss_limit"

        if self.state.trades_today >= self.config.max_trades_per_day:
            return False, "max_trades_per_day"

        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            self.halt("max_consecutive_losses")
            return False, "max_consecutive_losses"

        min_gap = self.config.min_seconds_between_trades
        if min_gap > 0 and (time.time() - self.state.last_trade_time) < min_gap:
            return False, "rate_limited"

        return True, "ok"

    def position_size(self, price: float, leverage: int | None = None) -> float:
        """Quantity to trade, from equity, the notional cap and leverage."""
        if price <= 0:
            return 0.0
        leverage = int(leverage or self.config.default_leverage)
        leverage = max(1, min(leverage, self.config.max_leverage))
        notional = self.state.equity * self.config.max_position_fraction * leverage
        return max(notional / price, 0.0)

    # ------------------------------------------------------------------
    def register_trade(self) -> None:
        self._roll_day_if_needed()
        self.state.trades_today += 1
        self.state.last_trade_time = time.time()

    def register_result(self, pnl: float) -> None:
        self.state.equity += pnl
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

    def update_equity(self, equity: float) -> None:
        self.state.equity = float(equity)

    def halt(self, reason: str) -> None:
        if not self.state.halted:
            logger.warning("Risk halt engaged: %s", reason)
        self.state.halted = True
        self.state.halt_reason = reason

    def resume(self) -> None:
        self.state.halted = False
        self.state.halt_reason = ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "equity": round(self.state.equity, 4),
            "day": self.state.day,
            "trades_today": self.state.trades_today,
            "consecutive_losses": self.state.consecutive_losses,
            "halted": self.state.halted,
            "halt_reason": self.state.halt_reason,
        }
