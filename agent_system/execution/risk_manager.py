"""Risk limits enforced before any order is sent.

This layer is deliberately paranoid and independent of the policy. The model is
a black box that can degrade at any time; these limits are the part of the system
that must never be wrong. Every check fails **closed** -- on doubt, no trade.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent_system.core.config import Settings, settings
from agent_system.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskState:
    starting_equity: float
    current_equity: float
    day_start_equity: float
    day_stamp: str
    realized_pnl_today: float = 0.0
    consecutive_losses: int = 0
    open_positions: int = 0
    halted: bool = False
    halt_reason: str = ""
    trades_today: int = 0
    last_trade_time: float = field(default_factory=lambda: 0.0)


class RiskManager:
    """Position sizing plus hard trading halts."""

    #: Stop trading after this many losses in a row until the day rolls over.
    MAX_CONSECUTIVE_LOSSES = 5

    #: Minimum seconds between entries, to stop runaway order loops.
    MIN_SECONDS_BETWEEN_TRADES = 1.0

    #: Upper bound on entries per day.
    MAX_TRADES_PER_DAY = 200

    def __init__(self, starting_equity: float | None = None, config: Settings = settings) -> None:
        self.config = config
        equity = float(starting_equity if starting_equity is not None else config.initial_capital)
        self.state = RiskState(
            starting_equity=equity,
            current_equity=equity,
            day_start_equity=equity,
            day_stamp=self._today(),
        )

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _roll_day_if_needed(self) -> None:
        """Reset daily counters at UTC midnight and lift a daily halt."""
        today = self._today()
        if today != self.state.day_stamp:
            logger.info("New trading day %s - resetting daily risk counters", today)
            self.state.day_stamp = today
            self.state.day_start_equity = self.state.current_equity
            self.state.realized_pnl_today = 0.0
            self.state.consecutive_losses = 0
            self.state.trades_today = 0
            if self.state.halt_reason in {"daily_loss_limit", "consecutive_losses"}:
                self.state.halted = False
                self.state.halt_reason = ""

    def update_equity(self, equity: float) -> None:
        self._roll_day_if_needed()
        self.state.current_equity = float(equity)
        self._check_daily_loss()

    def _check_daily_loss(self) -> None:
        if self.state.day_start_equity <= 0:
            return
        loss = (self.state.day_start_equity - self.state.current_equity) / self.state.day_start_equity
        if loss >= self.config.daily_loss_limit_fraction:
            self.halt(f"daily_loss_limit")
            logger.error(
                "DAILY LOSS LIMIT HIT: -%.2f%% (limit %.2f%%). Trading halted.",
                loss * 100, self.config.daily_loss_limit_fraction * 100,
            )

    def record_trade_result(self, pnl: float) -> None:
        self._roll_day_if_needed()
        self.state.realized_pnl_today += float(pnl)
        if pnl < 0:
            self.state.consecutive_losses += 1
            if self.state.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
                self.halt("consecutive_losses")
                logger.error(
                    "%d consecutive losses - trading halted until next day",
                    self.state.consecutive_losses,
                )
        else:
            self.state.consecutive_losses = 0

    def halt(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason

    def resume(self) -> None:
        """Manual override; intentionally not called automatically."""
        self.state.halted = False
        self.state.halt_reason = ""

    def can_trade(self) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for opening a new position."""
        self._roll_day_if_needed()

        if self.state.halted:
            return False, self.state.halt_reason
        if self.state.current_equity <= 0:
            return False, "no_equity"
        if self.state.open_positions >= self.config.max_open_positions:
            return False, "max_open_positions"
        if self.state.trades_today >= self.MAX_TRADES_PER_DAY:
            return False, "max_trades_per_day"
        if time.monotonic() - self.state.last_trade_time < self.MIN_SECONDS_BETWEEN_TRADES:
            return False, "rate_limited"
        return True, "ok"

    def position_size(self, price: float, leverage: int | None = None) -> float:
        """Quantity to trade, capped by equity fraction and max leverage."""
        if price <= 0:
            return 0.0
        leverage = min(leverage or self.config.default_leverage, self.config.max_leverage)
        notional = self.state.current_equity * self.config.max_position_fraction * leverage
        return max(notional / price, 0.0)

    def stop_loss_price(self, entry_price: float, side: int) -> float:
        offset = entry_price * self.config.stop_loss_fraction
        return entry_price - offset if side > 0 else entry_price + offset

    def take_profit_price(self, entry_price: float, side: int) -> float:
        offset = entry_price * self.config.take_profit_fraction
        return entry_price + offset if side > 0 else entry_price - offset

    def register_entry(self) -> None:
        self.state.open_positions += 1
        self.state.trades_today += 1
        self.state.last_trade_time = time.monotonic()

    def register_exit(self) -> None:
        self.state.open_positions = max(0, self.state.open_positions - 1)

    def snapshot(self) -> dict[str, object]:
        return {
            "equity": self.state.current_equity,
            "day_start_equity": self.state.day_start_equity,
            "realized_pnl_today": self.state.realized_pnl_today,
            "consecutive_losses": self.state.consecutive_losses,
            "open_positions": self.state.open_positions,
            "trades_today": self.state.trades_today,
            "halted": self.state.halted,
            "halt_reason": self.state.halt_reason,
        }
