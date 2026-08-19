"""Paper broker: virtual money, real prices.

This is the default execution backend and the one the agent should live in for
days before any real capital is involved. It applies the *same* fees, slippage
and risk checks as live trading, so paper results are a meaningful predictor
rather than a fantasy.

State is persisted to disk so a restart resumes the same virtual account.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agent_system.core.config import Settings, settings
from agent_system.core.logger import get_logger
from agent_system.execution.position_tracker import PositionTracker
from agent_system.execution.risk_manager import RiskManager

logger = get_logger(__name__)


class PaperBroker:
    """Simulated exchange with realistic costs and persistent balance."""

    def __init__(
        self,
        config: Settings = settings,
        risk_manager: RiskManager | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        self.config = config
        config.ensure_directories()
        self.symbol = config.symbol
        self.risk = risk_manager or RiskManager(config.initial_capital, config)
        self.positions = PositionTracker()
        self.state_path = Path(state_path or Path(config.state_dir) / "paper_account.json")

        self.balance = float(config.initial_capital)
        self.equity = self.balance
        self.fees_paid = 0.0
        self.trade_log: list[dict[str, Any]] = []
        self._load()

    # --- Persistence -----------------------------------------------------
    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.balance = float(payload.get("balance", self.balance))
            self.equity = float(payload.get("equity", self.balance))
            self.fees_paid = float(payload.get("fees_paid", 0.0))
            self.trade_log = payload.get("trade_log", [])[-500:]
            self.risk.update_equity(self.equity)
            logger.info("Resumed paper account: balance %.2f", self.balance)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Could not load paper state: %s", exc)

    def save(self) -> None:
        payload = {
            "balance": self.balance,
            "equity": self.equity,
            "fees_paid": self.fees_paid,
            "updated_at": time.time(),
            "trade_log": self.trade_log[-500:],
        }
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def reset(self) -> None:
        """Wipe the virtual account back to its starting capital."""
        self.balance = float(self.config.initial_capital)
        self.equity = self.balance
        self.fees_paid = 0.0
        self.trade_log = []
        self.positions = PositionTracker()
        self.risk = RiskManager(self.config.initial_capital, self.config)
        self.save()

    # --- Costs -----------------------------------------------------------
    def _fill_price(self, price: float, side: int) -> float:
        """Slippage always moves against us -- optimistic fills hide real losses."""
        return price * (1.0 + side * self.config.slippage_fraction)

    def _fee(self, price: float, quantity: float) -> float:
        return price * quantity * self.config.taker_fee_fraction

    # --- Trading ---------------------------------------------------------
    def open_position(self, direction: str, price: float) -> dict[str, Any]:
        direction = direction.upper()
        side = 1 if direction == "LONG" else -1

        allowed, reason = self.risk.can_trade()
        if not allowed:
            return {"status": "rejected", "reason": reason}
        if self.positions.has_position(self.symbol):
            return {"status": "rejected", "reason": "position_already_open"}

        fill = self._fill_price(price, side)
        quantity = self.risk.position_size(fill)
        if quantity <= 0:
            return {"status": "rejected", "reason": "zero_quantity"}

        fee = self._fee(fill, quantity)
        self.balance -= fee
        self.fees_paid += fee

        self.positions.open(
            symbol=self.symbol,
            side=side,
            entry_price=fill,
            quantity=quantity,
            stop_loss=self.risk.stop_loss_price(fill, side),
            take_profit=self.risk.take_profit_price(fill, side),
        )
        self.risk.register_entry()

        record = {
            "event": "open",
            "side": direction,
            "price": fill,
            "quantity": quantity,
            "fee": fee,
            "time": time.time(),
        }
        self.trade_log.append(record)
        logger.info("[PAPER] OPEN %s %.6f @ %.2f (fee %.4f)", direction, quantity, fill, fee)
        return {"status": "filled", **record}

    def close_position(self, price: float, reason: str = "signal") -> dict[str, Any]:
        position = self.positions.get(self.symbol)
        if position is None:
            return {"status": "rejected", "reason": "no_open_position"}

        fill = self._fill_price(price, -position.side)
        fee = self._fee(fill, position.quantity)
        gross = position.unrealized_pnl(fill)
        net = gross - fee

        self.balance += net
        self.fees_paid += fee
        self.positions.close(self.symbol, fill)
        self.risk.register_exit()
        self.risk.record_trade_result(net)
        self.equity = self.balance
        self.risk.update_equity(self.equity)

        record = {
            "event": "close",
            "reason": reason,
            "price": fill,
            "pnl": net,
            "fee": fee,
            "balance": self.balance,
            "time": time.time(),
        }
        self.trade_log.append(record)
        logger.info(
            "[PAPER] CLOSE @ %.2f | PnL %+.4f | balance %.2f (%s)",
            fill, net, self.balance, reason,
        )
        return {"status": "closed", **record}

    def mark_to_market(self, price: float) -> float:
        """Update equity from the current price and apply protective exits."""
        position = self.positions.get(self.symbol)
        if position is not None:
            hit_stop = position.stop_loss is not None and (
                (position.side > 0 and price <= position.stop_loss)
                or (position.side < 0 and price >= position.stop_loss)
            )
            hit_target = position.take_profit is not None and (
                (position.side > 0 and price >= position.take_profit)
                or (position.side < 0 and price <= position.take_profit)
            )
            if hit_stop or hit_target:
                self.close_position(price, "stop_loss" if hit_stop else "take_profit")
                return self.equity

        unrealized = position.unrealized_pnl(price) if position else 0.0
        self.equity = self.balance + unrealized
        self.risk.update_equity(self.equity)
        return self.equity

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": "paper",
            "balance": self.balance,
            "equity": self.equity,
            "fees_paid": self.fees_paid,
            "return_pct": (self.equity / self.config.initial_capital - 1.0) * 100.0,
            "positions": self.positions.snapshot(),
            "risk": self.risk.snapshot(),
            "trades": len(self.positions.closed),
        }
