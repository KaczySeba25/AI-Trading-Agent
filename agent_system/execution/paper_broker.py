"""Paper trading account with disk-backed state.

Virtual money, real bookkeeping. Fees and slippage are charged exactly as the
environment charges them, so a paper session and a backtest of the same policy
over the same data agree. The account survives restarts, which is what makes a
multi-day paper run meaningful.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_system.core.config import Settings, settings
from agent_system.core.logger import get_logger

logger = get_logger(__name__)

#: Trades kept in the on-disk log. Enough to audit a session, bounded so the
#: file cannot grow without limit over days of running.
MAX_TRADE_LOG = 500


@dataclass
class PaperPosition:
    side: int
    entry_price: float
    quantity: float
    opened_at: float = field(default_factory=time.time)
    stop_loss: float | None = None
    take_profit: float | None = None

    def unrealized(self, price: float) -> float:
        return (price - self.entry_price) * self.quantity * self.side


class PaperBroker:
    """Simulated account that persists to ``state/paper_account.json``."""

    def __init__(self, config: Settings = settings, state_path: Path | None = None) -> None:
        self.config = config
        self.symbol = config.symbol
        self.state_path = Path(state_path or Path(config.state_dir) / "paper_account.json")
        self.fee_fraction = config.entry_fee_fraction
        self.slippage_fraction = config.slippage_fraction

        self.balance = float(config.initial_capital)
        self.equity = self.balance
        self.peak_equity = self.balance
        self.position: PaperPosition | None = None
        self.trade_log: list[dict[str, Any]] = []
        self.closed_trades = 0
        self.fees_paid = 0.0
        self.created_at = time.time()

        self.load()

    # ------------------------------------------------------------------
    def open_position(
        self,
        side: int,
        price: float,
        quantity: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        if self.position is not None:
            return {"status": "skipped", "reason": "position_already_open"}
        if quantity <= 0 or price <= 0:
            return {"status": "rejected", "reason": "invalid_size"}

        # Slippage is always adverse: we buy slightly above and sell slightly
        # below the quoted price. Modelling it as symmetric noise would make
        # the paper account flatter than reality.
        fill_price = price * (1.0 + self.slippage_fraction * side)
        fee = fill_price * quantity * self.fee_fraction
        self.balance -= fee
        self.fees_paid += fee

        self.position = PaperPosition(
            side=side,
            entry_price=fill_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self.mark_to_market(price)
        return {"status": "open", "fill_price": fill_price, "fee": fee}

    def close_position(self, price: float, reason: str = "signal") -> dict[str, Any]:
        position = self.position
        if position is None:
            return {"status": "skipped", "reason": "no_position"}

        fill_price = price * (1.0 - self.slippage_fraction * position.side)
        gross = (fill_price - position.entry_price) * position.quantity * position.side
        fee = fill_price * position.quantity * self.fee_fraction

        self.balance += gross - fee
        self.fees_paid += fee
        self.closed_trades += 1
        self.position = None

        record = {
            "timestamp": time.time(),
            "side": "long" if position.side > 0 else "short",
            "entry_price": round(position.entry_price, 4),
            "exit_price": round(fill_price, 4),
            "quantity": round(position.quantity, 8),
            "pnl": round(gross - fee, 6),
            "reason": reason,
        }
        self.trade_log.append(record)
        self.trade_log = self.trade_log[-MAX_TRADE_LOG:]

        self.mark_to_market(price)
        return {"status": "closed", "record": record}

    def mark_to_market(self, price: float) -> float:
        unrealized = self.position.unrealized(price) if self.position else 0.0
        self.equity = self.balance + unrealized
        self.peak_equity = max(self.peak_equity, self.equity)
        return self.equity

    def check_protective_exits(self, price: float) -> dict[str, Any] | None:
        position = self.position
        if position is None:
            return None
        if position.side > 0:
            if position.stop_loss is not None and price <= position.stop_loss:
                return self.close_position(price, "stop_loss")
            if position.take_profit is not None and price >= position.take_profit:
                return self.close_position(price, "take_profit")
        else:
            if position.stop_loss is not None and price >= position.stop_loss:
                return self.close_position(price, "stop_loss")
            if position.take_profit is not None and price <= position.take_profit:
                return self.close_position(price, "take_profit")
        return None

    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        drawdown = 0.0
        if self.peak_equity > 0:
            drawdown = max(0.0, (self.peak_equity - self.equity) / self.peak_equity)
        wins = [item for item in self.trade_log if item["pnl"] > 0]
        return {
            "symbol": self.symbol,
            "balance": round(self.balance, 4),
            "equity": round(self.equity, 4),
            "return_pct": round((self.equity / self.config.initial_capital - 1) * 100, 4),
            "position": asdict(self.position) if self.position else None,
            "closed_trades": self.closed_trades,
            "win_rate": round(len(wins) / len(self.trade_log), 4) if self.trade_log else 0.0,
            "fees_paid": round(self.fees_paid, 4),
            "drawdown": round(drawdown, 4),
            "recent_trades": self.trade_log[-5:],
        }

    def save(self) -> None:
        payload = {
            "symbol": self.symbol,
            "balance": self.balance,
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "position": asdict(self.position) if self.position else None,
            "trade_log": self.trade_log,
            "closed_trades": self.closed_trades,
            "fees_paid": self.fees_paid,
            "created_at": self.created_at,
            "updated_at": time.time(),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w", dir=str(self.state_path.parent), delete=False, suffix=".tmp",
            encoding="utf-8",
        )
        try:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(handle.name, self.state_path)
        except Exception:
            handle.close()
            Path(handle.name).unlink(missing_ok=True)
            raise

    def load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Paper account at %s unreadable (%s); starting fresh", self.state_path, exc)
            return

        self.balance = float(payload.get("balance", self.balance))
        self.equity = float(payload.get("equity", self.balance))
        self.peak_equity = float(payload.get("peak_equity", self.equity))
        self.trade_log = list(payload.get("trade_log", []))[-MAX_TRADE_LOG:]
        self.closed_trades = int(payload.get("closed_trades", 0))
        self.fees_paid = float(payload.get("fees_paid", 0.0))
        self.created_at = float(payload.get("created_at", time.time()))

        position = payload.get("position")
        if position:
            self.position = PaperPosition(**position)
        logger.info("Restored paper account: equity %.2f, %d closed trades",
                    self.equity, self.closed_trades)

    def reset(self) -> None:
        self.balance = float(self.config.initial_capital)
        self.equity = self.balance
        self.peak_equity = self.balance
        self.position = None
        self.trade_log = []
        self.closed_trades = 0
        self.fees_paid = 0.0
        self.created_at = time.time()
        self.save()
