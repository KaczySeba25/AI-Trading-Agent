"""Tracks open positions and realised results."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackedPosition:
    symbol: str
    side: int  # +1 long, -1 short
    entry_price: float
    quantity: float
    opened_at: float = field(default_factory=time.time)
    stop_loss: float | None = None
    take_profit: float | None = None

    def unrealized_pnl(self, price: float) -> float:
        return self.side * (price - self.entry_price) * self.quantity

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": "LONG" if self.side > 0 else "SHORT",
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "opened_at": self.opened_at,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }


class PositionTracker:
    """In-memory record of open and closed positions."""

    def __init__(self) -> None:
        self.positions: dict[str, TrackedPosition] = {}
        self.closed: list[dict[str, Any]] = []
        self.realized_pnl = 0.0

    def open(
        self,
        symbol: str,
        side: int,
        entry_price: float,
        quantity: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> TrackedPosition:
        position = TrackedPosition(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self.positions[symbol] = position
        return position

    def close(self, symbol: str, exit_price: float) -> dict[str, Any] | None:
        position = self.positions.pop(symbol, None)
        if position is None:
            return None

        pnl = position.unrealized_pnl(exit_price)
        self.realized_pnl += pnl
        record = {
            **position.as_dict(),
            "exit_price": exit_price,
            "pnl": pnl,
            "closed_at": time.time(),
        }
        self.closed.append(record)
        return record

    def get(self, symbol: str) -> TrackedPosition | None:
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def total_unrealized(self, prices: dict[str, float]) -> float:
        return sum(
            position.unrealized_pnl(prices.get(symbol, position.entry_price))
            for symbol, position in self.positions.items()
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "open": [position.as_dict() for position in self.positions.values()],
            "closed_count": len(self.closed),
            "realized_pnl": self.realized_pnl,
        }
