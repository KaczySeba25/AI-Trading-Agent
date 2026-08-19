"""Tracks open positions."""

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
        return (price - self.entry_price) * self.quantity * self.side

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": "long" if self.side > 0 else "short",
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "opened_at": self.opened_at,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }


class PositionTracker:
    """In-memory registry of open positions, keyed by symbol."""

    def __init__(self) -> None:
        self._positions: dict[str, TrackedPosition] = {}

    def open(self, position: TrackedPosition) -> TrackedPosition:
        self._positions[position.symbol] = position
        return position

    def close(self, symbol: str) -> TrackedPosition | None:
        return self._positions.pop(symbol, None)

    def get(self, symbol: str) -> TrackedPosition | None:
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def total_unrealized(self, prices: dict[str, float]) -> float:
        return sum(
            position.unrealized_pnl(prices.get(symbol, position.entry_price))
            for symbol, position in self._positions.items()
        )

    def snapshot(self) -> list[dict[str, Any]]:
        return [position.as_dict() for position in self._positions.values()]

    def __len__(self) -> int:
        return len(self._positions)
