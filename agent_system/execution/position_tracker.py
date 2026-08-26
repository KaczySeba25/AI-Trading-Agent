"""Position state tracking for execution safety."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LivePosition:
    symbol: str
    side: str
    entry_price: float
    quantity: float

    @property
    def is_open(self) -> bool:
        return self.quantity > 0 and self.side in {"LONG", "SHORT"}


class PositionTracker:
    def __init__(self) -> None:
        self._positions: dict[str, LivePosition] = {}

    def set_position(self, position: LivePosition) -> None:
        self._positions[position.symbol] = position

    def clear_position(self, symbol: str) -> None:
        self._positions.pop(symbol, None)

    def get_position(self, symbol: str) -> LivePosition | None:
        return self._positions.get(symbol)

    def open_positions_count(self) -> int:
        return sum(1 for position in self._positions.values() if position.is_open)

    def all_positions(self) -> list[LivePosition]:
        return list(self._positions.values())
