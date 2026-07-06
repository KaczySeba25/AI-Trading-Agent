"""In-memory rolling market data buffer."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable


@dataclass(frozen=True)
class MarketTick:
    timestamp: int
    price: float
    volume: float
    bid: float
    ask: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "timestamp": self.timestamp,
            "price": self.price,
            "volume": self.volume,
            "bid": self.bid,
            "ask": self.ask,
        }


class MarketDataBuffer:
    def __init__(self, window_seconds: int = 60) -> None:
        self.window_seconds = window_seconds
        self._ticks: Deque[MarketTick] = deque()

    def add(self, tick: MarketTick) -> None:
        self._ticks.append(tick)
        self.prune()

    def prune(self, now_ms: int | None = None) -> None:
        current_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        cutoff_ms = current_ms - (self.window_seconds * 1000)
        while self._ticks and self._ticks[0].timestamp < cutoff_ms:
            self._ticks.popleft()

    def latest(self) -> MarketTick | None:
        return self._ticks[-1] if self._ticks else None

    def snapshot(self) -> list[MarketTick]:
        return list(self._ticks)

    def __len__(self) -> int:
        self.prune()
        return len(self._ticks)

    def __iter__(self) -> Iterable[MarketTick]:
        return iter(self.snapshot())
