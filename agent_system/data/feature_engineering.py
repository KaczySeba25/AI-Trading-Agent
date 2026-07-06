"""Real-time market feature engineering."""

from __future__ import annotations

import math
from dataclasses import dataclass

from agent_system.data.data_buffer import MarketTick


@dataclass(frozen=True)
class MarketFeatures:
    return_1s: float
    return_5s: float
    return_10s: float
    volatility: float
    rsi_14: float
    vwap_distance: float
    order_book_imbalance: float

    def as_vector(self) -> list[float]:
        return [
            self.return_1s,
            self.return_5s,
            self.return_10s,
            self.volatility,
            self.rsi_14,
            self.vwap_distance,
            self.order_book_imbalance,
        ]

    def as_dict(self) -> dict[str, float]:
        return {
            "return_1s": self.return_1s,
            "return_5s": self.return_5s,
            "return_10s": self.return_10s,
            "volatility": self.volatility,
            "rsi_14": self.rsi_14,
            "vwap_distance": self.vwap_distance,
            "order_book_imbalance": self.order_book_imbalance,
        }


class FeatureEngine:
    """Computes non-blocking features from the rolling tick buffer."""

    def compute(self, ticks: list[MarketTick]) -> MarketFeatures:
        if not ticks:
            return self.empty()

        latest = ticks[-1]
        prices = [tick.price for tick in ticks]
        returns = self._price_returns(prices)

        return MarketFeatures(
            return_1s=self._return_since(ticks, latest.timestamp - 1_000),
            return_5s=self._return_since(ticks, latest.timestamp - 5_000),
            return_10s=self._return_since(ticks, latest.timestamp - 10_000),
            volatility=self._std(returns),
            rsi_14=self._rsi(prices, period=14),
            vwap_distance=self._vwap_distance(ticks),
            order_book_imbalance=self._order_book_imbalance(latest),
        )

    @staticmethod
    def empty() -> MarketFeatures:
        return MarketFeatures(
            return_1s=0.0,
            return_5s=0.0,
            return_10s=0.0,
            volatility=0.0,
            rsi_14=50.0,
            vwap_distance=0.0,
            order_book_imbalance=0.0,
        )

    @staticmethod
    def _return_since(ticks: list[MarketTick], target_timestamp: int) -> float:
        latest = ticks[-1]
        baseline = ticks[0]
        for tick in reversed(ticks):
            if tick.timestamp <= target_timestamp:
                baseline = tick
                break
        if baseline.price == 0:
            return 0.0
        return (latest.price - baseline.price) / baseline.price

    @staticmethod
    def _price_returns(prices: list[float]) -> list[float]:
        output: list[float] = []
        for previous, current in zip(prices, prices[1:]):
            if previous:
                output.append((current - previous) / previous)
        return output

    @staticmethod
    def _std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return math.sqrt(variance)

    @staticmethod
    def _rsi(prices: list[float], period: int) -> float:
        if len(prices) <= period:
            return 50.0

        deltas = [current - previous for previous, current in zip(prices, prices[1:])]
        recent = deltas[-period:]
        gains = [delta for delta in recent if delta > 0]
        losses = [-delta for delta in recent if delta < 0]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _vwap_distance(ticks: list[MarketTick]) -> float:
        volume_sum = sum(tick.volume for tick in ticks)
        if volume_sum == 0:
            return 0.0
        vwap = sum(tick.price * tick.volume for tick in ticks) / volume_sum
        latest_price = ticks[-1].price
        if vwap == 0:
            return 0.0
        return (latest_price - vwap) / vwap

    @staticmethod
    def _order_book_imbalance(tick: MarketTick) -> float:
        mid = (tick.bid + tick.ask) / 2.0
        spread = tick.ask - tick.bid
        if mid <= 0:
            return 0.0
        return spread / mid
