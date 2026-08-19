"""Unified feature engineering.

This is the single feature pipeline for the whole system. Historical candles and
live websocket ticks are both converted to :class:`MarketTick` first, so training
and inference always see an identically shaped observation.

Design rule: **every feature is scale-free** (returns, ratios, z-scores, bounded
oscillators). Raw price levels such as ``close`` or ``ema_200`` are deliberately
excluded -- they are non-stationary, so a policy trained on BTC at 40k silently
breaks at 100k. Only stationary inputs generalise across regimes.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from agent_system.data.data_buffer import MarketTick

#: Number of values returned by :meth:`Features.as_vector`.
FEATURE_DIM = 14

#: Minimum ticks required before features carry real signal.
MIN_TICKS_FOR_FEATURES = 30


@dataclass(frozen=True)
class Features:
    """A stationary, scale-free view of recent market state."""

    ret_1: float = 0.0
    ret_5: float = 0.0
    ret_20: float = 0.0
    volatility: float = 0.0
    rsi: float = 0.0
    macd: float = 0.0
    macd_hist: float = 0.0
    bollinger_position: float = 0.0
    atr: float = 0.0
    volume_zscore: float = 0.0
    spread_bps: float = 0.0
    trend_ratio: float = 0.0
    range_position: float = 0.0
    tick_imbalance: float = 0.0

    def as_vector(self) -> list[float]:
        """Ordered feature values, safe to feed straight into a network."""
        return [_finite(value) for value in asdict(self).values()]

    def as_dict(self) -> dict[str, float]:
        return {key: _finite(value) for key, value in asdict(self).items()}


def _finite(value: float) -> float:
    """Replace NaN/inf with 0.0 and clamp extremes.

    A single NaN reaching the policy poisons every downstream gradient, so this
    guard is applied at the boundary rather than trusted to callers.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(-1e6, min(1e6, number))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = _mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _ema(values: Sequence[float], span: int) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


def _ema_series(values: Sequence[float], span: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (span + 1.0)
    out = [values[0]]
    for value in values[1:]:
        out.append(alpha * value + (1.0 - alpha) * out[-1])
    return out


def _pct_change(values: Sequence[float], lookback: int) -> float:
    if len(values) <= lookback:
        return 0.0
    past = values[-1 - lookback]
    if past == 0:
        return 0.0
    return (values[-1] - past) / past


class FeatureEngine:
    """Turns a window of ticks into a :class:`Features` snapshot."""

    def __init__(self, min_ticks: int = MIN_TICKS_FOR_FEATURES) -> None:
        self.min_ticks = min_ticks

    def compute(self, ticks: Sequence[MarketTick]) -> Features:
        """Compute features from ticks ordered oldest -> newest.

        Returns all-zero features when there is not enough history, which keeps
        the observation shape stable during warm-up instead of raising.
        """
        if len(ticks) < 2:
            return Features()

        prices = [tick.price for tick in ticks]
        volumes = [tick.volume for tick in ticks]
        last_price = prices[-1]
        if last_price <= 0:
            return Features()

        returns = [
            (prices[index] - prices[index - 1]) / prices[index - 1]
            for index in range(1, len(prices))
            if prices[index - 1] > 0
        ]

        return Features(
            ret_1=_pct_change(prices, 1),
            ret_5=_pct_change(prices, 5),
            ret_20=_pct_change(prices, 20),
            volatility=_stdev(returns[-20:]),
            rsi=self._rsi(prices),
            macd=self._macd(prices)[0],
            macd_hist=self._macd(prices)[1],
            bollinger_position=self._bollinger_position(prices),
            atr=self._atr(prices),
            volume_zscore=self._volume_zscore(volumes),
            spread_bps=self._spread_bps(ticks[-1]),
            trend_ratio=self._trend_ratio(prices),
            range_position=self._range_position(prices),
            tick_imbalance=self._tick_imbalance(returns),
        )

    def _rsi(self, prices: Sequence[float], period: int = 14) -> float:
        """Wilder RSI rescaled to [-1, 1] so it matches the other features."""
        if len(prices) <= period:
            return 0.0
        window = prices[-(period + 1):]
        gains = [max(window[i] - window[i - 1], 0.0) for i in range(1, len(window))]
        losses = [max(window[i - 1] - window[i], 0.0) for i in range(1, len(window))]
        average_gain = _mean(gains)
        average_loss = _mean(losses)
        if average_loss == 0:
            return 1.0 if average_gain > 0 else 0.0
        rsi = 100.0 - (100.0 / (1.0 + average_gain / average_loss))
        return (rsi - 50.0) / 50.0

    def _macd(self, prices: Sequence[float]) -> tuple[float, float]:
        """MACD and histogram, both normalised by price so they stay scale-free."""
        if len(prices) < 26:
            return 0.0, 0.0
        fast = _ema_series(prices, 12)
        slow = _ema_series(prices, 26)
        macd_line = [f - s for f, s in zip(fast, slow)]
        signal = _ema(macd_line[-9:], 9)
        reference = prices[-1] or 1.0
        return macd_line[-1] / reference, (macd_line[-1] - signal) / reference

    def _bollinger_position(self, prices: Sequence[float], period: int = 20) -> float:
        """Where price sits inside the bands: -1 lower, 0 middle, +1 upper."""
        if len(prices) < period:
            return 0.0
        window = prices[-period:]
        middle = _mean(window)
        deviation = _stdev(window)
        if deviation == 0:
            return 0.0
        return max(-3.0, min(3.0, (prices[-1] - middle) / (2.0 * deviation)))

    def _atr(self, prices: Sequence[float], period: int = 14) -> float:
        """Tick-level ATR proxy, normalised by price."""
        if len(prices) < period + 1:
            return 0.0
        moves = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        reference = prices[-1] or 1.0
        return _mean(moves[-period:]) / reference

    def _volume_zscore(self, volumes: Sequence[float], period: int = 20) -> float:
        if len(volumes) < period:
            return 0.0
        window = volumes[-period:]
        deviation = _stdev(window)
        if deviation == 0:
            return 0.0
        return max(-5.0, min(5.0, (volumes[-1] - _mean(window)) / deviation))

    def _spread_bps(self, tick: MarketTick) -> float:
        """Bid/ask spread in basis points -- a direct proxy for execution cost."""
        mid = (tick.bid + tick.ask) / 2.0
        if mid <= 0:
            return 0.0
        return max(0.0, min(1000.0, (tick.ask - tick.bid) / mid * 10_000.0))

    def _trend_ratio(self, prices: Sequence[float]) -> float:
        if len(prices) < 20:
            return 0.0
        slow = _mean(prices[-20:])
        if slow == 0:
            return 0.0
        return _mean(prices[-5:]) / slow - 1.0

    def _range_position(self, prices: Sequence[float], period: int = 20) -> float:
        """Position inside the recent high/low range, mapped to [-1, 1]."""
        if len(prices) < period:
            return 0.0
        window = prices[-period:]
        lowest, highest = min(window), max(window)
        if highest == lowest:
            return 0.0
        return ((prices[-1] - lowest) / (highest - lowest)) * 2.0 - 1.0

    def _tick_imbalance(self, returns: Sequence[float], period: int = 20) -> float:
        """Share of up-ticks vs down-ticks, mapped to [-1, 1]."""
        if not returns:
            return 0.0
        window = returns[-period:]
        ups = sum(1 for value in window if value > 0)
        downs = sum(1 for value in window if value < 0)
        total = ups + downs
        if total == 0:
            return 0.0
        return (ups - downs) / total
