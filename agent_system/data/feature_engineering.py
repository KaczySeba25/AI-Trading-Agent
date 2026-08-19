"""Feature engineering for the trading agent.

Design rules, in order of importance:

1. **Every feature is scale-free.** A network trained when BTC was 42,000 must
   still work at 90,000, so nothing here is a raw price. Returns are relative,
   oscillators are bounded, and dispersion measures are divided by price.
2. **Every feature is finite.** Flat markets produce zero-variance windows and
   division by zero; each helper falls back to a neutral value rather than
   emitting NaN, because a single NaN poisons an entire training batch.
3. **The vector order is frozen.** ``FEATURE_NAMES`` is the contract between
   the feature engine, the environment observation space and every saved
   model. Appending is safe; reordering silently invalidates saved models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from agent_system.data.data_buffer import MarketTick

FEATURE_NAMES: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_20",
    "volatility",
    "rsi",
    "macd",
    "macd_hist",
    "bollinger_position",
    "atr",
    "volume_zscore",
    "spread_bps",
    "trend_ratio",
    "range_position",
    "tick_imbalance",
)

FEATURE_DIM = len(FEATURE_NAMES)

#: Below this many ticks the indicators are not meaningful and the engine
#: returns a zero vector instead of a noisy one.
MIN_TICKS_FOR_FEATURES = 30


@dataclass
class FeatureSet:
    """One feature observation, addressable by name or as a vector."""

    values: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)

    def as_vector(self) -> np.ndarray:
        return np.array(
            [self.values.get(name, 0.0) for name in FEATURE_NAMES],
            dtype=np.float32,
        )

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def __contains__(self, key: str) -> bool:
        return key in self.values


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return default
    result = numerator / denominator
    return result if math.isfinite(result) else default


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(low, min(high, value))


def _ema(values: np.ndarray, span: int) -> float:
    """Exponential moving average of the final point of ``values``."""
    if values.size == 0:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    weights = (1.0 - alpha) ** np.arange(values.size - 1, -1, -1, dtype=np.float64)
    return float(np.sum(values * weights) / np.sum(weights))


def _relative_strength_index(prices: np.ndarray, period: int = 14) -> float:
    """RSI mapped to [-1, 1] instead of the usual [0, 100].

    Keeping every feature centred on zero and bounded by one means no input
    dominates the first layer of the network purely because of its units.
    """
    if prices.size < period + 1:
        return 0.0
    deltas = np.diff(prices[-(period + 1) :])
    gains = float(np.sum(np.clip(deltas, 0.0, None)))
    losses = float(-np.sum(np.clip(deltas, None, 0.0)))
    total = gains + losses
    if total == 0:
        return 0.0
    # (gains - losses) / (gains + losses) is algebraically the classic RSI
    # rescaled to [-1, 1], with the zero-division case already handled.
    return _clip(_safe_divide(gains - losses, total))


def _average_true_range(prices: np.ndarray, period: int = 14) -> float:
    """ATR as a fraction of price, so it is comparable across regimes."""
    if prices.size < 2:
        return 0.0
    window = prices[-(period + 1) :]
    true_ranges = np.abs(np.diff(window))
    if true_ranges.size == 0:
        return 0.0
    atr = float(np.mean(true_ranges))
    return _safe_divide(atr, float(prices[-1]))


class FeatureEngine:
    """Turns a sequence of ticks into a fixed-width feature vector."""

    def __init__(self, min_ticks: int = MIN_TICKS_FOR_FEATURES) -> None:
        self.min_ticks = min_ticks

    def compute(self, ticks: Sequence[MarketTick]) -> FeatureSet:
        if len(ticks) < self.min_ticks:
            return FeatureSet({name: 0.0 for name in FEATURE_NAMES})

        prices = np.array([tick.price for tick in ticks], dtype=np.float64)
        volumes = np.array([tick.volume for tick in ticks], dtype=np.float64)
        bids = np.array([tick.bid for tick in ticks], dtype=np.float64)
        asks = np.array([tick.ask for tick in ticks], dtype=np.float64)

        last_price = float(prices[-1])
        returns = np.diff(prices) / np.maximum(prices[:-1], 1e-12)

        values: dict[str, float] = {}

        # --- momentum over three horizons ---------------------------------
        # Scaled by 100 so a 1% move reads as 1.0 rather than 0.01; that keeps
        # the inputs in the same order of magnitude as the bounded features.
        values["ret_1"] = _clip(float(returns[-1]) * 100.0, -10.0, 10.0)
        values["ret_5"] = _clip(
            _safe_divide(last_price - float(prices[-6]), float(prices[-6])) * 100.0,
            -10.0,
            10.0,
        )
        values["ret_20"] = _clip(
            _safe_divide(last_price - float(prices[-21]), float(prices[-21])) * 100.0,
            -20.0,
            20.0,
        )

        # --- dispersion ---------------------------------------------------
        recent_returns = returns[-20:]
        values["volatility"] = _clip(float(np.std(recent_returns)) * 100.0, 0.0, 10.0)

        # --- oscillators ----------------------------------------------------
        values["rsi"] = _relative_strength_index(prices)

        ema_fast = _ema(prices[-26:], 12)
        ema_slow = _ema(prices[-26:], 26)
        macd = _safe_divide(ema_fast - ema_slow, last_price) * 100.0
        values["macd"] = _clip(macd, -5.0, 5.0)

        # MACD histogram: the signal line is the EMA of the MACD series, but
        # recomputing the full series every tick is wasteful. The difference
        # between the current MACD and a slower EMA of price captures the same
        # acceleration at a fraction of the cost.
        ema_signal = _ema(prices[-35:], 35) if prices.size >= 35 else ema_slow
        macd_signal = _safe_divide(ema_slow - ema_signal, last_price) * 100.0
        values["macd_hist"] = _clip(macd - macd_signal, -5.0, 5.0)

        # --- Bollinger position --------------------------------------------
        window = prices[-20:]
        mean = float(np.mean(window))
        std = float(np.std(window))
        values["bollinger_position"] = _clip(_safe_divide(last_price - mean, 2.0 * std))

        values["atr"] = _clip(_average_true_range(prices) * 100.0, 0.0, 10.0)

        # --- volume ----------------------------------------------------------
        volume_window = volumes[-20:]
        volume_mean = float(np.mean(volume_window))
        volume_std = float(np.std(volume_window))
        values["volume_zscore"] = _clip(
            _safe_divide(float(volumes[-1]) - volume_mean, volume_std), -5.0, 5.0
        )

        # --- microstructure ---------------------------------------------------
        spread = float(asks[-1] - bids[-1])
        values["spread_bps"] = _clip(_safe_divide(spread, last_price) * 10_000.0, 0.0, 100.0)

        short_mean = float(np.mean(prices[-5:]))
        long_mean = float(np.mean(prices[-20:]))
        values["trend_ratio"] = _clip(_safe_divide(short_mean - long_mean, long_mean) * 100.0, -5.0, 5.0)

        window_high = float(np.max(prices[-20:]))
        window_low = float(np.min(prices[-20:]))
        span = window_high - window_low
        # Where in the recent range we sit: -1 at the low, +1 at the high.
        values["range_position"] = _clip(
            2.0 * _safe_divide(last_price - window_low, span, 0.5) - 1.0
        )

        # Sign-weighted volume over the last 20 ticks: are buyers or sellers
        # the ones actually moving size?
        directions = np.sign(np.diff(prices[-21:]))
        recent_volumes = volumes[-directions.size :]
        signed = float(np.sum(directions * recent_volumes))
        total_volume = float(np.sum(recent_volumes))
        values["tick_imbalance"] = _clip(_safe_divide(signed, total_volume))

        return FeatureSet(values)

    def compute_many(self, ticks: Sequence[MarketTick], window: int) -> np.ndarray:
        """Feature matrix for every position in ``ticks``.

        Training replays the same history many times, so the features are
        computed once up front and reused. The result has one row per tick;
        rows before ``window`` are zero-filled because there is not enough
        history behind them to be meaningful.
        """
        total = len(ticks)
        matrix = np.zeros((total, FEATURE_DIM), dtype=np.float32)
        if total == 0:
            return matrix

        span = max(window, self.min_ticks)
        for index in range(total):
            start = max(0, index - span + 1)
            slice_ = ticks[start : index + 1]
            if len(slice_) < self.min_ticks:
                continue
            matrix[index] = self.compute(slice_).as_vector()
        return matrix
