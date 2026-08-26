"""Real-time market feature engineering.

Extended with three additional indicators:
  - Order Flow Imbalance (OFI): directional trade pressure from bid/ask changes
  - Garman-Klass Volatility (GK): high-low range volatility estimator
  - Fisher Transform: price-normalised oscillator sharpening reversal signals
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from agent_system.data.data_buffer import MarketTick


@dataclass(frozen=True)
class MarketFeatures:
    # --- Original 14 features ---
    return_1s: float
    return_5s: float
    return_10s: float
    volatility: float
    rsi_14: float
    vwap_distance: float
    order_book_imbalance: float
    ema_fast_distance: float
    ema_slow_distance: float
    macd: float
    atr: float
    bollinger_position: float
    volume_zscore: float
    candle_momentum: float
    # --- 3 new features ---
    ofi: float                  # Order Flow Imbalance
    gk_volatility: float        # Garman-Klass Volatility
    fisher_transform: float     # Fisher Transform oscillator

    def as_vector(self) -> list[float]:
        return [
            self.return_1s,
            self.return_5s,
            self.return_10s,
            self.volatility,
            self.rsi_14,
            self.vwap_distance,
            self.order_book_imbalance,
            self.ema_fast_distance,
            self.ema_slow_distance,
            self.macd,
            self.atr,
            self.bollinger_position,
            self.volume_zscore,
            self.candle_momentum,
            self.ofi,
            self.gk_volatility,
            self.fisher_transform,
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
            "ema_fast_distance": self.ema_fast_distance,
            "ema_slow_distance": self.ema_slow_distance,
            "macd": self.macd,
            "atr": self.atr,
            "bollinger_position": self.bollinger_position,
            "volume_zscore": self.volume_zscore,
            "candle_momentum": self.candle_momentum,
            "ofi": self.ofi,
            "gk_volatility": self.gk_volatility,
            "fisher_transform": self.fisher_transform,
        }


# Total feature count — used by GymTradingEnv observation_space
FEATURE_COUNT = 17


class FeatureEngine:
    """Computes non-blocking features from the rolling tick buffer."""

    def compute(self, ticks: list[MarketTick]) -> MarketFeatures:
        if not ticks:
            return self.empty()

        latest = ticks[-1]
        prices = [tick.price for tick in ticks]
        volumes = [tick.volume for tick in ticks]
        returns = self._price_returns(prices)
        ema_fast = self._ema(prices, 12)
        ema_slow = self._ema(prices, 26)

        return MarketFeatures(
            return_1s=self._return_since(ticks, latest.timestamp - 1_000),
            return_5s=self._return_since(ticks, latest.timestamp - 5_000),
            return_10s=self._return_since(ticks, latest.timestamp - 10_000),
            volatility=self._std(returns),
            rsi_14=self._rsi(prices, period=14),
            vwap_distance=self._vwap_distance(ticks),
            order_book_imbalance=self._order_book_imbalance(latest),
            ema_fast_distance=self._distance(latest.price, ema_fast),
            ema_slow_distance=self._distance(latest.price, ema_slow),
            macd=self._distance(ema_fast, ema_slow),
            atr=self._atr(prices, period=14),
            bollinger_position=self._bollinger_position(prices, period=20),
            volume_zscore=self._zscore(volumes),
            candle_momentum=self._candle_momentum(prices, lookback=20),
            ofi=self._order_flow_imbalance(ticks),
            gk_volatility=self._garman_klass_volatility(prices, period=20),
            fisher_transform=self._fisher_transform(prices, period=10),
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
            ema_fast_distance=0.0,
            ema_slow_distance=0.0,
            macd=0.0,
            atr=0.0,
            bollinger_position=0.0,
            volume_zscore=0.0,
            candle_momentum=0.0,
            ofi=0.0,
            gk_volatility=0.0,
            fisher_transform=0.0,
        )

    # ------------------------------------------------------------------
    # Original helpers
    # ------------------------------------------------------------------

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

    @staticmethod
    def _ema(values: list[float], period: int) -> float:
        if not values:
            return 0.0
        alpha = 2.0 / (period + 1)
        ema = values[0]
        for value in values[1:]:
            ema = alpha * value + (1.0 - alpha) * ema
        return ema

    @staticmethod
    def _distance(value: float, baseline: float) -> float:
        if baseline == 0:
            return 0.0
        return (value - baseline) / baseline

    @staticmethod
    def _atr(prices: list[float], period: int) -> float:
        if len(prices) <= 1:
            return 0.0
        changes = [abs(current - previous) for previous, current in zip(prices, prices[1:])]
        recent = changes[-period:]
        if not recent:
            return 0.0
        latest = prices[-1]
        if latest == 0:
            return 0.0
        return (sum(recent) / len(recent)) / latest

    @classmethod
    def _bollinger_position(cls, prices: list[float], period: int) -> float:
        if len(prices) < period:
            return 0.0
        recent = prices[-period:]
        mean = sum(recent) / len(recent)
        std = cls._std([price - mean for price in recent])
        if std == 0:
            return 0.0
        upper = mean + (2.0 * std)
        lower = mean - (2.0 * std)
        width = upper - lower
        if width == 0:
            return 0.0
        return ((prices[-1] - lower) / width) * 2.0 - 1.0

    @classmethod
    def _zscore(cls, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        recent = values[-30:]
        mean = sum(recent) / len(recent)
        std = cls._std(recent)
        if std == 0:
            return 0.0
        return (recent[-1] - mean) / std

    @staticmethod
    def _candle_momentum(prices: list[float], lookback: int) -> float:
        if len(prices) <= lookback or prices[-lookback] == 0:
            return 0.0
        return (prices[-1] - prices[-lookback]) / prices[-lookback]

    # ------------------------------------------------------------------
    # NEW FEATURES
    # ------------------------------------------------------------------

    @staticmethod
    def _order_flow_imbalance(ticks: list[MarketTick], lookback: int = 20) -> float:
        """Order Flow Imbalance (OFI).

        Measures directional pressure by comparing consecutive bid/ask changes.
        Positive OFI  => buying pressure dominant.
        Negative OFI  => selling pressure dominant.
        Result is normalised by total activity to stay in [-1, 1].
        """
        window = ticks[-lookback - 1:] if len(ticks) > lookback else ticks
        if len(window) < 2:
            return 0.0

        buy_pressure = 0.0
        sell_pressure = 0.0
        for prev, curr in zip(window, window[1:]):
            # Bid increased  => aggressive buyer arrived
            if curr.bid >= prev.bid:
                buy_pressure += curr.volume
            # Ask decreased  => aggressive seller arrived
            if curr.ask <= prev.ask:
                sell_pressure += curr.volume

        total = buy_pressure + sell_pressure
        if total == 0.0:
            return 0.0
        return (buy_pressure - sell_pressure) / total

    @staticmethod
    def _garman_klass_volatility(prices: list[float], period: int = 20) -> float:
        """Garman-Klass Volatility estimator.

        Uses rolling High/Low approximation from the tick prices.
        GK is more efficient than close-to-close std because it
        incorporates the intra-period range.
        Result is normalised (annualised-like factor removed) so it
        lives on the same scale as the existing `volatility` feature.
        """
        window = prices[-period:] if len(prices) >= period else prices
        if len(window) < 2:
            return 0.0

        # Approximate OHLC from sequential ticks
        high = max(window)
        low = min(window)
        open_price = window[0]
        close_price = window[-1]

        if open_price <= 0 or low <= 0:
            return 0.0

        log_hl = math.log(high / low) if low > 0 else 0.0
        log_co = math.log(close_price / open_price) if open_price > 0 else 0.0

        gk = 0.5 * (log_hl ** 2) - (2.0 * math.log(2.0) - 1.0) * (log_co ** 2)
        return math.sqrt(max(gk, 0.0))

    @classmethod
    def _fisher_transform(cls, prices: list[float], period: int = 10) -> float:
        """Fisher Transform.

        Converts prices to a Gaussian normal distribution, sharpening
        overbought/oversold turning points.  Output range is roughly
        (-3, +3); clipped to (-2.99, 2.99) to avoid log(0).
        """
        window = prices[-period:] if len(prices) >= period else prices
        if len(window) < 2:
            return 0.0

        highest = max(window)
        lowest = min(window)
        price_range = highest - lowest

        if price_range == 0.0:
            return 0.0

        # Normalise latest price to (-1, 1)
        raw = 2.0 * ((prices[-1] - lowest) / price_range) - 1.0
        # Clip strictly inside (-1, 1) to keep log finite
        raw = max(-0.999, min(0.999, raw))

        return 0.5 * math.log((1.0 + raw) / (1.0 - raw))
