"""Tests for OFI, Garman-Klass Volatility and Fisher Transform.

Run with:  pytest tests/test_new_features.py -v
"""

from __future__ import annotations

import math
import pytest

from agent_system.data.data_buffer import MarketTick
from agent_system.data.feature_engineering import FEATURE_COUNT, FeatureEngine, MarketFeatures


# ── helpers ───────────────────────────────────────────────────────────────────

def make_tick(
    price: float,
    volume: float = 1.0,
    bid: float | None = None,
    ask: float | None = None,
    ts: int = 0,
) -> MarketTick:
    return MarketTick(
        timestamp=ts,
        price=price,
        volume=volume,
        bid=bid if bid is not None else price - 0.5,
        ask=ask if ask is not None else price + 0.5,
    )


def rising_ticks(n: int = 30, start: float = 100.0, step: float = 0.5) -> list[MarketTick]:
    return [make_tick(start + i * step, ts=i * 1000) for i in range(n)]


def flat_ticks(n: int = 30, price: float = 100.0) -> list[MarketTick]:
    return [make_tick(price, ts=i * 1000) for i in range(n)]


engine = FeatureEngine()


# ── FEATURE_COUNT ─────────────────────────────────────────────────────────────

class TestFeatureCount:
    def test_constant_equals_vector_length(self) -> None:
        ticks = rising_ticks()
        feat = engine.compute(ticks)
        assert len(feat.as_vector()) == FEATURE_COUNT

    def test_empty_vector_length(self) -> None:
        feat = FeatureEngine.empty()
        assert len(feat.as_vector()) == FEATURE_COUNT

    def test_constant_value(self) -> None:
        assert FEATURE_COUNT == 17


# ── Order Flow Imbalance ─────────────────────────────────────────────────────

class TestOrderFlowImbalance:
    def test_range(self) -> None:
        ticks = rising_ticks()
        feat = engine.compute(ticks)
        assert -1.0 <= feat.ofi <= 1.0

    def test_buying_pressure(self) -> None:
        """Rising bids should produce positive OFI."""
        ticks = [
            make_tick(100.0, bid=99.0, ask=101.0, ts=0),
            make_tick(100.5, bid=99.5, ask=101.5, ts=1000),
            make_tick(101.0, bid=100.0, ask=102.0, ts=2000),
        ]
        feat = engine.compute(ticks)
        assert feat.ofi > 0.0

    def test_selling_pressure(self) -> None:
        """Falling asks should produce negative OFI."""
        ticks = [
            make_tick(100.0, bid=99.0, ask=101.0, ts=0),
            make_tick(99.5,  bid=98.5, ask=100.5, ts=1000),
            make_tick(99.0,  bid=98.0, ask=100.0, ts=2000),
        ]
        feat = engine.compute(ticks)
        assert feat.ofi < 0.0

    def test_single_tick_returns_zero(self) -> None:
        feat = engine.compute([make_tick(100.0)])
        assert feat.ofi == 0.0

    def test_empty_returns_zero(self) -> None:
        assert FeatureEngine.empty().ofi == 0.0


# ── Garman-Klass Volatility ───────────────────────────────────────────────────

class TestGarmanKlassVolatility:
    def test_non_negative(self) -> None:
        ticks = rising_ticks()
        feat = engine.compute(ticks)
        assert feat.gk_volatility >= 0.0

    def test_flat_market_near_zero(self) -> None:
        ticks = flat_ticks()
        feat = engine.compute(ticks)
        assert feat.gk_volatility == pytest.approx(0.0, abs=1e-10)

    def test_volatile_greater_than_calm(self) -> None:
        calm_ticks = [make_tick(100.0 + (i % 2) * 0.01, ts=i * 1000) for i in range(30)]
        volatile_ticks = [make_tick(100.0 + (i % 2) * 5.0,  ts=i * 1000) for i in range(30)]
        calm_feat    = engine.compute(calm_ticks)
        volatile_feat = engine.compute(volatile_ticks)
        assert volatile_feat.gk_volatility > calm_feat.gk_volatility

    def test_empty_returns_zero(self) -> None:
        assert FeatureEngine.empty().gk_volatility == 0.0


# ── Fisher Transform ──────────────────────────────────────────────────────────

class TestFisherTransform:
    def test_finite(self) -> None:
        ticks = rising_ticks()
        feat = engine.compute(ticks)
        assert math.isfinite(feat.fisher_transform)

    def test_range_roughly_bounded(self) -> None:
        """Fisher output stays within (-3, 3) for normal price series."""
        ticks = rising_ticks(50)
        feat = engine.compute(ticks)
        assert -3.0 < feat.fisher_transform < 3.0

    def test_overbought_positive(self) -> None:
        """Price at top of window range → positive Fisher."""
        # Create a window where last price is the highest
        ticks = [make_tick(90.0 + i, ts=i * 1000) for i in range(15)]
        feat = engine.compute(ticks)
        assert feat.fisher_transform > 0.0

    def test_oversold_negative(self) -> None:
        """Price at bottom of window range → negative Fisher."""
        ticks = [make_tick(90.0 + (14 - i), ts=i * 1000) for i in range(15)]
        feat = engine.compute(ticks)
        assert feat.fisher_transform < 0.0

    def test_flat_market_zero(self) -> None:
        ticks = flat_ticks()
        feat = engine.compute(ticks)
        assert feat.fisher_transform == pytest.approx(0.0, abs=1e-6)

    def test_empty_returns_zero(self) -> None:
        assert FeatureEngine.empty().fisher_transform == 0.0


# ── as_dict keys ──────────────────────────────────────────────────────────────

class TestAsDictKeys:
    def test_new_keys_present(self) -> None:
        feat = FeatureEngine.empty()
        d = feat.as_dict()
        assert "ofi" in d
        assert "gk_volatility" in d
        assert "fisher_transform" in d

    def test_dict_and_vector_same_length(self) -> None:
        feat = FeatureEngine.empty()
        assert len(feat.as_dict()) == len(feat.as_vector()) == FEATURE_COUNT
