import math

from agent_system.data.data_buffer import MarketTick
from agent_system.data.feature_engineering import FEATURE_DIM, FeatureEngine

from conftest import make_ticks


def test_feature_vector_has_expected_size() -> None:
    features = FeatureEngine().compute(make_ticks(60))

    assert len(features.as_vector()) == FEATURE_DIM
    assert set(features.as_dict()).issuperset(
        {"macd", "atr", "bollinger_position", "volume_zscore"}
    )


def test_features_are_finite_even_with_degenerate_input() -> None:
    """Flat prices produce zero-division candidates -- none may leak through."""
    flat = [
        MarketTick(timestamp=index * 1000, price=100.0, volume=0.0, bid=100.0, ask=100.0)
        for index in range(50)
    ]

    for value in FeatureEngine().compute(flat).as_vector():
        assert math.isfinite(value)


def test_insufficient_history_returns_zeros() -> None:
    assert FeatureEngine().compute([]).as_vector() == [0.0] * FEATURE_DIM


def test_features_are_scale_free() -> None:
    """The same shape at 40k and 400k must produce near-identical features.

    This is the property that lets a policy trained at one price level keep
    working at another.
    """
    cheap = FeatureEngine().compute(make_ticks(200, start_price=40_000.0)).as_vector()
    expensive = FeatureEngine().compute(make_ticks(200, start_price=400_000.0)).as_vector()

    for left, right in zip(cheap, expensive):
        assert abs(left - right) < 0.05


def test_rsi_is_bounded() -> None:
    rising = make_ticks(120, trend=0.01, volatility=0.0001)
    features = FeatureEngine().compute(rising)

    assert -1.0 <= features.rsi <= 1.0
    assert features.rsi > 0  # a strong uptrend must read as overbought
