import time

from agent_system.data.data_buffer import MarketTick
from agent_system.data.feature_engineering import FeatureEngine


def test_feature_vector_has_expected_size() -> None:
    now = int(time.time() * 1000)
    ticks = [
        MarketTick(
            timestamp=now + index * 60_000,
            price=100.0 + index,
            volume=1.0 + index * 0.1,
            bid=99.99 + index,
            ask=100.01 + index,
        )
        for index in range(40)
    ]

    features = FeatureEngine().compute(ticks)

    assert len(features.as_vector()) == 14
    assert set(features.as_dict()).issuperset(
        {"macd", "atr", "bollinger_position", "volume_zscore"}
    )
