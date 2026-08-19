import time

from agent_system.execution.risk_manager import RiskManager


def test_allows_trading_by_default(config) -> None:
    manager = RiskManager(1000.0, config)
    allowed, reason = manager.can_trade()

    assert allowed
    assert reason == "ok"


def test_limits_are_loose_enough_for_high_frequency_trading(config) -> None:
    """The risk layer must not be what stops the agent from trading often."""
    manager = RiskManager(1000.0, config)
    for _ in range(300):
        manager.register_trade()

    allowed, _ = manager.can_trade()
    assert allowed


def test_blocks_after_daily_loss_limit(config) -> None:
    manager = RiskManager(1000.0, config)
    manager.update_equity(1000.0 * (1 - config.daily_loss_limit_fraction) - 1)

    allowed, reason = manager.can_trade()

    assert not allowed
    assert reason == "daily_loss_limit"


def test_blocks_after_consecutive_losses(config) -> None:
    manager = RiskManager(1000.0, config)
    for _ in range(config.max_consecutive_losses):
        manager.register_result(-0.01)

    allowed, reason = manager.can_trade()

    assert not allowed
    assert reason == "max_consecutive_losses"


def test_a_win_resets_the_loss_streak(config) -> None:
    manager = RiskManager(1000.0, config)
    for _ in range(3):
        manager.register_result(-1.0)
    manager.register_result(5.0)

    assert manager.state.consecutive_losses == 0


def test_rate_limit_applies_only_when_configured(config) -> None:
    config.min_seconds_between_trades = 60.0
    manager = RiskManager(1000.0, config)
    manager.state.last_trade_time = time.time()

    allowed, reason = manager.can_trade()

    assert not allowed
    assert reason == "rate_limited"


def test_position_size_scales_with_equity_and_leverage(config) -> None:
    manager = RiskManager(10_000.0, config)

    single = manager.position_size(100.0, leverage=1)
    doubled = manager.position_size(100.0, leverage=2)

    assert doubled == single * 2


def test_position_size_is_capped_at_max_leverage(config) -> None:
    manager = RiskManager(10_000.0, config)

    requested = manager.position_size(100.0, leverage=999)
    capped = manager.position_size(100.0, leverage=config.max_leverage)

    assert requested == capped


def test_live_trading_without_credentials_is_blocked(config) -> None:
    config.enable_live_trading = True
    config.binance_api_key = None
    config.binance_api_secret = None
    manager = RiskManager(1000.0, config)

    allowed, reason = manager.can_trade()

    assert not allowed
    assert reason == "missing_credentials"
