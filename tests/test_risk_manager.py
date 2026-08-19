"""Risk manager tests -- the safety net that must never fail open."""

from __future__ import annotations

from agent_system.execution.risk_manager import RiskManager


def test_daily_loss_limit_halts_trading(test_settings) -> None:
    risk = RiskManager(1000.0, test_settings)
    assert risk.can_trade()[0]

    # Drop below the 5% daily limit.
    risk.update_equity(1000.0 * (1.0 - test_settings.daily_loss_limit_fraction - 0.01))

    allowed, reason = risk.can_trade()
    assert not allowed
    assert reason == "daily_loss_limit"


def test_consecutive_losses_halt_trading(test_settings) -> None:
    risk = RiskManager(1000.0, test_settings)

    for _ in range(RiskManager.MAX_CONSECUTIVE_LOSSES):
        risk.record_trade_result(-1.0)

    allowed, reason = risk.can_trade()
    assert not allowed
    assert reason == "consecutive_losses"


def test_a_win_resets_the_loss_streak(test_settings) -> None:
    risk = RiskManager(1000.0, test_settings)

    risk.record_trade_result(-1.0)
    risk.record_trade_result(-1.0)
    risk.record_trade_result(5.0)

    assert risk.state.consecutive_losses == 0
    assert risk.can_trade()[0]


def test_position_size_respects_max_leverage(test_settings) -> None:
    risk = RiskManager(1000.0, test_settings)

    quantity = risk.position_size(price=100.0, leverage=999)
    expected = 1000.0 * test_settings.max_position_fraction * test_settings.max_leverage / 100.0

    assert quantity == expected


def test_position_size_is_zero_for_invalid_price(test_settings) -> None:
    assert RiskManager(1000.0, test_settings).position_size(price=0.0) == 0.0


def test_stop_and_target_sit_on_the_correct_side(test_settings) -> None:
    risk = RiskManager(1000.0, test_settings)

    assert risk.stop_loss_price(100.0, side=1) < 100.0
    assert risk.take_profit_price(100.0, side=1) > 100.0
    assert risk.stop_loss_price(100.0, side=-1) > 100.0
    assert risk.take_profit_price(100.0, side=-1) < 100.0


def test_rate_limit_blocks_rapid_fire_entries(test_settings) -> None:
    risk = RiskManager(1000.0, test_settings)
    risk.register_entry()

    allowed, reason = risk.can_trade()
    assert not allowed
    assert reason == "rate_limited"


def test_no_equity_blocks_trading(test_settings) -> None:
    risk = RiskManager(1000.0, test_settings)
    risk.update_equity(0.0)

    assert not risk.can_trade()[0]
