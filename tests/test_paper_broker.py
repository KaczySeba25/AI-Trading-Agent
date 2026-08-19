"""Paper broker tests -- virtual money must behave like real money."""

from __future__ import annotations

import pytest

from agent_system.execution.paper_broker import PaperBroker


def test_open_and_close_round_trip_costs_fees(test_settings) -> None:
    broker = PaperBroker(test_settings)

    assert broker.open_position("LONG", 40_000.0)["status"] == "filled"
    result = broker.close_position(40_000.0, "test")

    assert result["status"] == "closed"
    # Same price in and out, so the account can only be down by costs.
    assert broker.balance < test_settings.initial_capital
    assert broker.fees_paid > 0


def test_profitable_long_increases_balance(test_settings) -> None:
    broker = PaperBroker(test_settings)
    broker.open_position("LONG", 40_000.0)
    broker.close_position(42_000.0, "test")

    assert broker.balance > test_settings.initial_capital


def test_profitable_short_increases_balance(test_settings) -> None:
    broker = PaperBroker(test_settings)
    broker.open_position("SHORT", 40_000.0)
    broker.close_position(38_000.0, "test")

    assert broker.balance > test_settings.initial_capital


def test_cannot_open_two_positions_at_once(test_settings) -> None:
    broker = PaperBroker(test_settings)
    broker.open_position("LONG", 40_000.0)

    second = broker.open_position("LONG", 40_000.0)
    assert second["status"] == "rejected"


def test_stop_loss_triggers_on_mark_to_market(test_settings) -> None:
    broker = PaperBroker(test_settings)
    broker.open_position("LONG", 40_000.0)
    entry = broker.positions.get(test_settings.symbol).entry_price

    # Move well beyond the stop.
    broker.mark_to_market(entry * (1.0 - test_settings.stop_loss_fraction * 2))

    assert not broker.positions.has_position(test_settings.symbol)
    assert any(entry["event"] == "close" for entry in broker.trade_log)


def test_take_profit_triggers_on_mark_to_market(test_settings) -> None:
    broker = PaperBroker(test_settings)
    broker.open_position("LONG", 40_000.0)
    entry = broker.positions.get(test_settings.symbol).entry_price

    broker.mark_to_market(entry * (1.0 + test_settings.take_profit_fraction * 2))

    assert not broker.positions.has_position(test_settings.symbol)


def test_slippage_always_moves_against_the_trader(test_settings) -> None:
    broker = PaperBroker(test_settings)

    broker.open_position("LONG", 40_000.0)
    assert broker.positions.get(test_settings.symbol).entry_price > 40_000.0

    broker.close_position(40_000.0)
    broker.risk.state.last_trade_time = 0.0  # bypass the rate limiter

    broker.open_position("SHORT", 40_000.0)
    assert broker.positions.get(test_settings.symbol).entry_price < 40_000.0


def test_state_survives_a_restart(test_settings) -> None:
    broker = PaperBroker(test_settings)
    broker.open_position("LONG", 40_000.0)
    broker.close_position(41_000.0)
    broker.save()
    balance = broker.balance

    restarted = PaperBroker(test_settings)
    assert restarted.balance == pytest.approx(balance)


def test_reset_restores_starting_capital(test_settings) -> None:
    broker = PaperBroker(test_settings)
    broker.open_position("LONG", 40_000.0)
    broker.close_position(30_000.0)
    assert broker.balance < test_settings.initial_capital

    broker.reset()
    assert broker.balance == test_settings.initial_capital
