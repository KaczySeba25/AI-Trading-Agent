from agent_system.execution.paper_broker import PaperBroker


def test_open_and_close_records_a_trade(config) -> None:
    broker = PaperBroker(config)
    broker.open_position(side=1, price=100.0, quantity=1.0)
    result = broker.close_position(110.0)

    assert result["status"] == "closed"
    assert result["record"]["pnl"] > 0
    assert broker.closed_trades == 1


def test_slippage_always_works_against_the_trader(config) -> None:
    """Buy fills above the quote, sell fills below it -- never the reverse."""
    broker = PaperBroker(config)
    broker.open_position(side=1, price=100.0, quantity=1.0)
    assert broker.position.entry_price > 100.0

    broker.close_position(100.0)
    broker.open_position(side=-1, price=100.0, quantity=1.0)
    assert broker.position.entry_price < 100.0


def test_round_trip_at_a_flat_price_loses_money(config) -> None:
    """Costs are real: opening and closing at the same price must lose."""
    broker = PaperBroker(config)
    starting_equity = broker.equity
    broker.open_position(side=1, price=100.0, quantity=1.0)
    broker.close_position(100.0)

    assert broker.equity < starting_equity


def test_state_survives_a_restart(config) -> None:
    broker = PaperBroker(config)
    broker.open_position(side=1, price=100.0, quantity=0.5)
    broker.close_position(105.0)
    equity = broker.equity
    broker.save()

    restored = PaperBroker(config)

    assert restored.closed_trades == 1
    assert abs(restored.equity - equity) < 1e-9


def test_open_position_survives_a_restart(config) -> None:
    broker = PaperBroker(config)
    broker.open_position(side=-1, price=100.0, quantity=0.25)
    broker.save()

    restored = PaperBroker(config)

    assert restored.position is not None
    assert restored.position.side == -1


def test_protective_exit_triggers_on_stop(config) -> None:
    broker = PaperBroker(config)
    broker.open_position(side=1, price=100.0, quantity=1.0, stop_loss=99.0, take_profit=105.0)

    assert broker.check_protective_exits(99.5) is None
    result = broker.check_protective_exits(98.5)

    assert result is not None
    assert result["record"]["reason"] == "stop_loss"


def test_reset_restores_initial_capital(config) -> None:
    broker = PaperBroker(config)
    broker.open_position(side=1, price=100.0, quantity=1.0)
    broker.close_position(80.0)
    broker.reset()

    assert broker.equity == config.initial_capital
    assert broker.closed_trades == 0
