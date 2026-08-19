"""Order manager tests -- orders must always respect exchange filters."""

from __future__ import annotations

from decimal import Decimal

from agent_system.execution.exchange_info import SymbolFilters
from agent_system.execution.order_manager import OrderManager
from agent_system.execution.position_tracker import PositionTracker
from agent_system.execution.risk_manager import RiskManager


class MockApi:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, int]:
        self.calls.append(("leverage", symbol, leverage))
        return {"leverage": leverage}

    def place_order(self, params: dict[str, object]) -> dict[str, object]:
        self.calls.append(("order", params))
        return {"orderId": 1, **params}


def build_manager(config, equity: float = 10_000.0) -> tuple[OrderManager, MockApi]:
    filters = SymbolFilters(
        symbol="BTCUSDT",
        tick_size=Decimal("0.10"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("100"),
    )
    api = MockApi()
    manager = OrderManager(
        api=api,
        risk_manager=RiskManager(starting_equity=equity, config=config),
        position_tracker=PositionTracker(),
        symbol_filters=filters,
        config=config,
    )
    return manager, api


def test_order_manager_quantizes_to_symbol_filters(test_settings) -> None:
    manager, api = build_manager(test_settings)

    result = manager.enter_limit("LONG", 62685.987)

    assert result["status"] == "submitted"
    assert api.calls[1][1]["price"] == "62685.9"  # snapped to the 0.10 tick grid
    # 10_000 equity * 2% * 3x leverage / 62685.9 = 0.00957 -> 0.009 on the 0.001 lot grid.
    assert api.calls[1][1]["quantity"] == "0.009"


def test_order_is_rejected_below_min_notional(test_settings) -> None:
    # Tiny equity produces a quantity worth less than the 100 minNotional.
    manager, api = build_manager(test_settings, equity=50.0)

    result = manager.enter_limit("LONG", 62685.987)

    assert result["status"] == "rejected"
    assert not any(call[0] == "order" for call in api.calls)


def test_second_entry_is_rejected_while_a_position_is_open(test_settings) -> None:
    manager, _ = build_manager(test_settings)
    manager.enter_limit("LONG", 62685.987)

    result = manager.enter_limit("LONG", 62685.987)

    assert result["status"] == "rejected"
    assert result["reason"] == "position_already_open"


def test_entry_is_blocked_when_risk_manager_halts(test_settings) -> None:
    manager, api = build_manager(test_settings)
    manager.risk.halt("manual_test")

    result = manager.enter_limit("LONG", 62685.987)

    assert result["status"] == "rejected"
    assert result["reason"] == "manual_test"
    assert not api.calls


def test_exit_market_closes_and_records_pnl(test_settings) -> None:
    manager, _ = build_manager(test_settings)
    manager.enter_limit("LONG", 60000.0)

    result = manager.exit_market(61000.0)

    assert result["status"] == "closed"
    assert result["record"]["pnl"] > 0
    assert not manager.positions.has_position("BTCUSDT")


def test_protective_exit_fires_on_stop_loss(test_settings) -> None:
    manager, _ = build_manager(test_settings)
    manager.enter_limit("LONG", 60000.0)
    position = manager.positions.get("BTCUSDT")

    result = manager.check_protective_exits(position.stop_loss - 1)

    assert result is not None
    assert result["status"] == "closed"


def test_protective_exit_does_nothing_inside_the_band(test_settings) -> None:
    manager, _ = build_manager(test_settings)
    manager.enter_limit("LONG", 60000.0)

    assert manager.check_protective_exits(60000.0) is None
