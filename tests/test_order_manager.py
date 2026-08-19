from decimal import Decimal

from agent_system.execution.exchange_info import SymbolFilters
from agent_system.execution.order_manager import OrderManager
from agent_system.execution.position_tracker import PositionTracker
from agent_system.execution.risk_manager import RiskManager


class MockApi:
    def __init__(self) -> None:
        self.calls = []

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, int]:
        self.calls.append(("leverage", symbol, leverage))
        return {"leverage": leverage}

    def place_order(self, params: dict[str, object]) -> dict[str, object]:
        self.calls.append(("order", params))
        return {"orderId": 1, **params}


def test_order_manager_uses_symbol_filters() -> None:
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
        risk_manager=RiskManager(starting_equity=10_000),
        position_tracker=PositionTracker(),
        symbol_filters=filters,
    )

    result = manager.enter_limit("LONG", 62685.987)

    assert result["status"] == "submitted"
    assert api.calls[1][1]["price"] == "62685.9"
    assert api.calls[1][1]["quantity"] == "0.003"
