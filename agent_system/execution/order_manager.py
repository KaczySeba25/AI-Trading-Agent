"""Safe order construction for Binance Futures Testnet."""

from __future__ import annotations

from agent_system.core.config import Settings, settings
from agent_system.core.exceptions import TradingAgentError
from agent_system.execution.exchange_info import ExchangeInfoClient, SymbolFilters
from agent_system.execution.position_tracker import LivePosition, PositionTracker
from agent_system.execution.risk_manager import RiskManager
from agent_system.execution.testnet_api import BinanceTestnetClient


class OrderManager:
    def __init__(
        self,
        api: BinanceTestnetClient,
        risk_manager: RiskManager,
        position_tracker: PositionTracker,
        symbol_filters: SymbolFilters | None = None,
        config: Settings = settings,
    ) -> None:
        self.api = api
        self.risk_manager = risk_manager
        self.position_tracker = position_tracker
        self.symbol_filters = symbol_filters
        self.config = config

    def enter_limit(self, side: str, limit_price: float) -> dict[str, object]:
        normalized_side = side.upper()
        if normalized_side not in {"LONG", "SHORT"}:
            return {"status": "rejected", "reason": "invalid_side"}

        if self.position_tracker.get_position(self.config.symbol):
            return {"status": "rejected", "reason": "position_already_open"}

        capacity = self.risk_manager.assess_position_capacity(
            self.position_tracker.open_positions_count()
        )
        if not capacity.allowed:
            return {"status": "rejected", "reason": capacity.reason}

        decision = self.risk_manager.assess_entry(limit_price)
        if not decision.allowed:
            return {"status": "rejected", "reason": decision.reason}

        try:
            price, quantity = self._normalize_order(limit_price, decision.quantity)
        except TradingAgentError as exc:
            return {"status": "rejected", "reason": str(exc)}

        self.api.set_leverage(self.config.symbol, decision.leverage)
        order_side = "BUY" if normalized_side == "LONG" else "SELL"
        response = self.api.place_order(
            {
                "symbol": self.config.symbol,
                "side": order_side,
                "type": "LIMIT",
                "timeInForce": "GTC",
                "quantity": quantity,
                "price": price,
            }
        )
        self.position_tracker.set_position(
            LivePosition(
                symbol=self.config.symbol,
                side=normalized_side,
                entry_price=float(price),
                quantity=float(quantity),
            )
        )
        return {"status": "submitted", "response": response}

    def exit_market(self) -> dict[str, object]:
        position = self.position_tracker.get_position(self.config.symbol)
        if not position:
            return {"status": "rejected", "reason": "no_open_position"}

        order_side = "SELL" if position.side == "LONG" else "BUY"
        response = self.api.place_order(
            {
                "symbol": position.symbol,
                "side": order_side,
                "type": "MARKET",
                "quantity": self._format_quantity(position.quantity),
                "reduceOnly": "true",
            }
        )
        self.position_tracker.clear_position(position.symbol)
        return {"status": "submitted", "response": response}

    @staticmethod
    def _format_quantity(quantity: float) -> str:
        return f"{quantity:.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def _format_price(price: float) -> str:
        return f"{price:.2f}"

    def _normalize_order(self, price: float, quantity: float) -> tuple[str, str]:
        filters = self.symbol_filters
        if filters is None:
            filters = ExchangeInfoClient(self.config).get_symbol_filters(self.config.symbol)
            self.symbol_filters = filters

        normalized_price = filters.quantize_price(price)
        normalized_quantity = filters.quantize_quantity(quantity)
        filters.validate_order(normalized_price, normalized_quantity)
        return (
            filters.decimal_to_string(normalized_price),
            filters.decimal_to_string(normalized_quantity),
        )
