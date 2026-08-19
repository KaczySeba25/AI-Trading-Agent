"""Order construction and submission.

Order of operations on every entry, none of which may be skipped:

1. ask the risk manager for permission,
2. size the position from *current* equity,
3. quantise price and quantity to the exchange's filters,
4. validate against minQty / minNotional,
5. only then submit.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from agent_system.core.config import Settings, settings
from agent_system.core.exceptions import ExecutionError
from agent_system.core.logger import get_logger
from agent_system.execution.exchange_info import SymbolFilters
from agent_system.execution.position_tracker import PositionTracker
from agent_system.execution.risk_manager import RiskManager

logger = get_logger(__name__)


class OrderManager:
    """Translates a policy decision into a filter-compliant exchange order."""

    def __init__(
        self,
        api: Any,
        risk_manager: RiskManager,
        position_tracker: PositionTracker,
        symbol_filters: SymbolFilters,
        config: Settings = settings,
    ) -> None:
        self.api = api
        self.risk = risk_manager
        self.positions = position_tracker
        self.filters = symbol_filters
        self.config = config
        self.symbol = symbol_filters.symbol

    def enter_limit(self, direction: str, price: float, leverage: int | None = None) -> dict[str, Any]:
        """Place a post-only limit entry. Returns a status dict, never raises on refusal."""
        direction = direction.upper()
        if direction not in {"LONG", "SHORT"}:
            raise ValueError(f"direction must be LONG or SHORT, got {direction!r}")

        # Check the position book first: "you already hold this" is a more precise
        # explanation than whatever generic limit the risk manager would report.
        if self.positions.has_position(self.symbol):
            return {"status": "rejected", "reason": "position_already_open"}

        allowed, reason = self.risk.can_trade()
        if not allowed:
            logger.warning("Entry refused by risk manager: %s", reason)
            return {"status": "rejected", "reason": reason}

        side = 1 if direction == "LONG" else -1
        leverage = min(leverage or self.config.default_leverage, self.config.max_leverage)

        quantized_price = self.filters.quantize_price(price)
        raw_quantity = self.risk.position_size(float(quantized_price), leverage)
        quantized_quantity = self.filters.quantize_quantity(raw_quantity)

        try:
            self.filters.validate_order(quantized_price, quantized_quantity)
        except ExecutionError as exc:
            logger.warning("Order failed filter validation: %s", exc)
            return {"status": "rejected", "reason": str(exc)}

        self.api.set_leverage(self.symbol, leverage)

        params = {
            "symbol": self.symbol,
            "side": "BUY" if side > 0 else "SELL",
            "type": "LIMIT",
            "timeInForce": "GTC",
            "price": self.filters.decimal_to_string(quantized_price),
            "quantity": self.filters.decimal_to_string(quantized_quantity),
        }
        response = self.api.place_order(params)

        entry_price = float(quantized_price)
        self.positions.open(
            symbol=self.symbol,
            side=side,
            entry_price=entry_price,
            quantity=float(quantized_quantity),
            stop_loss=self.risk.stop_loss_price(entry_price, side),
            take_profit=self.risk.take_profit_price(entry_price, side),
        )
        self.risk.register_entry()

        logger.info(
            "%s %s %s @ %s (leverage %dx)",
            direction, params["quantity"], self.symbol, params["price"], leverage,
        )
        return {"status": "submitted", "order": response, "params": params}

    def exit_market(self, price: float) -> dict[str, Any]:
        """Close the open position at market."""
        position = self.positions.get(self.symbol)
        if position is None:
            return {"status": "rejected", "reason": "no_open_position"}

        quantity = self.filters.quantize_quantity(Decimal(str(position.quantity)))
        params = {
            "symbol": self.symbol,
            "side": "SELL" if position.side > 0 else "BUY",
            "type": "MARKET",
            "quantity": self.filters.decimal_to_string(quantity),
            "reduceOnly": "true",
        }
        response = self.api.place_order(params)

        record = self.positions.close(self.symbol, price)
        self.risk.register_exit()
        if record:
            self.risk.record_trade_result(record["pnl"])
            logger.info("Closed %s @ %.2f | PnL %.4f", self.symbol, price, record["pnl"])

        return {"status": "closed", "order": response, "record": record}

    def check_protective_exits(self, price: float) -> dict[str, Any] | None:
        """Close the position if its stop-loss or take-profit has been touched."""
        position = self.positions.get(self.symbol)
        if position is None:
            return None

        hit_stop = position.stop_loss is not None and (
            (position.side > 0 and price <= position.stop_loss)
            or (position.side < 0 and price >= position.stop_loss)
        )
        hit_target = position.take_profit is not None and (
            (position.side > 0 and price >= position.take_profit)
            or (position.side < 0 and price <= position.take_profit)
        )

        if hit_stop or hit_target:
            logger.info("Protective exit (%s) at %.2f", "stop" if hit_stop else "target", price)
            return self.exit_market(price)
        return None
