"""Order placement: sizing, quantization, validation and protective exits."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from agent_system.core.config import Settings, settings
from agent_system.core.exceptions import ExecutionError
from agent_system.core.logger import get_logger
from agent_system.execution.exchange_info import SymbolFilters
from agent_system.execution.position_tracker import PositionTracker, TrackedPosition
from agent_system.execution.risk_manager import RiskManager

logger = get_logger(__name__)


class OrderManager:
    """Turns a direction and a price into a valid, risk-checked order.

    The ordering of the checks matters. Risk limits are evaluated before any
    exchange call, so a halted account never touches the network; quantization
    happens before validation, because it is the rounded values that get sent
    and therefore the rounded values that must satisfy the minimums.
    """

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

    # ------------------------------------------------------------------
    def enter_limit(
        self,
        direction: str,
        price: float,
        leverage: int | None = None,
    ) -> dict[str, Any]:
        direction = direction.upper()
        if direction not in {"LONG", "SHORT"}:
            raise ExecutionError(f"Unknown direction: {direction!r}")

        if self.positions.has_position(self.symbol):
            return {"status": "skipped", "reason": "position_already_open"}

        allowed, reason = self.risk.can_trade()
        if not allowed:
            return {"status": "rejected", "reason": reason}

        leverage = int(leverage or self.config.default_leverage)
        leverage = max(1, min(leverage, self.config.max_leverage))

        raw_quantity = self.risk.position_size(price, leverage)
        quantized_price = self.filters.quantize_price(price)
        quantized_quantity = self.filters.quantize_quantity(raw_quantity)

        try:
            self.filters.validate_order(quantized_price, quantized_quantity)
        except ExecutionError as exc:
            return {"status": "rejected", "reason": str(exc)}

        self.api.set_leverage(self.symbol, leverage)

        side = "BUY" if direction == "LONG" else "SELL"
        params = {
            "symbol": self.symbol,
            "side": side,
            "type": "LIMIT",
            # Post-only: the order is cancelled rather than filled if it would
            # cross the spread. That guarantees the maker fee, which roughly
            # halves the round-trip cost -- decisive when trading this often.
            "timeInForce": "GTX" if self.config.prefer_maker else "GTC",
            "price": self.filters.decimal_to_string(quantized_price),
            "quantity": self.filters.decimal_to_string(quantized_quantity),
        }

        response = self.api.place_order(params)
        self.risk.register_trade()

        side_sign = 1 if direction == "LONG" else -1
        fill_price = float(quantized_price)
        stop_fraction = self.config.stop_loss_fraction
        target_fraction = self.config.take_profit_fraction
        self.positions.open(
            TrackedPosition(
                symbol=self.symbol,
                side=side_sign,
                entry_price=fill_price,
                quantity=float(quantized_quantity),
                stop_loss=fill_price * (1 - stop_fraction * side_sign),
                take_profit=fill_price * (1 + target_fraction * side_sign),
            )
        )

        logger.info(
            "Submitted %s %s %s @ %s",
            direction,
            params["quantity"],
            self.symbol,
            params["price"],
        )
        return {"status": "submitted", "params": params, "response": response}

    # ------------------------------------------------------------------
    def exit_market(self, price: float) -> dict[str, Any]:
        position = self.positions.get(self.symbol)
        if position is None:
            return {"status": "skipped", "reason": "no_position"}

        side = "SELL" if position.side > 0 else "BUY"
        quantity = self.filters.quantize_quantity(position.quantity)
        params = {
            "symbol": self.symbol,
            "side": side,
            "type": "MARKET",
            "quantity": self.filters.decimal_to_string(quantity),
            "reduceOnly": "true",
        }
        response = self.api.place_order(params)

        pnl = position.unrealized_pnl(price)
        self.positions.close(self.symbol)
        self.risk.register_result(pnl)

        record = {
            "symbol": self.symbol,
            "side": "long" if position.side > 0 else "short",
            "entry_price": position.entry_price,
            "exit_price": price,
            "quantity": position.quantity,
            "pnl": pnl,
        }
        logger.info("Closed %s position, pnl %.4f", record["side"], pnl)
        return {"status": "closed", "record": record, "response": response}

    # ------------------------------------------------------------------
    def check_protective_exits(self, price: float) -> dict[str, Any] | None:
        """Close the position if price has hit the stop or the target."""
        position = self.positions.get(self.symbol)
        if position is None:
            return None

        if position.side > 0:
            hit_stop = position.stop_loss is not None and price <= position.stop_loss
            hit_target = position.take_profit is not None and price >= position.take_profit
        else:
            hit_stop = position.stop_loss is not None and price >= position.stop_loss
            hit_target = position.take_profit is not None and price <= position.take_profit

        if not (hit_stop or hit_target):
            return None

        result = self.exit_market(price)
        result["reason"] = "stop_loss" if hit_stop else "take_profit"
        return result

    def sync_equity(self, equity: float) -> None:
        self.risk.update_equity(equity)
