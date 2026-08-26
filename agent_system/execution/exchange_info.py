"""Public Binance exchange metadata and order filter handling."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any

import requests

from agent_system.core.config import Settings, settings
from agent_system.core.exceptions import TradingAgentError


@dataclass(frozen=True)
class SymbolFilters:
    symbol: str
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    min_notional: Decimal

    def quantize_price(self, price: float) -> Decimal:
        return self._floor_to_step(Decimal(str(price)), self.tick_size)

    def quantize_quantity(self, quantity: float) -> Decimal:
        return self._floor_to_step(Decimal(str(quantity)), self.step_size)

    def validate_order(self, price: Decimal, quantity: Decimal) -> None:
        if quantity < self.min_qty:
            raise TradingAgentError(
                f"Quantity {quantity} is below Binance minQty {self.min_qty}"
            )
        notional = price * quantity
        if notional < self.min_notional:
            raise TradingAgentError(
                f"Notional {notional} is below Binance minNotional {self.min_notional}"
            )

    @staticmethod
    def decimal_to_string(value: Decimal) -> str:
        normalized = value.normalize()
        if normalized == normalized.to_integral():
            return str(normalized.quantize(Decimal("1")))
        return format(normalized, "f")

    @staticmethod
    def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
        if step <= 0:
            return value
        units = (value / step).to_integral_value(rounding=ROUND_DOWN)
        return units * step


class ExchangeInfoClient:
    def __init__(
        self,
        config: Settings = settings,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()

    def get_symbol_filters(self, symbol: str | None = None) -> SymbolFilters:
        selected_symbol = (symbol or self.config.symbol).upper()
        response = self.session.get(
            f"{self.config.binance_testnet_base_url}/fapi/v1/exchangeInfo",
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        return parse_symbol_filters(payload, selected_symbol)


def parse_symbol_filters(payload: dict[str, Any], symbol: str) -> SymbolFilters:
    for symbol_info in payload.get("symbols", []):
        if symbol_info.get("symbol") != symbol:
            continue
        filters = {
            item.get("filterType"): item
            for item in symbol_info.get("filters", [])
            if isinstance(item, dict)
        }
        price_filter = filters.get("PRICE_FILTER", {})
        lot_size = filters.get("LOT_SIZE", {})
        min_notional = filters.get("MIN_NOTIONAL", {})
        return SymbolFilters(
            symbol=symbol,
            tick_size=Decimal(str(price_filter.get("tickSize", "0.01"))),
            step_size=Decimal(str(lot_size.get("stepSize", "0.001"))),
            min_qty=Decimal(str(lot_size.get("minQty", "0.001"))),
            min_notional=Decimal(
                str(
                    min_notional.get(
                        "notional",
                        min_notional.get("minNotional", "5"),
                    )
                )
            ),
        )
    raise TradingAgentError(f"Symbol {symbol} not found in exchangeInfo")
