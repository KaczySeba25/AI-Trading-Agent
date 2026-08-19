"""Binance symbol filters.

An order whose price or quantity violates the symbol's filters is rejected by the
exchange, so every value is quantised **down** to the allowed grid before it is
sent. Decimal is used throughout: binary floats silently produce values like
``0.30000000000000004``, which the exchange rejects.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any, Mapping

import requests

from agent_system.core.config import Settings, settings
from agent_system.core.exceptions import ExecutionError, MarketDataError
from agent_system.core.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SymbolFilters:
    """Trading constraints for one symbol."""

    symbol: str
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    min_notional: Decimal

    def quantize_price(self, price: float | Decimal) -> Decimal:
        """Round a price down onto the tick grid."""
        if self.tick_size <= 0:
            return Decimal(str(price))
        value = Decimal(str(price))
        return (value / self.tick_size).to_integral_value(rounding=ROUND_DOWN) * self.tick_size

    def quantize_quantity(self, quantity: float | Decimal) -> Decimal:
        """Round a quantity down onto the lot-size grid."""
        if self.step_size <= 0:
            return Decimal(str(quantity))
        value = Decimal(str(quantity))
        return (value / self.step_size).to_integral_value(rounding=ROUND_DOWN) * self.step_size

    def validate_order(self, price: Decimal, quantity: Decimal) -> None:
        """Raise if the order would be rejected by the exchange."""
        if quantity < self.min_qty:
            raise ExecutionError(
                f"Quantity {quantity} below minQty {self.min_qty} for {self.symbol}"
            )
        notional = price * quantity
        if self.min_notional > 0 and notional < self.min_notional:
            raise ExecutionError(
                f"Notional {notional} below minNotional {self.min_notional} for {self.symbol}"
            )

    @staticmethod
    def decimal_to_string(value: Decimal) -> str:
        """Format without scientific notation or trailing zeros."""
        normalized = value.normalize()
        if normalized == normalized.to_integral_value():
            normalized = normalized.quantize(Decimal(1))
        return format(normalized, "f")


def parse_symbol_filters(payload: Mapping[str, Any], symbol: str) -> SymbolFilters:
    """Extract filters for ``symbol`` from an ``exchangeInfo`` payload."""
    symbol = symbol.upper()
    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        raise MarketDataError("exchangeInfo payload missing 'symbols'")

    entry = next((item for item in symbols if item.get("symbol") == symbol), None)
    if entry is None:
        raise MarketDataError(f"Symbol {symbol} not found in exchangeInfo")

    tick_size = Decimal("0")
    step_size = Decimal("0")
    min_qty = Decimal("0")
    min_notional = Decimal("0")

    for filter_entry in entry.get("filters", []):
        filter_type = filter_entry.get("filterType")
        if filter_type == "PRICE_FILTER":
            tick_size = Decimal(str(filter_entry.get("tickSize", "0")))
        elif filter_type == "LOT_SIZE":
            step_size = Decimal(str(filter_entry.get("stepSize", "0")))
            min_qty = Decimal(str(filter_entry.get("minQty", "0")))
        elif filter_type in {"MIN_NOTIONAL", "NOTIONAL"}:
            raw = filter_entry.get("notional") or filter_entry.get("minNotional") or "0"
            min_notional = Decimal(str(raw))

    return SymbolFilters(
        symbol=symbol,
        tick_size=tick_size,
        step_size=step_size,
        min_qty=min_qty,
        min_notional=min_notional,
    )


def fetch_symbol_filters(
    symbol: str | None = None,
    config: Settings = settings,
    session: Any | None = None,
) -> SymbolFilters:
    """Download public ``exchangeInfo`` -- no credentials required."""
    symbol = (symbol or config.symbol).upper()
    session = session or requests
    url = f"{config.binance_testnet_base_url}/fapi/v1/exchangeInfo"

    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise MarketDataError(f"Could not fetch exchangeInfo: {exc}") from exc

    filters = parse_symbol_filters(payload, symbol)
    logger.info(
        "Loaded filters for %s: tick=%s step=%s minQty=%s minNotional=%s",
        filters.symbol, filters.tick_size, filters.step_size,
        filters.min_qty, filters.min_notional,
    )
    return filters
