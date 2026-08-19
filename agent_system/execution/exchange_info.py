"""Symbol filters: tick size, lot size and minimum notional.

Binance rejects any order whose price is not a multiple of the tick size or
whose quantity is not a multiple of the step size. Getting this wrong is the
single most common cause of a rejected order, so all rounding happens here,
in ``Decimal``, once.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from typing import Any

import requests

from agent_system.core.config import Settings, settings
from agent_system.core.exceptions import ExecutionError, MarketDataError
from agent_system.core.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SymbolFilters:
    """Trading rules for one symbol."""

    symbol: str
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    min_notional: Decimal

    # -- rounding --------------------------------------------------------
    @staticmethod
    def _quantize(value: Decimal | float | str, increment: Decimal) -> Decimal:
        """Round ``value`` down to a multiple of ``increment``.

        Always down, never to nearest: rounding a quantity up can exceed the
        available balance, and rounding a limit price up can cross the spread
        and turn a maker order into a taker one.
        """
        try:
            decimal_value = Decimal(str(value))
        except InvalidOperation as exc:
            raise ExecutionError(f"Cannot quantize value: {value!r}") from exc
        if increment <= 0:
            return decimal_value
        return (decimal_value / increment).to_integral_value(rounding=ROUND_DOWN) * increment

    def quantize_price(self, price: Decimal | float | str) -> Decimal:
        return self._quantize(price, self.tick_size)

    def quantize_quantity(self, quantity: Decimal | float | str) -> Decimal:
        return self._quantize(quantity, self.step_size)

    # -- validation ------------------------------------------------------
    def validate_order(self, price: Decimal, quantity: Decimal) -> None:
        if quantity < self.min_qty:
            raise ExecutionError(
                f"Quantity {quantity} below minimum {self.min_qty} for {self.symbol}"
            )
        notional = price * quantity
        if notional < self.min_notional:
            raise ExecutionError(
                f"Notional {notional} below minimum {self.min_notional} for {self.symbol}"
            )

    @staticmethod
    def decimal_to_string(value: Decimal) -> str:
        """Format for the API: no exponent, no trailing zeros.

        ``Decimal("0.004000")`` must be sent as ``"0.004"`` -- Binance rejects
        scientific notation, and trailing zeros can trip precision checks.
        """
        normalized = value.normalize()
        if normalized == normalized.to_integral_value():
            try:
                normalized = normalized.quantize(Decimal(1))
            except InvalidOperation:
                pass
        text = format(normalized, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"


def parse_symbol_filters(payload: dict[str, Any], symbol: str) -> SymbolFilters:
    """Extract the filters for ``symbol`` from an exchangeInfo payload."""
    symbol = symbol.upper()
    symbols = payload.get("symbols") or []
    entry = next((item for item in symbols if item.get("symbol") == symbol), None)
    if entry is None:
        raise MarketDataError(f"Symbol {symbol} not present in exchangeInfo payload")

    filters = {item.get("filterType"): item for item in entry.get("filters", [])}

    price_filter = filters.get("PRICE_FILTER", {})
    lot_filter = filters.get("LOT_SIZE", {})
    notional_filter = filters.get("MIN_NOTIONAL", {}) or filters.get("NOTIONAL", {})

    def _decimal(source: dict[str, Any], key: str, default: str) -> Decimal:
        try:
            return Decimal(str(source.get(key, default)))
        except InvalidOperation:
            return Decimal(default)

    return SymbolFilters(
        symbol=symbol,
        tick_size=_decimal(price_filter, "tickSize", "0.01"),
        step_size=_decimal(lot_filter, "stepSize", "0.001"),
        min_qty=_decimal(lot_filter, "minQty", "0.001"),
        min_notional=_decimal(notional_filter, "notional", "0")
        or _decimal(notional_filter, "minNotional", "0"),
    )


def fetch_symbol_filters(
    config: Settings = settings,
    session: Any | None = None,
    symbol: str | None = None,
) -> SymbolFilters:
    """Download exchangeInfo and parse the filters for one symbol."""
    symbol = (symbol or config.symbol).upper()
    session = session or requests.Session()
    base_url = config.binance_testnet_base_url
    url = f"{base_url}/fapi/v1/exchangeInfo"
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise MarketDataError(f"Failed to fetch exchange info: {exc}") from exc
    return parse_symbol_filters(payload, symbol)
