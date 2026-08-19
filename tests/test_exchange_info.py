from decimal import Decimal

from agent_system.execution.exchange_info import parse_symbol_filters


def test_parse_and_quantize_symbol_filters() -> None:
    filters = parse_symbol_filters(
        {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                        {
                            "filterType": "LOT_SIZE",
                            "minQty": "0.001",
                            "stepSize": "0.001",
                        },
                        {"filterType": "MIN_NOTIONAL", "notional": "100"},
                    ],
                }
            ]
        },
        "BTCUSDT",
    )

    assert filters.quantize_price(62685.987) == Decimal("62685.90")
    assert filters.quantize_quantity(0.004987) == Decimal("0.004")
    filters.validate_order(Decimal("62685.90"), Decimal("0.004"))
    assert filters.decimal_to_string(Decimal("0.004000")) == "0.004"
