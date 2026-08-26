"""Safe Testnet diagnostics that never print secrets or submit orders."""

from __future__ import annotations

from agent_system.core.config import Settings, settings
from agent_system.execution.exchange_info import ExchangeInfoClient
from agent_system.execution.testnet_api import BinanceTestnetClient


def run_testnet_diagnostic(config: Settings = settings) -> dict[str, object]:
    client = BinanceTestnetClient(config)
    if not client.has_credentials:
        return {
            "status": "missing_credentials",
            "has_api_key": bool(config.binance_api_key),
            "has_api_secret": bool(config.binance_api_secret),
            "orders_submitted": 0,
        }

    filters = ExchangeInfoClient(config).get_symbol_filters(config.symbol)
    account = client.account()
    balances = client.balance()
    open_orders = client.open_orders(config.symbol)
    usdt_balance = next((item for item in balances if item.get("asset") == "USDT"), {})

    return {
        "status": "ok",
        "symbol": config.symbol,
        "testnet_base_url": config.binance_testnet_base_url,
        "can_read_account": bool(account.get("assets") is not None),
        "available_balance_usdt": usdt_balance.get("availableBalance"),
        "open_orders_count": len(open_orders),
        "symbol_filters": {
            "tick_size": str(filters.tick_size),
            "step_size": str(filters.step_size),
            "min_qty": str(filters.min_qty),
            "min_notional": str(filters.min_notional),
        },
        "enable_testnet_trading": config.enable_testnet_trading,
        "orders_submitted": 0,
    }
