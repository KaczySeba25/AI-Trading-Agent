"""Read-only connectivity and credential diagnostics.

Never submits an order. Running this before enabling any trading confirms that
credentials, clock skew and account access are all sane.
"""

from __future__ import annotations

import time
from typing import Any

from agent_system.core.config import Settings, settings
from agent_system.core.exceptions import TradingAgentError
from agent_system.core.logger import get_logger
from agent_system.execution.binance_api import BinanceFuturesApi

logger = get_logger(__name__)


def run_testnet_diagnostic(
    config: Settings = settings,
    api: Any | None = None,
) -> dict[str, Any]:
    """Validate credentials and read access without placing orders."""
    result: dict[str, Any] = {
        "status": "unknown",
        "orders_submitted": 0,  # invariant: this function never trades
        "symbol": config.symbol,
        "base_url": config.binance_testnet_base_url,
        "checks": {},
    }

    if not (config.binance_api_key and config.binance_api_secret):
        result["status"] = "missing_credentials"
        result["message"] = (
            "Set BINANCE_API_KEY and BINANCE_API_SECRET in .env.local to enable "
            "account diagnostics. Public data and paper trading work without them."
        )
        return result

    client = api or BinanceFuturesApi(config=config, allow_orders=False)

    try:
        started = time.monotonic()
        client.ping()
        result["checks"]["ping_ms"] = round((time.monotonic() - started) * 1000, 1)

        server_time = client.server_time().get("serverTime", 0)
        skew_ms = abs(int(time.time() * 1000) - int(server_time))
        result["checks"]["clock_skew_ms"] = skew_ms
        # Binance rejects signed requests when local time drifts too far.
        if skew_ms > 5000:
            result["checks"]["clock_warning"] = "Local clock differs from exchange by >5s"

        account = client.account()
        result["checks"]["can_read_account"] = True
        result["checks"]["can_trade"] = bool(account.get("canTrade", False))
        result["checks"]["total_wallet_balance"] = account.get("totalWalletBalance")
        result["checks"]["available_balance"] = account.get("availableBalance")

        result["status"] = "ok"
        logger.info("Testnet diagnostic passed (skew %d ms)", skew_ms)
    except TradingAgentError as exc:
        result["status"] = "error"
        result["message"] = str(exc)
        logger.error("Testnet diagnostic failed: %s", exc)

    return result
