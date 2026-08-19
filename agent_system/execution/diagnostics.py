"""Read-only testnet connectivity check.

Run before trusting a deployment. It never places an order -- the whole point
is to answer "would trading work?" without risking anything if the answer is
no.
"""

from __future__ import annotations

import time
from typing import Any

from agent_system.core.config import Settings, settings
from agent_system.core.exceptions import ExecutionError, TradingAgentError
from agent_system.core.logger import get_logger
from agent_system.execution.binance_api import BinanceFuturesApi

logger = get_logger(__name__)

#: A signed request is rejected if the local clock drifts past recvWindow,
#: so warn well before that point.
MAX_ACCEPTABLE_CLOCK_SKEW_MS = 5000


def run_testnet_diagnostic(
    config: Settings = settings,
    api: Any | None = None,
) -> dict[str, Any]:
    """Check connectivity, clock skew and permissions. Places no orders."""
    result: dict[str, Any] = {
        "status": "unknown",
        "orders_submitted": 0,
        "checks": {},
    }

    if not config.has_credentials:
        result["status"] = "missing_credentials"
        result["checks"]["credentials"] = False
        result["message"] = (
            "Set BINANCE_API_KEY and BINANCE_API_SECRET in .env.local to run this check."
        )
        return result

    result["checks"]["credentials"] = True
    client = api or BinanceFuturesApi(config, allow_orders=False)

    try:
        started = time.perf_counter()
        client.ping()
        result["ping_ms"] = round((time.perf_counter() - started) * 1000, 2)
        result["checks"]["reachable"] = True
    except (ExecutionError, TradingAgentError) as exc:
        result["status"] = "unreachable"
        result["checks"]["reachable"] = False
        result["error"] = str(exc)
        return result

    try:
        server_time = int(client.server_time().get("serverTime", 0))
        local_time = int(time.time() * 1000)
        skew = server_time - local_time
        result["clock_skew_ms"] = skew
        if abs(skew) > MAX_ACCEPTABLE_CLOCK_SKEW_MS:
            result["checks"]["clock_warning"] = True
            logger.warning("Clock skew %d ms exceeds recvWindow; signed requests may fail", skew)
        else:
            result["checks"]["clock_warning"] = False
    except (ExecutionError, TradingAgentError, ValueError) as exc:
        result["checks"]["clock_warning"] = True
        result["clock_error"] = str(exc)

    try:
        account = client.account()
        result["checks"]["can_read_account"] = True
        result["checks"]["can_trade"] = bool(account.get("canTrade", False))
        balances = [
            {"asset": item.get("asset"), "balance": item.get("balance")}
            for item in (account.get("assets") or [])
            if float(item.get("balance", 0) or 0) != 0
        ]
        result["balances"] = balances[:10]
        result["total_wallet_balance"] = account.get("totalWalletBalance")
    except (ExecutionError, TradingAgentError) as exc:
        result["checks"]["can_read_account"] = False
        result["account_error"] = str(exc)

    result["status"] = "ok" if result["checks"].get("can_read_account") else "degraded"
    return result
