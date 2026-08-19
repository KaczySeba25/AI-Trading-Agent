"""Minimal signed client for the Binance USD-M Futures API."""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any
from urllib.parse import urlencode

import requests

from agent_system.core.config import Settings, settings
from agent_system.core.exceptions import ExecutionError
from agent_system.core.logger import get_logger

logger = get_logger(__name__)

#: Binance rejects a signed request whose timestamp is older than this.
RECV_WINDOW_MS = 5000


class BinanceFuturesApi:
    """Signed REST client.

    Order placement is gated behind ``allow_orders``. The default is off, so
    a misconfigured run can read balances but cannot spend money.
    """

    def __init__(
        self,
        config: Settings = settings,
        session: Any | None = None,
        allow_orders: bool = False,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.allow_orders = allow_orders
        self.base_url = (
            config.binance_testnet_base_url
            if not config.enable_live_trading
            else "https://fapi.binance.com"
        )
        if config.binance_api_key:
            self.session.headers.update({"X-MBX-APIKEY": config.binance_api_key})

    # ------------------------------------------------------------------
    def _sign(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self.config.has_credentials:
            raise ExecutionError("API credentials are required for signed requests")
        payload = dict(params)
        payload["timestamp"] = int(time.time() * 1000)
        payload["recvWindow"] = RECV_WINDOW_MS
        query = urlencode(payload)
        signature = hmac.new(
            self.config.binance_api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        payload["signature"] = signature
        return payload

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None,
                 signed: bool = False) -> Any:
        url = f"{self.base_url}{path}"
        params = dict(params or {})
        if signed:
            params = self._sign(params)
        try:
            response = self.session.request(method, url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise ExecutionError(f"{method} {path} failed: {exc}") from exc

    # ------------------------------------------------------------------
    def ping(self) -> dict[str, Any]:
        return self._request("GET", "/fapi/v1/ping")

    def server_time(self) -> dict[str, Any]:
        return self._request("GET", "/fapi/v1/time")

    def exchange_info(self) -> dict[str, Any]:
        return self._request("GET", "/fapi/v1/exchangeInfo")

    def account(self) -> dict[str, Any]:
        return self._request("GET", "/fapi/v2/account", signed=True)

    def balance(self) -> Any:
        return self._request("GET", "/fapi/v2/balance", signed=True)

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        return self._request(
            "POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": int(leverage)}, signed=True
        )

    def place_order(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self.allow_orders:
            raise ExecutionError(
                "Order placement is disabled. Enable it explicitly before trading."
            )
        if self.config.enable_live_trading and not self.config.enable_testnet_trading:
            logger.warning("Submitting a LIVE order: %s", params.get("symbol"))
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def cancel_all(self, symbol: str) -> dict[str, Any]:
        return self._request(
            "DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True
        )

    def __repr__(self) -> str:
        # Never let a secret reach a log line or a traceback.
        return (
            f"BinanceFuturesApi(base_url={self.base_url!r}, "
            f"credentials={'present' if self.config.has_credentials else 'absent'}, "
            f"allow_orders={self.allow_orders})"
        )
