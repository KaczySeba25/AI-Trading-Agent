"""Signed Binance Futures REST client.

Safety design:

* Signing happens locally with HMAC-SHA256; the secret is never logged or sent.
* ``allow_orders`` must be explicitly true, and it is derived from configuration
  flags rather than from anything the policy can influence.
* :func:`repr` and error messages never include credentials.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any
from urllib.parse import urlencode

import requests

from agent_system.core.config import Settings, settings
from agent_system.core.exceptions import ConfigurationError, ExecutionError
from agent_system.core.logger import get_logger

logger = get_logger(__name__)


class BinanceFuturesApi:
    """Thin authenticated wrapper over the Binance Futures REST API."""

    def __init__(
        self,
        config: Settings = settings,
        session: Any | None = None,
        allow_orders: bool | None = None,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.base_url = config.binance_testnet_base_url.rstrip("/")
        self.allow_orders = (
            allow_orders
            if allow_orders is not None
            else (config.enable_testnet_trading or config.enable_live_trading)
        )

    def __repr__(self) -> str:  # never leak secrets in tracebacks
        return f"BinanceFuturesApi(base_url={self.base_url!r}, allow_orders={self.allow_orders})"

    def _require_credentials(self) -> tuple[str, str]:
        if not self.config.binance_api_key or not self.config.binance_api_secret:
            raise ConfigurationError(
                "BINANCE_API_KEY and BINANCE_API_SECRET must be set for private endpoints"
            )
        return self.config.binance_api_key, self.config.binance_api_secret

    def _sign(self, params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        api_key, api_secret = self._require_credentials()
        payload = dict(params)
        payload.setdefault("timestamp", int(time.time() * 1000))
        payload.setdefault("recvWindow", 5000)
        query = urlencode(payload)
        signature = hmac.new(
            api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        payload["signature"] = signature
        return payload, {"X-MBX-APIKEY": api_key}

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
    ) -> Any:
        url = f"{self.base_url}{path}"
        params = params or {}
        headers: dict[str, str] = {}

        if signed:
            params, headers = self._sign(params)

        try:
            response = self.session.request(
                method, url, params=params, headers=headers, timeout=15
            )
        except requests.RequestException as exc:
            raise ExecutionError(f"Request to {path} failed: {exc}") from exc

        if response.status_code >= 400:
            # Binance returns a JSON error body; surface it without credentials.
            try:
                detail = response.json()
            except ValueError:
                detail = response.text[:200]
            raise ExecutionError(f"{method} {path} -> {response.status_code}: {detail}")

        try:
            return response.json()
        except ValueError as exc:
            raise ExecutionError(f"Invalid JSON from {path}") from exc

    # --- Public endpoints ------------------------------------------------
    def ping(self) -> dict[str, Any]:
        return self._request("GET", "/fapi/v1/ping")

    def server_time(self) -> dict[str, Any]:
        return self._request("GET", "/fapi/v1/time")

    def mark_price(self, symbol: str | None = None) -> dict[str, Any]:
        return self._request(
            "GET", "/fapi/v1/premiumIndex", {"symbol": (symbol or self.config.symbol).upper()}
        )

    # --- Private endpoints -----------------------------------------------
    def account(self) -> dict[str, Any]:
        return self._request("GET", "/fapi/v2/account", signed=True)

    def balance(self) -> Any:
        return self._request("GET", "/fapi/v2/balance", signed=True)

    def position_risk(self, symbol: str | None = None) -> Any:
        params = {"symbol": (symbol or self.config.symbol).upper()}
        return self._request("GET", "/fapi/v2/positionRisk", params, signed=True)

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        return self._request(
            "POST",
            "/fapi/v1/leverage",
            {"symbol": symbol.upper(), "leverage": int(leverage)},
            signed=True,
        )

    def place_order(self, params: dict[str, Any]) -> dict[str, Any]:
        """Submit an order. Blocked unless ``allow_orders`` was enabled."""
        if not self.allow_orders:
            raise ExecutionError(
                "Order submission disabled. Set ENABLE_TESTNET_TRADING=true "
                "(or ENABLE_LIVE_TRADING=true) to allow orders."
            )
        logger.info("Submitting order: %s", {k: v for k, v in params.items() if k != "signature"})
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def cancel_all(self, symbol: str) -> dict[str, Any]:
        return self._request(
            "DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol.upper()}, signed=True
        )
