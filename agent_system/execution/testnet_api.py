"""Binance Futures Testnet REST API client."""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any
from urllib.parse import urlencode

import requests

from agent_system.core.config import Settings, settings
from agent_system.core.exceptions import TradingAgentError
from agent_system.core.logger import get_logger


logger = get_logger(__name__)


class BinanceTestnetClient:
    def __init__(self, config: Settings = settings, session: requests.Session | None = None) -> None:
        self.config = config
        self.session = session or requests.Session()

    @property
    def has_credentials(self) -> bool:
        return bool(self.config.binance_api_key and self.config.binance_api_secret)

    def signed_request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        attempts: int = 3,
    ) -> dict[str, Any]:
        if not self.has_credentials:
            raise TradingAgentError("Missing Binance API credentials in environment")

        payload = dict(params or {})
        payload["timestamp"] = int(time.time() * 1000)
        query = urlencode(payload)
        signature = hmac.new(
            self.config.binance_api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        payload["signature"] = signature

        url = f"{self.config.binance_testnet_base_url}{path}"
        headers = {"X-MBX-APIKEY": self.config.binance_api_key or ""}
        delay = 0.5
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=payload,
                    headers=headers,
                    timeout=10,
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("Binance API request failed; retrying")
                time.sleep(delay)
                delay = min(delay * 2, 5.0)
        if last_error:
            raise TradingAgentError("Binance API request failed after retries") from last_error
        raise TradingAgentError("Binance API request failed")

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        return self.signed_request(
            "POST",
            "/fapi/v1/leverage",
            {"symbol": symbol, "leverage": leverage},
        )

    def place_order(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.signed_request("POST", "/fapi/v1/order", params)
