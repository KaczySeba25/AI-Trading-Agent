"""Diagnostics tests -- must never place an order."""

from __future__ import annotations

import time

from agent_system.core.config import Settings
from agent_system.execution.diagnostics import run_testnet_diagnostic


class FakeApi:
    def __init__(self, skew_ms: int = 0) -> None:
        self.skew_ms = skew_ms

    def ping(self) -> dict:
        return {}

    def server_time(self) -> dict:
        return {"serverTime": int(time.time() * 1000) + self.skew_ms}

    def account(self) -> dict:
        return {"canTrade": True, "totalWalletBalance": "1000", "availableBalance": "900"}


def test_diagnostic_does_not_require_credentials() -> None:
    config = Settings(binance_api_key=None, binance_api_secret=None)

    result = run_testnet_diagnostic(config)

    assert result["status"] == "missing_credentials"
    assert result["orders_submitted"] == 0


def test_diagnostic_reports_ok_with_working_credentials() -> None:
    config = Settings(binance_api_key="key", binance_api_secret="secret")

    result = run_testnet_diagnostic(config, api=FakeApi())

    assert result["status"] == "ok"
    assert result["orders_submitted"] == 0
    assert result["checks"]["can_read_account"]


def test_diagnostic_flags_clock_skew() -> None:
    """Binance rejects signed requests when the local clock drifts."""
    config = Settings(binance_api_key="key", binance_api_secret="secret")

    result = run_testnet_diagnostic(config, api=FakeApi(skew_ms=20_000))

    assert "clock_warning" in result["checks"]
