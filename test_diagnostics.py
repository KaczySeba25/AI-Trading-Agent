from agent_system.core.config import Settings
from agent_system.execution.diagnostics import run_testnet_diagnostic


def test_testnet_diagnostic_does_not_require_credentials() -> None:
    config = Settings(**{"binance_api_key": None, "binance_api_secret": None})

    result = run_testnet_diagnostic(config)

    assert result["status"] == "missing_credentials"
    assert result["orders_submitted"] == 0
