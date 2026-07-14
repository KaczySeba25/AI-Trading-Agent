"""Central configuration for the trading agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")
load_dotenv(ROOT_DIR / ".env.local", override=True)


@dataclass(frozen=True)
class Settings:
    symbol: str = os.getenv("TRADING_SYMBOL", "BTCUSDT").upper()
    initial_capital: float = float(os.getenv("INITIAL_CAPITAL", "500"))
    max_open_positions: int = int(os.getenv("MAX_OPEN_POSITIONS", "8"))
    public_market_data_base_url: str = os.getenv(
        "PUBLIC_MARKET_DATA_BASE_URL",
        "https://api.binance.com",
    )
    binance_ws_base_url: str = os.getenv(
        "BINANCE_WS_BASE_URL",
        "wss://stream.binance.com:9443/stream",
    )
    buffer_window_seconds: int = int(os.getenv("BUFFER_WINDOW_SECONDS", "60"))
    reconnect_initial_delay_seconds: float = float(
        os.getenv("RECONNECT_INITIAL_DELAY_SECONDS", "1.0")
    )
    reconnect_max_delay_seconds: float = float(
        os.getenv("RECONNECT_MAX_DELAY_SECONDS", "30.0")
    )
    stream_stats_interval_seconds: float = float(
        os.getenv("STREAM_STATS_INTERVAL_SECONDS", "5.0")
    )
    binance_testnet_base_url: str = os.getenv(
        "BINANCE_TESTNET_BASE_URL",
        "https://testnet.binancefuture.com",
    )
    binance_api_key: str | None = os.getenv("BINANCE_API_KEY")
    binance_api_secret: str | None = os.getenv("BINANCE_API_SECRET")
    enable_testnet_trading: bool = (
        os.getenv("ENABLE_TESTNET_TRADING", "false").lower() == "true"
    )
    max_position_fraction: float = float(os.getenv("MAX_POSITION_FRACTION", "0.02"))
    default_leverage: int = int(os.getenv("DEFAULT_LEVERAGE", "3"))
    max_leverage: int = int(os.getenv("MAX_LEVERAGE", "5"))
    stop_loss_fraction: float = float(os.getenv("STOP_LOSS_FRACTION", "0.003"))
    daily_loss_limit_fraction: float = float(
        os.getenv("DAILY_LOSS_LIMIT_FRACTION", "0.05")
    )


settings = Settings()
