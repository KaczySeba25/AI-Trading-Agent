"""Central configuration for the trading agent.

Values come from environment variables, optionally loaded from ``.env`` and then
``.env.local`` (the latter wins and is git-ignored, so secrets never get
committed). Every field has a safe default: the system must be runnable with no
credentials at all, in which case it stays in paper mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")
load_dotenv(ROOT_DIR / ".env.local", override=True)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    # --- Market ---------------------------------------------------------
    symbol: str = field(default_factory=lambda: os.getenv("TRADING_SYMBOL", "BTCUSDT").upper())
    initial_capital: float = field(default_factory=lambda: _env_float("INITIAL_CAPITAL", 500.0))
    max_open_positions: int = field(default_factory=lambda: _env_int("MAX_OPEN_POSITIONS", 8))

    # --- Data sources ---------------------------------------------------
    public_market_data_base_url: str = field(
        default_factory=lambda: os.getenv("PUBLIC_MARKET_DATA_BASE_URL", "https://api.binance.com")
    )
    binance_ws_base_url: str = field(
        default_factory=lambda: os.getenv("BINANCE_WS_BASE_URL", "wss://stream.binance.com:9443/stream")
    )
    buffer_window_seconds: int = field(default_factory=lambda: _env_int("BUFFER_WINDOW_SECONDS", 300))
    reconnect_initial_delay_seconds: float = field(
        default_factory=lambda: _env_float("RECONNECT_INITIAL_DELAY_SECONDS", 1.0)
    )
    reconnect_max_delay_seconds: float = field(
        default_factory=lambda: _env_float("RECONNECT_MAX_DELAY_SECONDS", 30.0)
    )
    stream_stats_interval_seconds: float = field(
        default_factory=lambda: _env_float("STREAM_STATS_INTERVAL_SECONDS", 5.0)
    )

    # --- Credentials (optional; absence forces paper mode) --------------
    binance_testnet_base_url: str = field(
        default_factory=lambda: os.getenv("BINANCE_TESTNET_BASE_URL", "https://testnet.binancefuture.com")
    )
    binance_api_key: str | None = field(default_factory=lambda: os.getenv("BINANCE_API_KEY") or None)
    binance_api_secret: str | None = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET") or None)
    enable_testnet_trading: bool = field(default_factory=lambda: _env_bool("ENABLE_TESTNET_TRADING", False))

    #: Master safety switch for real money. Guarded again at the execution layer.
    enable_live_trading: bool = field(default_factory=lambda: _env_bool("ENABLE_LIVE_TRADING", False))

    # --- Risk -----------------------------------------------------------
    max_position_fraction: float = field(default_factory=lambda: _env_float("MAX_POSITION_FRACTION", 0.02))
    default_leverage: int = field(default_factory=lambda: _env_int("DEFAULT_LEVERAGE", 3))
    max_leverage: int = field(default_factory=lambda: _env_int("MAX_LEVERAGE", 5))
    stop_loss_fraction: float = field(default_factory=lambda: _env_float("STOP_LOSS_FRACTION", 0.003))
    take_profit_fraction: float = field(default_factory=lambda: _env_float("TAKE_PROFIT_FRACTION", 0.006))
    daily_loss_limit_fraction: float = field(
        default_factory=lambda: _env_float("DAILY_LOSS_LIMIT_FRACTION", 0.05)
    )

    # --- Costs (must be realistic or backtests lie) ---------------------
    taker_fee_fraction: float = field(default_factory=lambda: _env_float("TAKER_FEE_FRACTION", 0.0004))
    slippage_fraction: float = field(default_factory=lambda: _env_float("SLIPPAGE_FRACTION", 0.0002))

    # --- Learning -------------------------------------------------------
    state_dir: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("STATE_DIR", "state"))
    model_dir: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("MODEL_DIR", "models"))
    data_dir: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("DATA_DIR", "data"))
    log_dir: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("LOG_DIR", "logs"))
    observation_window: int = field(default_factory=lambda: _env_int("OBSERVATION_WINDOW", 32))
    train_steps: int = field(default_factory=lambda: _env_int("TRAIN_STEPS", 2048))
    learning_rate: float = field(default_factory=lambda: _env_float("LEARNING_RATE", 3e-4))
    seed: int = field(default_factory=lambda: _env_int("SEED", 42))

    def ensure_directories(self) -> None:
        """Create runtime directories on demand (they are git-ignored)."""
        for directory in (self.state_dir, self.model_dir, self.data_dir, self.log_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)

    @property
    def has_credentials(self) -> bool:
        return bool(self.binance_api_key and self.binance_api_secret)

    def describe(self) -> dict[str, object]:
        """Human-readable summary that never exposes secret values."""
        return {
            "symbol": self.symbol,
            "initial_capital": self.initial_capital,
            "credentials_present": self.has_credentials,
            "testnet_trading": self.enable_testnet_trading,
            "live_trading": self.enable_live_trading,
            "max_position_fraction": self.max_position_fraction,
            "daily_loss_limit_fraction": self.daily_loss_limit_fraction,
            "taker_fee_fraction": self.taker_fee_fraction,
            "slippage_fraction": self.slippage_fraction,
        }


settings = Settings()
