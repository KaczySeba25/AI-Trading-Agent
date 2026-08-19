"""Central configuration.

Every value can be overridden with an environment variable or a line in
``.env.local``. Defaults are tuned for *active* trading: many small positions
opened and closed rather than a handful of long holds. The two settings that
decide whether active trading can be profitable at all are ``maker_fee_fraction``
and ``slippage_fraction`` -- see the note on costs below.
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
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Runtime configuration.

    A note on trading costs, because they dominate every other decision here.
    A round trip pays the fee twice plus slippage twice. With Binance Futures
    taker fees (0.04%) that is 0.12% per round trip, so a strategy trading every
    few minutes must find more than 0.12% of edge per trade just to break even.
    That is a very high bar. Posting limit orders pays the maker fee (0.02%)
    instead, which cuts the round trip to roughly 0.06% and is the difference
    between a viable active strategy and a slow bleed. ``prefer_maker`` keeps
    the agent on the maker side by default.
    """

    # --- Market ---
    symbol: str = field(default_factory=lambda: os.getenv("TRADING_SYMBOL", "BTCUSDT").upper())
    initial_capital: float = field(default_factory=lambda: _env_float("INITIAL_CAPITAL", 500.0))
    max_open_positions: int = field(default_factory=lambda: _env_int("MAX_OPEN_POSITIONS", 1))

    # --- Data sources ---
    public_market_data_base_url: str = field(
        default_factory=lambda: os.getenv("PUBLIC_MARKET_DATA_BASE_URL", "https://api.binance.com")
    )
    binance_ws_base_url: str = field(
        default_factory=lambda: os.getenv(
            "BINANCE_WS_BASE_URL", "wss://stream.binance.com:9443/stream"
        )
    )
    buffer_window_seconds: int = field(
        default_factory=lambda: _env_int("BUFFER_WINDOW_SECONDS", 300)
    )
    reconnect_initial_delay_seconds: float = field(
        default_factory=lambda: _env_float("RECONNECT_INITIAL_DELAY_SECONDS", 1.0)
    )
    reconnect_max_delay_seconds: float = field(
        default_factory=lambda: _env_float("RECONNECT_MAX_DELAY_SECONDS", 30.0)
    )
    stream_stats_interval_seconds: float = field(
        default_factory=lambda: _env_float("STREAM_STATS_INTERVAL_SECONDS", 5.0)
    )

    # --- Credentials / venue ---
    binance_testnet_base_url: str = field(
        default_factory=lambda: os.getenv(
            "BINANCE_TESTNET_BASE_URL", "https://testnet.binancefuture.com"
        )
    )
    binance_api_key: str | None = field(default_factory=lambda: os.getenv("BINANCE_API_KEY") or None)
    binance_api_secret: str | None = field(
        default_factory=lambda: os.getenv("BINANCE_API_SECRET") or None
    )
    enable_testnet_trading: bool = field(
        default_factory=lambda: _env_bool("ENABLE_TESTNET_TRADING", False)
    )
    enable_live_trading: bool = field(
        default_factory=lambda: _env_bool("ENABLE_LIVE_TRADING", False)
    )

    # --- Position sizing and risk -----------------------------------------
    # Live sizing starts deliberately tiny: 2% of equity as notional, no
    # leverage. Real money should only get bigger because the operator turned
    # this dial up on purpose, never because a default was generous.
    # The simulator uses a different convention -- see TradingEnvironment.
    max_position_fraction: float = field(
        default_factory=lambda: _env_float("MAX_POSITION_FRACTION", 0.02)
    )
    default_leverage: int = field(default_factory=lambda: _env_int("DEFAULT_LEVERAGE", 1))
    max_leverage: int = field(default_factory=lambda: _env_int("MAX_LEVERAGE", 5))
    daily_loss_limit_fraction: float = field(
        default_factory=lambda: _env_float("DAILY_LOSS_LIMIT_FRACTION", 0.05)
    )

    # --- Exit policy -------------------------------------------------------
    # Fixed percentage stops fight the market: in a quiet hour a 0.3% stop is so
    # far away it never triggers, and in a volatile one it is knocked out by
    # noise. Scaling the stop by recent ATR keeps the *probability* of being
    # stopped roughly constant instead.
    use_atr_exits: bool = field(default_factory=lambda: _env_bool("USE_ATR_EXITS", True))
    atr_stop_multiple: float = field(default_factory=lambda: _env_float("ATR_STOP_MULTIPLE", 1.5))
    atr_target_multiple: float = field(
        default_factory=lambda: _env_float("ATR_TARGET_MULTIPLE", 2.5)
    )
    stop_loss_fraction: float = field(
        default_factory=lambda: _env_float("STOP_LOSS_FRACTION", 0.004)
    )
    take_profit_fraction: float = field(
        default_factory=lambda: _env_float("TAKE_PROFIT_FRACTION", 0.008)
    )
    min_stop_fraction: float = field(default_factory=lambda: _env_float("MIN_STOP_FRACTION", 0.0015))
    max_stop_fraction: float = field(default_factory=lambda: _env_float("MAX_STOP_FRACTION", 0.02))
    trailing_stop_enabled: bool = field(
        default_factory=lambda: _env_bool("TRAILING_STOP_ENABLED", True)
    )
    trailing_stop_atr_multiple: float = field(
        default_factory=lambda: _env_float("TRAILING_STOP_ATR_MULTIPLE", 1.2)
    )
    max_holding_bars: int = field(default_factory=lambda: _env_int("MAX_HOLDING_BARS", 60))

    # --- Trading costs -----------------------------------------------------
    maker_fee_fraction: float = field(
        default_factory=lambda: _env_float("MAKER_FEE_FRACTION", 0.0002)
    )
    taker_fee_fraction: float = field(
        default_factory=lambda: _env_float("TAKER_FEE_FRACTION", 0.0004)
    )
    slippage_fraction: float = field(default_factory=lambda: _env_float("SLIPPAGE_FRACTION", 0.0001))
    prefer_maker: bool = field(default_factory=lambda: _env_bool("PREFER_MAKER", True))

    # --- Activity limits ---------------------------------------------------
    # Deliberately loose: the agent is meant to be in and out of the market
    # constantly. They exist to stop a runaway loop, not to pace the strategy.
    min_seconds_between_trades: float = field(
        default_factory=lambda: _env_float("MIN_SECONDS_BETWEEN_TRADES", 0.0)
    )
    max_trades_per_day: int = field(default_factory=lambda: _env_int("MAX_TRADES_PER_DAY", 2000))
    max_consecutive_losses: int = field(
        default_factory=lambda: _env_int("MAX_CONSECUTIVE_LOSSES", 12)
    )

    # --- Reward shaping ----------------------------------------------------
    # These steer *how* the agent trades. inactivity_penalty is the one that
    # turns a passive model into an active one; churn_penalty is its
    # counterweight, charged per trade so activity has to pay for itself.
    inactivity_penalty: float = field(
        default_factory=lambda: _env_float("INACTIVITY_PENALTY", 0.002)
    )
    churn_penalty: float = field(default_factory=lambda: _env_float("CHURN_PENALTY", 0.01))
    holding_penalty: float = field(default_factory=lambda: _env_float("HOLDING_PENALTY", 0.0005))
    drawdown_penalty: float = field(default_factory=lambda: _env_float("DRAWDOWN_PENALTY", 0.05))
    reward_clip: float = field(default_factory=lambda: _env_float("REWARD_CLIP", 10.0))

    # --- Paths and learning ---
    state_dir: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("STATE_DIR", "state"))
    model_dir: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("MODEL_DIR", "models"))
    data_dir: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("DATA_DIR", "data"))
    log_dir: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("LOG_DIR", "logs"))
    #: Fraction of equity risked per simulated position. Distinct from
    #: ``max_position_fraction`` (a live-order notional cap) because the
    #: simulator sizes by exposure, not by order value.
    sim_position_fraction: float = field(
        default_factory=lambda: _env_float("SIM_POSITION_FRACTION", 0.5)
    )
    observation_window: int = field(default_factory=lambda: _env_int("OBSERVATION_WINDOW", 32))
    train_steps: int = field(default_factory=lambda: _env_int("TRAIN_STEPS", 2048))
    learning_rate: float = field(default_factory=lambda: _env_float("LEARNING_RATE", 3e-4))
    entropy_coefficient: float = field(
        default_factory=lambda: _env_float("ENTROPY_COEFFICIENT", 0.02)
    )
    seed: int = field(default_factory=lambda: _env_int("SEED", 42))

    # ------------------------------------------------------------------
    @property
    def has_credentials(self) -> bool:
        return bool(self.binance_api_key and self.binance_api_secret)

    @property
    def entry_fee_fraction(self) -> float:
        """Fee actually charged on a fill, given the maker/taker preference."""
        return self.maker_fee_fraction if self.prefer_maker else self.taker_fee_fraction

    @property
    def round_trip_cost_fraction(self) -> float:
        """Total cost of opening and closing one position, as a fraction.

        This is the hurdle every trade must clear. Exposed so the environment,
        the reward function and the reports all quote the same number.
        """
        return 2.0 * (self.entry_fee_fraction + self.slippage_fraction)

    def ensure_directories(self) -> None:
        for directory in (self.state_dir, self.model_dir, self.data_dir, self.log_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "initial_capital": self.initial_capital,
            "credentials_present": self.has_credentials,
            "testnet_trading": self.enable_testnet_trading,
            "live_trading": self.enable_live_trading,
            "max_position_fraction": self.max_position_fraction,
            "default_leverage": self.default_leverage,
            "daily_loss_limit_fraction": self.daily_loss_limit_fraction,
            "prefer_maker": self.prefer_maker,
            "round_trip_cost_pct": round(self.round_trip_cost_fraction * 100, 4),
            "use_atr_exits": self.use_atr_exits,
            "max_trades_per_day": self.max_trades_per_day,
        }


settings = Settings()
