"""Walk-forward training on historical data.

Two rules are enforced here, and both exist to stop the agent fooling itself:

1. **Validation is always the newest slice**, never a random split. Shuffling
   time series lets the model train on the future and "predict" the past.
2. **Successive cycles walk forward** through history instead of re-training on
   the same window, so the agent sees many market regimes rather than
   memorising one.
"""

from __future__ import annotations

from typing import Sequence

from agent_system.core.config import Settings, settings
from agent_system.core.logger import get_logger
from agent_system.data.data_buffer import MarketTick
from agent_system.rl.agent import create_agent
from agent_system.rl.environment import TradingEnvironment

logger = get_logger(__name__)

#: Ticks per walk-forward training window.
WINDOW_SIZE = 2000


def split_train_validation(
    ticks: Sequence[MarketTick],
    validation_fraction: float = 0.2,
) -> tuple[list[MarketTick], list[MarketTick]]:
    """Chronological split: oldest ticks train, newest validate."""
    ticks = list(ticks)
    if not ticks:
        return [], []

    validation_fraction = min(max(validation_fraction, 0.0), 0.9)
    split_at = int(len(ticks) * (1.0 - validation_fraction))
    split_at = max(1, min(split_at, len(ticks) - 1))
    return ticks[:split_at], ticks[split_at:]


def select_training_window(
    ticks: Sequence[MarketTick],
    cycle: int,
    cycles: int,
    window_size: int = WINDOW_SIZE,
) -> list[MarketTick]:
    """Pick this cycle's slice, sliding forward as cycles progress."""
    ticks = list(ticks)
    if len(ticks) <= window_size or cycles <= 1:
        return ticks[-window_size:] if len(ticks) > window_size else ticks

    cycle_index = max(0, min(cycle - 1, cycles - 1))
    span = len(ticks) - window_size
    start = int(span * cycle_index / (cycles - 1))
    return ticks[start : start + window_size]


def evaluate_policy(
    agent,
    ticks: Sequence[MarketTick],
    config: Settings = settings,
) -> dict[str, float]:
    """Run a deterministic pass over ``ticks`` and return performance metrics."""
    env = TradingEnvironment(ticks, config=config)
    observation, _ = env.reset()
    done = False

    while not done:
        action = agent.act(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    return env.metrics()


def train_on_history(
    ticks: Sequence[MarketTick],
    cycle: int = 1,
    cycles: int = 1,
    train_steps: int | None = None,
    config: Settings = settings,
    agent=None,
) -> dict[str, object]:
    """Train one walk-forward cycle and evaluate on unseen newer data."""
    config.ensure_directories()
    train_steps = train_steps or config.train_steps

    window = select_training_window(ticks, cycle, cycles)
    train_ticks, validation_ticks = split_train_validation(window, 0.2)

    minimum = config.observation_window + 3
    if len(train_ticks) < minimum or len(validation_ticks) < minimum:
        raise ValueError(
            f"Not enough data: train={len(train_ticks)} validation={len(validation_ticks)}, "
            f"need >= {minimum} each"
        )

    train_env = TradingEnvironment(train_ticks, config=config)
    if agent is None:
        agent = create_agent(train_env, config=config)

    logger.info(
        "Cycle %d/%d | train=%d ticks | validation=%d ticks | steps=%d",
        cycle, cycles, len(train_ticks), len(validation_ticks), train_steps,
    )
    agent.learn(train_env, train_steps)

    train_metrics = evaluate_policy(agent, train_ticks, config)
    validation_metrics = evaluate_policy(agent, validation_ticks, config)

    logger.info(
        "Cycle %d | train return %.2f%% | validation return %.2f%% | trades %d | dd %.2f%%",
        cycle,
        train_metrics["return_pct"],
        validation_metrics["return_pct"],
        int(validation_metrics["trades"]),
        validation_metrics["drawdown"] * 100,
    )

    return {
        "agent": agent,
        "cycle": cycle,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "train_ticks": len(train_ticks),
        "validation_ticks": len(validation_ticks),
    }
