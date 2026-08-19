"""Training on historical data with a held-out validation split."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from agent_system.core.config import Settings, settings
from agent_system.core.logger import get_logger
from agent_system.data.data_buffer import MarketTick
from agent_system.data.feature_engineering import FeatureEngine
from agent_system.rl.agent import create_agent
from agent_system.rl.environment import TradingEnvironment

logger = get_logger(__name__)

#: Default number of ticks used per training cycle.
DEFAULT_WINDOW_SIZE = 2000


def split_train_validation(
    ticks: Sequence[MarketTick],
    validation_fraction: float = 0.2,
) -> tuple[list[MarketTick], list[MarketTick]]:
    """Split chronologically, newest data reserved for validation.

    Never shuffle time series before splitting. Random splits let the model
    see the future of a period it is validated on, which produces beautiful
    validation numbers and a policy that loses money live.

    The validation slice is floored at half the data so that even a small
    history yields a validation set big enough for its metrics to mean
    anything.
    """
    ticks = list(ticks)
    if len(ticks) < 4:
        return ticks, []

    fraction = min(max(validation_fraction, 0.05), 0.5)
    split_at = int(len(ticks) * (1.0 - fraction))
    split_at = min(split_at, len(ticks) // 2)
    split_at = max(split_at, 1)
    return ticks[:split_at], ticks[split_at:]


def select_training_window(
    ticks: Sequence[MarketTick],
    cycle: int,
    cycles: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> list[MarketTick]:
    """Pick a contiguous slice of history for this cycle.

    Successive cycles walk forward through time instead of re-reading the
    same slice, so over a run the agent sees every regime in the history
    rather than overfitting the most recent one.
    """
    ticks = list(ticks)
    if len(ticks) <= window_size:
        return ticks

    cycles = max(cycles, 1)
    cycle = max(min(cycle, cycles), 1)
    max_start = len(ticks) - window_size
    if cycles == 1:
        start = max_start
    else:
        start = int(round(max_start * (cycle - 1) / (cycles - 1)))
    start = max(0, min(start, max_start))
    return ticks[start : start + window_size]


def evaluate_policy(
    agent: Any,
    ticks: Sequence[MarketTick],
    config: Settings = settings,
    feature_engine: FeatureEngine | None = None,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run a policy over ``ticks`` and return the environment metrics."""
    if len(ticks) < 64:
        return {"error": "not_enough_ticks", "ticks": len(ticks)}

    env = TradingEnvironment(ticks, config=config, feature_engine=feature_engine)
    observation, _ = env.reset()
    total_reward = 0.0

    while True:
        action = agent.act(observation, deterministic=deterministic)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    metrics = env.metrics()
    metrics["total_reward"] = round(total_reward, 4)
    return metrics


def train_on_history(
    ticks: Sequence[MarketTick],
    config: Settings = settings,
    agent: Any | None = None,
    steps: int | None = None,
    validation_fraction: float = 0.2,
) -> dict[str, Any]:
    """Train on the older part of ``ticks`` and validate on the newer part."""
    train_ticks, validation_ticks = split_train_validation(ticks, validation_fraction)
    if len(train_ticks) < 64:
        raise ValueError(f"Not enough training data: {len(train_ticks)} ticks")

    feature_engine = FeatureEngine()
    steps = int(steps or config.train_steps)

    # Features are expensive and deterministic, so compute them once and hand
    # the same matrix to every environment built from this slice.
    train_features = feature_engine.compute_many(train_ticks, config.observation_window)
    train_env = TradingEnvironment(
        train_ticks,
        config=config,
        feature_engine=feature_engine,
        precomputed_features=train_features,
    )

    agent = agent or create_agent(config)
    logger.info("Training %s for %d steps on %d ticks", type(agent).__name__, steps, len(train_ticks))
    agent.learn(train_env, steps)

    train_metrics = evaluate_policy(agent, train_ticks, config, feature_engine)
    validation_metrics = (
        evaluate_policy(agent, validation_ticks, config, feature_engine)
        if len(validation_ticks) >= 64
        else {"error": "not_enough_validation_ticks"}
    )

    return {
        "agent": agent,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "train_ticks": len(train_ticks),
        "validation_ticks": len(validation_ticks),
    }
