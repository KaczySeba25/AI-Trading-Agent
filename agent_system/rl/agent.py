"""PPO agent wrapper with a dependency-free fallback.

Stable-Baselines3 (and therefore PyTorch) is treated as an *optional* dependency.
On machines where torch cannot be installed, :class:`HeuristicAgent` keeps the
whole pipeline -- environment, backtester, paper loop, dashboard -- runnable and
testable. The public surface (`act`, `learn`, `save`, `load`) is identical, so
callers never branch on which backend is active.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from agent_system.core.config import Settings, settings
from agent_system.core.logger import get_logger
from agent_system.rl.environment import (
    ACTION_CLOSE,
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
    TradingEnvironment,
)

logger = get_logger(__name__)

try:  # pragma: no cover - depends on the host environment
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    SB3_AVAILABLE = True
except ImportError:  # pragma: no cover
    PPO = None  # type: ignore[assignment]
    Monitor = None  # type: ignore[assignment]
    DummyVecEnv = None  # type: ignore[assignment]
    SB3_AVAILABLE = False


class TradingPolicy(Protocol):
    """Minimal interface every agent backend implements."""

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int: ...
    def learn(self, env: TradingEnvironment, steps: int) -> None: ...
    def save(self, path: str | Path) -> Path: ...


class HeuristicAgent:
    """Trend/mean-reversion baseline used when SB3 is unavailable.

    It is intentionally simple, but it is a *real* baseline: any learned policy
    that cannot beat it after costs is not worth promoting.
    """

    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self.trained_steps = 0

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.size < 5:
            return ACTION_HOLD

        # Last 4 values are account state; the rest is the feature window.
        account = observation[-4:]
        features = observation[:-4].reshape(-1, 14)
        latest = features[-1]

        ret_5 = float(latest[1])
        rsi = float(latest[4])
        bollinger = float(latest[7])
        position_side = float(account[0])

        if position_side != 0:
            # Exit when the original edge has clearly gone.
            if position_side > 0 and (rsi > 0.6 or bollinger > 1.0):
                return ACTION_CLOSE
            if position_side < 0 and (rsi < -0.6 or bollinger < -1.0):
                return ACTION_CLOSE
            return ACTION_HOLD

        if rsi < -0.4 and ret_5 > 0:
            return ACTION_LONG
        if rsi > 0.4 and ret_5 < 0:
            return ACTION_SHORT
        return ACTION_HOLD

    def learn(self, env: TradingEnvironment, steps: int) -> None:
        # Nothing to fit; recorded so reports stay uniform across backends.
        self.trained_steps += steps

    def save(self, path: str | Path) -> Path:
        path = Path(path).with_suffix(".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"heuristic_agent trained_steps={self.trained_steps}\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path, config: Settings = settings) -> "HeuristicAgent":
        return cls(config)


class PPOAgent:
    """Stable-Baselines3 PPO with settings-driven hyperparameters."""

    def __init__(
        self,
        env: TradingEnvironment | None = None,
        config: Settings = settings,
        model: Any | None = None,
    ) -> None:
        if not SB3_AVAILABLE:  # pragma: no cover
            raise ImportError("stable-baselines3 is not installed; use HeuristicAgent")

        self.config = config
        self.model = model
        if model is None:
            if env is None:
                raise ValueError("PPOAgent requires either an env or a preloaded model")
            self.model = self._build(env)

    def _build(self, env: TradingEnvironment) -> Any:
        vec_env = DummyVecEnv([lambda: Monitor(env)])
        return PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=self.config.learning_rate,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # keeps exploring instead of collapsing to always-hold
            policy_kwargs={"net_arch": [256, 256]},
            tensorboard_log=str(self.config.log_dir / "ppo"),
            verbose=0,
            seed=self.config.seed,
        )

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.model.predict(np.asarray(observation), deterministic=deterministic)
        return int(np.asarray(action).flatten()[0])

    def learn(self, env: TradingEnvironment, steps: int) -> None:
        self.model.set_env(DummyVecEnv([lambda: Monitor(env)]))
        self.model.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=False)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        return path.with_suffix(".zip")

    @classmethod
    def load(cls, path: str | Path, config: Settings = settings) -> "PPOAgent":
        model = PPO.load(str(path))
        return cls(config=config, model=model)


def create_agent(env: TradingEnvironment | None = None, config: Settings = settings) -> TradingPolicy:
    """Return the best available backend, preferring PPO."""
    if SB3_AVAILABLE and env is not None:
        return PPOAgent(env=env, config=config)
    if not SB3_AVAILABLE:
        logger.warning("stable-baselines3 unavailable - falling back to HeuristicAgent")
    return HeuristicAgent(config=config)


def load_agent(path: str | Path, config: Settings = settings) -> TradingPolicy:
    """Load a saved agent, picking the backend from the file extension."""
    path = Path(path)
    if path.suffix == ".zip" or path.with_suffix(".zip").exists():
        if SB3_AVAILABLE:
            return PPOAgent.load(path, config)
        logger.warning("Saved PPO model found but SB3 unavailable - using HeuristicAgent")
    return HeuristicAgent.load(path, config)
