"""Trading policies: a PPO agent and a heuristic fallback.

Stable-Baselines3 is optional on purpose. It is a heavy dependency with a
torch pin, and the rest of the system -- paper trading, backtests, the live
loop -- must keep working on a machine where it will not install. When SB3 is
missing, ``create_agent`` returns a rule-based policy with the same interface,
so nothing downstream needs to know the difference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from agent_system.core.config import Settings, settings
from agent_system.core.logger import get_logger
from agent_system.data.feature_engineering import FEATURE_DIM, FEATURE_NAMES
from agent_system.rl.environment import CLOSE, HOLD, LONG, SHORT

logger = get_logger(__name__)

try:  # pragma: no cover - depends on the install
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    SB3_AVAILABLE = True
except Exception:  # pragma: no cover - broad: SB3 can fail on torch import too
    PPO = None  # type: ignore[assignment]
    Monitor = None  # type: ignore[assignment]
    DummyVecEnv = None  # type: ignore[assignment]
    SB3_AVAILABLE = False


class Policy(Protocol):
    """The interface every policy must satisfy."""

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int: ...

    def learn(self, env: Any, steps: int) -> "Policy": ...

    def save(self, path: str | Path) -> Path: ...


_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


class HeuristicAgent:
    """Rule-based mean-reversion policy.

    Used when SB3 is unavailable and as the cold-start policy before the
    first model is trained. It trades a simple, well-understood edge: fade
    stretched moves back towards the mean, but only when the stretch is large
    enough to pay for the round trip.
    """

    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self.window = config.observation_window
        # Only fade a move that is worth more than the cost of trading it.
        # Everything below this line is noise the exchange gets paid for.
        self.entry_threshold = 0.55

    # -- interface -------------------------------------------------------
    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        features = self._last_features(observation)
        if features is None:
            return HOLD

        position = float(observation[-6])
        bars_held_ratio = float(observation[-2])

        bollinger = features[_FEATURE_INDEX["bollinger_position"]]
        rsi = features[_FEATURE_INDEX["rsi"]]
        trend = features[_FEATURE_INDEX["trend_ratio"]]
        imbalance = features[_FEATURE_INDEX["tick_imbalance"]]

        # Composite stretch score: agreement between a band position, an
        # oscillator and order flow is a far better signal than any one alone.
        score = 0.5 * bollinger + 0.3 * rsi + 0.2 * imbalance

        if position != 0.0:
            # Exit once the stretch is worked off, or the trade has gone stale.
            if bars_held_ratio >= 1.0:
                return CLOSE
            if position > 0 and score > 0.1:
                return CLOSE
            if position < 0 and score < -0.1:
                return CLOSE
            return HOLD

        if score <= -self.entry_threshold and trend <= 0.5:
            return LONG
        if score >= self.entry_threshold and trend >= -0.5:
            return SHORT
        return HOLD

    def learn(self, env: Any, steps: int) -> "HeuristicAgent":
        logger.info("HeuristicAgent has no trainable parameters; skipping %d steps", steps)
        return self

    def save(self, path: str | Path) -> Path:
        path = Path(path).with_suffix(".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"heuristic-agent\nentry_threshold={self.entry_threshold}\nwindow={self.window}\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path, config: Settings = settings) -> "HeuristicAgent":
        return cls(config)

    # -- helpers ---------------------------------------------------------
    def _last_features(self, observation: np.ndarray) -> np.ndarray | None:
        observation = np.asarray(observation, dtype=np.float32).flatten()
        if observation.size < FEATURE_DIM + 6:
            return None
        return observation[-(FEATURE_DIM + 6) : -6]


class PPOAgent:
    """Wraps a Stable-Baselines3 PPO model behind the ``Policy`` interface."""

    def __init__(self, config: Settings = settings, model: Any | None = None) -> None:
        if not SB3_AVAILABLE:
            raise RuntimeError("stable-baselines3 is not installed")
        self.config = config
        self.model = model

    def _build(self, env: Any) -> Any:
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config.learning_rate,
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            # A higher entropy bonus than the SB3 default. With a discrete
            # action space and a reward dominated by costs, PPO collapses onto
            # "always hold" very early; keeping entropy up forces it to keep
            # sampling real trades long enough to learn which ones pay.
            ent_coef=self.config.entropy_coefficient,
            clip_range=0.2,
            verbose=0,
            seed=self.config.seed,
            device="cpu",
            policy_kwargs={"net_arch": [256, 256]},
            tensorboard_log=str(Path(self.config.log_dir) / "ppo"),
        )

    def learn(self, env: Any, steps: int) -> "PPOAgent":
        wrapped = DummyVecEnv([lambda: Monitor(env)])
        if self.model is None:
            self.model = self._build(wrapped)
        else:
            self.model.set_env(wrapped)
        self.model.learn(total_timesteps=int(steps), reset_num_timesteps=False)
        return self

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        if self.model is None:
            return HOLD
        action, _ = self.model.predict(np.asarray(observation), deterministic=deterministic)
        return int(np.asarray(action).flatten()[0])

    def save(self, path: str | Path) -> Path:
        if self.model is None:
            raise RuntimeError("Nothing to save: model has not been trained")
        path = Path(path).with_suffix(".zip")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        return path

    @classmethod
    def load(cls, path: str | Path, config: Settings = settings) -> "PPOAgent":
        if not SB3_AVAILABLE:
            raise RuntimeError("stable-baselines3 is not installed")
        model = PPO.load(str(Path(path).with_suffix(".zip")), device="cpu")
        return cls(config, model=model)


def create_agent(config: Settings = settings, prefer_ppo: bool = True):
    """Best available policy: PPO when SB3 is installed, heuristic otherwise."""
    if prefer_ppo and SB3_AVAILABLE:
        return PPOAgent(config)
    if prefer_ppo:
        logger.warning("stable-baselines3 unavailable; falling back to HeuristicAgent")
    return HeuristicAgent(config)


def load_agent(path: str | Path, config: Settings = settings):
    """Load whichever policy was saved at ``path``, tolerating a missing file."""
    path = Path(path)
    zip_path = path.with_suffix(".zip")
    txt_path = path.with_suffix(".txt")

    if zip_path.exists() and SB3_AVAILABLE:
        try:
            return PPOAgent.load(zip_path, config)
        except Exception as exc:  # pragma: no cover - corrupt checkpoint
            logger.error("Could not load PPO model from %s: %s", zip_path, exc)
    if txt_path.exists():
        return HeuristicAgent.load(txt_path, config)
    return None
