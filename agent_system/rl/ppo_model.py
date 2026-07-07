"""Stable-Baselines3 PPO model integration."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO

from agent_system.data.data_buffer import MarketTick
from agent_system.environment.trading_env import TradingEnvironment


class GymTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ticks: list[MarketTick]) -> None:
        super().__init__()
        self.env = TradingEnvironment(ticks)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.array(self.env.reset(), dtype=np.float32), {}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        state, reward, done, info = self.env.step(int(action))
        return np.array(state, dtype=np.float32), float(reward), done, False, info


def build_ppo_model(env: gym.Env, tensorboard_log: str | None = "logs/ppo") -> PPO:
    os.makedirs(tensorboard_log or "logs/ppo", exist_ok=True)
    return PPO(
        "MlpPolicy",
        env,
        batch_size=256,
        gamma=0.95,
        learning_rate=3e-4,
        n_steps=256,
        policy_kwargs={
            "activation_fn": nn.ReLU,
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        },
        tensorboard_log=tensorboard_log,
        verbose=0,
    )


def save_checkpoint(model: PPO, path: str | Path) -> None:
    requested_path = Path(path)
    final_path = requested_path if requested_path.suffix == ".zip" else requested_path.with_suffix(".zip")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_name(f"{final_path.stem}.tmp{final_path.suffix}")
    model.save(str(temp_path))
    shutil.move(str(temp_path), str(final_path))


def load_checkpoint(path: str | Path, env: gym.Env | None = None) -> PPO:
    return PPO.load(str(path), env=env)
