"""PPO training helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path

from agent_system.core.logger import get_logger
from agent_system.data.data_buffer import MarketTick
from agent_system.rl.ppo_model import GymTradingEnv, build_ppo_model, save_checkpoint


logger = get_logger(__name__)


def train_ppo(
    ticks: list[MarketTick],
    total_timesteps: int = 256,
    model_dir: str = "models",
    log_dir: str = "logs/ppo",
) -> dict[str, str | int | float]:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    env = GymTradingEnv(ticks)
    model = build_ppo_model(env, tensorboard_log=log_dir)

    started = time.time()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.time() - started

    checkpoint_path = Path(model_dir) / "ppo_latest.zip"
    save_checkpoint(model, checkpoint_path)

    result: dict[str, str | int | float] = {
        "status": "trained",
        "total_timesteps": total_timesteps,
        "elapsed_seconds": round(elapsed, 3),
        "checkpoint": str(checkpoint_path),
    }
    report_path = Path(log_dir) / "training_summary.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("PPO training complete: %s", result)
    return result
