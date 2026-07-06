"""Longer public-history training workflow."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from stable_baselines3 import PPO

from agent_system.core.logger import get_logger
from agent_system.data.historical_client import HistoricalMarketDataClient
from agent_system.rl.model_evaluator import decide_model_promotion
from agent_system.rl.ppo_model import GymTradingEnv, build_ppo_model, save_checkpoint
from agent_system.rl.online_loop import OnlineLearningLoop


logger = get_logger(__name__)


class PublicHistoryTrainer:
    def __init__(
        self,
        model_dir: str = "models",
        report_dir: str = "data",
        log_dir: str = "logs/ppo",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.report_dir = Path(report_dir)
        self.log_dir = Path(log_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        interval: str = "1m",
        limit: int = 500,
        cycles: int = 3,
        train_steps: int = 1024,
    ) -> dict[str, object]:
        ticks = HistoricalMarketDataClient().fetch_klines(interval=interval, limit=limit)
        if len(ticks) < 2:
            raise ValueError("Need at least two historical ticks")

        current_path = self.model_dir / "ppo_latest.zip"
        best_path = self.model_dir / "ppo_best.zip"
        evaluator = OnlineLearningLoop(model_dir=str(self.model_dir), report_dir=str(self.report_dir))

        cycle_reports: list[dict[str, object]] = []
        best_score: dict[str, object] | None = None

        for cycle in range(1, cycles + 1):
            env = GymTradingEnv(ticks)
            if current_path.exists():
                try:
                    model = PPO.load(str(current_path), env=env)
                except Exception:
                    logger.warning("Existing checkpoint is unreadable; starting fresh")
                    model = build_ppo_model(env, tensorboard_log=str(self.log_dir))
            else:
                model = build_ppo_model(env, tensorboard_log=str(self.log_dir))

            started = time.time()
            model.learn(total_timesteps=train_steps)
            save_checkpoint(model, current_path)
            elapsed = round(time.time() - started, 3)

            evaluation = evaluator._evaluate(current_path, GymTradingEnv(ticks))
            decision = decide_model_promotion(best_score, evaluation)
            if decision.accepted:
                shutil.copy2(current_path, best_path)
                best_score = evaluation

            cycle_report: dict[str, object] = {
                "cycle": cycle,
                "train_steps": train_steps,
                "elapsed_seconds": elapsed,
                "evaluation": evaluation,
                "accepted": decision.accepted,
                "reason": decision.reason,
            }
            cycle_reports.append(cycle_report)
            logger.info("History training cycle complete: %s", cycle_report)

        report: dict[str, object] = {
            "timestamp": int(time.time()),
            "interval": interval,
            "limit": limit,
            "ticks": len(ticks),
            "cycles": cycles,
            "train_steps": train_steps,
            "best_score": best_score,
            "cycle_reports": cycle_reports,
        }
        report_path = self.report_dir / f"public_history_training_{report['timestamp']}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
