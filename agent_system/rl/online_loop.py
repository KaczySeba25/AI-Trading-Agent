"""Online learning loop with model replacement and rollback."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from stable_baselines3 import PPO

from agent_system.data.data_buffer import MarketTick
from agent_system.environment.trading_env import TradingAction
from agent_system.monitoring.metrics import summarize_performance
from agent_system.rl.ppo_model import GymTradingEnv, build_ppo_model, save_checkpoint


class OnlineLearningLoop:
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

    def run_cycle(
        self,
        ticks: list[MarketTick],
        collect_steps: int = 10_000,
        train_steps: int = 256,
    ) -> dict[str, object]:
        selected_ticks = ticks[-collect_steps:] if len(ticks) > collect_steps else ticks
        if len(selected_ticks) < 2:
            raise ValueError("OnlineLearningLoop requires at least two ticks")

        current_path = self.model_dir / "ppo_latest.zip"
        candidate_path = self.model_dir / "ppo_candidate.zip"
        backup_path = self.model_dir / "ppo_rollback.zip"

        env = GymTradingEnv(selected_ticks)
        baseline_score = self._evaluate(current_path, env) if current_path.exists() else None

        if current_path.exists():
            shutil.copy2(current_path, backup_path)
            model = PPO.load(str(current_path), env=env)
        else:
            model = build_ppo_model(env, tensorboard_log=str(self.log_dir))

        model.learn(total_timesteps=train_steps)
        save_checkpoint(model, candidate_path)

        candidate_score = self._evaluate(candidate_path, GymTradingEnv(selected_ticks))
        improved = baseline_score is None or candidate_score["equity"] >= baseline_score["equity"]
        if improved:
            shutil.copy2(candidate_path, current_path)
            decision = "replace_model"
        elif backup_path.exists():
            shutil.copy2(backup_path, current_path)
            decision = "rollback_model"
        else:
            decision = "keep_existing"

        report: dict[str, object] = {
            "timestamp": int(time.time()),
            "collect_steps": len(selected_ticks),
            "train_steps": train_steps,
            "baseline": baseline_score,
            "candidate": candidate_score,
            "decision": decision,
        }
        report_path = self.report_dir / f"online_learning_{report['timestamp']}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def _evaluate(self, model_path: Path, env: GymTradingEnv) -> dict[str, object]:
        model = PPO.load(str(model_path), env=env)
        observation, _ = env.reset()
        done = False
        equity_curve: list[float] = []
        trade_pnls: list[float] = []
        returns: list[float] = []
        previous_equity: float | None = None
        last_realized = 0.0

        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, done, _, info = env.step(int(action))
            equity = float(info["equity"])
            realized = float(info["realized_pnl"])
            equity_curve.append(equity)
            if previous_equity:
                returns.append((equity - previous_equity) / previous_equity)
            if realized != last_realized:
                trade_pnls.append(realized - last_realized)
                last_realized = realized
            previous_equity = equity

        metrics = summarize_performance(trade_pnls, equity_curve, returns)
        return {
            "equity": equity_curve[-1] if equity_curve else 0.0,
            "trades": len(trade_pnls),
            "metrics": metrics,
            "last_action_space": [action.value for action in TradingAction],
        }
