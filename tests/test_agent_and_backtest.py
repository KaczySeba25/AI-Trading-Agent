"""Agent and backtest integration tests."""

from __future__ import annotations

import numpy as np

from agent_system.rl.agent import HeuristicAgent, create_agent, load_agent
from agent_system.rl.backtest import buy_and_hold_return, run_backtest
from agent_system.rl.environment import TradingEnvironment

from conftest import make_ticks


def test_heuristic_agent_returns_valid_actions(test_settings, ticks) -> None:
    env = TradingEnvironment(ticks, config=test_settings)
    observation, _ = env.reset()
    agent = HeuristicAgent(test_settings)

    for _ in range(50):
        action = agent.act(observation)
        assert action in {0, 1, 2, 3}
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break


def test_agent_handles_a_degenerate_observation(test_settings) -> None:
    assert HeuristicAgent(test_settings).act(np.zeros(3, dtype=np.float32)) == 0


def test_create_agent_always_returns_something_usable(test_settings, ticks) -> None:
    env = TradingEnvironment(ticks, config=test_settings)
    agent = create_agent(env, test_settings)

    assert hasattr(agent, "act")
    assert hasattr(agent, "learn")
    assert hasattr(agent, "save")


def test_agent_round_trips_through_disk(test_settings, tmp_path) -> None:
    agent = HeuristicAgent(test_settings)
    path = agent.save(tmp_path / "agent")

    assert path.exists()
    assert load_agent(tmp_path / "agent", test_settings) is not None


def test_buy_and_hold_benchmark() -> None:
    rising = make_ticks(100, start_price=100.0, trend=0.01, volatility=0.0)

    assert buy_and_hold_return(rising) > 0
    assert buy_and_hold_return([]) == 0.0


def test_backtest_report_contains_benchmark_and_alpha(test_settings, wave_ticks) -> None:
    report = run_backtest(
        HeuristicAgent(test_settings), wave_ticks, test_settings, save_report=False
    )

    assert "metrics" in report
    assert "benchmark_return_pct" in report
    assert "alpha_pct" in report
    assert report["alpha_pct"] == (
        report["metrics"]["return_pct"] - report["benchmark_return_pct"]
    )
    assert len(report["equity_curve"]) > 1


def test_backtest_writes_a_report_file(test_settings, ticks) -> None:
    report = run_backtest(
        HeuristicAgent(test_settings), ticks, test_settings, save_report=True, label="unit"
    )

    assert "report_path" in report
