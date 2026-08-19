from agent_system.rl.agent import HeuristicAgent
from agent_system.rl.backtest import buy_and_hold_return, compare_agents, run_backtest
from tests.conftest import make_ticks


def test_buy_and_hold_matches_price_change() -> None:
    ticks = make_ticks(100)
    expected = (ticks[-1].price / ticks[0].price - 1) * 100

    assert abs(buy_and_hold_return(ticks) - expected) < 1e-6


def test_buy_and_hold_handles_empty_input() -> None:
    assert buy_and_hold_return([]) == 0.0


def test_backtest_reports_alpha_against_the_benchmark(config) -> None:
    report = run_backtest(
        HeuristicAgent(config), make_ticks(400), config, save_report=False, label="test"
    )

    assert report["alpha_pct"] == round(
        report["metrics"]["return_pct"] - report["benchmark_return_pct"], 4
    )
    assert set(report["actions"]) == {"hold", "long", "short", "close"}
    assert len(report["equity_curve"]) > 1


def test_backtest_quotes_the_cost_hurdle(config) -> None:
    """Reports must state the round-trip cost the strategy has to beat."""
    report = run_backtest(
        HeuristicAgent(config), make_ticks(400), config, save_report=False, label="test"
    )

    assert report["round_trip_cost_pct"] > 0


def test_compare_agents_runs_every_policy(config) -> None:
    result = compare_agents(
        {"a": HeuristicAgent(config), "b": HeuristicAgent(config)},
        make_ticks(400),
        config,
    )

    assert set(result["agents"]) == {"a", "b"}
    assert "benchmark_return_pct" in result
