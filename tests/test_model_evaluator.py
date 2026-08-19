"""Promotion gate tests -- the guard against deploying a bad model."""

from __future__ import annotations

from agent_system.rl.model_evaluator import decide_model_promotion


def test_rejects_candidate_without_enough_trades() -> None:
    decision = decide_model_promotion(
        None,
        {"equity": 501, "trades": 0, "metrics": {"drawdown": 0.01, "profit_factor": 2.0}},
    )

    assert not decision.accepted
    assert decision.reason == "not_enough_trades"


def test_accepts_first_model_when_there_is_no_champion() -> None:
    decision = decide_model_promotion(
        None,
        {"equity": 520, "trades": 10, "metrics": {"drawdown": 0.02, "profit_factor": 1.5}},
    )

    assert decision.accepted
    assert decision.reason == "no_champion_yet"


def test_accepts_candidate_with_better_equity_and_low_drawdown() -> None:
    decision = decide_model_promotion(
        {"equity": 500, "trades": 3, "metrics": {"drawdown": 0.01}},
        {"equity": 501, "trades": 3, "metrics": {"drawdown": 0.02, "profit_factor": 1.2}},
    )

    assert decision.accepted
    assert decision.reason == "candidate_not_worse"


def test_rejects_low_profit_factor() -> None:
    decision = decide_model_promotion(
        None,
        {"equity": 501, "trades": 4, "metrics": {"drawdown": 0.01, "profit_factor": 0.8}},
    )

    assert not decision.accepted
    assert decision.reason == "profit_factor_too_low"


def test_rejects_excessive_drawdown_even_when_profitable() -> None:
    """A 40% drawdown is unacceptable no matter how good the return looks."""
    decision = decide_model_promotion(
        None,
        {"equity": 2000, "trades": 50, "metrics": {"drawdown": 0.40, "profit_factor": 3.0}},
    )

    assert not decision.accepted
    assert decision.reason == "drawdown_too_high"


def test_rejects_candidate_worse_than_champion() -> None:
    decision = decide_model_promotion(
        {"equity": 1000, "trades": 10, "metrics": {"drawdown": 0.02}},
        {"equity": 800, "trades": 10, "metrics": {"drawdown": 0.03, "profit_factor": 1.1}},
    )

    assert not decision.accepted
    assert decision.reason == "worse_than_champion"


def test_rejects_drawdown_regression() -> None:
    decision = decide_model_promotion(
        {"equity": 1000, "trades": 10, "metrics": {"drawdown": 0.02}},
        {"equity": 1010, "trades": 10, "metrics": {"drawdown": 0.09, "profit_factor": 1.1}},
    )

    assert not decision.accepted
    assert decision.reason == "drawdown_regression"


def test_no_candidate_is_rejected() -> None:
    assert not decide_model_promotion(None, None).accepted
