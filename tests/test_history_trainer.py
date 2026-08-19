"""Walk-forward training tests -- guarding against look-ahead bias."""

from __future__ import annotations

from agent_system.rl.agent import HeuristicAgent
from agent_system.rl.history_trainer import (
    evaluate_policy,
    select_training_window,
    split_train_validation,
    train_on_history,
)

from conftest import make_ticks


def test_split_keeps_the_newest_data_for_validation() -> None:
    train, validation = split_train_validation(make_ticks(100), 0.5)

    assert len(train) == 50
    assert len(validation) == 50
    # Validation must lie strictly in the future of training.
    assert validation[0].timestamp > train[-1].timestamp


def test_split_handles_empty_input() -> None:
    assert split_train_validation([], 0.2) == ([], [])


def test_split_always_leaves_both_sides_non_empty() -> None:
    train, validation = split_train_validation(make_ticks(10), 0.99)

    assert train and validation


def test_training_window_walks_forward_through_history() -> None:
    ticks = make_ticks(5000)

    first = select_training_window(ticks, cycle=1, cycles=3)
    last = select_training_window(ticks, cycle=3, cycles=3)

    assert len(first) == 2000
    assert len(last) == 2000
    assert last[0].timestamp > first[0].timestamp


def test_training_window_handles_short_history() -> None:
    ticks = make_ticks(100)

    assert len(select_training_window(ticks, cycle=1, cycles=3)) == 100


def test_evaluate_policy_returns_metrics(test_settings) -> None:
    metrics = evaluate_policy(HeuristicAgent(test_settings), make_ticks(200), test_settings)

    assert "return_pct" in metrics
    assert "drawdown" in metrics


def test_train_on_history_reports_both_splits(test_settings) -> None:
    result = train_on_history(
        make_ticks(400),
        cycle=1,
        cycles=1,
        train_steps=16,
        config=test_settings,
        agent=HeuristicAgent(test_settings),
    )

    assert "train_metrics" in result
    assert "validation_metrics" in result
    assert result["train_ticks"] > result["validation_ticks"]
