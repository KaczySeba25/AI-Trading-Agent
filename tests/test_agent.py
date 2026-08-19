import numpy as np

from agent_system.data.feature_engineering import FEATURE_DIM
from agent_system.rl.agent import HeuristicAgent, create_agent, load_agent
from agent_system.rl.environment import CLOSE, HOLD, LONG, SHORT, TradingEnvironment
from tests.conftest import make_ticks


def _observation(**features) -> np.ndarray:
    """Build an observation with the named features set on the last row."""
    from agent_system.data.feature_engineering import FEATURE_NAMES

    window = np.zeros((32, FEATURE_DIM), dtype=np.float32)
    for name, value in features.items():
        window[-1, FEATURE_NAMES.index(name)] = value
    account = np.zeros(6, dtype=np.float32)
    return np.concatenate([window.flatten(), account])


def test_heuristic_returns_a_valid_action(config) -> None:
    agent = HeuristicAgent(config)
    action = agent.act(_observation())

    assert action in {HOLD, LONG, SHORT, CLOSE}


def test_heuristic_buys_when_oversold(config) -> None:
    agent = HeuristicAgent(config)
    action = agent.act(_observation(bollinger_position=-1.0, rsi=-0.9, tick_imbalance=-0.8))

    assert action == LONG


def test_heuristic_sells_when_overbought(config) -> None:
    agent = HeuristicAgent(config)
    action = agent.act(_observation(bollinger_position=1.0, rsi=0.9, tick_imbalance=0.8))

    assert action == SHORT


def test_heuristic_holds_when_signal_is_weak(config) -> None:
    agent = HeuristicAgent(config)
    action = agent.act(_observation(bollinger_position=0.05, rsi=0.02))

    assert action == HOLD


def test_heuristic_survives_a_short_observation(config) -> None:
    agent = HeuristicAgent(config)

    assert agent.act(np.zeros(3)) == HOLD


def test_heuristic_trades_actively_over_a_session(config) -> None:
    """The whole point of the retune: the policy must actually trade."""
    env = TradingEnvironment(make_ticks(800), config=config)
    agent = HeuristicAgent(config)
    observation, _ = env.reset()

    for _ in range(400):
        observation, _, terminated, truncated, _ = env.step(agent.act(observation))
        if terminated or truncated:
            break

    metrics = env.metrics()
    assert metrics["trades"] >= 10
    assert metrics["trades_per_100_steps"] >= 2.0


def test_create_agent_falls_back_when_ppo_not_requested(config) -> None:
    agent = create_agent(config, prefer_ppo=False)

    assert isinstance(agent, HeuristicAgent)


def test_heuristic_save_and_load(config, tmp_path) -> None:
    agent = HeuristicAgent(config)
    path = agent.save(tmp_path / "heuristic")

    assert path.exists()
    assert isinstance(load_agent(tmp_path / "heuristic", config), HeuristicAgent)


def test_load_agent_returns_none_when_missing(config, tmp_path) -> None:
    assert load_agent(tmp_path / "nothing_here", config) is None
