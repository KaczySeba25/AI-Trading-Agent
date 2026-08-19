# AI Trading Agent

An autonomous reinforcement-learning trading agent for Binance Futures. It
trains on historical candles, trades a virtual account, keeps what it learns
across restarts, and promotes a new model only when that model beats the one
currently in production.

It trades **paper money by default**. Real trading requires deliberately
setting `ENABLE_LIVE_TRADING=true`; nothing in the default configuration can
spend real funds.

## Quick start

No API keys and no network access are needed for any of the commands below.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/generate_sample_data.py      # writes data/BTCUSDT.csv
python -m agent_system.main --mode backtest --csv data/BTCUSDT.csv
```

> The sample data is **random noise**. It exists so the code paths can be
> exercised offline. No strategy can be profitable on it after costs, and a
> negative return there says nothing about the agent. See
> [Judging profitability](#judging-profitability).

## Modes

```bash
python -m agent_system.main --mode <mode> [options]
```

| Mode | What it does | Network |
| --- | --- | --- |
| `status` | Prints saved state: cycles completed, champion metrics, replay buffer size. | No |
| `backtest` | Runs the policy over a CSV and reports return, alpha vs buy-and-hold, drawdown and trade stats. | No |
| `paper` | Trades a simulated account tick by tick with full fee and slippage modelling. | No |
| `train` | Trains a PPO policy on historical candles with a walk-forward split. | No |
| `autonomous` | The main loop: train, evaluate, promote or reject, paper-trade, repeat. Persists between runs. | Optional |
| `stream` | Connects to the public Binance WebSocket and prints live features. | Yes |
| `testnet-diagnostic` | Verifies Testnet credentials and read access. Never submits an order. | Yes |

Common options: `--csv PATH`, `--steps N`, `--cycles N` (`0` = run forever),
`--train-steps N`, `--interval 1m`, `--days 7`, `--max-rows 10000`,
`--sleep-seconds S`, `--reset`.

Modes that need candles try Binance first and fall back to a local CSV
(`--csv`, otherwise `data/{SYMBOL}.csv`), so everything works offline.

### Running it continuously

```bash
python -m agent_system.main --mode autonomous --cycles 0 --sleep-seconds 300
```

Each cycle trains a candidate, evaluates it and either promotes it to champion
or throws it away, then runs a paper session. State is written to `state/` and
models to `models/` after every cycle, so `Ctrl+C` or a reboot costs you at
most the cycle in flight. It handles `SIGINT`/`SIGTERM` cleanly and is meant to
be left running for days.

## How it decides to trade

The environment models what actually erodes returns at high turnover:

- **Costs on every fill.** Maker and taker fees plus slippage, charged on entry
  and exit. `round_trip_cost_fraction` is roughly 0.06% maker-to-maker, and the
  agent sees it as an observation feature.
- **Exits.** ATR-scaled stop and target, a trailing stop, and a maximum holding
  period. A target is never allowed below twice the round-trip cost, so no
  trade is taken that only a zero-fee world would win.
- **Reward.** Per-step equity change, minus a drawdown penalty, minus an
  inactivity penalty (pushing turnover up) and a churn penalty (pushing back).
  Trading frequency is the balance of those two terms — tune them in `.env`.

Promotion is gated in `agent_system/rl/model_evaluator.py`: a candidate needs
at least 3 trades, drawdown under 15%, **profit factor above 1.0**, and it must
not be worse than the sitting champion. On random data this correctly refuses
to promote anything.

## Judging profitability

Two things are being asked of the agent at once — trade often, and make money —
and they pull against each other. Every round trip pays the spread and fees, so
frequency only pays if per-trade edge clears that hurdle.

To evaluate it honestly you need **real candles**:

```bash
python -m agent_system.main --mode backtest --interval 1m --days 30
```

Then check, in this order: profit factor above 1.0, `alpha_pct` positive (it
beat buy-and-hold, not just the market), and `fees_paid` against gross profit.
Backtest on one period, validate on a later one you never tuned against, then
paper-trade for weeks before considering anything else.

## Configuration

Copy `.env.example` to `.env.local` and edit. Every value is documented there:
symbol and capital, credentials, risk limits, fee assumptions, exit behaviour,
reward shaping and learning rates.

Set the fee variables to **your** actual tier. Optimistic fees are the fastest
way to make a losing strategy look profitable.

## Development

```bash
python -m pytest -q                    # 56 tests, no network, torch optional
python scripts/check_no_secrets.py     # entropy-based credential scan
```

The RL dependencies (`torch`, `stable-baselines3`) are optional: without them
the agent falls back to a rule-based policy and everything still runs. CI
deliberately tests that path — see [`ci/README.md`](ci/README.md) for enabling
the GitHub Actions workflow.

`state/`, `models/`, `logs/` and `data/*.csv` are generated at runtime and
gitignored.

## Layout

```
agent_system/
  core/        configuration, logging, exceptions
  data/        WebSocket feed, REST klines, feature engineering, buffers
  rl/          environment, agent, memory, evaluator, backtest, autonomous loop
  execution/   paper broker, risk manager, order manager, exchange filters
tests/         56 tests
scripts/       sample data generator, secret scanner, Binance setup helper
```

## Safety

- Paper trading by default; live trading requires an explicit opt-in flag.
- The order manager rounds to the symbol's `tickSize`, `stepSize`, `minQty` and
  `MIN_NOTIONAL` before submitting anything.
- The risk manager enforces a daily loss limit, a consecutive-loss cutout, a
  per-day trade cap and position sizing; it halts on breach.
- `testnet-diagnostic` never places an order.
- Secrets are read from the environment and never logged. `.env` and
  `.env.local` are gitignored, and `scripts/check_no_secrets.py` fails the
  build if a key gets committed.

Nothing here is financial advice. Cryptocurrency trading can lose you money
quickly, and an agent that performed well in a backtest can still fail live.
