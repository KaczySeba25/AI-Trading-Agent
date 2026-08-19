# AI Trading Agent

Autonomous reinforcement-learning trading agent for Binance Futures (BTCUSDT by default).

It learns from historical data, validates every new model on unseen newer data, and only
promotes a model to "champion" if it clears explicit quality gates. The champion then trades
**virtual money** on live prices, remembers what it learned across restarts, and keeps
looping — designed to run unattended for days.

> **Safety.** Real-money trading is off by default and requires two deliberate opt-ins
> (`ENABLE_TESTNET_TRADING` / `ENABLE_LIVE_TRADING` plus real API keys). Nothing in the
> default configuration can place an order on an exchange.

## Architecture

```
agent_system/
  core/         config (env + .env.local), logging, exception hierarchy
  data/         Binance WebSocket stream, rolling tick buffer, feature engine,
                historical klines client
  rl/           trading environment, agent (PPO + heuristic fallback), replay
                memory & journal, promotion gate, walk-forward trainer,
                backtester, 24/7 autonomous loop
  execution/    exchange filters, risk manager, position tracker, Binance REST
                client, order manager, paper broker, testnet diagnostics
  main.py       command-line entry point
```

One feature engine, one environment, one agent — every mode (backtest, training, paper,
autonomous) runs the exact same code path, so a backtest result means something.

### Key design decisions

| Concern | Decision |
| --- | --- |
| Look-ahead bias | Walk-forward split; validation data is always **newer** than training data |
| Reward | Per-step *increment* in equity (not the equity level), minus a drawdown penalty |
| Costs | Taker fee + slippage charged on every fill; slippage always moves against the trader |
| Feature scale | All 14 features are dimensionless, so a model trained at \$40k works at \$400k |
| Deployment safety | Champion/challenger: a bad training cycle produces a rejected candidate, not a drained account |
| Persistence | Model, replay buffer, promotion journal and paper account are written atomically after every cycle |

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip.exe
cp .env.example .env.local                    # optional; no keys needed for anything below
```

```bash
.venv/bin/python -m agent_system.main --mode status
.venv/bin/python -m agent_system.main --mode backtest --csv data/BTCUSDT.csv
.venv/bin/python -m agent_system.main --mode train --cycles 3 --train-steps 4096
.venv/bin/python -m agent_system.main --mode paper --steps 500
.venv/bin/python -m agent_system.main --mode autonomous --cycles 0 --sleep-seconds 300
```

## Modes

| Mode | What it does |
| --- | --- |
| `status` | Prints the effective configuration and what the agent currently remembers |
| `stream` | Connects to the public Binance WebSocket and prints live tick statistics |
| `backtest` | Runs the champion (or the heuristic baseline) over history and reports return, alpha vs buy & hold, win rate, profit factor, drawdown and Sharpe |
| `train` | Walk-forward training cycles; each candidate is validated on unseen newer data and promoted only if it passes the gates |
| `paper` | Trades the champion on virtual money and persists the account |
| `autonomous` | The 24/7 loop: fetch history → train → validate → promote → paper-trade → persist → sleep → repeat. `--cycles 0` runs forever |
| `testnet-diagnostic` | Read-only credential/clock/permission check. **Never submits an order** |

Useful flags: `--interval 1m --days 30 --max-rows 20000 --csv PATH --cycles N
--train-steps N --steps N --sleep-seconds N --reset`.

Every mode works offline: if the Binance API is unreachable, the CLI falls back to
`data/<SYMBOL>.csv`.

## Running 24/7

```bash
nohup .venv/bin/python -m agent_system.main --mode autonomous \
      --cycles 0 --interval 1m --days 30 --sleep-seconds 300 > logs/autonomous.log 2>&1 &
```

The loop handles `SIGINT`/`SIGTERM` gracefully: it finishes the current cycle, flushes state
and exits. State lives in:

- `models/champion*` — the promoted model
- `models/candidate*` — the model currently under evaluation
- `state/journal.json` — every cycle, its metrics and the promotion decision
- `state/replay_buffer.pkl` — experience that survives restarts
- `state/paper_account.json` — the virtual account

Deleting `state/` and `models/` resets the agent to a blank slate.

## Promotion gates

A candidate becomes the champion only if **all** of these hold:

1. at least 3 closed trades (not a fluke),
2. max drawdown ≤ 15%,
3. profit factor ≥ 1.0,
4. final equity ≥ 99.5% of the champion's,
5. drawdown no worse than 2× the champion's.

Rejection reasons are recorded in the journal, so you can see *why* a model was refused.

## Going live (deliberate, later)

1. Run `--mode autonomous` on paper for several days and read `state/journal.json`.
2. Create Binance Futures **Testnet** keys, put them in `.env.local`, run
   `--mode testnet-diagnostic`.
3. Set `ENABLE_TESTNET_TRADING=true` and let it trade the testnet.
4. Only then consider real keys and `ENABLE_LIVE_TRADING=true`, starting with the smallest
   possible `MAX_POSITION_FRACTION`.

Steps 3 and 4 are your decision. The agent will never enable them on its own.

## Validation

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q agent_system
.venv/bin/python scripts/check_no_secrets.py
```

## Environment variables

See [`.env.example`](.env.example) for the full annotated list. Secrets belong in
`.env.local`, which is git-ignored — never commit keys.
