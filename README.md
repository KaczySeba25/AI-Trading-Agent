# AI Trading Agent

Autonomous RL-based cryptocurrency trading system.

## Status

Initial autonomous trading scaffold is implemented without requiring private API credentials.

## Current Modes

```powershell
.venv\Scripts\python.exe -m agent_system.main --mode stream
.venv\Scripts\python.exe -m agent_system.main --mode train-smoke
.venv\Scripts\python.exe -m agent_system.main --mode online-smoke
.venv\Scripts\python.exe -m agent_system.main --mode paper-smoke
.venv\Scripts\python.exe -m agent_system.main --mode live-paper --min-ticks 120 --timeout-seconds 45
.venv\Scripts\python.exe -m agent_system.main --mode paper-loop --cycles 0 --min-ticks 120 --timeout-seconds 45 --sleep-seconds 10
.venv\Scripts\python.exe -m agent_system.main --mode public-history-train --interval 1m --limit 500 --train-steps 1024
.venv\Scripts\python.exe -m agent_system.main --mode public-history-train --interval 1m --limit 500 --train-steps 1024 --online-cycle
.venv\Scripts\python.exe -m agent_system.main --mode history-learning-loop --interval 1m --limit 500 --cycles 3 --train-steps 1024
.venv\Scripts\python.exe -m agent_system.main --mode autonomous-learning-loop --interval 1m --days 365 --max-rows 50000 --cycles 0 --history-cycles 3 --train-steps 2048 --min-ticks 120 --timeout-seconds 60 --sleep-seconds 300
.venv\Scripts\python.exe -m agent_system.main --mode testnet-diagnostic
```

`stream` connects to public Binance WebSocket market data. Execution modules use Binance Futures Testnet credentials from environment variables and do not print secrets.
`live-paper` collects live public market data, runs the PPO policy through simulated execution, and writes a report without sending any real order.
`paper-loop` repeats live paper cycles. `--cycles 0` means run continuously.
`public-history-train` downloads public Binance BTCUSDT klines, trains PPO, and writes paper replay reports. Add `--online-cycle` when you want the heavier candidate/rollback learning cycle too.
`history-learning-loop` repeats public-history training cycles, evaluates each model, and promotes only candidates that pass basic trade/drawdown/equity gates.
`autonomous-learning-loop` runs continuous public-history learning plus live paper checks. `--cycles 0` means run continuously.
`testnet-diagnostic` validates Binance Futures Testnet credentials and account read access without submitting orders.

## Validation

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q agent_system
.venv\Scripts\python.exe scripts\check_no_secrets.py
```

## Execution Safety

The order manager loads public Binance Futures Testnet `exchangeInfo` before submitting orders and rounds prices/quantities to the symbol filters:

- `PRICE_FILTER.tickSize`
- `LOT_SIZE.stepSize`
- `LOT_SIZE.minQty`
- `MIN_NOTIONAL`

No private credentials are required for this validation. Private Testnet credentials are only needed when actually submitting orders.

## Environment Variables

- `TRADING_SYMBOL`, default `BTCUSDT`
- `INITIAL_CAPITAL`, default `500`
- `MAX_OPEN_POSITIONS`, default `8`
- `PUBLIC_MARKET_DATA_BASE_URL`, default `https://api.binance.com`
- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `ENABLE_TESTNET_TRADING`, default `false`

## Local Binance Setup

Use the interactive local setup script. It writes secrets only to `.env.local`, which is ignored by git, and runs a read-only Testnet diagnostic.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_local_binance.ps1
```

The script keeps `ENABLE_TESTNET_TRADING=false`, so it does not submit orders.
- `BINANCE_TESTNET_BASE_URL`, default `https://testnet.binancefuture.com`
- `MAX_POSITION_FRACTION`, default `0.02`
- `DEFAULT_LEVERAGE`, default `3`
- `MAX_LEVERAGE`, default `5`
- `STOP_LOSS_FRACTION`, default `0.003`
- `DAILY_LOSS_LIMIT_FRACTION`, default `0.05`
