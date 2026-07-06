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
```

`stream` connects to public Binance WebSocket market data. Execution modules use Binance Futures Testnet credentials from environment variables and do not print secrets.
`live-paper` collects live public market data, runs the PPO policy through simulated execution, and writes a report without sending any real order.
`paper-loop` repeats live paper cycles. `--cycles 0` means run continuously.

## Validation

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q agent_system
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
- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `BINANCE_TESTNET_BASE_URL`, default `https://testnet.binancefuture.com`
- `MAX_POSITION_FRACTION`, default `0.02`
- `DEFAULT_LEVERAGE`, default `3`
- `MAX_LEVERAGE`, default `5`
- `STOP_LOSS_FRACTION`, default `0.003`
- `DAILY_LOSS_LIMIT_FRACTION`, default `0.05`
