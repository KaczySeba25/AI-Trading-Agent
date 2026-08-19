"""Generate a synthetic OHLCV file for offline development.

WARNING -- READ THIS BEFORE DRAWING ANY CONCLUSION FROM IT.

This data is random. It is a geometric random walk with a volatility cycle
bolted on so the ATR-based logic has something to react to. It contains no
real market structure, no genuine mean reversion, and no exploitable edge.

That has one unavoidable consequence: **no strategy can be profitable on this
file after costs**. Any positive return you see here is luck in one seed, and
it will not survive a different seed. Use this data to check that the code
runs, that the shapes line up and that trades are being opened and closed --
never to decide whether the agent makes money. For that you need real klines,
which ``--mode backtest`` downloads when the network is available.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def generate(rows: int, start_price: float, seed: int) -> list[list[object]]:
    rng = np.random.default_rng(seed)
    timestamp = 1_704_067_200_000  # 2024-01-01T00:00:00Z
    price = start_price
    out: list[list[object]] = []

    for index in range(rows):
        # Volatility cycles slowly so the ATR-scaled stops face both calm and
        # stormy regimes rather than one constant level of noise.
        vol = 0.0006 * (1.0 + 0.6 * math.sin(index / 180.0))
        drift = -0.5 * vol**2
        step = math.exp(drift + vol * float(rng.standard_normal()))

        open_price = price
        close = price * step
        high = max(open_price, close) * (1 + abs(float(rng.standard_normal())) * vol * 0.5)
        low = min(open_price, close) * (1 - abs(float(rng.standard_normal())) * vol * 0.5)
        volume = float(abs(rng.normal(12.0, 4.0))) + 0.1

        out.append(
            [
                timestamp + index * 60_000,
                round(open_price, 2),
                round(high, 2),
                round(low, 2),
                round(close, 2),
                round(volume, 4),
            ]
        )
        price = close

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--start-price", type=float, default=42000.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="data/BTCUSDT.csv")
    args = parser.parse_args()

    rows = generate(args.rows, args.start_price, args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic candles to {path} (seed={args.seed})")
    print("This data is random: do NOT use it to judge profitability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
