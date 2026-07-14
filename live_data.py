import json
import asyncio
import websockets
import pandas as pd
import requests
from datetime import datetime, timedelta

class LiveDataFeed:
    def __init__(self, symbol="btcusdt", interval="1m", max_history=50000):
        self.symbol = symbol.lower()
        self.interval = interval
        self.max_history = max_history
        self.df = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

        self._load_history_month()

    def _load_history_month(self):
        print("Ładuję miesiąc historii z Binance REST API...")

        end = datetime.utcnow()
        start = end - timedelta(days=30)

        base_url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": self.symbol.upper(),
            "interval": self.interval,
            "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
            "limit": 1000
        }

        all_rows = []

        while True:
            r = requests.get(base_url, params=params)
            data = r.json()

            if not data:
                break

            for k in data:
                ts = datetime.fromtimestamp(k[0]/1000)
                row = {
                    "timestamp": ts,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5])
                }
                all_rows.append(row)

            last_ts = data[-1][0]
            params["startTime"] = last_ts + 1

        self.df = pd.DataFrame(all_rows)
        if len(self.df) > self.max_history:
            self.df = self.df.iloc[-self.max_history:]

        print(f"Załadowano {len(self.df)} świec (ok. miesiąc historii).")

    async def connect(self):
        url = f"wss://stream.binance.com:9443/ws/{self.symbol}@kline_{self.interval}"
        async with websockets.connect(url) as ws:
            print("Connected to Binance WebSocket")

            while True:
                msg = await ws.recv()
                data = json.loads(msg)

                k = data["k"]

                row = {
                    "timestamp": datetime.fromtimestamp(k["t"]/1000),
                    "open": float(k["o"]),
                    "high": float(k["h"]),
                    "low": float(k["l"]),
                    "close": float(k["c"]),
                    "volume": float(k["v"])
                }

                self.df.loc[len(self.df)] = row

                if len(self.df) > self.max_history:
                    self.df = self.df.iloc[-self.max_history:]

                print(f"Live candle: {row['close']}")

    def get_history(self):
        return self.df.copy()
