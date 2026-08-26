import numpy as np
import pandas as pd

class FeatureExtractor88:
    def __init__(self):
        pass

    def safe(self, value):
        try:
            if value is None:
                return 0.0
            if isinstance(value, float) or isinstance(value, int):
                if np.isnan(value):
                    return 0.0
                return float(value)
            return float(value)
        except:
            return 0.0

    def compute(self, df):
        out = {}

        out["close"] = self.safe(df["close"].iloc[-1])
        out["open"] = self.safe(df["open"].iloc[-1])
        out["high"] = self.safe(df["high"].iloc[-1])
        out["low"] = self.safe(df["low"].iloc[-1])
        out["volume"] = self.safe(df["volume"].iloc[-1])

        out["delta_close"] = self.safe(df["close"].iloc[-1] - df["close"].iloc[-2])
        out["delta_volume"] = self.safe(df["volume"].iloc[-1] - df["volume"].iloc[-2])

        out["pct_change_1"] = self.safe(df["close"].pct_change().iloc[-1])
        out["pct_change_5"] = self.safe(df["close"].pct_change(5).iloc[-1])
        out["pct_change_15"] = self.safe(df["close"].pct_change(15).iloc[-1])

        out["volatility_10"] = self.safe(df["close"].pct_change().rolling(10).std().iloc[-1])
        out["volatility_20"] = self.safe(df["close"].pct_change().rolling(20).std().iloc[-1])
        out["volatility_50"] = self.safe(df["close"].pct_change().rolling(50).std().iloc[-1])

        out["momentum_5"] = self.safe(df["close"].iloc[-1] - df["close"].iloc[-5])
        out["momentum_10"] = self.safe(df["close"].iloc[-1] - df["close"].iloc[-10])
        out["momentum_20"] = self.safe(df["close"].iloc[-1] - df["close"].iloc[-20])

        for span in [5,10,20,50,100,200]:
            out[f"ema_{span}"] = self.safe(df["close"].ewm(span=span).mean().iloc[-1])

        for span in [5,10,20,50,100,200]:
            out[f"sma_{span}"] = self.safe(df["close"].rolling(span).mean().iloc[-1])

        delta = df["close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)

        rs14 = self.safe(up.rolling(14).mean().iloc[-1] / down.rolling(14).mean().iloc[-1])
        rs50 = self.safe(up.rolling(50).mean().iloc[-1] / down.rolling(50).mean().iloc[-1])

        out["rsi_14"] = self.safe(100 - (100 / (1 + rs14)))
        out["rsi_50"] = self.safe(100 - (100 / (1 + rs50)))

        ema12 = self.safe(df["close"].ewm(span=12).mean().iloc[-1])
        ema26 = self.safe(df["close"].ewm(span=26).mean().iloc[-1])
        out["macd"] = self.safe(ema12 - ema26)
        out["macd_signal"] = self.safe(df["close"].ewm(span=9).mean().iloc[-1])

        sma20 = self.safe(df["close"].rolling(20).mean().iloc[-1])
        std20 = self.safe(df["close"].rolling(20).std().iloc[-1])
        out["boll_upper"] = self.safe(sma20 + 2 * std20)
        out["boll_lower"] = self.safe(sma20 - 2 * std20)

        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        out["atr_14"] = self.safe(tr.rolling(14).mean().iloc[-1])

        out["vol_mean_20"] = self.safe(df["volume"].rolling(20).mean().iloc[-1])
        out["vol_mean_50"] = self.safe(df["volume"].rolling(50).mean().iloc[-1])
        out["vol_std_20"] = self.safe(df["volume"].rolling(20).std().iloc[-1])

        out["tick_spread"] = self.safe(df["high"].iloc[-1] - df["low"].iloc[-1])
        out["tick_body"] = self.safe(df["close"].iloc[-1] - df["open"].iloc[-1])
        out["tick_upper_wick"] = self.safe(df["high"].iloc[-1] - max(df["close"].iloc[-1], df["open"].iloc[-1]))
        out["tick_lower_wick"] = self.safe(min(df["close"].iloc[-1], df["open"].iloc[-1]) - df["low"].iloc[-1])

        out["std_20"] = self.safe(df["close"].rolling(20).std().iloc[-1])
        out["std_50"] = self.safe(df["close"].rolling(50).std().iloc[-1])
        out["mean_20"] = self.safe(df["close"].rolling(20).mean().iloc[-1])
        out["mean_50"] = self.safe(df["close"].rolling(50).mean().iloc[-1])

        out["range_20"] = self.safe((df["high"] - df["low"]).rolling(20).mean().iloc[-1])
        out["range_50"] = self.safe((df["high"] - df["low"]).rolling(50).mean().iloc[-1])

        arr = np.array([self.safe(v) for v in out.values()], dtype=np.float32)

        if len(arr) != 88:
            fixed = np.zeros(88, dtype=np.float32)
            fixed[:min(len(arr), 88)] = arr[:min(len(arr), 88)]
            return fixed

        return arr
