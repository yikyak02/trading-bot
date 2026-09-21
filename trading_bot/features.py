import numpy as np
import pandas as pd

FEATURES = ["return_1", "return_5", "return_20", "sma_ratio", "distance_sma50",
            "rsi", "macd_pct", "atr_pct", "volatility", "volume_ratio"]


def build_features(bars):
    pieces = []
    for symbol, source in bars.groupby("symbol", sort=True):
        frame = source.sort_values("date").copy()
        c = frame.close
        for lag in (1, 5, 20):
            frame[f"return_{lag}"] = c.pct_change(lag, fill_method=None)
        frame["sma20"] = c.rolling(20).mean()
        frame["sma50"] = c.rolling(50).mean()
        frame["sma_ratio"] = frame.sma20 / frame.sma50 - 1
        frame["distance_sma50"] = c / frame.sma50 - 1
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        frame["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).where(loss != 0, 100)
        frame.loc[(gain == 0) & (loss == 0), "rsi"] = 50
        frame["macd_pct"] = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()) / c
        tr = pd.concat([frame.high - frame.low, (frame.high - c.shift()).abs(),
                        (frame.low - c.shift()).abs()], axis=1).max(axis=1)
        frame["atr_pct"] = tr.rolling(14).mean() / c
        frame["volatility"] = frame.return_1.rolling(20).std()
        frame["volume_ratio"] = frame.volume / frame.volume.rolling(20).mean().replace(0, np.nan)
        # Close(t) decision, open(t+1) entry, open(t+2) reference exit.
        frame["target"] = frame.open.shift(-2) / frame.open.shift(-1) - 1
        frame["label_end"] = frame.date.shift(-2)
        pieces.append(frame)
    result = pd.concat(pieces).replace([np.inf, -np.inf], np.nan)
    return result.sort_values(["date", "symbol"]).reset_index(drop=True)
