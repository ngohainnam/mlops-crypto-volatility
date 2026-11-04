"""Feature engineering and target for next-hour volatility.

Every feature at row t uses only candles up to and including t (rolling windows and
positive shifts). The only negative shift is the target, which is the next candle.
"""

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1h", "ret_4h", "ret_24h",
    "vol_4h", "vol_24h", "vol_72h",
    "parkinson",
    "volume", "trades", "volume_ma24", "trades_ma24",
    "hl_range_pct",
    "hour", "dayofweek",
]
TARGET = "vol_next"


def parkinson(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson volatility of each candle: sqrt(ln(H/L)^2 / (4 ln 2))."""
    return np.sqrt(np.log(high / low) ** 2 / (4 * np.log(2)))


def build_features(klines: pd.DataFrame) -> pd.DataFrame:
    """All rows kept: early rows have NaN features, the last row has a NaN target."""
    log_close = np.log(klines["close"])
    r = log_close.diff()
    pk = parkinson(klines["high"], klines["low"])
    df = pd.DataFrame({"open_time": klines["open_time"]})
    for k in (1, 4, 24):
        df[f"ret_{k}h"] = log_close - log_close.shift(k)
    for k in (4, 24, 72):
        df[f"vol_{k}h"] = r.rolling(k).std()
    df["parkinson"] = pk
    df["volume"] = klines["volume"].astype(float)
    df["trades"] = klines["trades"].astype(float)
    df["volume_ma24"] = df["volume"].rolling(24).mean()
    df["trades_ma24"] = df["trades"].rolling(24).mean()
    df["hl_range_pct"] = (klines["high"] - klines["low"]) / klines["close"]
    df["hour"] = klines["open_time"].dt.hour
    df["dayofweek"] = klines["open_time"].dt.day_of_week
    df[TARGET] = pk.shift(-1)
    return df


def make_dataset(klines: pd.DataFrame) -> pd.DataFrame:
    """Training-ready rows: features and target all present."""
    return build_features(klines).dropna().reset_index(drop=True)
