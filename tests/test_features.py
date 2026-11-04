import math

import numpy as np
import pandas as pd
import pytest

from src.data.features import FEATURES, TARGET, build_features, make_dataset, parkinson


@pytest.fixture
def klines():
    """200 hourly candles from a seeded random walk."""
    rng = np.random.default_rng(42)
    n = 200
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.005, n)) * close
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.uniform(100, 1000, n),
            "trades": rng.integers(1000, 10000, n),
        }
    )


def test_parkinson_formula():
    out = parkinson(pd.Series([110.0]), pd.Series([100.0]))
    assert out.iloc[0] == pytest.approx(math.log(1.1) / math.sqrt(4 * math.log(2)))


def test_target_is_next_candle_parkinson(klines):
    df = build_features(klines)
    assert df[TARGET].iloc[10] == pytest.approx(parkinson(klines.high, klines.low).iloc[11])
    assert pd.isna(df[TARGET].iloc[-1])


def test_no_leakage(klines):
    # Removing future rows must not change any past feature value.
    full = build_features(klines)
    cut = build_features(klines.iloc[:-20])
    pd.testing.assert_frame_equal(full[FEATURES].iloc[:-20], cut[FEATURES])


def test_dataset_has_no_nans_and_all_columns(klines):
    ds = make_dataset(klines)
    assert list(ds.columns) == ["open_time", *FEATURES, TARGET]
    assert not ds.isna().any().any() and len(ds) > 0
