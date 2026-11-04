import numpy as np
import pandas as pd

from src.training.train import make_reference, time_split


def _ds(hours=24 * 30):
    t = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"open_time": t, "x": rng.normal(size=hours), "y": rng.normal(size=hours)})


def test_time_split_no_overlap_and_ordered():
    ds = _ds()
    train, test = time_split(ds, test_days=14)
    assert len(train) + len(test) == len(ds)
    assert train.open_time.max() < test.open_time.min()
    assert test.open_time.max() - test.open_time.min() <= pd.Timedelta(days=14)
    assert len(test) == 14 * 24  # hourly rows in (max - 14 days, max]


def test_make_reference_probs_sum_to_one():
    ref = make_reference(_ds(), ["x", "y"], bins=10)
    for feat in ("x", "y"):
        edges, probs = ref[feat]["edges"], ref[feat]["probs"]
        assert len(probs) == len(edges) - 1
        assert abs(sum(probs) - 1) < 1e-9
        assert edges[0] == -np.inf and edges[-1] == np.inf


def test_make_reference_constant_feature_has_no_duplicate_edges():
    ds = _ds().assign(x=1.0)
    edges = make_reference(ds, ["x"])["x"]["edges"]
    assert len(edges) == len(set(edges))
