import numpy as np
import pandas as pd

from src.serving.drift import DriftMonitor, psi
from src.training.train import make_reference

rng = np.random.default_rng(0)
REF_DATA = pd.DataFrame({"x": rng.normal(size=5000), "y": rng.normal(size=5000)})
REF = make_reference(REF_DATA, ["x", "y"])


def test_psi_identical_distribution_near_zero():
    assert psi(REF["x"], REF_DATA["x"].to_numpy()) < 0.02


def test_psi_shifted_distribution_high():
    assert psi(REF["x"], rng.normal(loc=2, size=5000)) > 0.2


def test_psi_empty_bin_is_finite():
    # every live value lands in one bin, the other nine are empty
    assert np.isfinite(psi(REF["x"], np.full(100, 10.0)))


def test_monitor_window_is_capped():
    m = DriftMonitor(REF, window=200)
    for i in range(100):
        m.add({"x": 10.0, "y": 10.0}, key=i)  # extreme values, pushed out of the window
    for i in range(100, 300):
        m.add({"x": float(REF_DATA.x[i]), "y": float(REF_DATA.y[i])}, key=i)
    assert len(m.rows) == 200
    assert m.scores()["x"] < 0.2


def test_monitor_skips_duplicate_key():
    m = DriftMonitor(REF)
    m.add({"x": 0.0, "y": 0.0}, key="t1")
    m.add({"x": 0.0, "y": 0.0}, key="t1")
    m.add({"x": 0.0, "y": 0.0})  # no key -> always added
    assert len(m.rows) == 2


def test_scores_empty_until_min_rows():
    # PSI of a handful of rows is noise (one row -> PSI ~12), so it must not be reported
    m = DriftMonitor(REF, min_rows=100)
    for i in range(99):
        m.add({"x": 0.0, "y": 0.0}, key=i)
    assert m.scores() == {}
    m.add({"x": 0.0, "y": 0.0}, key=99)
    assert "overall" in m.scores()


def test_duplicate_check_forgets_keys_that_left_the_window():
    m = DriftMonitor(REF, window=2)
    for k in ["a", "b", "c"]:
        m.add({"x": 0.0, "y": 0.0}, key=k)
    m.add({"x": 0.0, "y": 0.0}, key="a")  # "a" was evicted, so it counts again
    assert [k for k, _ in m.rows] == ["c", "a"]


def test_overall_is_mean():
    m = DriftMonitor(REF, min_rows=1)
    assert m.scores() == {}
    m.add({"x": 10.0, "y": 0.0})
    s = m.scores()
    assert s["overall"] == (s["x"] + s["y"]) / 2

