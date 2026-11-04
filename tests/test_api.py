from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data.features import FEATURES
from src.serving.app import create_app
from src.serving.drift import DriftMonitor

REF = {"parkinson": {"edges": [float("-inf"), 0.005, float("inf")], "probs": [0.5, 0.5]}}
PAYLOAD = {f: 0.01 for f in FEATURES} | {"hour": 3, "dayofweek": 2}


class StubModel:
    def predict(self, X):
        assert list(X.columns) == FEATURES  # model must get columns in training order
        return np.full(len(X), 0.01)


def fake_holder(model=True):
    return SimpleNamespace(model=StubModel() if model else None, version="1" if model else None,
                           drift=DriftMonitor(REF, min_rows=1) if model else None,
                           refresh=lambda: False)


def fake_fetch():
    t = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")
    close = 100 + np.cumsum(np.random.default_rng(0).normal(size=100))
    return pd.DataFrame({"open_time": t, "open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close, "volume": 10.0, "trades": 50})


def broken_fetch():
    raise ConnectionError("binance down")


@pytest.fixture
def make_client():
    clients = []

    def make(holder=None, fetch=fake_fetch):
        c = TestClient(create_app(holder or fake_holder(), fetch, poll=False))
        clients.append(c.__enter__())  # runs the startup hook like a real server
        return c

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def test_health_reports_version(make_client):
    assert make_client().get("/health").json() == {"status": "ok", "model_version": "1"}


def test_health_no_model(make_client):
    r = make_client(fake_holder(model=False)).get("/health").json()
    assert r == {"status": "no_model", "model_version": None}


def test_get_predict_returns_contract(make_client):
    r = make_client().get("/predict")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"symbol", "predicted_volatility", "model_version", "timestamp"}
    assert body["symbol"] == "BTCUSDT"
    assert body["predicted_volatility"] == pytest.approx(0.01)
    assert body["timestamp"].startswith("2026-01-05T03:00:00")  # last candle's open_time


def test_get_predict_fills_drift_window_once_per_candle(make_client):
    holder = fake_holder()
    c = make_client(holder)
    c.get("/predict")
    assert len(holder.drift.rows) == 100 - 72  # every candle with full features, not just the last
    c.get("/predict")
    assert len(holder.drift.rows) == 28  # same candles again are not double-counted


def test_post_predict_validates_payload(make_client):
    c = make_client()
    assert c.post("/predict", json={"ret_1h": 0.1}).status_code == 422
    assert c.post("/predict", json=PAYLOAD | {"volume": "lots"}).status_code == 422
    r = c.post("/predict", json=PAYLOAD)
    assert r.status_code == 200 and r.json()["model_version"] == "1"


def test_predict_503_without_model(make_client):
    c = make_client(fake_holder(model=False))
    assert c.get("/predict").status_code == 503
    assert c.post("/predict", json=PAYLOAD).status_code == 503


def test_get_predict_502_when_binance_fails(make_client):
    assert make_client(fetch=broken_fetch).get("/predict").status_code == 502


def test_metrics_exposes_custom_names(make_client):
    c = make_client()
    c.post("/predict", json=PAYLOAD)
    body = c.get("/metrics").text
    for name in ["prediction_requests_total", "prediction_latency_seconds",
                 "predicted_volatility", 'feature_drift_psi{feature="overall"}',
                 'model_version_info{version="1"}']:
        assert name in body
