"""Prediction API. Run: uvicorn src.serving.app:app --port 8000"""

import asyncio
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ConfigDict, create_model

from src.data.features import FEATURES, build_features
from src.data.fetch import fetch_latest
from src.serving import metrics as m
from src.serving.model import POLL_SECONDS, ModelHolder

SYMBOL = "BTCUSDT"
# 200 rows for the drift window + 72 warm-up candles for the longest rolling feature (vol_72h).
LIVE_CANDLES = 200 + 72
# One required float per model feature, built from FEATURES so the two can't drift apart.
FeatureRow = create_model("FeatureRow", __config__=ConfigDict(allow_inf_nan=False),
                          **{f: (float, ...) for f in FEATURES})


def create_app(holder: ModelHolder | None = None,
               fetch_fn: Callable[[], pd.DataFrame] = lambda: fetch_latest(LIVE_CANDLES),
               poll: bool = True) -> FastAPI:
    holder = holder or ModelHolder()

    def show_version():
        m.model_version_info.clear()
        if holder.version:
            m.model_version_info.labels(holder.version).set(1)

    async def poll_loop():
        while True:
            await asyncio.sleep(POLL_SECONDS)
            if await asyncio.to_thread(holder.refresh):
                show_version()

    @asynccontextmanager
    async def lifespan(_):
        await asyncio.to_thread(holder.refresh)
        show_version()
        task = asyncio.create_task(poll_loop()) if poll else None
        yield
        if task:
            task.cancel()

    app = FastAPI(title="BTC volatility API", lifespan=lifespan)

    def predict(row: dict[str, float], timestamp: str, endpoint: str,
                history: pd.DataFrame | None = None) -> dict:
        start = time.perf_counter()
        # Read once: a hot swap mid-request must not mix one version's model with another's drift.
        model, version, drift = holder.model, holder.version, holder.drift
        if model is None:
            raise HTTPException(503, "no champion model loaded yet")
        value = float(model.predict(pd.DataFrame([row], columns=FEATURES))[0])
        if history is None:
            drift.add(row)
        else:  # every recent candle, keyed by open_time so repeats are skipped
            for r in history.to_dict("records"):
                drift.add(r, key=r["open_time"])
        for feature, score in drift.scores().items():
            m.feature_drift_psi.labels(feature).set(score)
        m.predicted_volatility.set(value)
        m.prediction_requests_total.labels(endpoint).inc()
        m.prediction_latency_seconds.observe(time.perf_counter() - start)
        return {"symbol": SYMBOL, "predicted_volatility": value, "model_version": version,
                "timestamp": timestamp}

    @app.get("/health")
    def health():
        return {"status": "ok" if holder.model is not None else "no_model",
                "model_version": holder.version}

    # Plain `def` handlers: FastAPI runs them in a thread, so blocking HTTP/model calls are fine.
    @app.get("/predict")
    def predict_live():
        if holder.model is None:
            raise HTTPException(503, "no champion model loaded yet")
        try:
            feats = build_features(fetch_fn())
        except Exception as e:
            raise HTTPException(502, f"could not fetch candles from Binance: {e}") from e
        complete = feats.dropna(subset=FEATURES)
        if complete.empty or complete.index[-1] != feats.index[-1]:
            raise HTTPException(502, "not enough candles to build features")
        last = complete.iloc[-1]
        return predict({f: float(last[f]) for f in FEATURES},
                       timestamp=last["open_time"].isoformat(), endpoint="get", history=complete)

    @app.post("/predict")
    def predict_payload(row: FeatureRow):
        return predict(row.model_dump(), timestamp=datetime.now(UTC).isoformat(),
                       endpoint="post")

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
