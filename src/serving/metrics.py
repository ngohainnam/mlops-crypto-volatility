"""Prometheus metrics exposed on GET /metrics (names fixed by docs/MLOps.md)."""

from prometheus_client import Counter, Gauge, Histogram

prediction_requests_total = Counter("prediction_requests_total", "Prediction requests",
                                    ["endpoint"])
prediction_latency_seconds = Histogram("prediction_latency_seconds", "Prediction latency")
predicted_volatility = Gauge("predicted_volatility", "Latest predicted next-hour volatility")
feature_drift_psi = Gauge("feature_drift_psi", "PSI of live vs training features", ["feature"])
model_version_info = Gauge("model_version_info", "Loaded model version", ["version"])
