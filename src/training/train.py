"""Train the volatility model, log it to MLflow, register it, then run the promotion gate.

Run: MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.training.train [--max-depth 1 ...]
"""

import argparse

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.data.features import FEATURES, TARGET, make_dataset
from src.training.promote import MODEL_NAME, promote, rmse

# Features whose live distribution the API compares with training (PSI).
DRIFT_FEATURES = ["ret_1h", "vol_24h", "parkinson", "volume", "hl_range_pct"]


def time_split(ds: pd.DataFrame, test_days: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Older rows train, the last `test_days` test. Never shuffled."""
    cutoff = ds["open_time"].max() - pd.Timedelta(days=test_days)
    return ds[ds["open_time"] <= cutoff], ds[ds["open_time"] > cutoff]


def make_reference(df: pd.DataFrame, features: list[str], bins: int = 10) -> dict[str, dict]:
    """Quantile bins of each feature on training data; the API scores live data against them."""
    ref = {}
    for f in features:
        inner = np.unique(np.quantile(df[f], np.linspace(0, 1, bins + 1)[1:-1]))
        edges = np.concatenate([[-np.inf], inner, [np.inf]])
        counts, _ = np.histogram(df[f], bins=edges)
        ref[f] = {"edges": edges.tolist(), "probs": (counts / counts.sum()).tolist()}
    return ref


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--data", default="data/raw/klines.parquet")
    args = p.parse_args(argv)

    train, test = time_split(make_dataset(pd.read_parquet(args.data)))
    params = {"n_estimators": args.n_estimators, "max_depth": args.max_depth,
              "learning_rate": args.learning_rate}

    mlflow.set_experiment("btc-volatility")
    with mlflow.start_run():
        model = GradientBoostingRegressor(random_state=42, **params)
        model.fit(train[FEATURES], train[TARGET])
        pred = model.predict(test[FEATURES])
        mlflow.log_params({**params, "train_rows": len(train), "test_rows": len(test),
                           "test_start": str(test.open_time.min()),
                           "test_end": str(test.open_time.max())})
        mlflow.log_metrics({"rmse": rmse(test[TARGET], pred),
                            "mae": mean_absolute_error(test[TARGET], pred)})
        mlflow.log_dict(make_reference(train, DRIFT_FEATURES), "reference.json")
        info = mlflow.sklearn.log_model(model, name="model", registered_model_name=MODEL_NAME,
                                        input_example=test[FEATURES].head(2),
                                        # we built this model, so its tree nodes are trusted
                                        skops_trusted_types=["sklearn.tree._tree.Tree"])
        print(f"run {info.run_id}: registered v{info.registered_model_version}, "
              f"rmse={rmse(test[TARGET], pred):.6f}")

    promote(MlflowClient(), str(info.registered_model_version), test[FEATURES], test[TARGET])


if __name__ == "__main__":
    main()
