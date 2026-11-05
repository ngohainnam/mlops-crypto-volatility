"""Promotion gate: move the `champion` alias to a new model version only if it is better.

Both models are scored on the same test window, so the comparison is fair even if the
champion was trained months ago on different data.
"""

from collections.abc import Callable
from typing import Any

import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "btc-volatility"
ALIAS = "champion"


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def promote(client: MlflowClient, new_version: str, X_test: pd.DataFrame, y_test: pd.Series,
            load_model: Callable[[str], Any] = mlflow.sklearn.load_model) -> bool:
    """Return True if new_version became champion. Tags the version with the decision."""
    new_rmse = rmse(y_test, load_model(f"models:/{MODEL_NAME}/{new_version}").predict(X_test))
    try:
        champ = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version
    except MlflowException as e:
        # Only "alias not found" means no champion; any other error (outage) must not skip the gate.
        if e.error_code not in ("INVALID_PARAMETER_VALUE", "RESOURCE_DOES_NOT_EXIST"):
            raise
        champ = None

    if champ is None:
        promoted = True
        print(f"no champion yet; promoting v{new_version} (rmse={new_rmse:.6f})")
    else:
        champ_rmse = rmse(y_test, load_model(f"models:/{MODEL_NAME}/{champ}").predict(X_test))
        promoted = new_rmse < champ_rmse
        verdict = "promoted" if promoted else "rejected"
        print(f"challenger v{new_version} rmse={new_rmse:.6f} vs champion v{champ} "
              f"rmse={champ_rmse:.6f} -> {verdict}")

    if promoted:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, new_version)
    client.set_model_version_tag(MODEL_NAME, new_version, "promotion",
                                 "promoted" if promoted else "rejected")
    return promoted
