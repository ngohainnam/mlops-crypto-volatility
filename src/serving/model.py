"""Holds the current champion model and swaps in a newly promoted one without a restart."""

import logging
import os
from collections.abc import Callable
from typing import Any

import mlflow.artifacts
import mlflow.sklearn
from mlflow import MlflowClient

from src.serving.drift import DriftMonitor
from src.training.promote import ALIAS, MODEL_NAME

log = logging.getLogger(__name__)
POLL_SECONDS = int(os.environ.get("MODEL_POLL_SECONDS", "300"))


class ModelHolder:
    def __init__(self, client: MlflowClient | None = None,
                 load_model: Callable[[str], Any] = mlflow.sklearn.load_model,
                 load_dict: Callable[[str], dict] = mlflow.artifacts.load_dict):
        self.client = client  # created on first refresh, so importing the app has no side effects
        self.load_model = load_model
        self.load_dict = load_dict
        self.model: Any = None
        self.version: str | None = None
        self.drift: DriftMonitor | None = None

    def refresh(self) -> bool:
        """Load the champion if it changed. Never raises: a broken poll keeps the old model."""
        try:
            self.client = self.client or MlflowClient()
            mv = self.client.get_model_version_by_alias(MODEL_NAME, ALIAS)
            if mv.version == self.version:
                return False
            model = self.load_model(f"models:/{MODEL_NAME}/{mv.version}")
            drift = DriftMonitor(self.load_dict(f"runs:/{mv.run_id}/reference.json"))
        except Exception as e:  # no champion yet, MLflow down, artifact missing...
            log.warning("model refresh skipped: %s", e)
            return False
        # Swap only after everything loaded, so requests never see a half-updated holder.
        self.model, self.drift, self.version = model, drift, mv.version
        log.info("loaded %s v%s", MODEL_NAME, mv.version)
        return True
