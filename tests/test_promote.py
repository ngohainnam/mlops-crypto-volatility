from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException

from src.training.promote import ALIAS, MODEL_NAME, promote

X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
Y = pd.Series([1.0, 2.0, 3.0])


class Stub:
    """Model whose predictions are fixed; records what it was asked to predict on."""

    def __init__(self, preds):
        self.preds = np.array(preds)
        self.seen = None

    def predict(self, X):
        self.seen = X
        return self.preds


def make_client(champion_version="1"):
    client = MagicMock()
    if champion_version is None:
        client.get_model_version_by_alias.side_effect = MlflowException("no alias")
    else:
        client.get_model_version_by_alias.return_value = SimpleNamespace(version=champion_version)
    return client


def loader(models):
    return lambda uri: models[uri]


def uri(v):
    return f"models:/{MODEL_NAME}/{v}"


def test_promotes_when_no_champion():
    client = make_client(None)
    assert promote(client, "1", X, Y, load_model=loader({uri("1"): Stub([1, 2, 3])}))
    client.set_registered_model_alias.assert_called_once_with(MODEL_NAME, ALIAS, "1")
    client.set_model_version_tag.assert_called_once_with(MODEL_NAME, "1", "promotion", "promoted")


def test_promotes_when_better():
    client = make_client("1")
    models = {uri("1"): Stub([2, 3, 4]), uri("2"): Stub([1, 2, 3.1])}
    assert promote(client, "2", X, Y, load_model=loader(models))
    client.set_registered_model_alias.assert_called_once_with(MODEL_NAME, ALIAS, "2")
    client.set_model_version_tag.assert_called_once_with(MODEL_NAME, "2", "promotion", "promoted")


def test_rejects_when_worse():
    client = make_client("1")
    models = {uri("1"): Stub([1, 2, 3.1]), uri("2"): Stub([2, 3, 4])}
    assert not promote(client, "2", X, Y, load_model=loader(models))
    client.set_registered_model_alias.assert_not_called()
    client.set_model_version_tag.assert_called_once_with(MODEL_NAME, "2", "promotion", "rejected")


def test_rejects_when_equal():
    client = make_client("1")
    models = {uri("1"): Stub([2, 2, 2]), uri("2"): Stub([2, 2, 2])}
    assert not promote(client, "2", X, Y, load_model=loader(models))
    client.set_registered_model_alias.assert_not_called()


def test_both_scored_on_same_window():
    client = make_client("1")
    champ, chall = Stub([1, 2, 3]), Stub([1, 2, 3])
    promote(client, "2", X, Y, load_model=loader({uri("1"): champ, uri("2"): chall}))
    assert champ.seen is X and chall.seen is X
