from types import SimpleNamespace
from unittest.mock import MagicMock

from mlflow.exceptions import MlflowException

from src.serving.model import ModelHolder

REF = {"x": {"edges": [float("-inf"), 0.0, float("inf")], "probs": [0.5, 0.5]}}


def make_holder(version="1"):
    client = MagicMock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(version=version, run_id="r1")
    loaded = []
    holder = ModelHolder(client, load_model=lambda uri: loaded.append(uri) or f"model@{uri}",
                         load_dict=lambda uri: REF)
    return holder, client, loaded


def test_refresh_loads_first_champion():
    holder, _, loaded = make_holder()
    assert holder.refresh() is True
    assert holder.version == "1"
    assert holder.model == "model@models:/btc-volatility/1"
    assert holder.drift.reference == REF


def test_refresh_noop_when_same_version():
    holder, _, loaded = make_holder()
    holder.refresh()
    assert holder.refresh() is False
    assert len(loaded) == 1


def test_refresh_swaps_on_new_version():
    holder, client, _ = make_holder()
    holder.refresh()
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="2", run_id="r2")
    assert holder.refresh() is True
    assert holder.version == "2"
    assert holder.model == "model@models:/btc-volatility/2"


def test_refresh_no_champion_keeps_none():
    holder, client, _ = make_holder()
    client.get_model_version_by_alias.side_effect = MlflowException("alias not found")
    assert holder.refresh() is False
    assert holder.model is None and holder.version is None


def test_refresh_mlflow_down_keeps_current_model():
    holder, client, _ = make_holder()
    holder.refresh()
    client.get_model_version_by_alias.side_effect = ConnectionError("mlflow down")
    assert holder.refresh() is False
    assert holder.version == "1"
