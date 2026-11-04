import time

import pandas as pd
import pytest
import requests

from src.data import fetch

COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
HOUR_MS = 3_600_000


def raw_rows(start_ms, n):
    """Binance-shaped kline rows (strings for prices, like the real API)."""
    return [
        [start_ms + i * HOUR_MS, "100.0", "110.0", "90.0", "105.0", "12.5",
         start_ms + (i + 1) * HOUR_MS - 1, "1300.0", 42, "6.0", "630.0", "0"]
        for i in range(n)
    ]


class FakeResponse:
    def __init__(self, status, data=None, headers=None):
        self.status_code = status
        self._data = data
        self.headers = headers or {}

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


def serve(data):
    return lambda url, params, timeout: FakeResponse(200, data)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)


def test_paginates_until_short_page(monkeypatch):
    calls = []
    pages = [raw_rows(0, 1000), raw_rows(1000 * HOUR_MS, 5)]

    def fake_get(url, params, timeout):
        calls.append(params)
        return FakeResponse(200, pages[len(calls) - 1])

    monkeypatch.setattr(requests, "get", fake_get)
    df = fetch.fetch_klines(start_ms=0, base_url="http://x")
    assert len(df) == 1005
    assert len(calls) == 2
    assert calls[1]["startTime"] == 999 * HOUR_MS + 1


def test_parses_columns_and_types(monkeypatch):
    monkeypatch.setattr(requests, "get", serve(raw_rows(0, 3)))
    df = fetch.fetch_klines(start_ms=0, base_url="http://x")
    assert list(df.columns) == COLUMNS
    assert str(df["open_time"].dt.tz) == "UTC"
    assert df["close"].dtype == float
    assert df["trades"].dtype.kind == "i"


def test_retries_on_429_then_succeeds(monkeypatch):
    responses = iter([FakeResponse(429, headers={"Retry-After": "0"}),
                      FakeResponse(200, raw_rows(0, 3))])
    monkeypatch.setattr(requests, "get", lambda url, params, timeout: next(responses))
    assert len(fetch.fetch_klines(start_ms=0, base_url="http://x")) == 3


def test_gives_up_after_5_tries(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(1)
        return FakeResponse(500)

    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(requests.HTTPError):
        fetch.fetch_klines(start_ms=0, base_url="http://x")
    assert len(calls) == 5


def test_uses_base_url_env_var(monkeypatch):
    urls = []

    def fake_get(url, params, timeout):
        urls.append(url)
        return FakeResponse(200, raw_rows(0, 1))

    monkeypatch.setenv("BINANCE_BASE_URL", "https://data-api.binance.vision")
    monkeypatch.setattr(requests, "get", fake_get)
    fetch.fetch_klines(start_ms=0)
    assert urls[0] == "https://data-api.binance.vision/api/v3/klines"


def test_fetch_latest_drops_open_candle(monkeypatch):
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    start = now_ms - 3 * HOUR_MS - HOUR_MS // 2  # last of 4 candles closes in the future
    monkeypatch.setattr(requests, "get", serve(raw_rows(start, 4)))
    df = fetch.fetch_latest(limit=4)
    assert len(df) == 3
    assert (df["close_time"] < pd.Timestamp.now(tz="UTC")).all()


def test_fetch_klines_drops_open_candle(monkeypatch):
    # A half-formed last candle would become a wrong training target.
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    start = now_ms - 3 * HOUR_MS - HOUR_MS // 2
    monkeypatch.setattr(requests, "get", serve(raw_rows(start, 4)))
    assert len(fetch.fetch_klines(start_ms=start, base_url="http://x")) == 3
