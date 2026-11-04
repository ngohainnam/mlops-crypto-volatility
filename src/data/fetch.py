"""Download BTCUSDT klines (candles) from the Binance public API.

Run: python -m src.data.fetch  ->  data/raw/klines.parquet (last 365 days)
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

DEFAULT_BASE_URL = "https://api.binance.com"
PAGE_LIMIT = 1000  # Binance max per request
MAX_TRIES = 5
RETRY_STATUS = {418, 429, 500, 502, 503, 504}  # 418 = IP banned after ignoring 429s
OUT_PATH = Path("data/raw/klines.parquet")

COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
FLOAT_COLS = ["open", "high", "low", "close", "volume",
              "quote_volume", "taker_buy_base", "taker_buy_quote"]


def _get(url: str, params: dict) -> list:
    """GET with retries on rate limits, server errors and connection drops."""
    for attempt in range(MAX_TRIES):
        try:
            resp = requests.get(url, params=params, timeout=10)
        except requests.ConnectionError:
            if attempt == MAX_TRIES - 1:
                raise
            time.sleep(2**attempt)
            continue
        if resp.status_code in RETRY_STATUS and attempt < MAX_TRIES - 1:
            time.sleep(float(resp.headers.get("Retry-After", 2**attempt)))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("unreachable")


def _to_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS).drop(columns="ignore")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df[FLOAT_COLS] = df[FLOAT_COLS].astype(float)
    df["trades"] = df["trades"].astype("int64")
    df = df[df["close_time"] < pd.Timestamp.now(tz="UTC")]  # drop the still-forming candle
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def fetch_klines(start_ms: int, end_ms: int | None = None, symbol: str = "BTCUSDT",
                 interval: str = "1h", base_url: str | None = None) -> pd.DataFrame:
    """All klines from start_ms (to end_ms if given), paginating 1000 at a time."""
    base_url = base_url or os.environ.get("BINANCE_BASE_URL", DEFAULT_BASE_URL)
    url = f"{base_url}/api/v3/klines"
    rows: list = []
    start = start_ms
    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": start, "limit": PAGE_LIMIT}
        if end_ms is not None:
            params["endTime"] = end_ms
        page = _get(url, params)
        rows.extend(page)
        if len(page) < PAGE_LIMIT or (end_ms is not None and page[-1][0] >= end_ms):
            break
        start = page[-1][0] + 1
    return _to_frame(rows)


def fetch_latest(limit: int = 100, symbol: str = "BTCUSDT", interval: str = "1h",
                 base_url: str | None = None) -> pd.DataFrame:
    """The most recent closed candles (the still-forming candle is dropped)."""
    base_url = base_url or os.environ.get("BINANCE_BASE_URL", DEFAULT_BASE_URL)
    rows = _get(f"{base_url}/api/v3/klines",
                {"symbol": symbol, "interval": interval, "limit": limit})
    return _to_frame(rows)


def main() -> None:
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)
    df = fetch_klines(start_ms=int(start.timestamp() * 1000))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"saved {df.shape} to {OUT_PATH}: {df.open_time.min()} -> {df.open_time.max()}")


if __name__ == "__main__":
    main()
