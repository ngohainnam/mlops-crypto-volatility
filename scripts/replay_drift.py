"""Replay a historical window through POST /predict so the API's drift window fills with it.

uv run python -m scripts.replay_drift --start 2024-08-01 --end 2024-08-10
"""

import argparse
import re

import pandas as pd
import requests

from src.data.features import FEATURES, make_dataset
from src.data.fetch import fetch_klines

WARMUP = pd.Timedelta(hours=72)  # longest rolling feature (vol_72h) needs 72 earlier candles

p = argparse.ArgumentParser()
p.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
p.add_argument("--end", required=True, help="YYYY-MM-DD (UTC), exclusive")
p.add_argument("--url", default="http://localhost:8000")
args = p.parse_args()

start, end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")
ms = lambda t: int(t.timestamp() * 1000)  # noqa: E731
ds = make_dataset(fetch_klines(ms(start - WARMUP), ms(end) - 1))
rows = ds[ds["open_time"] >= start]
print(f"replaying {len(rows)} candles {start:%Y-%m-%d} -> {end:%Y-%m-%d}")
if len(rows) < 100:
    print("note: the API scores PSI on a 200-row window; fewer than 100 replay rows "
          "only partly displaces live data. Pick a longer window for a clear signal.")

with requests.Session() as s:
    for i, row in enumerate(rows[FEATURES].to_dict("records"), 1):
        s.post(f"{args.url}/predict", json=row, timeout=10).raise_for_status()
        if i % 50 == 0:
            print(f"  sent {i}/{len(rows)}")
    metrics = s.get(f"{args.url}/metrics", timeout=10).text

for feature, value in re.findall(r'feature_drift_psi\{feature="(\w+)"\} (\S+)', metrics):
    print(f"PSI {feature:14s} {float(value):.3f}")
