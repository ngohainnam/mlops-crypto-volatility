"""Generate traffic: call GET /predict every few seconds. Ctrl+C to stop.

uv run python -m scripts.load --url http://localhost:8000 --interval 3
"""

import argparse
import time

import requests

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://localhost:8000")
p.add_argument("--interval", type=float, default=3)
args = p.parse_args()

while True:
    try:
        r = requests.get(f"{args.url}/predict", timeout=30)
        print(r.status_code, r.text, flush=True)
    except requests.RequestException as e:  # keep looping; errors show up on the dashboard
        print("error:", e, flush=True)
    time.sleep(args.interval)
