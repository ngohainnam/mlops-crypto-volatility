"""PSI drift: compare live feature values with the training distribution saved by the trainer."""

from collections import deque

import numpy as np

EPS = 1e-6  # floor for bin proportions so an empty bin doesn't give inf/NaN


def psi(ref: dict, values: np.ndarray) -> float:
    """Population Stability Index of `values` against ref = {"edges": [...], "probs": [...]}."""
    counts, _ = np.histogram(values, bins=np.asarray(ref["edges"]))
    actual = np.clip(counts / max(len(values), 1), EPS, None)
    expected = np.clip(np.asarray(ref["probs"]), EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


class DriftMonitor:
    """Rolling window of the last `window` feature rows, scored with PSI per feature."""

    def __init__(self, reference: dict[str, dict], window: int = 200, min_rows: int = 100):
        self.reference = reference
        self.min_rows = min_rows  # PSI noise is ~(bins-1)/n: ~0.09 at 100 rows, ~12 at 1 row
        self.rows: deque[tuple[object, dict]] = deque(maxlen=window)
        self.keys: set = set()

    def add(self, row: dict[str, float], key: object | None = None) -> None:
        # Same candle predicted again (GET /predict polled often) must not be counted twice.
        if key is not None and key in self.keys:
            return
        if len(self.rows) == self.rows.maxlen:
            self.keys.discard(self.rows[0][0])
        self.rows.append((key, row))
        if key is not None:
            self.keys.add(key)

    def scores(self) -> dict[str, float]:
        if len(self.rows) < self.min_rows:
            return {}
        s = {f: psi(ref, np.array([r[f] for _, r in self.rows]))
             for f, ref in self.reference.items()}
        s["overall"] = sum(s.values()) / len(s)
        return s
