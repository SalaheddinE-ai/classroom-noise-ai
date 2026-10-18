"""Bounded per-stage timing statistics (median / p95), used for latency monitoring."""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

import numpy as np


class StageTimer:
    """Keeps the last ``maxlen`` durations (ms) of every named pipeline stage."""

    def __init__(self, maxlen: int = 300) -> None:
        self._values: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=maxlen))
        self._counters: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def record(self, stage: str, milliseconds: float) -> None:
        with self._lock:
            self._values[stage].append(float(milliseconds))

    def last(self, stage: str) -> Optional[float]:
        with self._lock:
            values = self._values.get(stage)
            return values[-1] if values else None

    def count(self, name: str, increment: int = 1) -> None:
        with self._lock:
            self._counters[name] += increment

    def summary(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            data = {k: list(v) for k, v in self._values.items() if v}
            counters = dict(self._counters)
        out: Dict[str, Dict[str, float]] = {}
        for stage, values in data.items():
            arr = np.asarray(values)
            out[stage] = {"count": len(values), "median_ms": round(float(np.median(arr)), 1),
                          "p95_ms": round(float(np.percentile(arr, 95)), 1),
                          "mean_ms": round(float(arr.mean()), 1), "max_ms": round(float(arr.max()), 1)}
        if counters:
            out["_counters"] = {k: float(v) for k, v in counters.items()}
        return out
