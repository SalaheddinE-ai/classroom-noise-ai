"""Bounded rolling buffer feeding the live microphone waveform of the dashboard."""
from __future__ import annotations

import threading
from collections import deque
from typing import Any, Deque, Dict, Tuple

import numpy as np

from core.segments import amplitude_envelope


class WaveformBuffer:
    """Peak envelope of the *combined* microphone signal (not speaker-specific).

    Each processed window contributes only its newest ``new_samples`` samples, because consecutive
    windows overlap; every audio sample is therefore drawn once. Memory is bounded by
    ``history_seconds * points_per_second``.
    """

    def __init__(self, sample_rate: int, history_seconds: float, points_per_second: int) -> None:
        self.sample_rate = sample_rate
        self.points_per_second = points_per_second
        self.history_seconds = history_seconds
        self._maxlen = int(history_seconds * points_per_second)
        self._points: Deque[Tuple[int, float]] = deque(maxlen=self._maxlen)  # (bin index, peak)
        self._lock = threading.Lock()

    def add(self, samples: np.ndarray, t_end: float, new_samples: int) -> None:
        """Append the last ``new_samples`` samples of ``samples`` (window that ended at ``t_end``)."""
        tail = np.asarray(samples, dtype=np.float32)[-max(new_samples, 1):]
        env = amplitude_envelope(tail, self.points_per_second, self.sample_rate)
        if env.size == 0:
            return
        t_start = t_end - env.size / self.points_per_second
        first = int(round(t_start * self.points_per_second))
        with self._lock:
            last_bin = self._points[-1][0] if self._points else None
            for i, value in enumerate(env):
                idx = first + i
                if last_bin is not None and idx <= last_bin:
                    continue  # never draw the same instant twice
                self._points.append((idx, float(value)))
                last_bin = idx

    def snapshot(self, seconds: float) -> Dict[str, Any]:
        """Dense series for the last ``seconds`` seconds: ``t0_epoch`` + fixed step + peaks (gaps = 0)."""
        with self._lock:
            pts = list(self._points)
        if not pts:
            return {"t0_epoch": None, "dt_seconds": 1.0 / self.points_per_second, "peaks": []}
        newest = pts[-1][0]
        floor = newest - int(seconds * self.points_per_second)
        pts = [p for p in pts if p[0] >= floor]
        first = pts[0][0]
        peaks = [0.0] * (newest - first + 1)
        for idx, value in pts:
            peaks[idx - first] = round(value, 4)
        return {"t0_epoch": first / self.points_per_second,
                "dt_seconds": 1.0 / self.points_per_second, "peaks": peaks}
