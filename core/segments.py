"""Small, dependency-light helpers for time intervals and amplitude envelopes (NumPy only)."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Interval = Tuple[float, float]


def merge_intervals(intervals: Sequence[Interval], gap: float = 0.0) -> List[Interval]:
    """Union of intervals; intervals closer than ``gap`` seconds are merged."""
    items = sorted((float(a), float(b)) for a, b in intervals if b > a)
    merged: List[Interval] = []
    for start, end in items:
        if merged and start <= merged[-1][1] + gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def total_duration(intervals: Sequence[Interval]) -> float:
    """Length of the union of ``intervals`` in seconds."""
    return float(sum(b - a for a, b in merge_intervals(intervals)))


def intersect_intervals(a: Sequence[Interval], b: Sequence[Interval]) -> List[Interval]:
    """Intersection of two interval lists (each is merged first)."""
    left, right = merge_intervals(a), merge_intervals(b)
    out: List[Interval] = []
    i = j = 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            out.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return out


def overlap_between_streams(streams: Sequence[Sequence[Interval]]) -> List[Interval]:
    """Time spans during which at least two of the given streams are active."""
    spans: List[Interval] = []
    for i in range(len(streams)):
        for j in range(i + 1, len(streams)):
            spans.extend(intersect_intervals(streams[i], streams[j]))
    return merge_intervals(spans)


def mask_to_intervals(mask: np.ndarray, frame_seconds: float, offset: float = 0.0,
                      min_duration: float = 0.0, gap: float = 0.0) -> List[Interval]:
    """Convert a boolean frame mask into ``(start, end)`` intervals in seconds."""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    starts, ends = edges[0::2], edges[1::2]
    out = [(offset + s * frame_seconds, offset + e * frame_seconds) for s, e in zip(starts, ends)]
    out = merge_intervals(out, gap)
    return [(a, b) for a, b in out if b - a >= min_duration]


def probs_to_intervals(probs: Sequence[float], frame_seconds: float, threshold: float,
                       min_duration: float = 0.1, gap: float = 0.3) -> List[Interval]:
    """Speech intervals from per-frame VAD probabilities."""
    return mask_to_intervals(np.asarray(probs) >= threshold, frame_seconds, 0.0, min_duration, gap)


def slice_samples(audio: np.ndarray, intervals: Sequence[Interval], sample_rate: int) -> np.ndarray:
    """Concatenate the samples covered by ``intervals`` (seconds relative to ``audio[0]``)."""
    parts = []
    for start, end in intervals:
        a, b = max(0, int(round(start * sample_rate))), min(len(audio), int(round(end * sample_rate)))
        if b > a:
            parts.append(audio[a:b])
    return np.concatenate(parts) if parts else np.zeros(0, dtype=audio.dtype)


def amplitude_envelope(audio: np.ndarray, points_per_second: int, sample_rate: int) -> np.ndarray:
    """Peak-amplitude envelope (max |x| per bin), rounded to 4 decimals. Empty audio -> empty array."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    bin_size = max(1, int(sample_rate // max(1, points_per_second)))
    n_bins = audio.size // bin_size
    if n_bins == 0:
        return np.zeros(0, dtype=np.float32)
    binned = np.abs(audio[: n_bins * bin_size]).reshape(n_bins, bin_size).max(axis=1)
    return np.round(binned, 4).astype(np.float32)


def pad_intervals(intervals: Sequence[Interval], pad: float, lo: float, hi: float) -> List[Interval]:
    """Extend every interval by ``pad`` seconds on both sides (clamped to ``[lo, hi]``) and merge."""
    return merge_intervals([(max(lo, a - pad), min(hi, b + pad)) for a, b in intervals])


def max_concurrency(streams: Sequence[Sequence[Interval]]) -> int:
    """Largest number of streams that are active at the same instant."""
    points = []
    for stream in streams:
        for a, b in merge_intervals(stream):
            points.append((a, 1))
            points.append((b, -1))
    best = current = 0
    for _, delta in sorted(points, key=lambda p: (p[0], p[1])):  # ends before starts at equal times
        current += delta
        best = max(best, current)
    return best
