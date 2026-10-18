"""Evaluation metrics (NumPy only, no models): identification, overlap, diarization, latency.

These functions only *compute* metrics from labelled data you provide. They never produce numbers
on their own: run them on real, consented recordings (see ``evaluation/run_eval.py``).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Interval = Tuple[float, float]


# ------------------------------------------------------------------- identification
def identification_metrics(trials: Sequence[Tuple[Optional[str], Optional[str]]]) -> Dict[str, Optional[float]]:
    """Open-set identification metrics from ``(true_student_or_None, predicted_student_or_None)``.

    ``true=None`` marks an impostor (a speaker who is not enrolled). ``predicted=None`` is a rejection
    (``Unknown Speaker``).

    * accuracy              - enrolled trials identified as the right student
    * FRR                   - enrolled trials rejected as unknown
    * misidentification     - enrolled trials attributed to the *wrong* student
    * FAR                   - impostor trials accepted as some enrolled student
    * unknown_rejection     - impostor trials correctly rejected (1 - FAR)
    A rate is ``None`` when the data contain no trial of that kind (never a fabricated 0).
    """
    known = [(t, p) for t, p in trials if t is not None]
    impostor = [(t, p) for t, p in trials if t is None]

    def rate(count: int, total: int) -> Optional[float]:
        return None if total == 0 else count / total

    return {
        "enrolled_trials": len(known),
        "impostor_trials": len(impostor),
        "accuracy": rate(sum(1 for t, p in known if p == t), len(known)),
        "frr": rate(sum(1 for _, p in known if p is None), len(known)),
        "misidentification_rate": rate(sum(1 for t, p in known if p is not None and p != t), len(known)),
        "far": rate(sum(1 for _, p in impostor if p is not None), len(impostor)),
        "unknown_rejection_rate": rate(sum(1 for _, p in impostor if p is None), len(impostor)),
    }


def sweep_thresholds(genuine: Sequence[float], impostor: Sequence[float],
                     thresholds: Optional[Iterable[float]] = None) -> Dict[str, object]:
    """FAR / FRR per threshold from raw cosine scores, plus the equal-error point.

    ``genuine``: score of each enrolled trial against its own student; ``impostor``: best score of
    each impostor trial against any student. Use it to pick ``CNAI_SPEAKER_MATCH_THRESHOLD`` on
    *your* data instead of assuming a universal value.
    """
    g, i = np.asarray(genuine, dtype=float), np.asarray(impostor, dtype=float)
    if g.size == 0 or i.size == 0:
        return {"points": [], "eer": None, "eer_threshold": None}
    grid = np.asarray(list(thresholds)) if thresholds is not None else np.linspace(0.0, 1.0, 101)
    points = [{"threshold": float(t), "far": float((i >= t).mean()), "frr": float((g < t).mean())} for t in grid]
    best = min(points, key=lambda p: abs(p["far"] - p["frr"]))
    return {"points": points, "eer": (best["far"] + best["frr"]) / 2, "eer_threshold": best["threshold"]}


# --------------------------------------------------------------------------- overlap
def _mask(intervals: Sequence[Interval], total: float, step: float) -> np.ndarray:
    mask = np.zeros(int(round(total / step)), dtype=bool)
    for a, b in intervals:
        mask[int(round(a / step)): int(round(b / step))] = True
    return mask


def overlap_prf(reference: Sequence[Interval], hypothesis: Sequence[Interval], total_seconds: float,
                step: float = 0.01) -> Dict[str, Optional[float]]:
    """Frame-level precision / recall / F1 of overlapping-speech detection."""
    ref, hyp = _mask(reference, total_seconds, step), _mask(hypothesis, total_seconds, step)
    tp, fp, fn = int((ref & hyp).sum()), int((~ref & hyp).sum()), int((ref & ~hyp).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if tp + fp + fn else None)
    return {"precision": precision, "recall": recall, "f1": f1}


def diarization_error_rate(reference: Dict[str, Sequence[Interval]], hypothesis: Dict[str, Sequence[Interval]],
                           total_seconds: float, step: float = 0.01) -> Optional[float]:
    """Frame-level DER with identity labels (no speaker mapping needed: enrolled names are the labels).

    DER = (missed speech + false alarm + speaker confusion) / total reference speaker-time.
    ``Unknown Speaker`` hypotheses should be passed under any label that is not a reference name.
    """
    n = int(round(total_seconds / step))
    ref = {k: _mask(v, total_seconds, step) for k, v in reference.items()}
    hyp = {k: _mask(v, total_seconds, step) for k, v in hypothesis.items()}
    ref_count = np.zeros(n, dtype=int)
    hyp_count = np.zeros(n, dtype=int)
    correct = np.zeros(n, dtype=int)
    for k, m in ref.items():
        ref_count += m
        if k in hyp:
            correct += m & hyp[k]
    for m in hyp.values():
        hyp_count += m
    total_ref = int(ref_count.sum())
    if total_ref == 0:
        return None
    miss = np.maximum(ref_count - hyp_count, 0).sum()
    false_alarm = np.maximum(hyp_count - ref_count, 0).sum()
    confusion = (np.minimum(ref_count, hyp_count) - correct).sum()
    return float((miss + false_alarm + confusion) / total_ref)


def attribution_accuracy(ref_sets: Sequence[Sequence[str]], hyp_sets: Sequence[Sequence[str]]) -> Dict[str, Optional[float]]:
    """Speaker attribution during overlap: per-region precision / recall of the *set* of named speakers."""
    tp = fp = fn = 0
    exact = 0
    for ref, hyp in zip(ref_sets, hyp_sets):
        r, h = set(ref), set(hyp)
        tp += len(r & h)
        fp += len(h - r)
        fn += len(r - h)
        exact += int(r == h)
    n = len(ref_sets)
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "exact_set_accuracy": exact / n if n else None,
        "regions": n,
    }


# ----------------------------------------------------------------------------- events
def event_duplication_rate(events: Sequence[Dict[str, object]], gap_seconds: float = 3.0) -> Optional[float]:
    """Share of events that repeat an earlier event (same speaker set, starting within ``gap_seconds``
    of the earlier one's end). ``events`` need ``start``, ``end`` (seconds) and ``speakers`` (names)."""
    if not events:
        return None
    ordered = sorted(events, key=lambda e: float(e["start"]))  # type: ignore[arg-type]
    duplicates = 0
    for i, ev in enumerate(ordered):
        sig = frozenset(ev["speakers"])  # type: ignore[arg-type]
        if any(frozenset(prev["speakers"]) == sig  # type: ignore[arg-type]
               and float(ev["start"]) <= float(prev["end"]) + gap_seconds  # type: ignore[arg-type]
               for prev in ordered[:i]):
            duplicates += 1
    return duplicates / len(ordered)


# ----------------------------------------------------------------------------- latency
def latency_stats(latencies_ms: Sequence[float], processing_ms: Sequence[float] = (),
                  audio_seconds: Sequence[float] = ()) -> Dict[str, Optional[float]]:
    """Median / p95 end-to-end latency and the real-time factor (processing time / audio duration)."""
    lat = np.asarray(latencies_ms, dtype=float)
    out: Dict[str, Optional[float]] = {
        "n": float(lat.size),
        "median_ms": float(np.median(lat)) if lat.size else None,
        "p95_ms": float(np.percentile(lat, 95)) if lat.size else None,
        "real_time_factor": None,
    }
    if len(processing_ms) and len(processing_ms) == len(audio_seconds) and sum(audio_seconds) > 0:
        out["real_time_factor"] = float(sum(processing_ms) / 1000.0 / sum(audio_seconds))
    return out
