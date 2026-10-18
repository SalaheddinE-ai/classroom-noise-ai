"""Enrollment-sample quality assessment (NumPy only).

Used when a voice sample is registered: an empty, clipped, too quiet or flat/noisy recording is
rejected, and every accepted sample gets a quality weight in [min_sample_weight, 1] so that one
poor sample cannot dominate a student's profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from core.audio_utils import clip_ratio


@dataclass
class QualityReport:
    ok: bool
    weight: float  # 0..1 (1 = best); only meaningful when ``ok``
    reasons: List[str] = field(default_factory=list)
    rms_dbfs: float = -120.0
    clip_ratio: float = 0.0
    dynamic_range_db: float = 0.0
    duration_seconds: float = 0.0


def assess_sample(wav: np.ndarray, sample_rate: int, settings) -> QualityReport:
    """Check duration, finiteness, level, clipping and dynamic range of a mono float32 recording."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    duration = wav.size / sample_rate if sample_rate else 0.0
    reasons: List[str] = []
    if wav.size == 0 or not np.all(np.isfinite(wav)):
        return QualityReport(False, 0.0, ["empty or corrupted audio"], duration_seconds=duration)

    rms = float(np.sqrt(np.mean(np.square(wav.astype(np.float64)))))
    rms_dbfs = 20.0 * np.log10(max(rms, 1e-9))
    clips = clip_ratio(wav)
    frame = max(1, int(0.03 * sample_rate))
    n = wav.size // frame
    if n >= 4:
        frame_rms = np.sqrt(np.mean(np.square(wav[: n * frame].reshape(n, frame)), axis=1)) + 1e-9
        dyn = float(20.0 * np.log10(np.percentile(frame_rms, 90) / np.percentile(frame_rms, 10)))
    else:
        dyn = 0.0

    if duration < settings.min_sample_seconds:
        reasons.append(f"too short ({duration:.1f}s < {settings.min_sample_seconds}s)")
    if rms_dbfs < settings.enrollment_min_rms_dbfs:
        reasons.append(f"too quiet ({rms_dbfs:.0f} dBFS)")
    if clips > settings.enrollment_max_clip_ratio:
        reasons.append(f"clipped ({clips:.1%} of samples at full scale)")
    if dyn < settings.enrollment_min_dynamic_range_db and rms_dbfs >= settings.enrollment_min_rms_dbfs:
        reasons.append(f"no clear speech pauses/dynamics (range {dyn:.0f} dB): noise or constant tone?")

    # weight: penalise clipping, low level and weak dynamics, never below min_sample_weight
    penalties = [
        min(1.0, clips / max(settings.enrollment_max_clip_ratio, 1e-6)) * 0.5,
        max(0.0, min(1.0, (-30.0 - rms_dbfs) / 20.0)) * 0.3,
        max(0.0, min(1.0, (15.0 - dyn) / 15.0)) * 0.3,
    ]
    weight = max(settings.min_sample_weight, 1.0 - sum(penalties))
    return QualityReport(not reasons, float(weight), reasons, float(rms_dbfs), float(clips), dyn, duration)
