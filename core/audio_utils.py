"""Small audio helpers shared by the whole project (no heavy imports at module load)."""
from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch

PCM16_SCALE: float = 32768.0


def pcm16_bytes_to_float(audio_bytes: bytes) -> np.ndarray:
    """Convert little-endian 16-bit PCM bytes to float32 samples in [-1, 1]."""
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / PCM16_SCALE


def compute_db(samples: np.ndarray, offset_db: float = 0.0) -> float:
    """Pseudo-dB level: ``20 * log10(RMS)`` on the int16 scale, plus a calibration offset.

    ``samples`` are float32 values in [-1, 1]. The result is floored at 0 dB.
    """
    if samples.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    return float(20.0 * np.log10(max(rms * PCM16_SCALE, 1.0)) + offset_db)


def load_audio_bytes(data: bytes, target_sr: int = 16000) -> "torch.Tensor":
    """Decode an audio file (WAV/FLAC/OGG...) to a mono float32 tensor at ``target_sr``."""
    import soundfile as sf
    import torch
    import torchaudio

    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.mean(axis=1))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.contiguous()


def validate_pcm16(audio_bytes: bytes, expected_samples: int | None = None) -> str | None:
    """Return a reason string when a PCM16 frame is unusable, else ``None``.

    Guards against malformed frames (odd byte count, empty, wrong length) before any model sees them.
    """
    if not audio_bytes:
        return "empty frame"
    if len(audio_bytes) % 2:
        return "odd byte count (not 16-bit PCM)"
    if expected_samples is not None and len(audio_bytes) // 2 != expected_samples:
        return f"unexpected length {len(audio_bytes) // 2} samples (expected {expected_samples})"
    return None


def clip_ratio(samples: np.ndarray, level: float = 0.999) -> float:
    """Fraction of samples at (or beyond) digital full scale."""
    if samples.size == 0:
        return 0.0
    return float(np.mean(np.abs(samples) >= level))
