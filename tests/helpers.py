"""Synthetic-audio helpers shared by the multi-speaker tests (no models, no downloads).

A "voice" is a pure tone; ``fake_embed`` is a 192-band magnitude spectrum. Different tones give
nearly orthogonal embeddings and a *mixture* of two tones gives an embedding close to both, which is
exactly the behaviour that makes real speaker embeddings unreliable on overlapping speech.
"""
from __future__ import annotations

import numpy as np

SR = 16000
FREQS = {"salah": 500.0, "sarah": 1500.0, "rajae": 3000.0}


def tone(freq: float, seconds: float, amp: float = 0.4, sr: int = SR, start: float = 0.0, total: float | None = None) -> np.ndarray:
    """Sine tone; optionally placed at ``start`` inside a silent buffer of ``total`` seconds."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    wave = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if total is None:
        return wave
    out = np.zeros(int(total * sr), dtype=np.float32)
    a = int(start * sr)
    out[a: a + n] = wave[: len(out) - a]
    return out


def fake_embed(audio: np.ndarray) -> np.ndarray:
    spec = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float32)))
    edges = np.linspace(0, len(spec), 193).astype(int)
    vec = np.array([spec[a:b].sum() if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])], dtype=np.float32)
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def pcm_bytes(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


def make_store(names=("salah", "sarah", "rajae"), samples=2):
    from database.vector_store import VectorStore

    store = VectorStore(192)
    for name in names:
        store.set_name(name, name.capitalize())
        for k in range(samples):
            store.add(name, fake_embed(tone(FREQS[name], 1.5, amp=0.3 + 0.1 * k)))
    return store
