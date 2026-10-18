"""Silero VAD wrapper (TorchScript model)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VADResult:
    """Outcome of a VAD pass over one audio window."""

    is_speech: bool
    speech_ratio: float  # fraction of 32 ms frames classified as speech
    max_prob: float
    # Per-frame speech probabilities (one per 32 ms frame); lets callers derive speech segments
    # without running the model a second time. ``None`` for results built without them.
    frame_probs: Optional[Tuple[float, ...]] = field(default=None, compare=False, repr=False)


class SileroVAD:
    """Frame-level speech probability with Silero VAD (512 samples @ 16 kHz)."""

    FRAME_SAMPLES = 512

    @property
    def frame_seconds(self) -> float:
        return self.FRAME_SAMPLES / self.sample_rate

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_ratio: float = 0.3,
    ) -> None:
        from silero_vad import load_silero_vad

        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD supports 8 kHz or 16 kHz only")
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_ratio = min_speech_ratio
        self.model = load_silero_vad(onnx=False)  # TorchScript (JIT) model
        logger.info("Silero VAD loaded")

    def analyze(self, audio: torch.Tensor) -> VADResult:
        """Compute per-frame speech probabilities and decide whether speech is present.

        Args:
            audio: 1-D float32 tensor in [-1, 1] sampled at ``sample_rate``.
        """
        audio = audio.detach().float().cpu().reshape(-1)
        n_frames = audio.numel() // self.FRAME_SAMPLES
        if n_frames == 0:
            return VADResult(False, 0.0, 0.0)
        frames = audio[: n_frames * self.FRAME_SAMPLES].view(n_frames, self.FRAME_SAMPLES)
        self.model.reset_states()
        probs = []
        with torch.no_grad():
            for frame in frames:
                probs.append(float(self.model(frame, self.sample_rate).item()))
        speech_ratio = sum(p >= self.threshold for p in probs) / n_frames
        return VADResult(
            is_speech=speech_ratio >= self.min_speech_ratio,
            speech_ratio=speech_ratio,
            max_prob=max(probs),
            frame_probs=tuple(probs),
        )
