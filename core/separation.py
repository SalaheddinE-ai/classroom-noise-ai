"""Optional audio source separation (SpeechBrain SepFormer).

Off by default (``CNAI_SEPARATION_ENABLED=false``). When enabled it turns one mixed window into
one waveform per speaker, which is the only way to obtain *genuinely isolated* audio per voice.
Pre-trained SepFormer checkpoints separate a fixed number of sources (the default model: 2), so
more simultaneous speakers than the model supports cannot be separated.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SourceSeparator:
    """Lazy wrapper around ``speechbrain.inference.separation.SepformerSeparation``."""

    def __init__(self, settings) -> None:  # settings: config.Settings (duck-typed)
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()
        self.available = False
        self.error: Optional[str] = None

    def load(self) -> bool:
        """Load the model once. Returns ``False`` (and records ``error``) when unavailable."""
        if self._model is not None:
            return True
        try:
            try:  # SpeechBrain >= 1.0
                from speechbrain.inference.separation import SepformerSeparation
            except ImportError:  # pragma: no cover - older SpeechBrain
                from speechbrain.pretrained import SepformerSeparation

            from core.speaker_id import resolve_device

            kwargs = {}
            try:  # avoid symlink permission errors on Windows
                from speechbrain.utils.fetching import LocalStrategy

                kwargs["local_strategy"] = LocalStrategy.COPY
            except Exception:  # pragma: no cover
                pass
            device = resolve_device(self.settings.device)
            savedir = self.settings.models_dir / "separation"
            self._model = SepformerSeparation.from_hparams(
                source=self.settings.separation_model,
                savedir=str(savedir),
                run_opts={"device": device},
                **kwargs,
            )
            self._model.eval()
            self.available = True
            logger.info("Source separation model loaded (%s)", self.settings.separation_model)
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.available = False
            logger.warning("Source separation unavailable: %s", self.error)
            return False

    def separate(self, audio: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        """Split a mono float32 window into one float32 waveform per detected source.

        Output waveforms have the same length and sample rate as the input.
        """
        if not self.available and not self.load():
            raise RuntimeError(f"source separation unavailable: {self.error}")
        import torch
        import torchaudio

        target_sr = self.settings.separation_sample_rate
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).reshape(1, -1)
        if sample_rate != target_sr:
            wav = torchaudio.functional.resample(wav, sample_rate, target_sr)
        with self._lock, torch.inference_mode():
            est = self._model.separate_batch(wav.to(self._model.device))  # (1, time, n_sources)
        est = est[0].detach().cpu().T  # (n_sources, time)
        if sample_rate != target_sr:
            est = torchaudio.functional.resample(est, target_sr, sample_rate)
        out = []
        for source in est.numpy():
            fixed = np.zeros(len(audio), dtype=np.float32)
            n = min(len(audio), source.shape[0])
            fixed[:n] = source[:n]
            out.append(fixed)
        return out
