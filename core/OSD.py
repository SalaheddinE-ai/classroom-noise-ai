"""Overlap Speech Detection (OSD).

Backends:
* ``pyannote``  : ``pyannote/segmentation-3.0`` (powerset output); accurate, needs a
                  Hugging Face token and accepted model terms.
* ``embedding`` : lightweight fallback - compares ECAPA embeddings of short sub-windows;
                  strong dissimilarity means several voices are present.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from config import Settings
from core.speaker_id import resolve_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OSDResult:
    """Outcome of overlap detection on one window."""

    is_overlap: bool
    overlap_ratio: float
    backend: str
    # Optional detail reused downstream so the same audio is never embedded twice. All of them
    # are ``None`` when the backend cannot provide them.
    # pyannote: boolean activity matrix (frames x local speakers) and its frame duration.
    activity: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    frame_seconds: Optional[float] = field(default=None, compare=False)
    # embedding fallback: per-window start (s), embedding and "voices inconsistent here" flag.
    window_starts: Optional[Tuple[float, ...]] = field(default=None, compare=False)
    window_seconds: Optional[float] = field(default=None, compare=False)
    window_embeddings: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    window_flags: Optional[Tuple[bool, ...]] = field(default=None, compare=False)


class OverlapSpeechDetector:
    """Detects simultaneous speakers in a 16 kHz mono window."""

    def __init__(
        self,
        settings: Settings,
        embed_fn: Optional[Callable[[torch.Tensor], np.ndarray]] = None,
    ) -> None:
        self.settings = settings
        self._embed_fn = embed_fn
        self.backend = "none"
        self._model = None
        self._powerset = None
        self._device = "cpu"

    def load(self) -> None:
        """Load the configured backend (``auto`` prefers pyannote, else embedding)."""
        wanted = self.settings.osd_backend
        if wanted in ("auto", "pyannote"):
            if self._try_load_pyannote():
                return
            if wanted == "pyannote":
                logger.warning("pyannote OSD unavailable, using the embedding fallback")
        if self._embed_fn is None:
            raise RuntimeError("No OSD backend available (provide embed_fn or install pyannote.audio)")
        self.backend = "embedding"
        logger.info("OSD backend: embedding dispersion (fallback)")

    def _try_load_pyannote(self) -> bool:
        token = self.settings.hf_token
        if not token:
            logger.info("No HF token configured: skipping pyannote OSD")
            return False
        try:
            from pyannote.audio import Model
            from pyannote.audio.utils.powerset import Powerset

            try:
                model = Model.from_pretrained(self.settings.pyannote_model, use_auth_token=token)
            except TypeError:
                model = Model.from_pretrained(self.settings.pyannote_model, token=token)
            if model is None:
                raise RuntimeError("model download refused (accept the model terms on Hugging Face)")
            self._device = resolve_device(self.settings.device)
            self._model = model.eval().to(self._device)
            specs = model.specifications
            self._powerset = Powerset(len(specs.classes), specs.powerset_max_classes)
            self.backend = "pyannote"
            logger.info("OSD backend: pyannote (%s)", self.settings.pyannote_model)
            return True
        except Exception as exc:
            logger.warning("Could not load pyannote OSD: %s", exc)
            return False

    def detect(self, audio: torch.Tensor) -> OSDResult:
        """Return whether at least two speakers overlap in ``audio``."""
        audio = audio.detach().float().reshape(-1)
        if self.backend == "pyannote":
            return self._detect_pyannote(audio)
        if self.backend == "embedding":
            return self._detect_embedding(audio)
        raise RuntimeError("OverlapSpeechDetector.load() must be called first")

    def _detect_pyannote(self, audio: torch.Tensor) -> OSDResult:
        with torch.inference_mode():
            scores = self._model(audio.to(self._device)[None, None, :])
            multilabel = self._powerset.to_multilabel(scores)
        active = multilabel[0].sum(dim=-1)  # simultaneous speakers per frame
        ratio = float((active >= 2).float().mean().item())
        activity_matrix = multilabel[0].detach().cpu().numpy() > 0.5
        n_frames = max(int(activity_matrix.shape[0]), 1)
        frame_seconds = (audio.numel() / self.settings.sample_rate) / n_frames
        return OSDResult(
            ratio >= self.settings.effective_overlap_ratio_threshold,
            ratio,
            "pyannote",
            activity=activity_matrix,
            frame_seconds=frame_seconds,
        )

    def _detect_embedding(self, audio: torch.Tensor) -> OSDResult:
        sr = self.settings.sample_rate
        win, hop = sr, sr // 2  # 1 s windows, 0.5 s hop
        starts = list(range(0, max(audio.numel() - win, 0) + 1, hop))
        segments = [audio[s : s + win] for s in starts if audio[s : s + win].numel() == win]
        if len(segments) < 2:
            return OSDResult(False, 0.0, "embedding")
        embs = np.stack([self._embed_fn(seg) for seg in segments])  # already L2-normalised
        sims = embs @ embs.T
        pair_sims = sims[np.triu_indices(len(segments), k=1)]
        thr = self.settings.effective_dispersion_threshold
        ratio = float((pair_sims < thr).mean())
        flags = tuple(bool((sims[i] < thr).any()) for i in range(len(segments)))
        return OSDResult(
            bool(pair_sims.min() < thr),
            ratio,
            "embedding",
            window_starts=tuple(s / sr for s in starts[: len(segments)]),
            window_seconds=win / sr,
            window_embeddings=embs,
            window_flags=flags,
        )
