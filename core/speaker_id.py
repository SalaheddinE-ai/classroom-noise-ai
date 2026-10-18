"""ECAPA-TDNN speaker embeddings (SpeechBrain) and student matching."""
from __future__ import annotations

import logging
import threading
from typing import Optional, Tuple

import numpy as np
import torch

from config import Settings
from database.db_session import session_scope
from database.models import Student
from database.vector_store import VectorStore

logger = logging.getLogger(__name__)


def resolve_device(preference: str) -> str:
    """Resolve ``auto`` / ``cpu`` / ``cuda`` to a usable torch device string."""
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preference == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable, falling back to CPU")
        return "cpu"
    return preference


class StudentRecognizer:
    """Computes 192-D ECAPA-TDNN embeddings and matches them to registered students."""

    def __init__(self, settings: Settings, vector_store: VectorStore) -> None:
        try:  # SpeechBrain >= 1.0
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # pragma: no cover - older SpeechBrain
            from speechbrain.pretrained import EncoderClassifier

        self.settings = settings
        self.vector_store = vector_store
        self.device = resolve_device(settings.device)
        self._lock = threading.Lock()

        kwargs = {}
        try:  # avoid symlink permission errors on Windows
            from speechbrain.utils.fetching import LocalStrategy

            kwargs["local_strategy"] = LocalStrategy.COPY
        except Exception:  # pragma: no cover
            pass

        savedir = settings.models_dir / "spkrec-ecapa-voxceleb"
        self._classifier = EncoderClassifier.from_hparams(
            source=settings.speaker_model_source,
            savedir=str(savedir),
            run_opts={"device": self.device},
            **kwargs,
        )
        self._classifier.eval()
        logger.info("ECAPA-TDNN loaded on %s", self.device)

    def extract_embedding(self, audio_tensor: torch.Tensor) -> np.ndarray:
        """Return the L2-normalised 192-D embedding of a 16 kHz mono waveform."""
        wav = audio_tensor.detach().float().reshape(1, -1).to(self.device)
        with self._lock, torch.inference_mode():
            emb = self._classifier.encode_batch(wav)
        vec = emb.squeeze().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def identify_student(self, embedding: np.ndarray) -> Tuple[Optional[Student], float]:
        """Find the closest registered student.

        Returns ``(student, cosine_score)`` for the best candidate, or ``(None, 0.0)``
        if nobody is registered. The caller applies the acceptance threshold.
        """
        results = self.vector_store.search(embedding, top_k=1)
        if not results:
            return None, 0.0
        student_id, score = results[0]
        with session_scope() as session:
            student = session.get(Student, student_id)
        return student, float(score)
