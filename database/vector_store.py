"""In-memory NumPy vector store for cosine-similarity speaker lookup."""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.orm import Session


class VectorStore:
    """Thread-safe store of L2-normalised embeddings keyed by student id.

    Cosine similarity: ``cos(A, B) = (A . B) / (||A|| ||B||)``. Because every
    stored vector is normalised, a search is a single matrix-vector product.

    Besides the historical best-sample :meth:`search`, :meth:`score_students` scores every
    student against the centroid of its most recent samples (more robust than one short clip).
    Student display names are cached here so matching never needs a database query.
    """

    def __init__(self, dim: int = 192) -> None:
        self.dim = dim
        self._lock = threading.RLock()
        self._matrix = np.empty((0, dim), dtype=np.float32)
        self._student_ids: List[str] = []
        self._sample_ids: List[Optional[int]] = []
        self._weights: List[float] = []  # sample quality weight in (0, 1]
        self._names: Dict[str, str] = {}

    # ------------------------------------------------------------------ math
    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        """Return the L2-normalised float32 copy of ``vector``."""
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    # --------------------------------------------------------------- mutation
    def add(self, student_id: str, embedding: np.ndarray, sample_id: Optional[int] = None,
            weight: float = 1.0) -> None:
        """Add one embedding for ``student_id`` (``sample_id`` = VoiceProfile.id when known).

        ``weight`` is the sample's quality weight; it scales the sample's pull on the profile centroid.
        """
        vec = self.normalize(embedding)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected a {self.dim}-D embedding, got {vec.shape[0]}-D")
        with self._lock:
            self._matrix = np.vstack([self._matrix, vec[None, :]])
            self._student_ids.append(student_id)
            self._sample_ids.append(sample_id)
            self._weights.append(float(min(max(weight, 0.01), 1.0)))

    def remove_student(self, student_id: str) -> int:
        """Remove every embedding of a student. Returns the number removed."""
        with self._lock:
            keep = [i for i, sid in enumerate(self._student_ids) if sid != student_id]
            removed = len(self._student_ids) - len(keep)
            self._matrix = self._matrix[keep] if keep else np.empty((0, self.dim), np.float32)
            self._student_ids = [self._student_ids[i] for i in keep]
            self._sample_ids = [self._sample_ids[i] for i in keep]
            self._weights = [self._weights[i] for i in keep]
            self._names.pop(student_id, None)
            return removed

    def remove_sample(self, student_id: str, sample_id: int) -> bool:
        """Remove a single voice sample. Returns ``True`` if something was removed."""
        with self._lock:
            idx = [
                i for i, (sid, pid) in enumerate(zip(self._student_ids, self._sample_ids))
                if sid == student_id and pid == sample_id
            ]
            if not idx:
                return False
            keep = [i for i in range(len(self._student_ids)) if i not in idx]
            self._matrix = self._matrix[keep] if keep else np.empty((0, self.dim), np.float32)
            self._student_ids = [self._student_ids[i] for i in keep]
            self._sample_ids = [self._sample_ids[i] for i in keep]
            self._weights = [self._weights[i] for i in keep]
            return True

    def clear(self) -> None:
        with self._lock:
            self._matrix = np.empty((0, self.dim), dtype=np.float32)
            self._student_ids = []
            self._sample_ids = []
            self._weights = []
            self._names = {}

    def load_from_db(self, session: Session) -> int:
        """(Re)load all voice profiles (and student names) from SQLite. Returns the profile count."""
        from sqlalchemy import select

        from database.models import Student, VoiceProfile

        self.clear()
        names = {s.id: s.name for s in session.execute(select(Student)).scalars().all()}
        profiles = session.execute(select(VoiceProfile).order_by(VoiceProfile.id)).scalars().all()
        for profile in profiles:
            self.add(profile.student_id, profile.embedding, profile.id,
                     profile.quality_weight if profile.quality_weight is not None else 1.0)
        with self._lock:
            self._names = names
        return len(profiles)

    # ------------------------------------------------------------------ names
    def set_name(self, student_id: str, name: str) -> None:
        with self._lock:
            self._names[student_id] = name

    def name_of(self, student_id: str) -> Optional[str]:
        with self._lock:
            return self._names.get(student_id)

    # ----------------------------------------------------------------- search
    def __len__(self) -> int:
        with self._lock:
            return len(self._student_ids)

    def sample_count(self, student_id: str) -> int:
        with self._lock:
            return sum(1 for sid in self._student_ids if sid == student_id)

    def search(self, query: np.ndarray, top_k: int = 1) -> List[Tuple[str, float]]:
        """Return the ``top_k`` best ``(student_id, cosine_score)`` pairs.

        When a student owns several embeddings, the best score is kept.
        """
        q = self.normalize(query)
        with self._lock:
            if not self._student_ids:
                return []
            scores = self._matrix @ q
            best: Dict[str, float] = {}
            for sid, score in zip(self._student_ids, scores):
                if score > best.get(sid, -2.0):
                    best[sid] = float(score)
        return sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    def score_students(
        self, query: np.ndarray, max_samples: int = 5, min_samples: int = 1
    ) -> Dict[str, float]:
        """Cosine similarity of ``query`` to each student's profile centroid.

        The centroid is the re-normalised quality-weighted mean of the student's ``max_samples``
        most recent embeddings (a poor sample has a smaller weight, never zero). Students with fewer than ``min_samples`` samples are left out (not ready).
        """
        q = self.normalize(query)
        with self._lock:
            if not self._student_ids:
                return {}
            rows: Dict[str, List[int]] = {}
            for i, sid in enumerate(self._student_ids):
                rows.setdefault(sid, []).append(i)
            matrix = self._matrix
            result: Dict[str, float] = {}
            for sid, idx in rows.items():
                if len(idx) < min_samples:
                    continue
                chosen = idx[-max_samples:]
                w = np.asarray([self._weights[i] for i in chosen], dtype=np.float32)[:, None]
                centroid = self.normalize((matrix[chosen] * w).sum(axis=0))
                result[sid] = float(centroid @ q)
        return result
