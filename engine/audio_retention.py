"""Optional, short-lived retention of per-speaker audio clips (off by default).

``CNAI_RAW_AUDIO_RETENTION_SECONDS=0`` (default) keeps no raw audio at all. When positive, a short
clip per speaker of a detected event is written under ``data/event_audio`` and deleted once it is
older than the configured retention time.
"""
from __future__ import annotations

import logging
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AudioRetention:
    MAX_CLIP_SECONDS = 6.0

    def __init__(self, directory: Path, retention_seconds: float, sample_rate: int) -> None:
        self.directory = directory
        self.retention_seconds = retention_seconds
        self.sample_rate = sample_rate
        self._last_purge = 0.0

    @property
    def enabled(self) -> bool:
        return self.retention_seconds > 0

    def save_clip(self, event_id: str, speaker_id: str, audio: Optional[np.ndarray]) -> Optional[str]:
        """Write a clip and return its file name (the reference stored in the event), or ``None``."""
        if not self.enabled or audio is None or len(audio) == 0:
            return None
        self.purge()
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            clip = np.clip(np.asarray(audio, dtype=np.float32)[: int(self.MAX_CLIP_SECONDS * self.sample_rate)], -1, 1)
            name = f"{event_id}_{speaker_id}.wav"
            with wave.open(str(self.directory / name), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.sample_rate)
                w.writeframes((clip * 32767).astype(np.int16).tobytes())
            return name
        except Exception:
            logger.exception("Could not store the audio clip")
            return None

    def purge(self, force: bool = False) -> int:
        """Delete clips older than the retention time. Returns the number removed."""
        now = time.time()
        if not force and now - self._last_purge < 5.0:
            return 0
        self._last_purge = now
        removed = 0
        if not self.directory.exists():
            return 0
        for path in self.directory.glob("*.wav"):
            try:
                if now - path.stat().st_mtime > self.retention_seconds:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
        return removed
