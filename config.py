"""Central configuration for classroom-noise-ai.

All values can be overridden with environment variables prefixed ``CNAI_``
(or through a ``.env`` file at the project root). ``HF_TOKEN`` is also accepted.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR: Path = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Application settings (Pydantic Settings)."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_prefix="CNAI_",
        extra="ignore",
        validate_assignment=True,
    )

    # ---- Paths -----------------------------------------------------------
    data_dir: Path = BASE_DIR / "data"
    database_url: Optional[str] = None

    # ---- Audio capture ---------------------------------------------------
    sample_rate: int = 16000
    chunk_seconds: float = Field(2.0, gt=0.5, le=10.0)
    hop_seconds: float = Field(1.0, gt=0.1, le=10.0)
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    audio_backend: Literal["sounddevice", "pyaudio"] = "sounddevice"

    # ---- Thresholds ------------------------------------------------------
    threshold_db: float = Field(68.0, ge=0, le=90)  # int16 full scale = ~90 dB
    # Pseudo-dB = 20*log10(RMS of int16 samples) + offset. Calibrate the offset
    # against a sound level meter to obtain approximate dB SPL.
    calibration_offset_db: float = 0.0
    vad_threshold: float = Field(0.5, ge=0.0, le=1.0)
    vad_min_speech_ratio: float = Field(0.3, ge=0.0, le=1.0)
    overlap_ratio_threshold: float = Field(0.2, ge=0.0, le=1.0)
    embedding_dispersion_threshold: float = Field(0.35, ge=-1.0, le=1.0)
    speaker_match_threshold: float = Field(0.75, ge=0.0, le=1.0)
    embedding_dim: int = 192
    cooldown_seconds: float = Field(20.0, ge=15.0, le=30.0)

    # ---- Multi-speaker identification ------------------------------------
    # Minimum amount of speech (seconds) a speaker stream needs before we try to identify it.
    min_identify_seconds: float = Field(1.0, ge=0.3, le=10.0)
    # Voice samples (most recent first) averaged into one profile per student.
    samples_per_student: int = Field(5, ge=1, le=50)
    # A student with fewer usable samples than this is never matched ("needs samples").
    min_voice_samples: int = Field(1, ge=1, le=20)
    # Best-vs-second-best margin: below it, an ambiguous match is reported as Unknown Speaker.
    min_match_margin: float = Field(0.05, ge=0.0, le=0.5)
    # Maximum number of simultaneous speakers reported per window.
    max_simultaneous_speakers: int = Field(3, ge=1, le=6)
    # 0 (strict) .. 1 (sensitive). When unset, overlap_ratio_threshold / embedding_dispersion_threshold apply.
    overlap_sensitivity: Optional[float] = Field(None, ge=0.0, le=1.0)
    # How unidentified speakers are handled: report them as "Unknown Speaker" (default),
    # keep them in the events but never speak a warning because of them, or drop them.
    unknown_speaker_handling: Literal["report", "log_only", "ignore"] = "report"
    # Per-speaker cooldown (seconds); the global cooldown above still applies on top of it.
    speaker_cooldown_seconds: float = Field(20.0, ge=0.0, le=600.0)
    # Detections closer than this (seconds) belong to the same alert.
    event_merge_gap_seconds: float = Field(3.0, ge=0.5, le=30.0)
    # Cosine similarity above which two unidentified streams are considered the same voice.
    unknown_cluster_similarity: float = Field(0.6, ge=0.0, le=1.0)
    # Window scan fallback (no separation / diarization): sliding window for embeddings.
    scan_window_seconds: float = Field(1.0, ge=0.5, le=3.0)
    scan_hop_seconds: float = Field(0.5, ge=0.1, le=3.0)

    # New sample vs. the student's existing profile: below this it is rejected (wrong person?).
    sample_consistency_threshold: float = Field(0.30, ge=0.0, le=1.0)
    # A first sample this close to ANOTHER student's profile is refused as a probable duplicate.
    duplicate_voice_threshold: float = Field(0.85, ge=0.0, le=1.0)
    min_sample_seconds: float = Field(1.5, ge=0.5, le=30.0)

    # ---- V2.1: VAD segmentation, temporal tracking, warning policy --------
    vad_min_speech_ms: int = Field(100, ge=30, le=2000)  # shorter speech runs are dropped
    vad_merge_gap_ms: int = Field(300, ge=0, le=2000)  # pauses shorter than this do not split speech
    vad_pad_ms: int = Field(60, ge=0, le=500)  # padding added around each speech run
    # An identity is "confirmed" only after this much distinct speech AND this many observations
    # that agree on the same student; until then it is reported as "tentative".
    tracker_confirm_seconds: float = Field(1.5, ge=0.3, le=30.0)
    tracker_confirm_hits: int = Field(2, ge=1, le=20)
    # A weak observation (similarity within this margin below the threshold) of an already
    # confirmed speaker of the same event is treated as a continuation, not as a new voice.
    tracker_hysteresis: float = Field(0.08, ge=0.0, le=0.3)
    # Mixture-embedding fallback needs this much extra similarity (mixed audio is less reliable).
    mixture_extra_threshold: float = Field(0.10, ge=0.0, le=0.4)
    # Overlap strategy: auto picks sepformer > pyannote_slots > mixture_embedding when available.
    overlap_mode: Literal["auto", "pyannote_slots", "sepformer", "mixture_embedding"] = "auto"
    # A first warning waits until a speaker is confirmed or the event has lasted this long
    # (debounce: avoids warning "generally" a moment before the student can be named).
    warning_grace_seconds: float = Field(2.0, ge=0.0, le=30.0)
    # Minimum gap between two spoken warnings (any speaker / type) so TTS is never chatty.
    warning_min_gap_seconds: float = Field(3.0, ge=0.0, le=60.0)
    # Name the identified students in spoken warnings (set false to keep warnings anonymous).
    tts_announce_names: bool = True

    # ---- Enrollment quality / privacy ---------------------------------
    # Keep the enrollment recordings (needed to replay samples). The embeddings are always kept.
    store_enrollment_audio: bool = False
    enrollment_max_clip_ratio: float = Field(0.02, ge=0.0, le=1.0)  # share of samples at full scale
    enrollment_min_rms_dbfs: float = Field(-50.0, ge=-90.0, le=0.0)  # quieter recordings are rejected
    # Peak-to-floor frame-energy range: a flat, noisy recording has little dynamic range.
    enrollment_min_dynamic_range_db: float = Field(8.0, ge=0.0, le=60.0)
    # Weight of the lowest-quality sample relative to a perfect one in the profile centroid.
    min_sample_weight: float = Field(0.3, ge=0.05, le=1.0)

    # ---- Optional heavy components (off by default) ---------------------
    # Source separation (SpeechBrain SepFormer): gives real per-speaker audio, at a CPU/GPU cost.
    separation_enabled: bool = False
    separation_model: str = "speechbrain/sepformer-wsj02mix"
    separation_sample_rate: int = Field(8000, ge=8000, le=48000)  # rate the separation model expects
    # "auto": use the pyannote segmentation activity when the pyannote OSD backend is loaded.
    diarization_backend: Literal["auto", "pyannote", "none"] = "auto"

    # ---- Visualisation / retention ---------------------------------------
    viz_history_seconds: float = Field(60.0, ge=10.0, le=900.0)
    viz_points_per_second: int = Field(20, ge=5, le=100)
    # 0 = never keep raw audio. > 0: keep short per-speaker clips of detected events this long.
    raw_audio_retention_seconds: float = Field(0.0, ge=0.0, le=86400.0)

    # ---- Models ----------------------------------------------------------
    device: Literal["auto", "cpu", "cuda"] = "auto"
    speaker_model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    osd_backend: Literal["auto", "pyannote", "embedding"] = "auto"
    pyannote_model: str = "pyannote/segmentation-3.0"
    hf_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("CNAI_HF_TOKEN", "HF_TOKEN", "hf_token"),
    )

    # ---- TTS -------------------------------------------------------------
    language: Literal["en", "fr"] = "en"
    tts_backend: Literal["auto", "piper", "kokoro", "edge"] = "auto"
    tts_volume: float = Field(1.0, ge=0.0, le=2.0)
    piper_model_path: Optional[Path] = None
    kokoro_voice_en: str = "af_heart"
    kokoro_voice_fr: str = "ff_siwis"
    edge_voice_en: str = "en-US-AriaNeural"
    edge_voice_fr: str = "fr-FR-DeniseNeural"

    # ---- Servers ---------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    ui_port: int = 8501

    # ---- Derived helpers -------------------------------------------------
    @property
    def effective_overlap_ratio_threshold(self) -> float:
        """Overlap ratio above which a window counts as overlapping speech."""
        if self.overlap_sensitivity is None:
            return self.overlap_ratio_threshold
        return max(0.05, 0.4 - 0.4 * self.overlap_sensitivity)

    @property
    def effective_dispersion_threshold(self) -> float:
        """Embedding-dispersion threshold of the fallback overlap detector."""
        if self.overlap_sensitivity is None:
            return self.embedding_dispersion_threshold
        return 0.15 + 0.4 * self.overlap_sensitivity

    @property
    def event_audio_dir(self) -> Path:
        return self.data_dir / "event_audio"

    @property
    def sqlite_url(self) -> str:
        """SQLAlchemy URL of the SQLite database."""
        return self.database_url or f"sqlite:///{(self.data_dir / 'classroom.db').as_posix()}"

    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "samples"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def tts_cache_dir(self) -> Path:
        return self.data_dir / "tts_cache"

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_seconds)

    @property
    def hop_samples(self) -> int:
        return int(self.sample_rate * self.hop_seconds)

    def ensure_dirs(self) -> None:
        """Create every runtime directory."""
        for path in (self.data_dir, self.samples_dir, self.models_dir, self.tts_cache_dir, self.event_audio_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
