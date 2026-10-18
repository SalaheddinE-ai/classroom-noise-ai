"""MonitorService: owns the models, the audio stream and the processing thread."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from config import Settings, get_settings
from core.audio_stream import AudioStream
from database.db_session import init_db, session_scope
from database.vector_store import VectorStore

logger = logging.getLogger(__name__)


class MonitorService:
    """Lazy model loading + background monitoring loop, shared by the API."""

    WARNING_HOLD_SECONDS = 4.0

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.vector_store = VectorStore(self.settings.embedding_dim)
        self.recognizer = None
        self.vad = None
        self.osd = None
        self.tts = None
        self.engine = None
        self.analyzer = None
        self.separation_error: Optional[str] = None

        self._load_lock = threading.Lock()
        self._models_loaded = False
        self._loading = False
        self._stream: Optional[AudioStream] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.last_error: Optional[str] = None
        self.processing_errors = 0
        self._run_lock = threading.Lock()  # serialises concurrent start / stop requests

    # --------------------------------------------------------------- models
    @property
    def models_loaded(self) -> bool:
        return self._models_loaded

    def load_models(self) -> None:
        """Load every model once (idempotent, thread-safe)."""
        with self._load_lock:
            if self._models_loaded:
                return
            self._loading = True
            try:
                import torch

                from core.multi_speaker import MultiSpeakerAnalyzer
                from core.OSD import OverlapSpeechDetector
                from core.speaker_id import StudentRecognizer
                from core.tts_engine import TTSEngine
                from core.vad import SileroVAD
                from engine.policy_engine import ClassroomPolicyEngine

                init_db()
                with session_scope() as session:
                    count = self.vector_store.load_from_db(session)
                logger.info("Loaded %d voice profile(s)", count)

                self.recognizer = StudentRecognizer(self.settings, self.vector_store)
                self.vad = SileroVAD(
                    self.settings.sample_rate,
                    self.settings.vad_threshold,
                    self.settings.vad_min_speech_ratio,
                )
                self.osd = OverlapSpeechDetector(self.settings, embed_fn=self.recognizer.extract_embedding)
                self.osd.load()
                self.tts = TTSEngine(self.settings)

                separator = None
                if self.settings.separation_enabled:  # optional heavy component, loaded once
                    from core.separation import SourceSeparator

                    separator = SourceSeparator(self.settings)
                    if not separator.load():
                        self.separation_error = separator.error
                        separator = None
                self.analyzer = MultiSpeakerAnalyzer(
                    self.settings,
                    self.vector_store,
                    lambda audio: self.recognizer.extract_embedding(torch.from_numpy(np.ascontiguousarray(audio))),
                    separator,
                )
                self.engine = ClassroomPolicyEngine(
                    self.settings, self.vad, self.osd, self.recognizer, self.tts, self.analyzer
                )
                self._models_loaded = True
                self.last_error = None
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Model loading failed")
                raise
            finally:
                self._loading = False

    def embed_audio(self, audio: np.ndarray) -> np.ndarray:
        """L2-normalised ECAPA-TDNN embedding of a mono float32 16 kHz waveform (loads models once)."""
        self.load_models()
        import torch

        return self.recognizer.extract_embedding(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)))

    def preload_async(self) -> None:
        """Warm the models in a background thread (used at API start-up)."""

        def _job() -> None:
            try:
                self.load_models()
            except Exception:
                pass  # already logged and stored in last_error

        threading.Thread(target=_job, name="model-preload", daemon=True).start()

    # ----------------------------------------------------------- monitoring
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start capturing and analysing the microphone (idempotent, safe under concurrent calls)."""
        with self._run_lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self.running:
            return
        self.load_models()
        self._stream = AudioStream(
            sample_rate=self.settings.sample_rate,
            chunk_seconds=self.settings.chunk_seconds,
            hop_seconds=self.settings.hop_seconds,
            device=self.settings.input_device,
            backend=self.settings.audio_backend,
            calibration_offset_db=self.settings.calibration_offset_db,
        )
        self._stream.start()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="monitor-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring and release the microphone."""
        with self._run_lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    def shutdown(self) -> None:
        self.stop()
        if self.tts is not None:
            self.tts.close()

    def _loop(self) -> None:
        assert self._stream is not None and self.engine is not None
        while not self._stop_event.is_set():
            chunk = self._stream.read_chunk(timeout=1.0)
            if chunk is None:
                continue
            try:
                self.engine.process_audio_chunk(chunk, captured_at=self._stream.last_chunk_time)
            except Exception:
                self.processing_errors += 1
                logger.exception("Processing loop error")

    # --------------------------------------------------------------- status
    def status(self) -> Dict[str, Any]:
        """Aggregated status for the dashboard (REST and WebSocket)."""
        snap: Dict[str, Any] = self.engine.snapshot() if self.engine else {}
        live_db = self._stream.recent_db(0.5) if (self._stream and self.running) else 0.0
        last_event = snap.get("last_event")

        if self._loading:
            ui_state = "Loading models"
        elif not self.running:
            ui_state = "Stopped"
        elif last_event and (time.time() - last_event["epoch"]) < self.WARNING_HOLD_SECONDS:
            ui_state = "Warning Triggered" if last_event["spoken"] else "Warning Suppressed (cooldown)"
        elif snap.get("processing") or snap.get("state") == "NOISE_ONLY" or live_db >= self.settings.threshold_db:
            ui_state = "Analyzing"
        else:
            ui_state = "Idle"

        mic_alive = bool(self._stream and self.running and (time.time() - self._stream.last_audio_time) < 2.0)
        mic_silent = bool(self.running and mic_alive and live_db <= 0.5)
        return {
            "mic_alive": mic_alive,
            "mic_silent": mic_silent,
            "speech_detected": snap.get("speech_detected", False),
            "speech_ratio": snap.get("speech_ratio", 0.0),
            "running": self.running,
            "models_loaded": self._models_loaded,
            "loading": self._loading,
            "error": self.last_error,
            "ui_state": ui_state,
            "live_db": round(live_db, 1),
            "threshold_db": self.engine.threshold_dB if self.engine else self.settings.threshold_db,
            "cooldown_seconds": self.engine.cooldown_seconds if self.engine else self.settings.cooldown_seconds,
            "match_threshold": self.engine.match_threshold if self.engine else self.settings.speaker_match_threshold,
            "language": self.settings.language,
            "osd_backend": snap.get("osd_backend"),
            "in_cooldown": snap.get("in_cooldown", False),
            "cooldown_remaining": snap.get("cooldown_remaining", 0.0),
            "recent_events": snap.get("recent_events", []),
            "registered_voices": len(self.vector_store),
            "current_event": snap.get("current_event"),
            "capabilities": self.analyzer.capabilities(snap.get("osd_backend")) if self.analyzer else {},
            "separation_error": self.separation_error,
            "unknown_speaker_handling": self.settings.unknown_speaker_handling,
            "viz_history_seconds": self.settings.viz_history_seconds,
            "latency": snap.get("latency", {}),
            "compute": self._compute_info(),
            "dropped_windows": self._stream.dropped_windows if self._stream else 0,
            "processing_errors": self.processing_errors,
            "server_time": time.time(),
        }

    def _compute_info(self) -> Dict[str, Any]:
        """Where the models run (shown in /status): proves whether CUDA is really in use."""
        info: Dict[str, Any] = {"embedding_device": getattr(self.recognizer, "device", None),
                                "cuda_available": False, "gpu": None, "torch": None}
        try:
            import torch

            info["torch"] = getattr(torch, "__version__", None)
            if torch.cuda.is_available():
                info["cuda_available"] = True
                info["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        return info

    def update_settings(self, **values: Any) -> Dict[str, Any]:
        """Apply runtime-tunable settings (no restart needed)."""
        for key in (
            "threshold_db", "cooldown_seconds", "speaker_match_threshold", "tts_volume", "language",
            "speaker_cooldown_seconds", "min_identify_seconds", "max_simultaneous_speakers",
            "overlap_sensitivity", "unknown_speaker_handling",
        ):
            if values.get(key) is not None:
                setattr(self.settings, key, values[key])
        if self.engine:
            self.engine.threshold_dB = self.settings.threshold_db
            self.engine.cooldown_seconds = self.settings.cooldown_seconds
            self.engine.match_threshold = self.settings.speaker_match_threshold
            self.engine.speaker_cooldown.seconds = self.settings.speaker_cooldown_seconds
            self.engine.grouper.unknown_handling = self.settings.unknown_speaker_handling
        if self.tts:
            self.tts.volume = self.settings.tts_volume
            self.tts.language = self.settings.language
        return self.status()
