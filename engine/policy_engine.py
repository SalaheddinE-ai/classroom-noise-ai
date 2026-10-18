"""Classroom policy engine: dB check -> VAD -> OSD -> speaker ID -> cooldown -> intervene."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import torch

from config import Settings
from core.audio_utils import compute_db, pcm16_bytes_to_float, validate_pcm16
from core.metrics import StageTimer
from core.OSD import OverlapSpeechDetector
from core.speaker_id import StudentRecognizer
from core.tts_engine import TTSEngine
from core.vad import SileroVAD
from database.db_session import session_scope
from database.models import EventType, InterventionLog, Student
from engine.audio_retention import AudioRetention
from engine.events import DetectionEvent, EventGrouper, SpeakerCooldown, cooldown_key
from engine.persistence import persist_event
from engine.visualization import WaveformBuffer

logger = logging.getLogger(__name__)


class State(str, Enum):
    """Result of processing one audio window."""

    IDLE = "IDLE"  # below the dB threshold
    NOISE_ONLY = "NOISE_ONLY"  # loud but no speech (door, chairs...)
    GROUP_WARNING = "GROUP_WARNING"  # overlapping speakers
    SINGLE_STUDENT_WARNING = "SINGLE_STUDENT_WARNING"  # identified student
    GENERAL_WARNING = "GENERAL_WARNING"  # unknown loud speaker


class ClassroomPolicyEngine:
    """Decides whether and how to intervene for each 2-second audio window."""

    DIAGNOSTIC_FLOOR_DB = 30.0  # below this the window is treated as silence (no VAD run)

    def __init__(
        self,
        settings: Settings,
        vad: SileroVAD,
        osd: OverlapSpeechDetector,
        recognizer: StudentRecognizer,
        tts: TTSEngine,
        analyzer: Optional[Any] = None,
    ) -> None:
        """``analyzer`` (a ``MultiSpeakerAnalyzer``) enables multi-speaker identification and the
        grouping of detections into events. Without it the engine keeps its original behaviour:
        one event per window, overlap -> generic group warning."""
        self.settings = settings
        self.vad = vad
        self.osd = osd
        self.recognizer = recognizer
        self.tts = tts
        self.analyzer = analyzer
        self.grouper = EventGrouper(
            settings.event_merge_gap_seconds, settings.unknown_cluster_similarity,
            settings.unknown_speaker_handling, settings.viz_points_per_second,
            settings.tracker_confirm_seconds, settings.tracker_confirm_hits, settings.tracker_hysteresis,
        )
        self.grouper.match_threshold = settings.speaker_match_threshold
        self.timers = StageTimer()
        self.speaker_cooldown = SpeakerCooldown(settings.speaker_cooldown_seconds)
        self.waveform = WaveformBuffer(settings.sample_rate, settings.viz_history_seconds,
                                       settings.viz_points_per_second)
        self.retention = AudioRetention(settings.event_audio_dir, settings.raw_audio_retention_seconds,
                                        settings.sample_rate)
        self._current_updated = 0.0
        self._last_update_spoke = False

        self.threshold_dB: float = settings.threshold_db
        self.cooldown_seconds: float = settings.cooldown_seconds
        self.match_threshold: float = settings.speaker_match_threshold
        self.last_trigger_time: float = 0.0

        self.events: Deque[Dict[str, Any]] = deque(maxlen=100)
        self._lock = threading.Lock()
        self._state = State.IDLE
        self._last_db = 0.0
        self._processing = False
        self._speech = False
        self._speech_ratio = 0.0

    # ---------------------------------------------------------------- state
    @property
    def is_cooldown(self) -> bool:
        """True while the post-intervention cooldown is active."""
        return (time.time() - self.last_trigger_time) < self.cooldown_seconds

    @property
    def cooldown_remaining(self) -> float:
        return max(0.0, self.cooldown_seconds - (time.time() - self.last_trigger_time))

    @staticmethod
    def _slim(event: Dict[str, Any]) -> Dict[str, Any]:
        """Event dict without the (bulky) per-speaker waveform series, for /status and the WebSocket."""
        if "speakers" not in event:
            return event
        slim = dict(event)
        slim["speakers"] = [{k: v for k, v in sp.items() if k != "waveform"} for sp in event["speakers"]]
        return slim

    def _with_status(self, events: List[Dict[str, Any]], now: float) -> List[Dict[str, Any]]:
        """Newest-first copy of ``events`` with the ``ongoing`` / ``completed`` status refreshed."""
        out = []
        for i, ev in enumerate(reversed(events)):
            if "status" in ev:
                ongoing = i == 0 and (now - self._current_updated) <= self.settings.event_merge_gap_seconds
                ev = {**ev, "status": "ongoing" if ongoing else "completed"}
            out.append(ev)
        return out

    def _latency_summary(self) -> Dict[str, Any]:
        """Per-stage median / p95 (ms) and the real-time factor (processing time / hop duration)."""
        summary = self.timers.summary()
        total = summary.get("total")
        if total:
            summary["real_time_factor"] = {
                "median": round(total["median_ms"] / (self.settings.hop_seconds * 1000), 3),
                "p95": round(total["p95_ms"] / (self.settings.hop_seconds * 1000), 3),
            }
        return summary

    def forget_student(self, student_id: str) -> None:
        """Scrub a deleted student's identity from the in-memory events (privacy)."""
        self.grouper.forget_student(student_id)
        with self._lock:
            for i, ev in enumerate(self.events):
                if ev.get("student_id") == student_id or "speakers" not in ev:
                    continue
                speakers = []
                for sp in ev["speakers"]:
                    if sp.get("student_id") == student_id:
                        sp = {**sp, "student_id": None, "student_name": "Unknown Speaker",
                              "is_identified": False, "identification_status": "deleted_profile"}
                    speakers.append(sp)
                self.events[i] = {**ev, "speakers": speakers, "student": None}

    def events_snapshot(self, limit: int = 50, full: bool = True) -> List[Dict[str, Any]]:
        """Newest-first events (``full=True`` keeps the speaker waveform series)."""
        with self._lock:
            evs = list(self.events)[-limit:]
        evs = self._with_status(evs, time.time())
        return evs if full else [self._slim(e) for e in evs]

    def visualization_snapshot(self, seconds: float) -> Dict[str, Any]:
        """Data for the Live Monitor: mixed-microphone waveform + events of the last ``seconds``."""
        now = time.time()
        events = [e for e in self.events_snapshot(self.events.maxlen or 100)
                  if "speakers" in e and e.get("epoch", 0) >= now - seconds - 60]
        return {
            "now": datetime.fromtimestamp(now).astimezone().isoformat(),
            "history_seconds": seconds,
            "waveform": self.waveform.snapshot(seconds),
            "events": events,
            "capabilities": self.analyzer.capabilities(self.osd.backend) if self.analyzer else
            {"source_separation": False, "diarization": False, "overlap_method": "none",
             "isolated_audio_per_speaker": False},
        }

    def snapshot(self) -> Dict[str, Any]:
        """Thread-safe view of the engine for the API / dashboard."""
        now = time.time()
        with self._lock:
            recent = self._with_status(list(self.events)[-20:], now)
            last = self._slim(recent[0]) if recent else None
            if last is not None and "speakers" in last:
                # Grouped events keep growing: for the status banner use the time / outcome of the
                # latest *update* instead of the event's creation time.
                last = {**last, "epoch": self._current_updated, "spoken": self._last_update_spoke}
            return {
                "state": self._state.value,
                "last_chunk_db": round(self._last_db, 1),
                "processing": self._processing,
                "speech_detected": self._speech,
                "speech_ratio": round(self._speech_ratio, 2),
                "threshold_db": self.threshold_dB,
                "cooldown_seconds": self.cooldown_seconds,
                "match_threshold": self.match_threshold,
                "in_cooldown": self.is_cooldown,
                "cooldown_remaining": round(self.cooldown_remaining, 1),
                "osd_backend": self.osd.backend,
                "last_event": last,
                "recent_events": [self._slim(e) for e in recent],
                "current_event": self._slim(recent[0]) if recent and recent[0].get("status") == "ongoing" else None,
                "latency": self._latency_summary(),
            }

    def _set_state(self, state: State, db: Optional[float] = None, processing: bool = False) -> State:
        with self._lock:
            self._state = state
            if db is not None:
                self._last_db = db
            self._processing = processing
        return state

    # ----------------------------------------------------------- main logic
    def process_audio_chunk(self, audio_bytes: bytes, captured_at: Optional[float] = None) -> State:
        """Run the full decision pipeline on one PCM16 window.

        ``captured_at`` (epoch seconds when the window became available) lets the engine report
        true end-to-end latency (queue wait + processing); without it only processing time is known.
        """
        t_start = time.perf_counter()
        problem = validate_pcm16(audio_bytes)
        if problem:  # malformed frame: never feed it to a model
            self.timers.count("invalid_frames")
            logger.warning("Dropped malformed audio frame: %s", problem)
            return self._set_state(State.IDLE, None)
        samples = pcm16_bytes_to_float(audio_bytes)
        db = compute_db(samples, self.settings.calibration_offset_db)
        now = time.time()
        self.waveform.add(samples, now, self.settings.hop_samples)  # live display, every window

        # Ignore our own loudspeaker output (feedback-loop guard).
        if self.tts.is_busy:
            return self._set_state(State.IDLE, db)

        audio = torch.from_numpy(samples)

        # Diagnostics: run VAD on any non-silent window so the dashboard can show whether
        # speech is heard, even when the loudness gate (below) blocks interventions.
        vad_result = None
        if db >= self.DIAGNOSTIC_FLOOR_DB:
            t = time.perf_counter()
            try:
                vad_result = self.vad.analyze(audio)
            except Exception:
                logger.exception("VAD failed")
                self.timers.count("vad_errors")
            self.timers.record("vad", (time.perf_counter() - t) * 1000)
        with self._lock:
            self._speech = bool(vad_result and vad_result.is_speech)
            self._speech_ratio = vad_result.speech_ratio if vad_result else 0.0

        # 1. Loudness gate (uncalibrated digital level; supplementary to the VAD, not a dB SPL reading)
        if db < self.threshold_dB:
            return self._set_state(State.IDLE, db)

        self._set_state(self._state, db, processing=True)

        try:
            # 2. Voice activity
            if vad_result is None or not vad_result.is_speech:
                return self._set_state(State.NOISE_ONLY, db)

            # 3. Overlapping speakers
            t = time.perf_counter()
            overlap = self.osd.detect(audio)
            self.timers.record("overlap_detection", (time.perf_counter() - t) * 1000)
            if self.analyzer is not None:
                result = self._handle_analysis(samples, now, db, vad_result, overlap, t_start, captured_at)
                self.timers.record("total", (time.perf_counter() - t_start) * 1000)
                return result
            if overlap.is_overlap:  # legacy behaviour: generic group warning, no biometrics
                return self._intervene(State.GROUP_WARNING, EventType.GROUP, db, None, None)

            # 4-5. Speaker identification
            embedding = self.recognizer.extract_embedding(audio)
            student, score = self.recognizer.identify_student(embedding)
            if student is not None and score > self.match_threshold:
                return self._intervene(State.SINGLE_STUDENT_WARNING, EventType.SINGLE_STUDENT, db, student, score)
            return self._intervene(State.GENERAL_WARNING, EventType.GENERAL, db, None, score)
        except Exception:
            logger.exception("Error while processing audio chunk")
            self.timers.count("processing_errors")
            return self._set_state(State.IDLE, db)

    # --------------------------------------------------- multi-speaker path
    def _key_eligible(self, key: str, now: float) -> bool:
        """Cooldown per warning key: students/unknown voices use the speaker cooldown, overlap its own."""
        seconds = self.cooldown_seconds if key == "overlap" else None
        return self.speaker_cooldown.eligible(key, now, seconds)

    def _handle_analysis(self, samples, now: float, db: float, vad_result, overlap,
                         t_start: float, captured_at: Optional[float]) -> State:
        """Identify every speaker, group the detection into an event, apply cooldowns, warn, log."""
        duration = len(samples) / self.settings.sample_rate
        t = time.perf_counter()
        analysis = self.analyzer.analyze(
            samples, now - duration, vad_result, overlap, self.match_threshold,
            getattr(self.vad, "frame_seconds", 512 / self.settings.sample_rate),
        )
        self.timers.record("identification", (time.perf_counter() - t) * 1000)
        grouped = self.grouper.update(analysis, db, now, self.match_threshold)
        if grouped is None:  # only ignored unknown speakers
            return self._set_state(State.IDLE, db)
        state, is_new = grouped
        self._current_updated = now

        if self.retention.enabled:
            for obs in analysis.speakers:
                # the grouper resolved obs -> event speaker; save one clip per event speaker
                for spk in state.speakers.values():
                    if spk.clip_ref is None and obs.clip is not None and (
                        (obs.is_identified and spk.student_id == obs.match.student_id)
                        or (not obs.is_identified and spk.student_id is None)
                    ):
                        spk.clip_ref = self.retention.save_clip(state.event_id, spk.speaker_id, obs.clip)
                        break

        draft = self.grouper.to_event(state, now)
        handling = self.settings.unknown_speaker_handling
        keys: List[str] = []
        if draft.is_overlap:
            keys.append("overlap")
        for sp in draft.speakers:
            if sp.is_identified:
                keys.append(cooldown_key(sp))
            elif handling == "report":
                keys.append("unknown")
        keys = list(dict.fromkeys(keys))
        eligible = [k for k in keys if self._key_eligible(k, now)]
        gap_ok = (now - self.last_trigger_time) >= self.settings.warning_min_gap_seconds
        # debounce: first warning only once someone is confirmed or the event has lasted a little
        settled = any(sp.is_identified for sp in draft.speakers) or (now - state.created) >= self.settings.warning_grace_seconds
        # inside one event, a further warning (e.g. a student confirmed late) waits for the cooldown
        repeat_ok = state.warnings_spoken == 0 or (now - self.last_trigger_time) >= self.cooldown_seconds

        names = [s.student_name for s in draft.speakers if s.is_identified] if self.settings.tts_announce_names else []
        event_type = EventType(draft.event_type)
        if event_type == EventType.GROUP and names:
            message = self.tts.render_message(event_type, None, names=names)
        else:
            message = self.tts.render_message(event_type, names[0] if names else None)
        if event_type == EventType.SINGLE_STUDENT and not names:
            message = self.tts.render_message(EventType.GENERAL, None)
        if state.warnings_spoken == 0 or eligible:
            state.message = message  # message of the first / next warning

        self._last_update_spoke = False
        if eligible and gap_ok and settled and repeat_ok:
            self._last_update_spoke = True
            self.tts.speak_nowait(message)
            self.last_trigger_time = now
            state.warnings_spoken += 1
            state.message = message
            self.speaker_cooldown.mark(keys, now)

        # latency + stage timings (measurements, not estimates)
        processing_ms = (time.perf_counter() - t_start) * 1000
        state.latency_ms = round((time.time() - captured_at) * 1000, 1) if captured_at else round(processing_ms, 1)
        state.metadata["stage_ms"] = {"identification": round(self.timers.last("identification") or 0.0, 1),
                                      "processing_total": round(processing_ms, 1)}
        state.metadata["clipping"] = bool(np.mean(np.abs(samples) >= 0.999) > 0.01)
        self.timers.record("end_to_end", state.latency_ms)

        event = self.grouper.to_event(state, now)
        persist_event(event)
        entry = event.model_dump(mode="json")
        with self._lock:
            if not is_new and self.events and self.events[-1].get("event_id") == event.event_id:
                self.events[-1] = entry
            else:
                if self.events and "status" in self.events[-1]:
                    self.events[-1] = {**self.events[-1], "status": "completed"}
                self.events.append(entry)
        if event.event_type == EventType.GROUP.value:
            new_state = State.GROUP_WARNING
        elif event.event_type == EventType.SINGLE_STUDENT.value:
            new_state = State.SINGLE_STUDENT_WARNING
        else:
            new_state = State.GENERAL_WARNING
        return self._set_state(new_state, db)

    def _intervene(
        self,
        state: State,
        event_type: EventType,
        db: float,
        student: Optional[Student],
        score: Optional[float],
    ) -> State:
        """Apply the cooldown rule, speak if allowed and log the event."""
        message = self.tts.render_message(event_type, student.name if student else None)

        # 6-7. Cooldown guard
        spoken = False
        if not self.is_cooldown:
            self.tts.speak_nowait(message)
            self.last_trigger_time = time.time()
            spoken = True

        self._log_event(event_type, student, db, score, message, spoken)
        with self._lock:
            self.events.append(
                {
                    "epoch": time.time(),
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "time_label": datetime.now().strftime("%H:%M:%S"),
                    "event_type": event_type.value,
                    "student": student.name if student else None,
                    "match_score": round(score, 3) if score is not None else None,
                    "db": round(db, 1),
                    "message": message,
                    "spoken": spoken,
                }
            )
        return self._set_state(state, db)

    @staticmethod
    def _log_event(
        event_type: EventType,
        student: Optional[Student],
        db: float,
        score: Optional[float],
        message: str,
        spoken: bool,
    ) -> None:
        try:
            with session_scope() as session:
                session.add(
                    InterventionLog(
                        event_type=event_type,
                        student_id=student.id if student else None,
                        dB_level=db,
                        message_spoken=message if spoken else f"[suppressed by cooldown] {message}",
                        match_score=score,
                        was_spoken=spoken,
                    )
                )
        except Exception:
            logger.exception("Could not persist intervention log")
