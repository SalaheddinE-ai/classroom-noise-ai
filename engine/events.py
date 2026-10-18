"""Structured detection events, grouping of consecutive detections and per-speaker cooldown.

The policy engine analyses overlapping 2 s windows every second. Turning every window into an
alert would flood the dashboard, so :class:`EventGrouper` merges detections that follow each other
within ``event_merge_gap_seconds`` into one :class:`DetectionEvent` that grows while the noise lasts.

Time convention: every datetime in the public schema is timezone-aware and serialised as ISO 8601
with a UTC offset (``2026-10-17T13:43:02.250+00:00``). Internally the grouper works in epoch seconds.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, computed_field

from core.multi_speaker import UNKNOWN_LABEL, ChunkAnalysis, SpeakerObservation
from core.segments import Interval, max_concurrency, merge_intervals, total_duration
from database.event_types import EventType

SCHEMA_VERSION = "2.1"
DECISION_CONFIRMED = "confirmed"  # enough consistent evidence
DECISION_TENTATIVE = "tentative"  # a candidate exists but evidence is insufficient / weak
DECISION_UNKNOWN = "unknown"
SIMILARITY_NOTE = "similarity = cosine similarity of speaker embeddings; it is NOT a probability or a confidence"

MAX_WAVEFORM_POINTS = 600  # per speaker and event (30 s at the default 20 points/s)
MAX_SEGMENTS = 200

CATEGORY_SINGLE = "Single Speaker"
CATEGORY_MULTIPLE = "Multiple Speakers"
CATEGORY_UNKNOWN = "Unknown Speaker"
CATEGORY_OVERLAP = "Overlapping Speech"


def epoch_to_dt(epoch: float) -> datetime:
    """Epoch seconds -> timezone-aware local datetime."""
    return datetime.fromtimestamp(epoch).astimezone()


# --------------------------------------------------------------------------- public schema
class TimeSpan(BaseModel):
    start: datetime
    end: datetime


class WaveformTrack(BaseModel):
    """Peak-amplitude envelope: ``peaks[i]`` is the peak at ``t0 + i * dt_seconds``."""

    t0: datetime
    dt_seconds: float
    peaks: List[float]


class SpeakerEntry(BaseModel):
    """One speaker of an event. ``speaker_id`` is stable for the whole event."""

    speaker_id: str
    student_id: Optional[str] = None
    student_name: str = UNKNOWN_LABEL
    start_time: datetime
    end_time: datetime
    segments: List[TimeSpan] = Field(default_factory=list)
    similarity_score: Optional[float] = None  # best cosine similarity to the profile (NOT a probability)
    runner_up_similarity: Optional[float] = None  # best similarity of the second-best student
    confidence: Optional[float] = None  # only set when a calibrated confidence exists (none today)
    # is_identified is True only for a *confirmed* identity. A tentative speaker carries the
    # candidate in student_id / student_name but is_identified=False.
    is_identified: bool = False
    decision_status: str = DECISION_UNKNOWN  # confirmed | tentative | unknown
    rejection_reason: Optional[str] = None  # why a speaker is not identified (below_threshold, ambiguous...)
    observations: int = 0  # number of agreeing window observations
    evidence_seconds: float = 0.0  # distinct speech duration backing the identity
    identification_status: str = "unidentified"  # matched | below_threshold | ambiguous | ...
    # separated      = real isolated audio of this speaker (source separation)
    # masked_mixture = the mixed microphone signal restricted to this speaker's activity (NOT isolated)
    # activity_only  = timestamps only, no audio signal
    signal_type: str = "activity_only"
    analysis_method: str = "single_stream"
    waveform: Optional[WaveformTrack] = None
    audio_segment_reference: Optional[str] = None


class DetectionEvent(BaseModel):
    """A grouped alert: one or several speakers over a continuous stretch of noise.

    Unavailable measurements are ``None`` (never a fabricated zero): e.g. ``overlap_duration_ms`` is
    ``None`` when the overlap was flagged but the detection method cannot locate it in time.
    All datetimes are timezone-aware ISO 8601; durations are in milliseconds.
    """

    schema_version: str = SCHEMA_VERSION
    event_id: str
    event_type: str  # legacy values SINGLE_STUDENT | GROUP | GENERAL
    category: str  # Single Speaker | Multiple Speakers | Unknown Speaker | Overlapping Speech
    labels: List[str] = Field(default_factory=list)
    timestamp: datetime  # when the event was first created
    start_time: datetime
    end_time: datetime
    status: str = "ongoing"  # ongoing | completed
    overlap_duration_ms: Optional[int] = None
    overlap_segments: List[TimeSpan] = Field(default_factory=list)
    speakers: List[SpeakerEntry] = Field(default_factory=list)
    speaker_count: int = 0
    unknown_speaker_count: Optional[int] = None
    estimated_active_speaker_count: Optional[int] = None  # max simultaneous speakers, None if unmeasurable
    is_overlap: bool = False
    detection_method: str = "vad"  # who flagged the overlap: pyannote | embedding | vad
    identification_method: str = "single_stream"  # single_stream | sepformer | pyannote_slots | mixture_embedding
    runner_up_similarity: Optional[float] = None
    decision_status: str = DECISION_UNKNOWN  # confirmed | tentative | unknown | mixed
    rejection_reason: Optional[str] = None
    warning_triggered: bool = False
    warnings_spoken: int = 0
    message: str = ""
    db: float = 0.0  # uncalibrated pseudo-dB (digital level), not dB SPL
    processing_latency_ms: Optional[float] = None
    notes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def started_at(self) -> datetime:
        return self.start_time

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ended_at(self) -> datetime:
        return self.end_time

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_ms(self) -> int:
        return int(round((self.end_time - self.start_time).total_seconds() * 1000))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def similarity(self) -> Optional[float]:
        """Best speaker similarity of the event (cosine, not a probability)."""
        scores = [s.similarity_score for s in self.speakers if s.similarity_score is not None]
        return round(max(scores), 3) if scores else None

    # ---- legacy keys of the pre-grouping /status ``recent_events`` entries -------------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def epoch(self) -> float:
        return self.timestamp.timestamp()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def time_label(self) -> str:
        return self.timestamp.strftime("%H:%M:%S")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spoken(self) -> bool:
        return self.warning_triggered

    @computed_field  # type: ignore[prop-decorator]
    @property
    def student(self) -> Optional[str]:
        names = [s.student_name for s in self.speakers if s.is_identified]
        return ", ".join(names) if names else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def match_score(self) -> Optional[float]:
        return self.similarity


# ------------------------------------------------------------------------------- cooldown
class SpeakerCooldown:
    """Per-speaker cooldown (the engine's global cooldown still applies on top of it)."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._last: Dict[str, float] = {}
        self._lock = threading.Lock()

    def eligible(self, key: str, now: float, seconds: Optional[float] = None) -> bool:
        """``seconds`` overrides the default cooldown for this check (e.g. the overlap cooldown)."""
        with self._lock:
            last = self._last.get(key)
        return last is None or (now - last) >= (self.seconds if seconds is None else seconds)

    def mark(self, keys: Iterable[str], now: float) -> None:
        with self._lock:
            for key in keys:
                self._last[key] = now

    def remaining(self, key: str, now: float) -> float:
        with self._lock:
            last = self._last.get(key)
        return 0.0 if last is None else max(0.0, self.seconds - (now - last))


def cooldown_key(speaker: SpeakerEntry) -> str:
    """Students have their own cooldown; every unidentified voice shares one."""
    return speaker.student_id if speaker.is_identified and speaker.student_id else "unknown"


# -------------------------------------------------------------------------------- grouper
@dataclass
class _SpeakerState:
    speaker_id: str
    student_id: Optional[str]
    name: str
    status: str  # confirmed | tentative | unknown
    similarity: Optional[float]
    runner_up: Optional[float]
    reason: str
    segments: List[Interval]
    signal_type: str
    method: str
    strong_segments: List[Interval] = field(default_factory=list)  # evidence from reliable observations
    hits: int = 0  # strong observations agreeing on this student
    sim_sum: float = 0.0
    centroid: Optional[np.ndarray] = None
    waveform: Dict[int, float] = field(default_factory=dict)
    pps: int = 20
    clip_ref: Optional[str] = None

    @property
    def is_identified(self) -> bool:
        return self.status == DECISION_CONFIRMED


@dataclass
class _EventState:
    event_id: str
    created: float
    start: float
    end: float
    last_update: float
    db: float
    detection_method: str
    identification_method: str = "single_stream"
    is_overlap: bool = False
    overlap_approximate: bool = False
    overlap_segments: List[Interval] = field(default_factory=list)
    speakers: Dict[str, _SpeakerState] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    warnings_spoken: int = 0
    message: str = ""
    counter: int = 0
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventGrouper:
    """Merges consecutive window analyses into events with stable speaker ids and a temporal tracker.

    Identity tracking rules (only past evidence is used, never future audio):

    * a student is **tentative** after the first matching observation;
    * it becomes **confirmed** once ``confirm_hits`` reliable observations agree and they cover at
      least ``confirm_seconds`` of distinct speech with a mean similarity >= the match threshold;
    * observations from the approximate mixture-embedding mode (or from audio that still contained
      another speaker) can never confirm an identity on their own;
    * a weak observation (within ``hysteresis`` below the threshold) of an already confirmed
      speaker is a continuation, so one noisy window does not create a new "unknown" voice;
    * a new event starts with a clean slate: identities are never carried into unrelated speech.
    """

    def __init__(self, merge_gap_seconds: float, unknown_similarity: float = 0.6,
                 unknown_handling: str = "report", points_per_second: int = 20,
                 confirm_seconds: float = 1.5, confirm_hits: int = 2, hysteresis: float = 0.08) -> None:
        self.merge_gap = merge_gap_seconds
        self.unknown_similarity = unknown_similarity
        self.unknown_handling = unknown_handling
        self.points_per_second = points_per_second
        self.confirm_seconds = confirm_seconds
        self.confirm_hits = confirm_hits
        self.hysteresis = hysteresis
        self.match_threshold = 0.75
        self._current: Optional[_EventState] = None
        self._lock = threading.Lock()

    @property
    def current(self) -> Optional[_EventState]:
        return self._current

    # ---------------------------------------------------------------- update
    def update(self, analysis: ChunkAnalysis, db: float, now: float,
               threshold: Optional[float] = None) -> Optional[Tuple[_EventState, bool]]:
        """Merge ``analysis`` into the current event (or open a new one).

        Returns ``(state, is_new)``; ``None`` when nothing reportable remains (all speakers
        dropped by ``unknown_speaker_handling="ignore"``).
        """
        if threshold is not None:
            self.match_threshold = threshold
        observations = [
            o for o in analysis.speakers
            if o.is_identified or self.unknown_handling != "ignore"
            or (self._current is not None and self._continuation_target(self._current, o) is not None)
        ]
        if not observations:
            return None
        with self._lock:
            state = self._current
            is_new = state is None or (now - state.last_update) > self.merge_gap
            if is_new:
                state = _EventState(
                    event_id=str(uuid.uuid4()), created=now, start=analysis.t0, end=analysis.t1,
                    last_update=now, db=db, detection_method=analysis.detection_method,
                    identification_method=analysis.method,
                )
                self._current = state
            assert state is not None
            state.last_update = now
            state.end = max(state.end, analysis.t1)
            state.start = min(state.start, analysis.t0)
            state.db = max(state.db, db)
            state.detection_method = _merge_method(state.detection_method, analysis.detection_method)
            state.identification_method = _merge_method(state.identification_method, analysis.method)
            if analysis.is_overlap:
                state.is_overlap = True
                state.overlap_approximate = state.overlap_approximate or analysis.overlap_approximate
                state.overlap_segments = merge_intervals(state.overlap_segments + analysis.overlap_segments, 0.2)[-MAX_SEGMENTS:]
            for note in analysis.notes:
                if note not in state.notes:
                    state.notes.append(note)
            for obs in observations:
                self._merge_observation(state, obs)
            return state, is_new

    def _merge_observation(self, state: _EventState, obs: SpeakerObservation) -> None:
        m = obs.match
        continuation = self._continuation_target(state, obs)
        spk = continuation or self._resolve(state, obs)
        spk.segments = merge_intervals(spk.segments + list(obs.segments), 0.3)[-MAX_SEGMENTS:]
        if continuation is not None:
            note = f"{spk.name}: weak window treated as continuation (similarity {m.similarity:.2f})"
            if note not in state.notes and len(state.notes) < 20:
                state.notes.append(note)
        else:
            if m.similarity is not None and (spk.similarity is None or m.similarity > spk.similarity):
                spk.similarity = float(m.similarity)
            if m.runner_up_similarity is not None:
                spk.runner_up = max(spk.runner_up or -1.0, float(m.runner_up_similarity))
            if not m.is_identified:
                spk.reason = m.reason
            elif spk.status == DECISION_UNKNOWN:
                spk.status = DECISION_TENTATIVE
            if m.is_identified:
                weak = obs.method == "mixture_embedding" or obs.quality == "mixed"
                if not weak:
                    spk.hits += 1
                    spk.sim_sum += float(m.similarity or 0.0)
                    spk.strong_segments = merge_intervals(spk.strong_segments + list(obs.segments), 0.3)[-MAX_SEGMENTS:]
                self._update_status(spk)
        # keep the most informative signal type: separated > masked_mixture > activity_only
        rank = {"separated": 2, "masked_mixture": 1, "activity_only": 0}
        if rank.get(obs.signal_type, 0) >= rank.get(spk.signal_type, 0):
            spk.signal_type, spk.method = obs.signal_type, obs.method
        if obs.waveform is not None and len(obs.waveform):
            spk.pps = self.points_per_second
            for i, value in enumerate(obs.waveform):
                key = int(round((obs.waveform_t0 + i / spk.pps) * spk.pps))
                spk.waveform[key] = max(float(value), spk.waveform.get(key, 0.0))
            if len(spk.waveform) > MAX_WAVEFORM_POINTS * 2:  # bound memory on very long events
                for key in sorted(spk.waveform)[: len(spk.waveform) - MAX_WAVEFORM_POINTS]:
                    del spk.waveform[key]

    def _update_status(self, spk: _SpeakerState) -> None:
        """tentative -> confirmed once distinct, consistent, reliable evidence is sufficient."""
        if spk.status == DECISION_CONFIRMED or spk.student_id is None:
            return
        evidence = total_duration(spk.strong_segments)
        mean_sim = spk.sim_sum / spk.hits if spk.hits else 0.0
        if spk.hits >= self.confirm_hits and evidence >= self.confirm_seconds and mean_sim >= self.match_threshold:
            spk.status = DECISION_CONFIRMED
        else:
            spk.status = DECISION_TENTATIVE

    def _continuation_target(self, state: _EventState, obs: SpeakerObservation) -> Optional[_SpeakerState]:
        """Confirmed speaker that a just-below-threshold observation most likely belongs to."""
        m = obs.match
        if m.is_identified or m.reason != "below_threshold" or not m.candidate_id or m.similarity is None:
            return None
        if m.similarity < self.match_threshold - self.hysteresis:
            return None
        for spk in state.speakers.values():
            if spk.status == DECISION_CONFIRMED and spk.student_id == m.candidate_id:
                return spk
        return None

    def _resolve(self, state: _EventState, obs: SpeakerObservation) -> _SpeakerState:
        """Find (or create) the event speaker an observation belongs to."""
        m = obs.match
        if m.is_identified and m.student_id:
            for spk in state.speakers.values():
                if spk.student_id == m.student_id and spk.status != DECISION_UNKNOWN:
                    return spk
            return self._new_speaker(state, m.student_id, m.student_name or UNKNOWN_LABEL, DECISION_TENTATIVE, obs)
        unknowns = [s for s in state.speakers.values() if s.status == DECISION_UNKNOWN]
        if obs.embedding is not None:
            for spk in unknowns:
                if spk.centroid is not None and float(spk.centroid @ _unit(obs.embedding)) >= self.unknown_similarity:
                    spk.centroid = _unit(spk.centroid + _unit(obs.embedding))
                    return spk
            spk = self._new_speaker(state, None, UNKNOWN_LABEL, DECISION_UNKNOWN, obs)
            spk.centroid = _unit(obs.embedding)
            return spk
        # no embedding (segment too short / unresolved): attach to the only unknown voice if any
        if len(unknowns) == 1:
            return unknowns[0]
        if unknowns:
            return max(unknowns, key=lambda s: total_duration(s.segments))
        return self._new_speaker(state, None, UNKNOWN_LABEL, DECISION_UNKNOWN, obs)

    @staticmethod
    def _new_speaker(state: _EventState, student_id: Optional[str], name: str, status: str,
                     obs: SpeakerObservation) -> _SpeakerState:
        state.counter += 1
        spk = _SpeakerState(
            speaker_id=f"S{state.counter}", student_id=student_id, name=name, status=status,
            similarity=obs.match.similarity, runner_up=obs.match.runner_up_similarity,
            reason=obs.match.reason, segments=[], signal_type=obs.signal_type, method=obs.method,
        )
        state.speakers[spk.speaker_id] = spk
        return spk

    def forget_student(self, student_id: str) -> None:
        """Drop a deleted student's identity from the event being built."""
        with self._lock:
            if self._current is None:
                return
            for spk in self._current.speakers.values():
                if spk.student_id == student_id:
                    spk.student_id, spk.name, spk.status = None, UNKNOWN_LABEL, DECISION_UNKNOWN
                    spk.reason = "deleted_profile"

    # ----------------------------------------------------------------- output
    def to_event(self, state: _EventState, now: float, warning_triggered: Optional[bool] = None) -> DetectionEvent:
        """Immutable public view of an event state."""
        entries: List[SpeakerEntry] = []
        for spk in state.speakers.values():
            segs = spk.segments or [(state.start, state.end)]
            waveform = None
            if spk.waveform:
                keys = sorted(spk.waveform)[-MAX_WAVEFORM_POINTS:]
                first, last = keys[0], keys[-1]
                waveform = WaveformTrack(
                    t0=epoch_to_dt(first / spk.pps), dt_seconds=1.0 / spk.pps,
                    peaks=[round(spk.waveform.get(k, 0.0), 4) for k in range(first, last + 1)],
                )
            confirmed = spk.status == DECISION_CONFIRMED
            entries.append(SpeakerEntry(
                speaker_id=spk.speaker_id, student_id=spk.student_id,
                student_name=spk.name if spk.student_id else UNKNOWN_LABEL,
                start_time=epoch_to_dt(min(a for a, _ in segs)), end_time=epoch_to_dt(max(b for _, b in segs)),
                segments=[TimeSpan(start=epoch_to_dt(a), end=epoch_to_dt(b)) for a, b in segs],
                similarity_score=round(spk.similarity, 4) if spk.similarity is not None else None,
                runner_up_similarity=round(spk.runner_up, 4) if spk.runner_up is not None else None,
                is_identified=confirmed, decision_status=spk.status,
                rejection_reason=None if spk.status != DECISION_UNKNOWN else spk.reason,
                observations=spk.hits, evidence_seconds=round(total_duration(spk.strong_segments), 2),
                identification_status="matched" if spk.student_id else spk.reason,
                signal_type=spk.signal_type, analysis_method=spk.method, waveform=waveform,
                audio_segment_reference=spk.clip_ref,
            ))
        n = len(entries)
        n_unconfirmed = sum(1 for e in entries if not e.is_identified)
        labels = [CATEGORY_MULTIPLE if n >= 2 else CATEGORY_SINGLE]
        if n_unconfirmed:
            labels.append(CATEGORY_UNKNOWN)
        if state.is_overlap:
            labels.append(CATEGORY_OVERLAP)
        if state.is_overlap:
            category = CATEGORY_OVERLAP
        elif n >= 2:
            category = CATEGORY_MULTIPLE
        elif n_unconfirmed:
            category = CATEGORY_UNKNOWN
        else:
            category = CATEGORY_SINGLE
        if state.is_overlap or n >= 2:
            event_type = EventType.GROUP
        elif n == 1 and not n_unconfirmed:
            event_type = EventType.SINGLE_STUDENT
        else:
            event_type = EventType.GENERAL

        statuses = {e.decision_status for e in entries}
        if statuses == {DECISION_CONFIRMED}:
            decision = DECISION_CONFIRMED
        elif len(statuses) > 1:
            decision = "mixed"
        else:
            decision = next(iter(statuses), DECISION_UNKNOWN)
        reasons = [e.rejection_reason for e in entries if e.rejection_reason]
        unknown_count = None if state.overlap_approximate else sum(1 for e in entries if e.decision_status == DECISION_UNKNOWN)
        measurable = bool(state.overlap_segments) or not state.is_overlap
        overlap_ms = int(round(total_duration(state.overlap_segments) * 1000)) if measurable else None
        if state.is_overlap and not state.overlap_approximate and state.overlap_segments:
            active = max_concurrency([s.segments or [] for s in state.speakers.values()])
        elif state.is_overlap:
            active = None  # cannot be measured reliably with the selected method
        else:
            active = 1 if n else 0
        runner = [e.runner_up_similarity for e in entries if e.runner_up_similarity is not None]
        triggered = state.warnings_spoken > 0 if warning_triggered is None else warning_triggered
        metadata = {
            "similarity_note": SIMILARITY_NOTE,
            "overlap_estimate": "approximate" if (state.is_overlap and state.overlap_approximate) else
                                ("measured" if state.is_overlap else None),
            **state.metadata,
        }
        return DetectionEvent(
            event_id=state.event_id, event_type=event_type.value, category=category, labels=labels,
            timestamp=epoch_to_dt(state.created), start_time=epoch_to_dt(state.start),
            end_time=epoch_to_dt(state.end),
            status="ongoing" if (now - state.last_update) <= self.merge_gap else "completed",
            overlap_duration_ms=overlap_ms,
            overlap_segments=[TimeSpan(start=epoch_to_dt(a), end=epoch_to_dt(b)) for a, b in state.overlap_segments],
            speakers=entries, speaker_count=n, unknown_speaker_count=unknown_count,
            estimated_active_speaker_count=active, is_overlap=state.is_overlap,
            detection_method=state.detection_method, identification_method=state.identification_method,
            runner_up_similarity=round(max(runner), 3) if runner else None,
            decision_status=decision, rejection_reason=reasons[0] if reasons else None,
            warning_triggered=triggered, warnings_spoken=state.warnings_spoken, message=state.message,
            db=round(state.db, 1), processing_latency_ms=state.latency_ms, notes=list(state.notes),
            metadata=metadata,
        )


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _merge_method(old: str, new: str) -> str:
    """Keep every distinct method seen during the event (e.g. ``embedding+window_scan``)."""
    if new in old.split(","):
        return old
    return f"{old},{new}" if old else new
