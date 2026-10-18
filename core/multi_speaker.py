"""Multi-speaker analysis: who is speaking in one audio window?

Three terms are kept strictly apart in this module (and in the dashboard):

* **Speaker identification** - matching an ECAPA-TDNN embedding to a registered student.
* **Diarization / overlap detection** - *when* each speaker is active (timestamps and labels only).
* **Source separation** - *audio* of each speaker, isolated from the mix.

ECAPA-TDNN is an embedding model: it does **not** separate overlapping voices. To identify several
simultaneous speakers this module therefore uses the best of three tiers that is available:

1. ``sepformer``          - an optional separator yields one waveform per voice; each is embedded
   and matched on its own (true per-speaker audio, ``signal_type="separated"``).
2. ``pyannote_slots``     - the pyannote segmentation activity says who is active when. Each local
   speaker is embedded on the frames where it speaks *alone* when possible (``"masked_mixture"``:
   the audio is still the mixed microphone signal, not isolated audio).
3. ``mixture_embedding``  - lightweight fallback: embed sliding windows of the mix and report the
   registered students whose profile matches a window above the threshold. This can only name a
   speaker that dominates some window; it never invents identities. If an overlap is detected but
   fewer than two voices can be resolved, an ``Unknown Speaker`` placeholder records the gap.

When there is no overlap a single stream is identified from the speech-only part of the window.
The module depends on NumPy only (no torch / SQLAlchemy) so it is cheap to import and to test.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.segments import (
    Interval,
    amplitude_envelope,
    mask_to_intervals,
    merge_intervals,
    overlap_between_streams,
    pad_intervals,
    probs_to_intervals,
    slice_samples,
    total_duration,
)

logger = logging.getLogger(__name__)

UNKNOWN_LABEL = "Unknown Speaker"
# Extra similarity required when the embedding was computed on audio that still contains
# another speaker (mixed): such embeddings are less reliable.
MIXED_EMBEDDING_PENALTY = 0.05

SIGNAL_SEPARATED = "separated"
SIGNAL_MASKED_MIXTURE = "masked_mixture"
SIGNAL_ACTIVITY = "activity_only"

METHOD_SINGLE = "single_stream"
METHOD_SEPARATION = "sepformer"  # overlap mode B: estimated sources, each embedded on its own
METHOD_DIARIZATION = "pyannote_slots"  # overlap mode A: speaker-activity slots + clean-region embeddings
METHOD_WINDOW_SCAN = "mixture_embedding"  # overlap mode C: approximate fallback on the mixed signal


# --------------------------------------------------------------------------- data classes
@dataclass(frozen=True)
class SpeakerMatch:
    """Outcome of matching one embedding against the registered students."""

    student_id: Optional[str]
    student_name: Optional[str]
    similarity: Optional[float]  # best centroid cosine (even when rejected); None if not computed
    is_identified: bool
    reason: str  # matched | below_threshold | ambiguous | no_students | no_embedding | taken
    # Diagnostics (never shown as probabilities): best candidate and runner-up, even when rejected.
    candidate_id: Optional[str] = None
    runner_up_id: Optional[str] = None
    runner_up_similarity: Optional[float] = None


@dataclass
class SpeakerObservation:
    """One speaker stream observed in one window (times are absolute epoch seconds)."""

    key: str
    match: SpeakerMatch
    segments: List[Interval]
    signal_type: str
    method: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    waveform: Optional[np.ndarray] = field(default=None, repr=False)  # peak envelope from ``t0``
    waveform_t0: float = 0.0
    quality: str = "clean"  # clean | mixed
    clip: Optional[np.ndarray] = field(default=None, repr=False)  # audio, only used for retention

    @property
    def is_identified(self) -> bool:
        return self.match.is_identified


@dataclass
class ChunkAnalysis:
    """Everything learned from one window."""

    t0: float
    t1: float
    speakers: List[SpeakerObservation]
    is_overlap: bool
    overlap_segments: List[Interval]
    method: str  # identification method (single_stream | sepformer | pyannote_slots | mixture_embedding)
    detection_method: str  # who detected the overlap (pyannote | embedding | ...)
    notes: List[str] = field(default_factory=list)
    # False when the overlap is flagged but cannot be located in time (duration must be reported as null)
    overlap_measurable: bool = True
    overlap_approximate: bool = False

    @property
    def overlap_duration(self) -> float:
        return total_duration(self.overlap_segments)


# ------------------------------------------------------------------------------ matcher
class SpeakerMatcher:
    """Matches embeddings to registered students (cosine similarity on profile centroids)."""

    def __init__(self, store, settings) -> None:
        self.store = store
        self.settings = settings

    def scores(self, embedding: np.ndarray) -> Dict[str, float]:
        """Similarity to every student that has enough samples."""
        return self.store.score_students(
            embedding,
            max_samples=self.settings.samples_per_student,
            min_samples=self.settings.min_voice_samples,
        )

    def match(self, embedding: Optional[np.ndarray], threshold: float, penalty: float = 0.0) -> SpeakerMatch:
        """Best valid identity for one embedding, or an unidentified result (never forced)."""
        return self.assign([embedding], threshold, [penalty])[0]

    def assign(
        self,
        embeddings: Sequence[Optional[np.ndarray]],
        threshold: float,
        penalties: Optional[Sequence[float]] = None,
    ) -> List[SpeakerMatch]:
        """Assign students to several simultaneous streams.

        A student can be attributed to at most one stream of the same window; the highest
        similarities are assigned first. A stream whose best and second-best students are closer
        than ``min_match_margin`` (both above threshold) is reported as ambiguous/unidentified.
        """
        n = len(embeddings)
        penalties = list(penalties) if penalties is not None else [0.0] * n
        per_stream: List[Optional[Dict[str, float]]] = []
        diag: Dict[int, Tuple[Optional[str], Optional[str], Optional[float]]] = {}
        results: List[Optional[SpeakerMatch]] = [None] * n
        for i, emb in enumerate(embeddings):
            if emb is None:
                results[i] = SpeakerMatch(None, None, None, False, "no_embedding")
                per_stream.append(None)
                continue
            scores = self.scores(emb)
            per_stream.append(scores)
            if not scores:
                results[i] = SpeakerMatch(None, None, None, False, "no_students")

        candidates: List[Tuple[float, int, str]] = []
        best_seen: Dict[int, float] = {}
        ambiguous: set = set()
        margin = self.settings.min_match_margin
        for i, scores in enumerate(per_stream):
            if not scores:
                continue
            ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            best_seen[i] = ordered[0][1]
            diag[i] = (ordered[0][0], ordered[1][0] if len(ordered) > 1 else None,
                       ordered[1][1] if len(ordered) > 1 else None)
            needed = threshold + penalties[i]
            if len(ordered) > 1 and ordered[0][1] >= needed and ordered[1][1] >= needed \
                    and ordered[0][1] - ordered[1][1] < margin:
                ambiguous.add(i)
            for sid, score in ordered:
                if score >= needed:
                    candidates.append((score, i, sid))

        taken_students: set = set()
        for score, i, sid in sorted(candidates, key=lambda c: c[0], reverse=True):
            if results[i] is not None or i in ambiguous or sid in taken_students:
                continue
            taken_students.add(sid)
            cand, ru_id, ru_sim = diag[i]
            runner = ru_id if ru_id != sid else None
            results[i] = SpeakerMatch(sid, self.store.name_of(sid), float(score), True, "matched",
                                      sid, runner, ru_sim if runner else None)

        for i in range(n):
            if results[i] is not None:
                continue
            if i in ambiguous:
                reason = "ambiguous"
            elif any(c[1] == i for c in candidates):
                reason = "taken"  # best student already attributed to a stronger stream
            else:
                reason = "below_threshold"
            cand, ru_id, ru_sim = diag.get(i, (None, None, None))
            results[i] = SpeakerMatch(None, None, best_seen.get(i), False, reason, cand, ru_id, ru_sim)
        return [r for r in results if r is not None]


# ----------------------------------------------------------------------------- analyzer
class MultiSpeakerAnalyzer:
    """Turns one audio window into a list of speaker observations."""

    ENERGY_FRAME_SECONDS = 0.02

    def __init__(
        self,
        settings,
        store,
        embed_fn: Callable[[np.ndarray], np.ndarray],
        separator=None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embed_fn = embed_fn
        self.separator = separator
        self.matcher = SpeakerMatcher(store, settings)

    # ------------------------------------------------------------------ capabilities
    @property
    def separation_active(self) -> bool:
        return bool(self.settings.separation_enabled and self.separator is not None
                    and getattr(self.separator, "available", True))

    def capabilities(self, osd_backend: Optional[str] = None) -> Dict[str, object]:
        """What the current configuration can and cannot do (shown in the dashboard)."""
        diarization = self.settings.diarization_backend != "none" and osd_backend == "pyannote"
        tier = self._select_mode(has_activity=diarization)[0]
        return {
            "source_separation": self.separation_active,
            "diarization": diarization,
            "overlap_method": tier,
            "isolated_audio_per_speaker": self.separation_active,
        }

    def _select_mode(self, has_activity: bool) -> Tuple[str, Optional[str]]:
        """Overlap strategy for this window: ``(method, note)``; ``note`` explains a fallback."""
        available = {
            METHOD_SEPARATION: self.separation_active,
            METHOD_DIARIZATION: bool(has_activity and self.settings.diarization_backend != "none"),
            METHOD_WINDOW_SCAN: True,
        }
        best = next(m for m in (METHOD_SEPARATION, METHOD_DIARIZATION, METHOD_WINDOW_SCAN) if available[m])
        wanted = self.settings.overlap_mode
        if wanted == "auto":
            return best, None
        if available[wanted]:
            return wanted, None
        return best, f"overlap_mode={wanted} unavailable (model/backend not active): using {best}"

    # ----------------------------------------------------------------------- main entry
    def analyze(
        self,
        audio: np.ndarray,
        t0: float,
        vad_result=None,
        osd_result=None,
        match_threshold: Optional[float] = None,
        vad_frame_seconds: float = 512 / 16000,
    ) -> ChunkAnalysis:
        """Identify the speakers of one mono float32 window starting at epoch ``t0``."""
        sr = self.settings.sample_rate
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = len(audio) / sr
        threshold = self.settings.speaker_match_threshold if match_threshold is None else match_threshold
        is_overlap = bool(osd_result is not None and osd_result.is_overlap)
        osd_backend = getattr(osd_result, "backend", None) if osd_result is not None else None

        speech = self._speech_intervals(vad_result, duration, vad_frame_seconds)
        notes: List[str] = []
        if not is_overlap:
            speakers = [self._single_stream(audio, t0, speech, threshold)]
            return ChunkAnalysis(t0, t0 + duration, speakers, False, [], METHOD_SINGLE,
                                 osd_backend or "vad", notes)

        has_activity = osd_result is not None and getattr(osd_result, "activity", None) is not None
        method, mode_note = self._select_mode(has_activity)
        if mode_note:
            notes.append(mode_note)
        analysis: Optional[ChunkAnalysis] = None
        if method == METHOD_SEPARATION:
            try:
                analysis = self._separation(audio, t0, threshold, osd_backend)
            except Exception as exc:  # never let an optional component stop monitoring
                logger.warning("Source separation failed (%s); falling back", exc)
                notes.append(f"sepformer failed ({type(exc).__name__}): fell back")
                method = METHOD_DIARIZATION if has_activity and self.settings.diarization_backend != "none" \
                    else METHOD_WINDOW_SCAN
        if analysis is None and method == METHOD_DIARIZATION:
            analysis = self._diarization(audio, t0, osd_result, threshold)
        if analysis is None:
            analysis = self._window_scan(audio, t0, speech, osd_result, threshold)
        analysis.notes = notes + analysis.notes
        analysis.is_overlap = True
        analysis.detection_method = osd_backend or "overlap"
        return analysis

    # ---------------------------------------------------------------------- single stream
    def _speech_intervals(self, vad_result, duration: float, frame_seconds: float) -> List[Interval]:
        probs = getattr(vad_result, "frame_probs", None) if vad_result is not None else None
        if probs:
            segs = probs_to_intervals(
                probs, frame_seconds, self.settings.vad_threshold,
                min_duration=self.settings.vad_min_speech_ms / 1000.0,
                gap=self.settings.vad_merge_gap_ms / 1000.0,
            )
            if segs:
                return pad_intervals(segs, self.settings.vad_pad_ms / 1000.0, 0.0, duration)
        return [(0.0, duration)]

    def _single_stream(self, audio: np.ndarray, t0: float, speech: List[Interval],
                       threshold: float) -> SpeakerObservation:
        sr = self.settings.sample_rate
        voiced = slice_samples(audio, speech, sr)
        embedding = None
        if len(voiced) / sr >= self.settings.min_identify_seconds:
            embedding = self.embed_fn(voiced)
        match = self.matcher.match(embedding, threshold)
        return SpeakerObservation(
            key="S1", match=match, segments=[(t0 + a, t0 + b) for a, b in speech],
            signal_type=SIGNAL_ACTIVITY, method=METHOD_SINGLE, embedding=embedding,
        )

    # ---------------------------------------------------------------- tier 1: separation
    def _separation(self, audio: np.ndarray, t0: float, threshold: float,
                    osd_backend: Optional[str]) -> ChunkAnalysis:
        sr = self.settings.sample_rate
        sources = self.separator.separate(audio, sr)
        rms = [float(np.sqrt(np.mean(np.square(s)))) if len(s) else 0.0 for s in sources]
        peak = max(rms) if rms else 0.0
        order = [i for i in np.argsort(rms)[::-1] if rms[i] > 1e-4 and rms[i] >= 0.1 * peak]
        order = order[: self.settings.max_simultaneous_speakers]
        frame = max(1, int(self.ENERGY_FRAME_SECONDS * sr))
        streams: List[Tuple[np.ndarray, List[Interval]]] = []
        for i in order:
            src = sources[i]
            n = len(src) // frame
            if n == 0:
                continue
            frame_rms = np.sqrt(np.mean(np.square(src[: n * frame].reshape(n, frame)), axis=1))
            active = frame_rms >= max(0.15 * float(frame_rms.max()), 1e-4)
            segs = mask_to_intervals(active, frame / sr, 0.0, min_duration=0.1, gap=0.2)
            if segs:
                streams.append((src, segs))
        notes: List[str] = []
        if len(streams) < 2:
            notes.append("separation found a single active source")
        embeddings: List[Optional[np.ndarray]] = []
        for src, segs in streams:
            voiced = slice_samples(src, segs, sr)
            embeddings.append(self.embed_fn(voiced) if len(voiced) / sr >= self.settings.min_identify_seconds else None)
        matches = self.matcher.assign(embeddings, threshold)
        speakers = []
        for idx, ((src, segs), emb, match) in enumerate(zip(streams, embeddings, matches), start=1):
            speakers.append(SpeakerObservation(
                key=f"S{idx}", match=match, segments=[(t0 + a, t0 + b) for a, b in segs],
                signal_type=SIGNAL_SEPARATED, method=METHOD_SEPARATION, embedding=emb,
                waveform=amplitude_envelope(src, self.settings.viz_points_per_second, sr),
                waveform_t0=t0, clip=src,
            ))
        overlap = overlap_between_streams([[(a, b) for a, b in segs] for _, segs in streams])
        return ChunkAnalysis(t0, t0 + len(audio) / sr, speakers, len(overlap) > 0,
                             [(t0 + a, t0 + b) for a, b in overlap], METHOD_SEPARATION,
                             METHOD_SEPARATION, notes)

    # ---------------------------------------------------------------- tier 2: diarization
    def _diarization(self, audio: np.ndarray, t0: float, osd_result, threshold: float) -> ChunkAnalysis:
        sr = self.settings.sample_rate
        activity = np.asarray(osd_result.activity, dtype=bool)
        frame_s = float(osd_result.frame_seconds)
        n_frames, n_local = activity.shape
        simultaneous = activity.sum(axis=1)
        overlap_local = mask_to_intervals(simultaneous >= 2, frame_s, 0.0, min_duration=0.1, gap=0.1)

        local: List[Tuple[int, List[Interval], List[Interval], float]] = []
        for k in range(n_local):
            col = activity[:, k]
            segs = mask_to_intervals(col, frame_s, 0.0, min_duration=0.2, gap=0.15)
            if not segs or total_duration(segs) < 0.3:
                continue
            clean = mask_to_intervals(col & (simultaneous == 1), frame_s, 0.0, min_duration=0.2, gap=0.1)
            local.append((k, segs, clean, total_duration(segs)))
        local.sort(key=lambda item: item[3], reverse=True)
        local = local[: self.settings.max_simultaneous_speakers]

        embeddings: List[Optional[np.ndarray]] = []
        qualities: List[str] = []
        for _, segs, clean, _ in local:
            if total_duration(clean) >= self.settings.min_identify_seconds:
                embeddings.append(self.embed_fn(slice_samples(audio, clean, sr)))
                qualities.append("clean")
            elif total_duration(segs) >= self.settings.min_identify_seconds:
                embeddings.append(self.embed_fn(slice_samples(audio, segs, sr)))
                qualities.append("mixed")
            else:
                embeddings.append(None)
                qualities.append("clean")
        penalties = [MIXED_EMBEDDING_PENALTY if q == "mixed" else 0.0 for q in qualities]
        matches = self.matcher.assign(embeddings, threshold, penalties)

        speakers = []
        for idx, ((k, segs, _, _), emb, match, quality) in enumerate(
                zip(local, embeddings, matches, qualities), start=1):
            mask = np.zeros(len(audio), dtype=np.float32)
            for a, b in segs:
                mask[int(a * sr): int(b * sr)] = 1.0
            speakers.append(SpeakerObservation(
                key=f"S{idx}", match=match, segments=[(t0 + a, t0 + b) for a, b in segs],
                signal_type=SIGNAL_MASKED_MIXTURE, method=METHOD_DIARIZATION, embedding=emb,
                waveform=amplitude_envelope(audio * mask, self.settings.viz_points_per_second, sr),
                waveform_t0=t0, quality=quality,
                clip=slice_samples(audio, segs, sr),
            ))
        notes = []
        if any(q == "mixed" for q in qualities):
            notes.append("some speakers never spoke alone in this window: identification used mixed audio")
        if len(speakers) < 2:
            notes.append("segmentation found fewer than two active speakers")
        return ChunkAnalysis(t0, t0 + len(audio) / sr, speakers, True,
                             [(t0 + a, t0 + b) for a, b in overlap_local], METHOD_DIARIZATION,
                             METHOD_DIARIZATION, notes)

    # ---------------------------------------------------------------- tier 3: window scan
    def _window_scan(self, audio: np.ndarray, t0: float, speech: List[Interval], osd_result,
                     threshold: float) -> ChunkAnalysis:
        sr = self.settings.sample_rate
        starts = getattr(osd_result, "window_starts", None)
        embs = getattr(osd_result, "window_embeddings", None)
        win = getattr(osd_result, "window_seconds", None) or self.settings.scan_window_seconds
        flags = getattr(osd_result, "window_flags", None)
        if starts is None or embs is None:
            win = self.settings.scan_window_seconds
            hop = self.settings.scan_hop_seconds
            n_win, hop_n, win_n = 0, int(hop * sr), int(win * sr)
            starts_l, emb_l = [], []
            for s in range(0, max(len(audio) - win_n, 0) + 1, max(hop_n, 1)):
                chunk = audio[s: s + win_n]
                if len(chunk) == win_n:
                    starts_l.append(s / sr)
                    emb_l.append(self.embed_fn(chunk))
                    n_win += 1
            starts, embs, flags = tuple(starts_l), (np.stack(emb_l) if emb_l else np.zeros((0, 1))), None

        per_student: Dict[str, List[Tuple[float, float]]] = {}
        best: Dict[str, float] = {}
        # mixed audio is unreliable: demand extra similarity before naming anybody
        mixture_threshold = threshold + self.settings.mixture_extra_threshold
        for start, emb in zip(starts, embs):
            scores = self.matcher.scores(emb)
            for sid, score in scores.items():
                if score >= mixture_threshold:
                    per_student.setdefault(sid, []).append((start, start + win))
                    best[sid] = max(best.get(sid, -1.0), score)
        ranked = sorted(per_student, key=lambda sid: best[sid], reverse=True)
        ranked = ranked[: self.settings.max_simultaneous_speakers]

        speakers: List[SpeakerObservation] = []
        for idx, sid in enumerate(ranked, start=1):
            segs = merge_intervals(per_student[sid])
            speakers.append(SpeakerObservation(
                key=f"S{idx}",
                match=SpeakerMatch(sid, self.store.name_of(sid), float(best[sid]), True, "matched"),
                segments=[(t0 + a, t0 + b) for a, b in segs],
                signal_type=SIGNAL_ACTIVITY, method=METHOD_WINDOW_SCAN,
                clip=slice_samples(audio, segs, sr),
            ))
        flagged = [(s, s + win) for s, f in zip(starts, flags or ()) if f]
        overlap = merge_intervals(flagged)
        notes = ["no source separation or diarization model active: speakers are named only when "
                 "they dominate at least one analysis window"]
        if not flagged:
            notes.append("overlap could not be localized in time")
        else:
            notes.append("overlap duration is approximate (windows whose voices are inconsistent)")
        if len(speakers) < 2:
            all_segs = overlap or speech
            speakers.append(SpeakerObservation(
                key=f"S{len(speakers) + 1}",
                match=SpeakerMatch(None, None, None, False, "unresolved_overlap"),
                segments=[(t0 + a, t0 + b) for a, b in all_segs],
                signal_type=SIGNAL_ACTIVITY, method=METHOD_WINDOW_SCAN,
            ))
            notes.append("overlap detected but fewer than two voices could be resolved")
        notes.append(f"approximate: names require similarity >= {mixture_threshold:.2f} (threshold + "
                     f"{self.settings.mixture_extra_threshold:.2f}) on at least one window")
        return ChunkAnalysis(t0, t0 + len(audio) / sr, speakers, True,
                             [(t0 + a, t0 + b) for a, b in overlap], METHOD_WINDOW_SCAN,
                             METHOD_WINDOW_SCAN, notes, overlap_measurable=bool(flagged),
                             overlap_approximate=True)
