"""Multi-speaker analysis: from one audio window to a list of identified / unknown speakers.

Three concepts are kept strictly apart here:

* **Diarization**  - *who speaks when*. Provided by pyannote segmentation slots (optional).
* **Identification** - *which registered student a voice belongs to* (ECAPA-TDNN + cosine).
* **Source separation** - *isolated audio per speaker*. Only produced by the optional SepFormer
  model; ECAPA-TDNN alone does not separate voices.

Strategies for windows with overlapping speech, tried in this order:

1. ``sepformer``        : separated sources -> one embedding per source (needs separation enabled).
2. ``pyannote_slots``   : per-slot activity -> embeddings on the non-overlapped parts of each slot.
3. ``mixture_embedding``: one embedding of the mixed signal matched against all students; every
                          student above ``overlap_match_threshold`` is reported. This is an
                          *estimate* and is flagged as such.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from config import Settings
from core.OSD import OSDResult
from core.segments import (
    Segment,
    concat_segments,
    envelope,
    frame_rms,
    merge_segments,
    overlap_from_masks,
    segments_to_mask,
    shift_segments,
    subtract_segments,
    total_duration,
)
from core.vad import VADResult
from engine.events import UNKNOWN_LABEL, AnalysisResult, SpeakerDetection

logger = logging.getLogger(__name__)

MIN_ACTIVE_SECONDS = 0.3  # a slot / source with less activity is treated as noise


class MultiSpeakerAnalyzer:
    """Turns ``(audio, VAD, OSD)`` into an :class:`AnalysisResult`."""

    def __init__(self, settings: Settings, recognizer: Any, separator: Optional[Any] = None) -> None:
        self.settings = settings
        self.recognizer = recognizer
        self.separator = separator

    # ------------------------------------------------------------------ API
    def analyze(self, audio: np.ndarray, vad: Optional[VADResult], osd: OSDResult, chunk_end: float) -> AnalysisResult:
        sr = self.settings.sample_rate
        duration = len(audio) / sr
        start = chunk_end - duration
        speech = merge_segments(list(vad.segments)) if vad is not None and vad.segments else [(0.0, duration)]

        result: Optional[AnalysisResult] = None
        if osd.is_overlap:
            if self.settings.separation_enabled and self.separator is not None:
                result = self._guard(self._separated, audio, speech, osd, start, chunk_end)
            if result is None and osd.slot_segments and self.settings.diarization_backend != "mixture":
                result = self._guard(self._slots, audio, speech, osd, start, chunk_end)
            if result is None:
                result = self._mixture(audio, speech, osd, start, chunk_end)
        else:
            result = self._single(audio, speech, osd, start, chunk_end)
        result.mixture_envelope = envelope(audio, self.settings.waveform_points)
        return result

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _guard(fn: Callable[..., Optional[AnalysisResult]], *args: Any) -> Optional[AnalysisResult]:
        try:
            return fn(*args)
        except Exception:
            logger.exception("Speaker analysis strategy %s failed; using the next one", fn.__name__)
            return None

    def _rank(self, embedding: np.ndarray) -> List[Tuple[Any, float]]:
        rank = getattr(self.recognizer, "rank_students", None)
        if rank is not None:
            return list(rank(embedding, top_k=None))
        student, score = self.recognizer.identify_student(embedding)
        return [(student, score)] if student is not None else []

    def _assign(self, embeddings: Sequence[np.ndarray], thresholds: Sequence[float]) -> List[Tuple[Optional[Any], Optional[float]]]:
        """One-to-one assignment of streams to students; unmatched streams stay unknown.

        The best below-threshold similarity is still returned (for display) but never
        turns into an identity.
        """
        ranked = [self._rank(e) for e in embeddings]
        candidates = [
            (score, i, stu) for i, lst in enumerate(ranked) for stu, score in lst if score > thresholds[i]
        ]
        candidates.sort(key=lambda c: -c[0])
        taken_students, taken_streams = set(), set()
        out: List[Tuple[Optional[Any], Optional[float]]] = [(None, None)] * len(embeddings)
        for score, i, stu in candidates:
            if i in taken_streams or stu.id in taken_students:
                continue
            out[i] = (stu, float(score))
            taken_streams.add(i)
            taken_students.add(stu.id)
        for i in range(len(embeddings)):
            if i not in taken_streams:
                out[i] = (None, float(ranked[i][0][1]) if ranked[i] else None)
        return out

    def _speaker(
        self,
        student: Optional[Any],
        score: Optional[float],
        segs_abs: List[Segment],
        wave: Optional[List[float]],
        kind: str,
        reliability: str,
    ) -> SpeakerDetection:
        identified = student is not None
        segs_abs = merge_segments(segs_abs) or [(0.0, 0.0)]
        return SpeakerDetection(
            speaker_id="spk",
            student_id=student.id if identified else None,
            student_name=student.name if identified else UNKNOWN_LABEL,
            start=segs_abs[0][0],
            end=segs_abs[-1][1],
            similarity_score=score,
            is_identified=identified,
            segments=segs_abs,
            waveform=wave,
            waveform_kind=kind,
            reliability=reliability,
        )

    def _masked_env(self, audio: np.ndarray, segs_rel: Sequence[Segment]) -> List[float]:
        mask = segments_to_mask(segs_rel, len(audio), self.settings.sample_rate)
        peak = float(np.abs(audio).max()) or 1.0
        return envelope(audio * mask, self.settings.waveform_points, scale=peak)

    def _embed(self, wave: np.ndarray) -> Optional[np.ndarray]:
        if len(wave) / self.settings.sample_rate < self.settings.min_identification_seconds:
            return None
        return self.recognizer.extract_embedding(wave)

    # ------------------------------------------------------------ strategies
    def _single(self, audio, speech, osd, start, chunk_end) -> AnalysisResult:
        sr = self.settings.sample_rate
        wave = concat_segments(audio, speech, sr)
        if len(wave) / sr < self.settings.min_identification_seconds:
            wave = audio  # too little speech: use the whole window (original behaviour)
        emb = self.recognizer.extract_embedding(wave)
        student, score = self.recognizer.identify_student(emb)
        accepted = student if (student is not None and score > self.settings.speaker_match_threshold) else None
        sp = self._speaker(
            accepted,
            float(score) if student is not None else None,
            shift_segments(speech, start),
            self._masked_env(audio, speech),
            "single_source",
            "clean",
        )
        return AnalysisResult([sp], False, [], False, "single_embedding", start, chunk_end)

    def _slots(self, audio, speech, osd, start, chunk_end) -> Optional[AnalysisResult]:
        sr = self.settings.sample_rate
        s = self.settings
        slots = sorted(osd.slot_segments.items(), key=lambda kv: -total_duration(kv[1]))
        slots = [(k, list(v)) for k, v in slots if total_duration(v) >= MIN_ACTIVE_SECONDS][: s.max_simultaneous_speakers]
        if len(slots) < 2:
            return None
        overlap = list(osd.overlap_segments)
        embeddings, thresholds, owners, reliab = [], [], [], {}
        for idx, (_, segs) in enumerate(slots):
            clean = subtract_segments(segs, overlap)
            if total_duration(clean) >= s.min_identification_seconds:
                wave, rel, thr = concat_segments(audio, clean, sr), "clean", s.speaker_match_threshold
            elif total_duration(segs) >= s.min_identification_seconds:
                wave, rel, thr = concat_segments(audio, segs, sr), "overlapped", s.overlap_match_threshold
            else:
                reliab[idx] = "too_short"
                continue
            emb = self._embed(wave)
            if emb is None:
                reliab[idx] = "too_short"
                continue
            embeddings.append(emb)
            thresholds.append(thr)
            owners.append(idx)
            reliab[idx] = rel
        assigned = self._assign(embeddings, thresholds) if embeddings else []
        by_slot = dict(zip(owners, assigned))
        speakers: List[SpeakerDetection] = []
        for idx, (_, segs) in enumerate(slots):
            student, score = by_slot.get(idx, (None, None))
            speakers.append(
                self._speaker(student, score, shift_segments(segs, start), self._masked_env(audio, segs), "masked_mixture", reliab.get(idx, "too_short"))
            )
        notes = []
        if any(r in ("overlapped", "too_short") for r in reliab.values()):
            notes.append("Some speakers were identified from overlapped or very short audio (lower reliability).")
        return AnalysisResult(speakers, True, shift_segments(overlap, start), False, "pyannote_slots", start, chunk_end, notes)

    def _mixture(self, audio, speech, osd, start, chunk_end) -> AnalysisResult:
        sr, s = self.settings.sample_rate, self.settings
        wave = concat_segments(audio, speech, sr)
        if len(wave) / sr < s.min_identification_seconds:
            wave = audio
        emb = self.recognizer.extract_embedding(wave)
        matches = [(stu, sc) for stu, sc in self._rank(emb) if sc >= s.overlap_match_threshold][: s.max_simultaneous_speakers]
        speech_abs = shift_segments(speech, start)
        env = self._masked_env(audio, speech)
        speakers = [self._speaker(stu, sc, speech_abs, None, "activity_only", "estimated") for stu, sc in matches]
        while len(speakers) < 2:  # the OSD established that at least two people are talking
            speakers.append(self._speaker(None, None, speech_abs, None, "activity_only", "estimated"))
        notes = [
            "Voices could not be separated: identities are estimated from the mixed signal "
            "and the speaker count is a lower bound. Enable pyannote or source separation for better results."
        ]
        overlap_abs = shift_segments(list(osd.overlap_segments), start)
        result = AnalysisResult(speakers, True, overlap_abs, not overlap_abs, "mixture_embedding", start, chunk_end, notes)
        result.mixture_envelope = env
        return result

    def _separated(self, audio, speech, osd, start, chunk_end) -> Optional[AnalysisResult]:
        sr, s = self.settings.sample_rate, self.settings
        sources = self.separator.separate(audio)
        frame_ms = 20.0
        frame_dur = frame_ms / 1000.0
        rms = [frame_rms(src, sr, frame_ms) for src in sources]
        if not rms or min(len(r) for r in rms) == 0:
            return None
        ceiling = max(float(r.max()) for r in rms)
        if ceiling <= 0:
            return None
        masks = [r > s.separation_activity_ratio * ceiling for r in rms]
        from core.segments import mask_to_segments

        segs_per_source = [merge_segments(mask_to_segments(m, frame_dur), gap=0.2) for m in masks]
        order = [i for i in np.argsort([-total_duration(g) for g in segs_per_source]) if total_duration(segs_per_source[i]) >= MIN_ACTIVE_SECONDS]
        order = order[: s.max_simultaneous_speakers]
        if len(order) < 2:
            return None
        peak = max(float(np.abs(sources[i]).max()) for i in order) or 1.0
        embeddings, owners = [], []
        for i in order:
            emb = self._embed(concat_segments(sources[i], segs_per_source[i], sr))
            if emb is not None:
                embeddings.append(emb)
                owners.append(i)
        assigned = self._assign(embeddings, [s.speaker_match_threshold] * len(embeddings)) if embeddings else []
        by_src = dict(zip(owners, assigned))
        speakers = []
        for i in order:
            student, score = by_src.get(i, (None, None))
            speakers.append(
                self._speaker(student, score, shift_segments(segs_per_source[i], start), envelope(sources[i], s.waveform_points, scale=peak), "separated", "clean" if i in by_src else "too_short")
            )
        overlap = overlap_from_masks([masks[i] for i in order], frame_dur)
        return AnalysisResult(speakers, True, shift_segments(overlap, start), False, "sepformer", start, chunk_end)
