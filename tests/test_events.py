"""Event grouping, temporal tracker, schema semantics and cooldown (no models)."""
import json

import numpy as np
import pytest

from core.multi_speaker import (
    ChunkAnalysis, SpeakerMatch, SpeakerObservation, METHOD_DIARIZATION, METHOD_SINGLE, METHOD_WINDOW_SCAN,
)
from engine.events import (
    CATEGORY_MULTIPLE, CATEGORY_OVERLAP, CATEGORY_SINGLE, CATEGORY_UNKNOWN, SCHEMA_VERSION,
    EventGrouper, SpeakerCooldown,
)
from tests.helpers import fake_embed, tone


def matched(sid, name, sim=0.9, runner=0.3):
    return SpeakerMatch(sid, name, sim, True, "matched", sid, "other", runner)


def unknown(sim=0.4, reason="below_threshold", cand="salah", emb=None):
    return SpeakerMatch(None, None, sim, False, reason, cand, None, None)


def obs(match, t0, t1, key="S1", method=METHOD_SINGLE, emb=None, quality="clean", signal="activity_only"):
    return SpeakerObservation(key, match, [(t0, t1)], signal, method, embedding=emb, quality=quality)


def chunk(t0, observations, overlap=None, method=METHOD_SINGLE, detection="vad", measurable=True, approx=False):
    overlap = overlap or []
    return ChunkAnalysis(t0, t0 + 2.0, observations, bool(overlap) or approx, overlap, method, detection,
                         overlap_measurable=measurable, overlap_approximate=approx)


def grouper(**kw):
    g = EventGrouper(3.0, **kw)
    g.match_threshold = 0.75
    return g


# ------------------------------------------------------------ grouping
def test_consecutive_windows_form_one_event_not_one_per_frame():
    g = grouper()
    ids = set()
    for i in range(6):                                  # 6 windows, 1 s apart
        state, is_new = g.update(chunk(100 + i, [obs(matched("s1", "Salah"), 100 + i, 102 + i)]), 70.0, 100 + i)
        ids.add(state.event_id)
        assert is_new == (i == 0)
    assert len(ids) == 1
    ev = g.to_event(state, 106)
    assert ev.speaker_count == 1 and ev.duration_ms == 7000  # 100 .. 107


def test_gap_longer_than_merge_gap_starts_new_event_with_clean_slate():
    g = grouper()
    s1, _ = g.update(chunk(100, [obs(matched("s1", "Salah"), 100, 102)]), 70, 100)
    s1b, _ = g.update(chunk(101, [obs(matched("s1", "Salah"), 101, 103)]), 70, 101)
    assert g.to_event(s1b, 101).speakers[0].decision_status == "confirmed"
    s2, is_new = g.update(chunk(110, [obs(matched("s1", "Salah"), 110, 112)]), 70, 110)
    assert is_new and s2.event_id != s1.event_id
    assert g.to_event(s2, 110).speakers[0].decision_status == "tentative"   # identity not carried over


def test_speaker_ids_are_stable_and_unique_within_event():
    g = grouper()
    a, b = fake_embed(tone(500, 1)), fake_embed(tone(3000, 1))
    for i in range(3):
        g.update(chunk(100 + i, [obs(unknown(emb=a), 100 + i, 102 + i, emb=a), obs(unknown(emb=b), 100 + i, 102 + i, "S2", emb=b)]), 70, 100 + i)
    ev = g.to_event(g.current, 103)
    assert [s.speaker_id for s in ev.speakers] == ["S1", "S2"] and ev.unknown_speaker_count == 2


# -------------------------------------------------------------- tracker
def test_identity_tentative_then_confirmed_with_enough_distinct_evidence():
    g = EventGrouper(3.0, confirm_seconds=1.5, confirm_hits=2)
    g.match_threshold = 0.75
    st, _ = g.update(chunk(100, [obs(matched("s1", "Salah"), 100, 101)]), 70, 100)
    sp = g.to_event(st, 100).speakers[0]
    assert sp.decision_status == "tentative" and not sp.is_identified and sp.student_name == "Salah"
    g.update(chunk(100.5, [obs(matched("s1", "Salah"), 100, 101)]), 70, 100.5)   # same audio again: no new evidence
    assert g.to_event(g.current, 100.5).speakers[0].decision_status == "tentative"
    g.update(chunk(101.5, [obs(matched("s1", "Salah"), 101.2, 102.2)]), 70, 101.5)
    sp = g.to_event(g.current, 101.5).speakers[0]
    assert sp.decision_status == "confirmed" and sp.is_identified and sp.evidence_seconds >= 1.5


def test_mixture_embedding_can_never_confirm():
    g = grouper()
    for i in range(5):
        g.update(chunk(100 + i, [obs(matched("s1", "Salah"), 100 + i, 102 + i, method=METHOD_WINDOW_SCAN)],
                       overlap=[(100 + i, 101 + i)], method=METHOD_WINDOW_SCAN, approx=True), 70, 100 + i)
    ev = g.to_event(g.current, 105)
    assert ev.speakers[0].decision_status == "tentative"
    assert ev.unknown_speaker_count is None and ev.estimated_active_speaker_count is None
    assert ev.metadata["overlap_estimate"] == "approximate"


def test_mixed_quality_observation_cannot_confirm():
    g = grouper()
    for i in range(4):
        g.update(chunk(100 + i, [obs(matched("s1", "Salah"), 100 + i, 102 + i, quality="mixed")]), 70, 100 + i)
    assert g.to_event(g.current, 104).speakers[0].decision_status == "tentative"


def test_hysteresis_keeps_confirmed_speaker_through_one_weak_window():
    g = grouper(hysteresis=0.08)
    g.update(chunk(100, [obs(matched("s1", "Salah"), 100, 102)]), 70, 100)
    g.update(chunk(101, [obs(matched("s1", "Salah"), 101, 103)]), 70, 101)
    g.update(chunk(102, [obs(unknown(sim=0.70, cand="s1"), 102, 104)]), 70, 102)   # 0.70 >= 0.75 - 0.08
    ev = g.to_event(g.current, 102)
    assert ev.speaker_count == 1 and ev.speakers[0].decision_status == "confirmed"
    assert any("continuation" in n for n in ev.notes)


def test_clearly_different_voice_is_not_absorbed_by_hysteresis():
    g = grouper(hysteresis=0.08)
    g.update(chunk(100, [obs(matched("s1", "Salah"), 100, 102)]), 70, 100)
    g.update(chunk(101, [obs(matched("s1", "Salah"), 101, 103)]), 70, 101)
    g.update(chunk(102, [obs(unknown(sim=0.40, cand="s1", emb=fake_embed(tone(5000, 1))), 102, 104, emb=fake_embed(tone(5000, 1)))]), 70, 102)
    ev = g.to_event(g.current, 102)
    assert ev.speaker_count == 2 and ev.unknown_speaker_count == 1     # real speaker change is visible


def test_unknown_handling_ignore_drops_unknown_only_detections():
    g = EventGrouper(3.0, unknown_handling="ignore")
    assert g.update(chunk(100, [obs(unknown(), 100, 102)]), 70, 100) is None


# ------------------------------------------------------- schema semantics
def test_event_categories_labels_and_legacy_types():
    g = grouper()
    st, _ = g.update(chunk(100, [obs(unknown(), 100, 102, emb=fake_embed(tone(500, 1)))]), 70, 100)
    ev = g.to_event(st, 100)
    assert ev.category == CATEGORY_UNKNOWN and ev.event_type == "GENERAL" and ev.decision_status == "unknown"
    assert ev.rejection_reason == "below_threshold" and ev.overlap_duration_ms is None or ev.overlap_duration_ms == 0

    g2 = grouper()
    for i in range(2):
        g2.update(chunk(100 + i, [obs(matched("s1", "Salah"), 100 + i, 102 + i)]), 70, 100 + i)
    ev = g2.to_event(g2.current, 101)
    assert ev.category == CATEGORY_SINGLE and ev.event_type == "SINGLE_STUDENT" and ev.student == "Salah"

    g3 = grouper()
    seg = [(100.5, 101.5)]
    st, _ = g3.update(chunk(100, [
        SpeakerObservation("S1", matched("s1", "Salah"), [(100, 102)], "masked_mixture", METHOD_DIARIZATION),
        SpeakerObservation("S2", matched("s2", "Sarah"), [(101, 103)], "masked_mixture", METHOD_DIARIZATION)],
        overlap=[(101, 102)], method=METHOD_DIARIZATION, detection="pyannote"), 70, 100)
    ev = g3.to_event(st, 100)
    assert ev.category == CATEGORY_OVERLAP and CATEGORY_MULTIPLE in ev.labels and ev.event_type == "GROUP"
    assert ev.overlap_duration_ms == 1000 and ev.estimated_active_speaker_count == 2
    assert ev.identification_method == METHOD_DIARIZATION and ev.detection_method == "pyannote"
    assert ev.runner_up_similarity == pytest.approx(0.3)


def test_unmeasurable_overlap_is_null_not_zero():
    g = grouper()
    st, _ = g.update(chunk(100, [obs(matched("s1", "Salah"), 100, 102, method=METHOD_WINDOW_SCAN)],
                           overlap=[], method=METHOD_WINDOW_SCAN, measurable=False, approx=True), 70, 100)
    ev = g.to_event(st, 100)
    assert ev.is_overlap and ev.overlap_duration_ms is None


def test_event_json_roundtrip_and_versioned_schema():
    g = grouper()
    st, _ = g.update(chunk(100, [obs(matched("s1", "Salah"), 100, 102)]), 71.23, 100)
    st.latency_ms = 123.4
    payload = json.loads(g.to_event(st, 100).model_dump_json())
    for key in ("schema_version", "event_id", "event_type", "started_at", "ended_at", "duration_ms", "speakers",
                "unknown_speaker_count", "estimated_active_speaker_count", "overlap_duration_ms", "detection_method",
                "identification_method", "similarity", "runner_up_similarity", "decision_status", "rejection_reason",
                "warning_triggered", "processing_latency_ms", "metadata"):
        assert key in payload, key
    assert payload["schema_version"] == SCHEMA_VERSION and payload["processing_latency_ms"] == 123.4
    assert "NOT a probability" in payload["metadata"]["similarity_note"]
    assert payload["started_at"].endswith(("+00:00", "Z")) or "+" in payload["started_at"] or "-" in payload["started_at"][10:]


def test_waveform_overlapping_windows_are_not_double_counted():
    g = grouper()
    wf = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    for t in (100.0, 100.1):             # second window repeats 2 of the 4 bins
        o = obs(matched("s1", "Salah"), t, t + 2)
        o.waveform, o.waveform_t0 = wf, t
        g.update(chunk(t, [o]), 70, t)
    peaks = g.to_event(g.current, 100.1).speakers[0].waveform.peaks
    assert len(peaks) <= 7 and max(peaks) == pytest.approx(0.4)


# ------------------------------------------------------------- cooldown
def test_speaker_cooldown_per_key_and_override():
    cd = SpeakerCooldown(20)
    assert cd.eligible("a", 0)
    cd.mark(["a"], 0)
    assert not cd.eligible("a", 10) and cd.eligible("b", 10)
    assert cd.eligible("a", 21) and cd.eligible("a", 10, seconds=5)
    assert cd.remaining("a", 5) == pytest.approx(15)
