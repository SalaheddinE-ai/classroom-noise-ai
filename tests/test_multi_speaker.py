"""Matcher + analyzer: thresholds, ambiguity, unknown handling, overlap tiers (synthetic audio, no models)."""
from types import SimpleNamespace

import numpy as np
import pytest

from config import Settings
from core.multi_speaker import (
    METHOD_DIARIZATION, METHOD_SEPARATION, METHOD_SINGLE, METHOD_WINDOW_SCAN,
    MultiSpeakerAnalyzer, SpeakerMatcher,
)
from core.OSD import OSDResult
from core.vad import VADResult
from database.vector_store import VectorStore
from tests.helpers import FREQS, SR, fake_embed, make_store, tone

T0 = 1000.0


def analyzer(settings=None, store=None, separator=None):
    s = settings or Settings(min_identify_seconds=0.5, speaker_match_threshold=0.75)
    return MultiSpeakerAnalyzer(s, store or make_store(), fake_embed, separator), s


# ------------------------------------------------------------------ matcher
def test_embedding_normalisation_and_cosine():
    v = np.array([3.0, 4.0], dtype=np.float32)
    assert np.linalg.norm(VectorStore.normalize(v)) == pytest.approx(1.0)
    assert VectorStore.cosine_similarity(v, 2 * v) == pytest.approx(1.0)
    assert VectorStore.cosine_similarity(v, np.array([-4.0, 3.0])) == pytest.approx(0.0)


def test_match_correct_student_and_diagnostics():
    m = SpeakerMatcher(make_store(), Settings()).match(fake_embed(tone(FREQS["sarah"], 1.0)), 0.75)
    assert m.is_identified and m.student_name == "Sarah" and m.reason == "matched"
    assert m.similarity > 0.9 and m.runner_up_id in {"salah", "rajae"} and m.runner_up_similarity < m.similarity


def test_below_threshold_is_rejected_not_forced():
    other = fake_embed(tone(5000.0, 1.0))  # nobody enrolled sounds like this
    m = SpeakerMatcher(make_store(), Settings()).match(other, 0.75)
    assert not m.is_identified and m.student_id is None and m.reason == "below_threshold"
    assert m.similarity is not None and m.candidate_id is not None  # diagnostics kept, identity not given


def test_threshold_is_configurable():
    mix = fake_embed(tone(FREQS["salah"], 1.0) + 0.6 * tone(FREQS["sarah"], 1.0))
    matcher = SpeakerMatcher(make_store(), Settings(min_match_margin=0.0))
    assert matcher.match(mix, 0.99).student_id is None
    assert matcher.match(mix, 0.5).student_id == "salah"


def test_ambiguity_margin_rejects_close_candidates():
    store = VectorStore(192)
    base = fake_embed(tone(800.0, 1.0))
    twin = fake_embed(tone(800.0, 1.0) + 0.02 * tone(900.0, 1.0))
    for sid, emb in (("a", base), ("b", twin)):
        store.set_name(sid, sid.upper())
        store.add(sid, emb)
    m = SpeakerMatcher(store, Settings(min_match_margin=0.05)).match(base, 0.5)
    assert not m.is_identified and m.reason == "ambiguous"
    assert SpeakerMatcher(store, Settings(min_match_margin=0.0)).match(base, 0.5).is_identified


def test_no_students_and_no_embedding():
    matcher = SpeakerMatcher(VectorStore(192), Settings())
    assert matcher.match(fake_embed(tone(500, 1)), 0.5).reason == "no_students"
    assert matcher.match(None, 0.5).reason == "no_embedding"


def test_min_voice_samples_makes_student_not_ready():
    store = make_store(samples=1)
    matcher = SpeakerMatcher(store, Settings(min_voice_samples=2))
    assert matcher.match(fake_embed(tone(FREQS["salah"], 1.0)), 0.5).reason == "no_students"


def test_student_cannot_be_assigned_to_two_streams():
    emb = fake_embed(tone(FREQS["salah"], 1.0))
    res = SpeakerMatcher(make_store(), Settings(min_match_margin=0.0)).assign([emb, emb * 0.999], 0.75)
    assert sum(r.is_identified for r in res) == 1
    assert {r.reason for r in res} == {"matched", "taken"}


def test_profile_centroid_uses_quality_weights():
    store = VectorStore(192)
    good, bad = fake_embed(tone(500.0, 1.0)), fake_embed(tone(2500.0, 1.0))
    store.add("s", good, weight=1.0)
    store.add("s", bad, weight=0.05)
    heavy = VectorStore(192)
    heavy.add("s", good, weight=1.0)
    heavy.add("s", bad, weight=1.0)
    assert store.score_students(good)["s"] > heavy.score_students(good)["s"]  # one poor sample cannot dominate


# ----------------------------------------------------------------- analyzer
def test_single_stream_identifies_and_unknown_when_uncertain():
    az, _ = analyzer()
    ok = az.analyze(tone(FREQS["rajae"], 2.0), T0)
    assert ok.method == METHOD_SINGLE and not ok.is_overlap
    assert ok.speakers[0].match.student_name == "Rajae"
    unk = az.analyze(tone(6000.0, 2.0), T0)
    assert not unk.speakers[0].is_identified and unk.speakers[0].match.reason == "below_threshold"


def test_too_short_speech_is_not_identified():
    az, _ = analyzer(Settings(min_identify_seconds=1.5))
    vad = VADResult(True, 0.3, 0.9, frame_probs=tuple([1.0] * 20 + [0.0] * 44))  # 0.64 s of speech
    res = az.analyze(tone(FREQS["salah"], 2.0), T0, vad_result=vad)
    assert res.speakers[0].match.reason == "no_embedding" and not res.speakers[0].is_identified


def test_vad_segments_merge_pauses_and_drop_blips():
    az, _ = analyzer(Settings(vad_min_speech_ms=100, vad_merge_gap_ms=300, vad_pad_ms=0))
    probs = [1.0] * 10 + [0.0] * 4 + [1.0] * 10 + [0.0] * 20 + [1.0] * 2 + [0.0] * 18   # blip of 2 frames (64 ms)
    res = az.analyze(tone(FREQS["salah"], 2.0), T0, vad_result=VADResult(True, 0.4, 1, frame_probs=tuple(probs)))
    segs = res.speakers[0].segments
    assert len(segs) == 1 and segs[0][1] - segs[0][0] == pytest.approx(24 * 512 / SR, abs=1e-6)


def osd_activity(frame=0.1, total=2.0):
    n = int(total / frame)
    act = np.zeros((n, 3), dtype=bool)
    act[0:15, 0] = True       # A: 0.0-1.5 s
    act[10:20, 1] = True      # B: 1.0-2.0 s  -> overlap 1.0-1.5 s
    return OSDResult(True, 0.25, "pyannote", activity=act, frame_seconds=frame)


def mix_ab():
    return tone(FREQS["salah"], 1.5, start=0.0, total=2.0) + tone(FREQS["sarah"], 1.0, start=1.0, total=2.0)


def test_pyannote_slots_names_both_speakers_with_masked_mixture_labels():
    az, _ = analyzer(Settings(min_identify_seconds=0.45, speaker_match_threshold=0.7, diarization_backend="auto"))
    res = az.analyze(mix_ab(), T0, osd_result=osd_activity())
    assert res.method == METHOD_DIARIZATION and res.is_overlap
    assert {s.match.student_name for s in res.speakers} == {"Salah", "Sarah"}
    assert all(s.signal_type == "masked_mixture" for s in res.speakers)       # never "separated"
    assert res.overlap_duration == pytest.approx(0.5, abs=0.11)
    assert res.detection_method == "pyannote"


def test_pyannote_slots_unknown_when_no_clean_evidence_and_penalty_applies():
    az, _ = analyzer(Settings(min_identify_seconds=0.9, speaker_match_threshold=0.7))
    res = az.analyze(mix_ab(), T0, osd_result=osd_activity())
    assert any(s.quality == "mixed" for s in res.speakers)   # B never spoke alone for 0.9 s
    assert any("mixed audio" in n for n in res.notes)


class StubSeparator:
    available = True

    def __init__(self, sources):
        self.sources = sources

    def separate(self, audio, sr):
        return self.sources


def test_sepformer_mode_matches_each_source_and_marks_real_audio():
    a = tone(FREQS["salah"], 1.5, start=0.0, total=2.0)
    b = tone(FREQS["sarah"], 1.0, start=1.0, total=2.0)
    s = Settings(min_identify_seconds=0.45, separation_enabled=True, speaker_match_threshold=0.75)
    az, _ = analyzer(s, separator=StubSeparator([a, b, np.zeros_like(a)]))   # silent 3rd output is ignored
    res = az.analyze(a + b, T0, osd_result=OSDResult(True, 0.3, "embedding"))
    assert res.method == METHOD_SEPARATION and len(res.speakers) == 2
    assert {x.match.student_name for x in res.speakers} == {"Salah", "Sarah"}
    assert all(x.signal_type == "separated" and x.waveform is not None for x in res.speakers)
    assert res.overlap_duration == pytest.approx(0.5, abs=0.1)


def test_sepformer_single_active_source_is_not_overlap():
    a = tone(FREQS["salah"], 2.0)
    s = Settings(min_identify_seconds=0.45, separation_enabled=True)
    az, _ = analyzer(s, separator=StubSeparator([a, np.zeros_like(a)]))
    res = az.analyze(a, T0, osd_result=OSDResult(True, 0.3, "embedding"))
    assert len(res.speakers) == 1 and res.overlap_segments == [] and "single active source" in " ".join(res.notes)


def test_separator_failure_falls_back_without_crashing():
    class Broken(StubSeparator):
        def separate(self, audio, sr):
            raise RuntimeError("boom")

    s = Settings(min_identify_seconds=0.45, separation_enabled=True)
    az, _ = analyzer(s, separator=Broken([]))
    res = az.analyze(mix_ab(), T0, osd_result=osd_activity())
    assert res.method == METHOD_DIARIZATION and any("sepformer failed" in n for n in res.notes)


def window_osd(embs_tones, flags):
    embs = np.stack([fake_embed(tone(f, 1.0)) if f else fake_embed(tone(FREQS["salah"], 1.0) + tone(FREQS["sarah"], 1.0))
                     for f in embs_tones])
    starts = tuple(0.5 * i for i in range(len(embs)))
    return OSDResult(True, 0.5, "embedding", window_starts=starts, window_seconds=1.0,
                     window_embeddings=embs, window_flags=tuple(flags))


def test_mixture_embedding_is_approximate_and_demands_extra_similarity():
    s = Settings(speaker_match_threshold=0.75, mixture_extra_threshold=0.10, min_identify_seconds=0.5)
    az, _ = analyzer(s)
    # window 0: pure Salah, window 1: mixture (ambiguous), window 2: pure Sarah
    res = az.analyze(mix_ab(), T0, osd_result=window_osd([FREQS["salah"], None, FREQS["sarah"]], [False, True, False]))
    assert res.method == METHOD_WINDOW_SCAN and res.overlap_approximate
    assert {x.match.student_name for x in res.speakers if x.is_identified} == {"Salah", "Sarah"}
    assert all(x.signal_type == "activity_only" for x in res.speakers)
    assert any("approximate" in n for n in res.notes)
    # a mixture-only evidence set names nobody and says so
    res2 = az.analyze(mix_ab(), T0, osd_result=window_osd([None, None, None], [True, True, True]))
    assert not any(x.is_identified for x in res2.speakers)
    assert res2.speakers[0].match.reason == "unresolved_overlap"


def test_mixture_embedding_overlap_not_measurable_when_unlocalised():
    az, _ = analyzer()
    res = az.analyze(mix_ab(), T0, osd_result=window_osd([FREQS["salah"], FREQS["sarah"]], [False, False]))
    assert res.overlap_segments == [] and res.overlap_measurable is False


def test_overlap_mode_selection_and_explicit_fallback_note():
    s = Settings(overlap_mode="sepformer", min_identify_seconds=0.45)
    az, _ = analyzer(s)    # no separator configured
    res = az.analyze(mix_ab(), T0, osd_result=osd_activity())
    assert res.method == METHOD_DIARIZATION and any("unavailable" in n for n in res.notes)
    s2 = Settings(overlap_mode="mixture_embedding", min_identify_seconds=0.45)
    az2, _ = analyzer(s2)
    assert az2.analyze(mix_ab(), T0, osd_result=osd_activity()).method == METHOD_WINDOW_SCAN
    assert az.capabilities("pyannote")["overlap_method"] == METHOD_DIARIZATION
    assert az.capabilities("embedding")["isolated_audio_per_speaker"] is False


def test_max_simultaneous_speakers_is_respected():
    s = Settings(min_identify_seconds=0.45, max_simultaneous_speakers=1)
    az, _ = analyzer(s)
    res = az.analyze(mix_ab(), T0, osd_result=osd_activity())
    assert len(res.speakers) == 1
