"""Multi-speaker identification, event grouping, cooldowns, persistence and serialization."""
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from api.schemas import EventOut
from config import Settings
from core.OSD import OSDResult
from core.speaker_analysis import MultiSpeakerAnalyzer
from core.vad import VADResult
from database.db_session import init_db, session_scope
from database.models import EventType, InterventionLog, Student
from database.vector_store import VectorStore
from engine.events import AnalysisResult, EventAggregator, SpeakerDetection, SpeakerTimeline
from engine.policy_engine import ClassroomPolicyEngine, State
from tests.helpers import FakeRecognizer, SR, fake_embed, to_pcm, tone

STUDENTS = {"Salah": 440, "Sarah": 880, "Rajae": 1320}
VAD2 = VADResult(True, 1.0, 0.9, segments=((0.0, 2.0),))
NOW = 1_800_000_000.0


def analyzer(**kw):
    kw.setdefault("min_identification_seconds", 0.5)
    s = Settings(**kw)
    return s, MultiSpeakerAnalyzer(s, FakeRecognizer(STUDENTS))


# ----------------------------------------------------------- vector store / matching
def test_vector_store_aggregation_modes_and_min_samples():
    rng = np.random.default_rng(1)
    good, bad = rng.normal(size=192), rng.normal(size=192)
    vs = VectorStore(192)
    vs.add("A", good); vs.add("A", bad)
    vs.add("B", rng.normal(size=192))
    top_max = dict(vs.search(good, None, aggregation="max"))
    top_mean = dict(vs.search(good, None, aggregation="topk_mean", k_samples=2))
    assert top_max["A"] == pytest.approx(1.0, abs=1e-5)
    assert top_mean["A"] < top_max["A"]                       # one bad sample lowers the mean
    assert "A" in dict(vs.search(good, None, aggregation="centroid"))
    assert set(dict(vs.search(good, None, min_samples=2))) == {"A"}  # B has a single sample
    assert vs.counts() == {"A": 2, "B": 1}


def test_embedding_normalisation_and_cosine():
    v = np.array([3.0, 4.0])
    n = VectorStore.normalize(v)
    assert np.linalg.norm(n) == pytest.approx(1.0) and n.dtype == np.float32
    assert VectorStore.cosine_similarity(v, v * 5) == pytest.approx(1.0)


# ------------------------------------------------------------------- analyzer
def test_single_known_speaker_and_unknown_below_threshold():
    s, an = analyzer()
    res = an.analyze(tone(440, 0, 2), VAD2, OSDResult(False, 0.0, "stub"), NOW)
    sp = res.speakers[0]
    assert res.method == "single_embedding" and sp.is_identified and sp.student_name == "Salah"
    assert sp.similarity_score > s.speaker_match_threshold and sp.confidence is None

    res = an.analyze(tone(2500, 0, 2), VAD2, OSDResult(False, 0.0, "stub"), NOW)   # nobody registered at 2500 Hz
    sp = res.speakers[0]
    assert not sp.is_identified and sp.student_name == "Unknown Speaker" and sp.student_id is None
    assert sp.similarity_score is not None and sp.similarity_score < s.speaker_match_threshold  # reported, never promoted


def test_two_simultaneous_speakers_via_diarization_slots():
    """Salah 0-1.2 s, Sarah 0.6-2.0 s: overlap 0.6-1.2 s. Each is identified from its non-overlapped part."""
    audio = tone(440, 0, 1.2) + tone(880, 0.6, 2.0)
    osd = OSDResult(True, 0.3, "pyannote", overlap_seconds=0.6, overlap_segments=((0.6, 1.2),),
                    slot_segments={0: ((0.0, 1.2),), 1: ((0.6, 2.0),)})
    s, an = analyzer()
    res = an.analyze(audio, VAD2, osd, NOW)
    names = sorted(sp.student_name for sp in res.speakers)
    assert res.method == "pyannote_slots" and names == ["Salah", "Sarah"]
    assert all(sp.waveform_kind == "masked_mixture" and sp.reliability == "clean" for sp in res.speakers)
    assert sum(e - st for st, e in res.overlap_intervals) == pytest.approx(0.6)
    by_name = {sp.student_name: sp for sp in res.speakers}
    assert by_name["Salah"].end == pytest.approx(NOW - 2.0 + 1.2) and by_name["Sarah"].start == pytest.approx(NOW - 2.0 + 0.6)


def test_three_speakers_slots_and_max_speakers_limit():
    audio = tone(440, 0, 1.0) + tone(880, 0.8, 1.9) + tone(1320, 1.5, 2.0)
    osd = OSDResult(True, 0.3, "pyannote", overlap_segments=((0.8, 1.0), (1.5, 1.9)),
                    slot_segments={0: ((0.0, 1.0),), 1: ((0.8, 1.9),), 2: ((1.5, 2.0),)})
    s, an = analyzer(min_identification_seconds=0.3)
    res = an.analyze(audio, VAD2, osd, NOW)
    assert sorted(sp.student_name for sp in res.speakers) == ["Rajae", "Salah", "Sarah"]
    s2, an2 = analyzer(min_identification_seconds=0.3, max_simultaneous_speakers=2)
    assert len(an2.analyze(audio, VAD2, osd, NOW).speakers) == 2


def test_mixture_fallback_is_flagged_as_estimate():
    audio = tone(440, 0, 2) + tone(880, 0, 2)
    s, an = analyzer()
    res = an.analyze(audio, VAD2, OSDResult(True, 0.5, "embedding"), NOW)
    assert res.method == "mixture_embedding" and res.overlap_estimated and res.notes
    assert sorted(sp.student_name for sp in res.speakers) == ["Salah", "Sarah"]
    assert all(sp.reliability == "estimated" and sp.waveform_kind == "activity_only" and sp.waveform is None for sp in res.speakers)
    # an unregistered pair: at least two *unknown* speakers, never invented identities
    res = an.analyze(tone(2500, 0, 2) + tone(3100, 0, 2), VAD2, OSDResult(True, 0.5, "embedding"), NOW)
    assert len(res.speakers) == 2 and not any(sp.is_identified for sp in res.speakers)


def test_source_separation_path_gives_separated_waveforms_and_measured_overlap():
    sources = [tone(440, 0, 1.5), tone(880, 0.5, 2.0)]
    sep = SimpleNamespace(separate=lambda a: sources)
    s = Settings(min_identification_seconds=0.5, separation_enabled=True)
    an = MultiSpeakerAnalyzer(s, FakeRecognizer(STUDENTS), separator=sep)
    res = an.analyze(sources[0] + sources[1], VAD2, OSDResult(True, 0.4, "embedding"), NOW)
    assert res.method == "sepformer" and sorted(sp.student_name for sp in res.speakers) == ["Salah", "Sarah"]
    assert all(sp.waveform_kind == "separated" and len(sp.waveform) == s.waveform_points for sp in res.speakers)
    assert sum(e - st for st, e in res.overlap_intervals) == pytest.approx(1.0, abs=0.08)


def test_failing_separator_falls_back_to_next_strategy():
    def boom(_):
        raise RuntimeError("model missing")
    s = Settings(min_identification_seconds=0.5, separation_enabled=True)
    an = MultiSpeakerAnalyzer(s, FakeRecognizer(STUDENTS), separator=SimpleNamespace(separate=boom))
    res = an.analyze(tone(440, 0, 2) + tone(880, 0, 2), VAD2, OSDResult(True, 0.5, "embedding"), NOW)
    assert res.method == "mixture_embedding"


def test_assignment_is_one_to_one():
    s, an = analyzer()
    e = fake_embed(tone(440, 0, 1))
    (a, sa), (b, sb) = an._assign([e, e], [0.75, 0.75])
    assert [a is not None, b is not None].count(True) == 1     # the same student is never assigned twice
    assert (b if a else a) is None


# ---------------------------------------------------------------- event grouping
def det(name, sid, start, end, score=0.9, identified=True):
    return SpeakerDetection("spk", sid, name, start, end, score, is_identified=identified, segments=[(start, end)])


def analysis(speakers, start, end, overlap=(), is_overlap=None, method="pyannote_slots"):
    ov = list(overlap)
    return AnalysisResult(speakers, bool(ov) if is_overlap is None else is_overlap, ov, False, method, start, end)


def test_detections_group_into_one_event_with_stable_speaker_ids_and_union_overlap():
    agg = EventAggregator(merge_gap=3.0)
    e1, created = agg.ingest(analysis([det("Salah", "s", 10, 12), det("Sarah", "r", 10.5, 12)], 10, 12, [(10.5, 11.5)]), 70.0)
    assert created and e1.event_type == EventType.GROUP and [s.speaker_id for s in e1.speakers] == ["spk_1", "spk_2"]
    e2, created = agg.ingest(analysis([det("Sarah", "r", 11, 13, 0.95), det("Rajae", "j", 12.5, 13)], 11, 13, [(11.0, 12.0)]), 72.0)
    assert not created and e2 is e1 and len(agg.events) == 1
    assert [s.student_name for s in e1.speakers] == ["Salah", "Sarah", "Rajae"]
    assert [s.speaker_id for s in e1.speakers] == ["spk_1", "spk_2", "spk_3"]          # ids never change
    assert e1.overlap_duration == pytest.approx(1.5)                                     # union of 10.5-11.5 and 11-12
    assert next(s for s in e1.speakers if s.student_name == "Sarah").similarity_score == 0.95   # best score kept
    assert e1.start == 10 and e1.end == 13 and e1.db_level == 72.0 and "Multiple Speakers" in e1.labels


def test_event_split_after_gap_and_grouping_can_be_disabled():
    agg = EventAggregator(merge_gap=3.0)
    agg.ingest(analysis([det("Salah", "s", 10, 12)], 10, 12, method="single_embedding"), 70)
    agg.ingest(analysis([det("Salah", "s", 20, 22)], 20, 22, method="single_embedding"), 70)
    assert len(agg.events) == 2
    off = EventAggregator(merge_gap=0)
    for t in (10, 11):
        off.ingest(analysis([det("Salah", "s", t, t + 2)], t, t + 2, method="single_embedding"), 70)
    assert len(off.events) == 2


def test_single_event_upgrades_to_group_and_unknown_labels():
    agg = EventAggregator()
    ev, _ = agg.ingest(analysis([det("Salah", "s", 10, 12)], 10, 12, method="single_embedding"), 70)
    assert ev.event_type == EventType.SINGLE_STUDENT and ev.labels == ["Single Speaker"]
    agg.ingest(analysis([det("Salah", "s", 11, 13), det("Unknown Speaker", None, 11, 13, 0.3, False)], 11, 13, [(11, 12)]), 70)
    assert ev.event_type == EventType.GROUP and "Unknown Speaker" in ev.labels and "Overlapping Speech" in ev.labels


def test_event_serialisation_matches_api_schema():
    agg = EventAggregator()
    ev, _ = agg.ingest(analysis([det("Salah", "s", 10, 12), det("Sarah", "r", 10, 12)], 10, 12, [(10, 11.8)]), 70.04)
    data = ev.to_dict()
    EventOut(**data)                                                       # validates against the Pydantic contract
    json.dumps(data)                                                       # JSON-serialisable (WebSocket)
    iso_re = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}$")
    assert all(iso_re.match(data[k]) for k in ("timestamp", "start_time", "end_time"))
    assert data["speaker_count"] == 2 and data["overlap_duration"] == pytest.approx(1.8)
    for key in ("event_id", "event_type", "speakers", "is_overlap", "detection_method", "warning_triggered"):
        assert key in data
    assert data["speakers"][0]["similarity_score"] == 0.9 and data["speakers"][0]["audio_segment_reference"] is None
    assert "waveform" not in ev.to_dict(include_waveform=False)["speakers"][0]


def test_timeline_tracks_prune_and_overlap():
    agg, tl = EventAggregator(), SpeakerTimeline(max_age=30)
    a = analysis([det("Salah", "s", 100, 102), det("Sarah", "r", 101, 103)], 100, 103, [(101, 102)])
    ev, _ = agg.ingest(a, 70)
    tl.add(ev, a)
    win = tl.window(now=103, seconds=30)
    assert sorted(t["label"] for t in win["tracks"]) == ["Salah", "Sarah"] and win["overlaps"] == [[101.0, 102.0]]
    late = tl.window(now=200, seconds=30)                                  # everything is older than the window
    assert late["tracks"] == [] and late["overlaps"] == []


# ------------------------------------------------------- policy engine end-to-end
class FakeTTS:
    is_busy = False
    language = "en"

    def __init__(self):
        self.spoken = []

    def render_message(self, event_type, name=None):
        return f"{event_type.value}:{name or ''}"

    def render_named_group(self, names):
        return "NAMED:" + "+".join(names)

    def speak_nowait(self, text):
        self.spoken.append(text)


def make_engine(osd_result, students=STUDENTS, **kw):
    init_db()
    settings = Settings(min_identification_seconds=0.5, **kw)
    rec = FakeRecognizer(students)
    with session_scope() as session:                                       # real rows, as in production
        for stu in rec.students.values():
            if session.get(Student, stu.id) is None:
                session.add(Student(id=stu.id, name=stu.name))
    vad = SimpleNamespace(analyze=lambda a: VAD2)
    osd = SimpleNamespace(detect=lambda a: osd_result, backend="stub")
    tts = FakeTTS()
    eng = ClassroomPolicyEngine(settings, vad, osd, rec, tts)
    return eng, tts


SLOTS = OSDResult(True, 0.3, "pyannote", overlap_segments=((0.6, 1.2),),
                  slot_segments={0: ((0.0, 1.2),), 1: ((0.6, 2.0),)})
PAIR_AUDIO = to_pcm(tone(440, 0, 1.2) + tone(880, 0.6, 2.0))


def test_engine_names_both_speakers_and_persists_one_event():
    eng, tts = make_engine(SLOTS)
    assert eng.process_audio_chunk(PAIR_AUDIO, end_time=NOW) == State.GROUP_WARNING
    assert tts.spoken == ["NAMED:Salah+Sarah"] or tts.spoken == ["NAMED:Sarah+Salah"]
    ev = eng.snapshot()["last_event"]
    assert ev["speaker_count"] == 2 and ev["overlap_duration"] == pytest.approx(0.6) and ev["warning_status"] == "spoken"
    # a second window of the same conversation extends the event: no new alert, no new TTS (cooldown)
    eng.process_audio_chunk(PAIR_AUDIO, end_time=NOW + 1.0)
    assert len(eng.events) == 1 and len(tts.spoken) == 1
    with session_scope() as s:
        rows = s.query(InterventionLog).filter_by(event_uid=ev["event_id"]).all()
        assert len(rows) == 1                                              # updated in place, not one row per window
        assert rows[0].speaker_count == 2 and rows[0].is_overlap and rows[0].detection_method == "pyannote_slots"
        assert {sp["student_name"] for sp in json.loads(rows[0].speakers_json)} == {"Salah", "Sarah"}
        assert "waveform" not in json.loads(rows[0].speakers_json)[0]      # no waveform / raw audio persisted


def test_third_speaker_joins_existing_event():
    eng, tts = make_engine(SLOTS)
    eng.process_audio_chunk(PAIR_AUDIO, end_time=NOW)
    third = OSDResult(True, 0.3, "pyannote", overlap_segments=((0.8, 1.0),),
                      slot_segments={0: ((0.0, 1.0),), 1: ((0.8, 2.0),)})
    eng.osd.detect = lambda a: third
    eng.process_audio_chunk(to_pcm(tone(880, 0, 1.0) + tone(1320, 0.8, 2.0)), end_time=NOW + 1.0)
    ev = eng.snapshot()["last_event"]
    assert ev["speaker_count"] == 3 and {s["student_name"] for s in ev["speakers"]} == {"Salah", "Sarah", "Rajae"}


def test_per_speaker_cooldown_and_global_cooldown():
    single = OSDResult(False, 0.0, "stub")
    eng, tts = make_engine(single, event_merge_gap_seconds=0.0)
    eng.cooldown_seconds = 15
    eng.process_audio_chunk(to_pcm(tone(440, 0, 2)), end_time=NOW)
    assert tts.spoken == ["SINGLE_STUDENT:Salah"]
    eng.process_audio_chunk(to_pcm(tone(880, 0, 2)), end_time=NOW + 1)          # different student, global cooldown active
    assert eng.snapshot()["last_event"]["warning_status"] == "suppressed_cooldown"
    eng.last_trigger_time = 0.0                                                    # global cooldown over
    eng.process_audio_chunk(to_pcm(tone(440, 0, 2)), end_time=NOW + 2)          # same student again: per-speaker cooldown
    assert eng.snapshot()["last_event"]["warning_status"] == "suppressed_speaker_cooldown"
    eng.process_audio_chunk(to_pcm(tone(880, 0, 2)), end_time=NOW + 3)          # other student is warned
    assert tts.spoken[-1] == "SINGLE_STUDENT:Sarah"


def test_unknown_speaker_policies():
    single = OSDResult(False, 0.0, "stub")
    stranger = to_pcm(tone(2500, 0, 2))
    eng, tts = make_engine(single, unknown_speaker_policy="general_warning")
    assert eng.process_audio_chunk(stranger, end_time=NOW) == State.GENERAL_WARNING and len(tts.spoken) == 1
    eng, tts = make_engine(single, unknown_speaker_policy="log_only")
    assert eng.process_audio_chunk(stranger, end_time=NOW) == State.GENERAL_WARNING
    assert not tts.spoken and eng.snapshot()["last_event"]["warning_status"] == "not_warned_policy"
    eng, tts = make_engine(single, unknown_speaker_policy="ignore")
    assert eng.process_audio_chunk(stranger, end_time=NOW) == State.NOISE_ONLY and not eng.events


def test_snapshot_exposes_timeline_and_per_speaker_view():
    eng, _ = make_engine(SLOTS)
    eng.process_audio_chunk(PAIR_AUDIO, end_time=NOW)
    snap = eng.snapshot()
    assert {t["label"] for t in snap["timeline"]["tracks"]} == {"Salah", "Sarah"}
    view = snap["latest_analysis"]
    assert view["method"] == "pyannote_slots" and len(view["mixture"]) == eng.settings.waveform_points
    assert {s["waveform_kind"] for s in view["speakers"]} == {"masked_mixture"}
    json.dumps(snap)                                                       # whole snapshot is JSON-serialisable


# ------------------------------------------------------------ schema migration
def test_schema_upgrade_adds_columns_and_keeps_rows(tmp_path):
    from sqlalchemy import create_engine, inspect, text
    from database.migrations import upgrade_schema

    eng = create_engine(f"sqlite:///{tmp_path/'old.db'}")
    with eng.begin() as c:
        c.execute(text("CREATE TABLE students (id VARCHAR(36) PRIMARY KEY, name VARCHAR(120) NOT NULL, age INTEGER, student_code VARCHAR(50), created_at DATETIME)"))
        c.execute(text("CREATE TABLE voice_profiles (id INTEGER PRIMARY KEY, student_id VARCHAR(36) NOT NULL, embedding_blob BLOB NOT NULL, audio_sample_path VARCHAR(500), created_at DATETIME)"))
        c.execute(text("CREATE TABLE intervention_logs (id INTEGER PRIMARY KEY, timestamp DATETIME, event_type VARCHAR(20) NOT NULL, student_id VARCHAR(36), db_level FLOAT NOT NULL, message_spoken TEXT, match_score FLOAT, was_spoken BOOLEAN)"))
        c.execute(text("INSERT INTO students VALUES ('1','Old Student',12,'X1',NULL)"))
        c.execute(text("INSERT INTO intervention_logs (id,event_type,db_level,message_spoken,was_spoken) VALUES (1,'GROUP',70.0,'hi',1)"))
    added = upgrade_schema(eng)
    assert "students.notes" in added and "intervention_logs.speakers_json" in added and "voice_profiles.duration_seconds" in added
    cols = {c["name"] for c in inspect(eng).get_columns("intervention_logs")}
    assert {"event_uid", "start_time", "overlap_duration", "speaker_count", "detection_method"} <= cols
    with eng.connect() as c:
        assert c.execute(text("SELECT name FROM students")).scalar() == "Old Student"
        assert c.execute(text("SELECT speaker_count, is_overlap FROM intervention_logs WHERE id=1")).one() == (1, 0)
    assert upgrade_schema(eng) == []                                       # idempotent
