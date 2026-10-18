"""DB migration, audio validation, enrollment quality, metrics, tracker timers, UI helpers, monitor lifecycle."""
import sqlite3
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from config import Settings
from core import segments as seg
from core.audio_utils import clip_ratio, validate_pcm16
from core.audio_stream import AudioStream
from core.metrics import StageTimer
from core.quality import assess_sample
from evaluation import metrics as M
from tests.helpers import SR, tone


# ---------------------------------------------------------------- migration
def test_old_database_is_migrated_additively(tmp_path):
    from sqlalchemy import create_engine, inspect

    from database.db_session import _add_missing_columns
    from database.models import Base

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE students (id VARCHAR(36) PRIMARY KEY, name VARCHAR(120) NOT NULL, age INTEGER,
                               student_code VARCHAR(50), created_at DATETIME);
        CREATE TABLE voice_profiles (id INTEGER PRIMARY KEY AUTOINCREMENT, student_id VARCHAR(36) NOT NULL,
                                     embedding_blob BLOB NOT NULL, audio_sample_path VARCHAR(500), created_at DATETIME);
        CREATE TABLE intervention_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME,
                                        event_type VARCHAR(20) NOT NULL, student_id VARCHAR(36), db_level FLOAT NOT NULL,
                                        message_spoken TEXT, match_score FLOAT, was_spoken BOOLEAN);
        INSERT INTO students VALUES ('s1', 'Old Student', 12, 'X1', '2025-01-01 10:00:00');
        INSERT INTO intervention_logs (event_type, db_level, message_spoken, was_spoken) VALUES ('GROUP', 70, 'hi', 1);
    """)
    con.commit()
    con.close()
    engine = create_engine(f"sqlite:///{path}")
    added = _add_missing_columns(engine)
    assert "students.notes" in added and "intervention_logs.event_id" in added and "voice_profiles.quality_weight" in added
    cols = {c["name"] for c in inspect(engine).get_columns("intervention_logs")}
    assert {"event_id", "category", "details_json", "overlap_duration"} <= cols
    assert _add_missing_columns(engine) == []                                         # idempotent
    with engine.connect() as c:
        row = c.exec_driver_sql("SELECT name, notes FROM students").one()
        old_log = c.exec_driver_sql("SELECT event_type, event_id, speaker_count FROM intervention_logs").one()
    assert row == ("Old Student", None) and old_log == ("GROUP", None, None)          # old rows stay valid
    Base.metadata.create_all(engine)                                                   # and the ORM can use it


# -------------------------------------------------------------- audio guards
def test_pcm_validation_and_clipping():
    assert validate_pcm16(b"") == "empty frame"
    assert "odd" in validate_pcm16(b"abc")
    assert validate_pcm16(b"abcd", expected_samples=3) is not None and validate_pcm16(b"abcd") is None
    assert clip_ratio(np.array([1.0, 0.5, -1.0, 0.0])) == 0.5


def test_resample_helper_keeps_expected_length():
    x = (np.sin(np.arange(4800) / 10) * 1000).astype(np.int16)
    assert len(AudioStream.resample_int16(x, 48000, 16000)) == 1600


def test_stream_counts_dropped_windows_and_stamps_capture_time():
    st = AudioStream(sample_rate=16000, chunk_seconds=2.0, hop_seconds=1.0)
    st._closed = False
    st._ingest(np.ones(16000 * 5, dtype=np.int16))          # consumer was 4 s behind
    before = time.time()
    assert st.read_chunk(timeout=0.05) is not None
    assert st.dropped_windows >= 3 and st.last_chunk_time >= before
    assert len(st._buffer) <= st._max_samples                  # buffer stays bounded


# ------------------------------------------------------------------ quality
def speechlike(freq=500, seconds=2.5, amp=0.3):
    out = np.zeros(int(seconds * SR), dtype=np.float32)
    burst = tone(freq, 0.45, amp)
    for k in range(int(seconds / 0.7)):
        a = int(k * 0.7 * SR)
        out[a: a + len(burst)] = burst[: len(out) - a]
    return out


def test_quality_accepts_good_and_rejects_bad_samples():
    s = Settings()
    good = assess_sample(speechlike(), SR, s)
    assert good.ok and 0.3 <= good.weight <= 1.0
    assert not assess_sample(np.zeros(SR * 3), SR, s).ok
    assert any("clipped" in r for r in assess_sample(np.clip(speechlike() * 30, -1, 1), SR, s).reasons)
    assert any("pauses" in r for r in assess_sample(tone(500, 3.0), SR, s).reasons)
    assert not assess_sample(np.array([], dtype=np.float32), SR, s).ok
    quiet = assess_sample(speechlike(amp=0.0004), SR, s)
    assert not quiet.ok and any("quiet" in r for r in quiet.reasons)
    assert assess_sample(speechlike(amp=0.02), SR, s).weight < good.weight         # weaker sample, smaller weight


# ----------------------------------------------------------------- segments
def test_interval_helpers():
    assert seg.merge_intervals([(0, 1), (1.2, 2), (5, 6)], gap=0.3) == [(0, 2), (5, 6)]
    assert seg.total_duration([(0, 1), (0.5, 2)]) == 2.0
    assert seg.overlap_between_streams([[(0, 2)], [(1, 3)], [(2.5, 4)]]) == [(1, 2), (2.5, 3)]
    assert seg.max_concurrency([[(0, 2)], [(1, 3)], [(1.5, 2.5)]]) == 3
    assert seg.max_concurrency([[(0, 1)], [(1, 2)]]) == 1                           # touching is not overlapping
    assert seg.mask_to_intervals(np.array([0, 1, 1, 0, 1], bool), 0.5) == [(0.5, 1.5), (2.0, 2.5)]
    assert seg.pad_intervals([(1, 2)], 0.5, 0, 2.2) == [(0.5, 2.2)]
    env = seg.amplitude_envelope(np.ones(16000, dtype=np.float32) * 0.5, 20, 16000)
    assert len(env) == 20 and env.max() == pytest.approx(0.5)


# ------------------------------------------------------------------ metrics
def test_stage_timer_percentiles_and_counters():
    t = StageTimer(maxlen=100)
    for v in range(1, 101):
        t.record("vad", v)
    t.count("errors", 2)
    s = t.summary()
    assert s["vad"]["median_ms"] == pytest.approx(50.5) and s["vad"]["p95_ms"] == pytest.approx(95.05, abs=0.1)
    assert s["_counters"]["errors"] == 2.0 and t.last("vad") == 100


def test_identification_metrics_far_frr_unknown_rejection():
    trials = [("a", "a"), ("a", "a"), ("a", None), ("b", "a"), (None, None), (None, "a"), (None, None), (None, None)]
    m = M.identification_metrics(trials)
    assert m["accuracy"] == pytest.approx(0.5) and m["frr"] == pytest.approx(0.25)
    assert m["misidentification_rate"] == pytest.approx(0.25)
    assert m["far"] == pytest.approx(0.25) and m["unknown_rejection_rate"] == pytest.approx(0.75)
    assert M.identification_metrics([("a", "a")])["far"] is None                      # no impostor trial -> None, not 0


def test_threshold_sweep_finds_equal_error_point():
    out = M.sweep_thresholds([0.9, 0.8, 0.85, 0.7], [0.2, 0.3, 0.75, 0.1])
    assert out["eer"] is not None and 0.7 <= out["eer_threshold"] <= 0.9
    assert M.sweep_thresholds([], [0.1])["eer"] is None


def test_overlap_prf_der_attribution_duplication_latency():
    prf = M.overlap_prf([(1, 3)], [(2, 4)], total_seconds=5)
    assert prf["precision"] == pytest.approx(0.5, abs=0.02) and prf["recall"] == pytest.approx(0.5, abs=0.02)
    assert M.overlap_prf([], [], 5)["f1"] is None
    ref = {"a": [(0, 2)], "b": [(1, 3)]}
    assert M.diarization_error_rate(ref, ref, 4) == pytest.approx(0.0)
    der = M.diarization_error_rate(ref, {"a": [(0, 2)], "x": [(1, 3)]}, 4)             # b confused with unknown x
    assert der == pytest.approx(0.5, abs=0.02)
    assert M.diarization_error_rate({}, {}, 4) is None
    att = M.attribution_accuracy([["a", "b"], ["a"]], [["a"], ["a", "c"]])
    assert att["precision"] == pytest.approx(2 / 3) and att["recall"] == pytest.approx(2 / 3) and att["exact_set_accuracy"] == 0
    events = [{"start": 0, "end": 5, "speakers": ["a"]}, {"start": 6, "end": 8, "speakers": ["a"]},
              {"start": 30, "end": 31, "speakers": ["a"]}]
    assert M.event_duplication_rate(events) == pytest.approx(1 / 3)
    lat = M.latency_stats([100, 200, 300, 1000], [50, 60], [1.0, 1.0])
    assert lat["median_ms"] == 250 and lat["real_time_factor"] == pytest.approx(0.055)


# ---------------------------------------------------------------- UI helpers
def test_ui_alert_view_models_single_multi_unknown():
    from ui import components as ui

    def spk(i, name, ident, sim, status="confirmed"):
        return {"speaker_id": f"S{i}", "student_id": name.lower() if ident else None, "student_name": name,
                "start_time": "2026-10-17T13:00:00+00:00", "end_time": "2026-10-17T13:00:02+00:00",
                "segments": [{"start": "2026-10-17T13:00:00+00:00", "end": "2026-10-17T13:00:02+00:00"}],
                "similarity_score": sim, "runner_up_similarity": 0.2, "is_identified": ident,
                "decision_status": status if ident or status == "tentative" else "unknown",
                "rejection_reason": None if ident else "below_threshold", "signal_type": "masked_mixture"}

    base = {"event_id": "e1", "start_time": "2026-10-17T13:00:00+00:00", "end_time": "2026-10-17T13:00:03+00:00",
            "labels": [], "status": "completed", "notes": [], "metadata": {}, "detection_method": "pyannote",
            "identification_method": "pyannote_slots", "decision_status": "confirmed", "warning_triggered": True,
            "processing_latency_ms": 120.0, "is_overlap": False, "overlap_duration_ms": None}
    single = {**base, "speakers": [spk(1, "Salah", True, 0.86)], "speaker_count": 1}
    vm = ui.alert_view_model(single)
    assert vm["title"] == "Single speaker detected" and "similarity: 0.86" in vm["speaker_lines"][0]
    multi = {**base, "speakers": [spk(1, "Salah", True, 0.86), spk(2, "Sarah", True, 0.81)], "speaker_count": 2,
             "is_overlap": True, "overlap_duration_ms": 1800, "labels": ["Multiple Speakers", "Overlapping Speech"]}
    vm = ui.alert_view_model(multi)
    assert vm["title"].startswith("Multiple speakers detected") and len(vm["speaker_lines"]) == 2
    assert vm["overlap_text"] == "Overlap duration: 1.8 s" and vm["warning_text"] == "Warning spoken"
    assert len(vm["speaker_rows"]) == 2 and vm["speaker_rows"][0]["Similarity (cosine)"] == "0.86"
    unmeasured = {**multi, "overlap_duration_ms": None}
    assert "not measurable" in ui.alert_view_model(unmeasured)["overlap_text"]
    unknown = {**base, "speakers": [spk(1, "Unknown Speaker", False, 0.41)], "speaker_count": 1,
               "warning_triggered": False}
    vm = ui.alert_view_model(unknown)
    assert vm["title"] == "Unknown speaker detected" and "not identified" in vm["speaker_lines"][0]
    assert "0.41" in vm["speaker_lines"][0] and vm["warning_text"].startswith("No warning")
    tentative = {**base, "speakers": [spk(1, "Salah", True, 0.80) | {"is_identified": False, "decision_status": "tentative"}],
                 "speaker_count": 1}
    assert "tentative" in ui.alert_view_model(tentative)["speaker_lines"][0]
    assert "NOT a probability" in ui.SIMILARITY_HELP

    rows = ui.timeline_rows([multi])
    assert {r["track"] for r in rows} == {"Salah", "Sarah"}
    assert ui.speaker_color(multi["speakers"][0]) == ui.speaker_color(single["speakers"][0])      # consistent colour
    assert ui.speaker_color(multi["speakers"][0]) != ui.speaker_color(multi["speakers"][1])
    assert [t[0] for t in ui.track_order([multi, unknown])][-1].startswith("Unknown Speaker")


def test_ui_figures_render_when_plotly_is_installed():
    pytest.importorskip("plotly")
    from ui import components as ui

    ev = {"event_id": "e", "start_time": "2026-10-17T13:00:00+00:00", "end_time": "2026-10-17T13:00:03+00:00",
          "overlap_segments": [{"start": "2026-10-17T13:00:01+00:00", "end": "2026-10-17T13:00:02+00:00"}],
          "speakers": [{"speaker_id": "S1", "student_id": "a", "student_name": "A", "is_identified": True,
                        "decision_status": "confirmed", "start_time": "2026-10-17T13:00:00+00:00",
                        "end_time": "2026-10-17T13:00:02+00:00", "segments": [], "signal_type": "separated",
                        "waveform": {"t0": "2026-10-17T13:00:00+00:00", "dt_seconds": 0.05, "peaks": [0.1, 0.3, 0.2]}}]}
    assert len(ui.build_timeline_figure([ev]).data) == 1
    assert ui.build_speaker_waveforms_figure([ev]) is not None
    assert ui.build_speaker_waveforms_figure([{**ev, "speakers": [{**ev["speakers"][0], "waveform": None}]}]) is None
    fig = ui.build_mic_waveform_figure({"t0_epoch": 1.0, "dt_seconds": 0.05, "peaks": [0.1, 0.2]})
    assert len(fig.data) == 2
    assert len(ui.build_timeline_figure([]).data) == 0


# ---------------------------------------------------------- monitor lifecycle
def test_monitor_start_stop_restart_and_concurrent_start(monkeypatch):
    from engine import runner as runner_mod

    started = []

    class FakeStream:
        last_chunk_time = 0.0
        last_audio_time = time.time()
        dropped_windows = 0

        def __init__(self, **kw):
            started.append(self)
            self._stop = threading.Event()

        def start(self):
            pass

        def stop(self):
            self._stop.set()

        def read_chunk(self, timeout=1.0):
            self._stop.wait(0.01)
            return None

        def recent_db(self, s=0.5):
            return 0.0

    monkeypatch.setattr(runner_mod, "AudioStream", FakeStream)
    svc = runner_mod.MonitorService(Settings())
    svc.engine = SimpleNamespace(process_audio_chunk=lambda *a, **k: None, snapshot=lambda: {},
                                 threshold_dB=68.0, cooldown_seconds=20.0, match_threshold=0.75)
    svc._models_loaded = True
    threads = [threading.Thread(target=svc.start) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert svc.running and len(started) == 1                 # five concurrent starts -> one stream
    svc.stop()
    assert not svc.running
    svc.start()                                               # restart works
    assert svc.running and len(started) == 2
    svc.stop()
