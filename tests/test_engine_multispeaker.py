"""Policy engine with the multi-speaker analyzer: grouping, cooldowns, persistence, API payloads."""
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from config import Settings
from core.multi_speaker import MultiSpeakerAnalyzer
from core.OSD import OSDResult
from core.tts_engine import TTSEngine
from core.vad import VADResult
from database.db_session import init_db, session_scope
from database.models import InterventionLog
from engine.policy_engine import ClassroomPolicyEngine, State
from tests.helpers import FREQS, fake_embed, make_store, pcm_bytes, tone


class FakeTTS:
    language = "en"
    is_busy = False

    def __init__(self):
        self.spoken = []

    def render_message(self, event_type, student_name=None, names=None):
        return TTSEngine.render_message(self, event_type, student_name, names)

    def speak_nowait(self, text):
        self.spoken.append(text)


class Clock:
    def __init__(self, t=10_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(time, "time", c)
    return c


def activity_osd():
    act = np.zeros((20, 3), dtype=bool)
    act[0:15, 0] = True
    act[10:20, 1] = True
    return OSDResult(True, 0.25, "pyannote", activity=act, frame_seconds=0.1)


def build(osd=None, **settings):
    init_db()
    s = Settings(min_identify_seconds=0.45, tracker_confirm_seconds=1.0, tracker_confirm_hits=2, **settings)
    store = make_store()
    analyzer = MultiSpeakerAnalyzer(s, store, fake_embed)
    vad = SimpleNamespace(analyze=lambda a: VADResult(True, 1.0, 0.9), frame_seconds=512 / 16000)
    osd_obj = SimpleNamespace(detect=lambda a: osd if osd is not None else OSDResult(False, 0.0, "stub"),
                              backend=osd.backend if osd is not None else "stub")
    tts = FakeTTS()
    eng = ClassroomPolicyEngine(s, vad, osd_obj, SimpleNamespace(), tts, analyzer)
    return eng, tts


def speak(freq, loud=0.6):
    return pcm_bytes(tone(freq, 2.0, amp=loud))


def overlap_audio():
    return pcm_bytes(tone(FREQS["salah"], 1.5, start=0, total=2.0, amp=0.4) + tone(FREQS["sarah"], 1.0, start=1.0, total=2.0, amp=0.4))


def test_single_speaker_windows_group_into_one_event_and_one_warning(clock):
    eng, tts = build()
    for _ in range(5):
        eng.process_audio_chunk(speak(FREQS["salah"]))
        clock.t += 1.0
    assert len(eng.events) == 1                                  # not one alert per frame
    ev = eng.events[-1]
    assert ev["category"] == "Single Speaker" and ev["decision_status"] == "confirmed"
    assert ev["speakers"][0]["student_name"] == "Salah"
    assert len(tts.spoken) == 1                                  # no repeated TTS for every frame
    with session_scope() as s:
        rows = s.query(InterventionLog).filter_by(event_id=ev["event_id"]).all()
    assert len(rows) == 1 and rows[0].speaker_count == 1 and rows[0].category == "Single Speaker"
    assert json.loads(rows[0].details_json)["speakers"][0]["student_name"] == "Salah"


def test_two_registered_speakers_overlapping_are_listed_individually(clock):
    eng, tts = build(osd=activity_osd(), speaker_match_threshold=0.7)
    for _ in range(3):
        assert eng.process_audio_chunk(overlap_audio()) == State.GROUP_WARNING
        clock.t += 1.0
    ev = eng.events[-1]
    assert ev["is_overlap"] and {s["student_name"] for s in ev["speakers"]} == {"Salah", "Sarah"}
    assert ev["identification_method"] == "pyannote_slots" and ev["detection_method"] == "pyannote"
    assert ev["overlap_duration_ms"] and ev["estimated_active_speaker_count"] == 2
    assert all(s["signal_type"] == "masked_mixture" for s in ev["speakers"])
    assert "Salah" in tts.spoken[0] and "Sarah" in tts.spoken[0]      # named group message
    assert len(eng.events) == 1


def test_names_can_be_kept_out_of_spoken_warnings(clock):
    eng, tts = build(tts_announce_names=False)
    for _ in range(3):
        eng.process_audio_chunk(speak(FREQS["salah"]))
        clock.t += 1.0
    assert tts.spoken and "Salah" not in tts.spoken[0]


def test_unknown_speaker_event_and_policy_log_only(clock):
    eng, tts = build(unknown_speaker_handling="log_only")
    eng.process_audio_chunk(speak(6000.0))
    ev = eng.events[-1]
    assert ev["category"] == "Unknown Speaker" and ev["speakers"][0]["student_name"] == "Unknown Speaker"
    assert ev["unknown_speaker_count"] == 1 and ev["rejection_reason"] == "below_threshold"
    assert tts.spoken == []                                      # reported but never announced


def test_per_speaker_cooldown_is_independent_of_global_cooldown(clock):
    eng, tts = build(cooldown_seconds=20, speaker_cooldown_seconds=60, warning_min_gap_seconds=0)
    for _ in range(3):
        eng.process_audio_chunk(speak(FREQS["salah"]))
        clock.t += 1.0
    assert len(tts.spoken) == 1
    clock.t += 30                                                # global cooldown over, Salah's is not
    for _ in range(3):
        eng.process_audio_chunk(speak(FREQS["salah"]))
        clock.t += 1.0
    assert len(tts.spoken) == 1 and len(eng.events) == 2         # new event, still silenced
    clock.t += 5                                                 # a separate event for another student
    for _ in range(3):                                           # an unrelated speaker is NOT suppressed
        eng.process_audio_chunk(speak(FREQS["sarah"]))
        clock.t += 1.0
    assert len(tts.spoken) == 2 and "Sarah" in tts.spoken[1]
    clock.t += 40                                                # Salah's cooldown (60 s) expires
    for _ in range(3):
        eng.process_audio_chunk(speak(FREQS["salah"]))
        clock.t += 1.0
    assert len(tts.spoken) == 3


def test_warning_min_gap_prevents_back_to_back_warnings(clock):
    eng, tts = build(warning_min_gap_seconds=30, speaker_cooldown_seconds=0, cooldown_seconds=15)
    for freq, pause in ((FREQS["salah"], 5), (FREQS["sarah"], 5), (FREQS["sarah"], 40)):
        clock.t += pause
        for _ in range(3):
            eng.process_audio_chunk(speak(freq))
            clock.t += 1.0
    assert len(tts.spoken) == 2        # Sarah's first event (5 s after Salah's warning) was held back


def test_malformed_and_silent_frames_are_ignored_safely(clock):
    eng, tts = build()
    assert eng.process_audio_chunk(b"") == State.IDLE
    assert eng.process_audio_chunk(b"\x01\x02\x03") == State.IDLE          # odd byte count
    assert eng.process_audio_chunk(bytes(64000)) == State.IDLE
    assert eng.timers.summary()["_counters"]["invalid_frames"] == 2.0
    assert tts.spoken == [] and len(eng.events) == 0


def test_latency_is_measured_per_stage_and_end_to_end(clock):
    eng, _ = build()
    eng.process_audio_chunk(speak(FREQS["salah"]), captured_at=clock.t - 0.25)
    summary = eng.snapshot()["latency"]
    assert {"vad", "overlap_detection", "identification", "total", "end_to_end"} <= set(summary)
    assert summary["end_to_end"]["median_ms"] == pytest.approx(250, abs=1)
    assert "real_time_factor" in summary
    assert eng.events[-1]["processing_latency_ms"] == pytest.approx(250, abs=1)


def test_snapshot_status_ongoing_then_completed(clock):
    eng, _ = build()
    eng.process_audio_chunk(speak(FREQS["salah"]))
    assert eng.snapshot()["current_event"]["status"] == "ongoing"
    clock.t += 10
    snap = eng.snapshot()
    assert snap["current_event"] is None and snap["recent_events"][0]["status"] == "completed"
    assert all("waveform" not in sp for sp in snap["recent_events"][0]["speakers"])   # slim payload


def test_forget_student_scrubs_memory_and_stored_events(clock):
    from engine.persistence import anonymize_student_in_logs

    eng, _ = build()
    for _ in range(3):
        eng.process_audio_chunk(speak(FREQS["salah"]))
        clock.t += 1.0
    eng.forget_student("salah")
    assert eng.events[-1]["speakers"][0]["student_name"] == "Unknown Speaker"
    assert anonymize_student_in_logs("salah") >= 1
    with session_scope() as s:
        row = s.query(InterventionLog).filter_by(event_id=eng.events[-1]["event_id"]).one()
    assert "Salah" not in row.details_json


def test_events_and_visualization_endpoints_and_logs_api(clock):
    from api.deps import get_service
    from api.main import app

    eng, _ = build(osd=activity_osd(), speaker_match_threshold=0.7)
    for _ in range(2):
        eng.process_audio_chunk(overlap_audio())
        clock.t += 1.0
    service = get_service()
    service.engine, service.analyzer = eng, eng.analyzer
    client = TestClient(app)
    events = client.get("/events").json()
    assert events and events[0]["schema_version"] == "2.1" and events[0]["speaker_count"] == 2
    one = client.get(f"/events/{events[0]['event_id']}").json()
    assert one["event_id"] == events[0]["event_id"] and client.get("/events/nope").status_code == 404
    viz = client.get("/live/visualization", params={"seconds": 30}).json()
    assert viz["waveform"]["peaks"] and viz["events"] and viz["capabilities"]["overlap_method"] == "pyannote_slots"
    status = client.get("/status").json()
    assert status["current_event"]["speaker_count"] == 2 and "latency" in status
    with client.websocket_connect("/ws/stream") as ws:                   # WebSocket serialises the same payload
        msg = ws.receive_json()
        assert msg["current_event"]["event_id"] == events[0]["event_id"]
    logs = client.get("/logs", params={"limit": 500}).json()
    structured = [l for l in logs if l["event_id"] == events[0]["event_id"]]
    assert structured and structured[0]["speaker_count"] == 2 and len(structured[0]["speakers"]) == 2
    assert structured[0]["event_type"] == "GROUP"
    service.engine = None


def test_legacy_engine_path_unchanged_without_analyzer():
    init_db()
    from tests.test_policy_engine import make_engine, LOUD
    eng, tts = make_engine(overlap=True)
    assert eng.process_audio_chunk(LOUD) == State.GROUP_WARNING and len(tts.spoken) == 1
    assert eng.events[-1]["timestamp"] and eng.events[-1]["time_label"]
