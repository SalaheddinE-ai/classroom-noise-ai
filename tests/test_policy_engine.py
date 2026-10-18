"""Policy engine decision-flow tests with stubbed models (no downloads, no audio hardware)."""
import time
from types import SimpleNamespace

import numpy as np
import pytest


from config import Settings
from core.OSD import OSDResult
from core.tts_engine import TTSEngine
from core.vad import VADResult
from database.db_session import init_db, session_scope
from database.models import EventType, InterventionLog, Student
from engine.policy_engine import ClassroomPolicyEngine, State


class FakeTTS:
    def __init__(self):
        self.spoken, self.is_busy = [], False
        self._real = TTSEngine.render_message
        self.language = "en"

    def render_message(self, event_type, name=None):
        return TTSEngine.render_message(self, event_type, name)

    def speak_nowait(self, text):
        self.spoken.append(text)


def pcm(amplitude):
    t = np.linspace(0, 300, 32000)
    return (np.sin(t) * amplitude).astype(np.int16).tobytes()


LOUD, QUIET = pcm(12000), pcm(100)


def make_engine(speech=True, overlap=False, student=None, score=0.0):
    init_db()
    vad = SimpleNamespace(analyze=lambda a: VADResult(speech, 1.0 if speech else 0.0, 0.9))
    osd = SimpleNamespace(detect=lambda a: OSDResult(overlap, 0.5 if overlap else 0.0, "stub"), backend="stub")
    rec = SimpleNamespace(
        extract_embedding=lambda a: np.ones(192, dtype=np.float32),
        identify_student=lambda e: (student, score),
    )
    tts = FakeTTS()
    eng = ClassroomPolicyEngine(Settings(), vad, osd, rec, tts)
    return eng, tts


def test_quiet_audio_is_idle():
    eng, tts = make_engine()
    assert eng.process_audio_chunk(QUIET) == State.IDLE and not tts.spoken


def test_loud_without_speech_is_noise_only():
    eng, tts = make_engine(speech=False)
    assert eng.process_audio_chunk(LOUD) == State.NOISE_ONLY and not tts.spoken


def test_overlap_triggers_group_and_skips_speaker_id():
    eng, tts = make_engine(overlap=True)
    eng.recognizer.identify_student = lambda e: (_ for _ in ()).throw(AssertionError("must be skipped"))
    assert eng.process_audio_chunk(LOUD) == State.GROUP_WARNING
    assert len(tts.spoken) == 1


def test_known_student_above_threshold_is_targeted():
    stu = Student(id="s-1", name="Yassine")
    eng, tts = make_engine(student=stu, score=0.82)
    assert eng.process_audio_chunk(LOUD) == State.SINGLE_STUDENT_WARNING
    assert "Yassine" in tts.spoken[0]


@pytest.mark.parametrize("score", [0.75, 0.60])
def test_score_at_or_below_threshold_is_general(score):
    stu = Student(id="s-2", name="Sara")
    eng, tts = make_engine(student=stu, score=score)
    assert eng.process_audio_chunk(LOUD) == State.GENERAL_WARNING
    assert "Sara" not in tts.spoken[0]


def test_unknown_when_nobody_registered():
    eng, tts = make_engine(student=None, score=0.0)
    assert eng.process_audio_chunk(LOUD) == State.GENERAL_WARNING


def test_cooldown_suppresses_tts_but_logs():
    eng, tts = make_engine()
    eng.process_audio_chunk(LOUD)
    assert eng.is_cooldown and 0 < eng.cooldown_remaining <= eng.cooldown_seconds
    eng.process_audio_chunk(LOUD)
    eng.process_audio_chunk(LOUD)
    assert len(tts.spoken) == 1                       # only the first was spoken
    assert [e["spoken"] for e in eng.events] == [True, False, False]
    with session_scope() as s:
        suppressed = s.query(InterventionLog).filter_by(was_spoken=False).count()
    assert suppressed >= 2


def test_cooldown_expiry_allows_new_speech():
    eng, tts = make_engine()
    eng.process_audio_chunk(LOUD)
    eng.last_trigger_time = time.time() - eng.cooldown_seconds - 1
    eng.process_audio_chunk(LOUD)
    assert len(tts.spoken) == 2


def test_threshold_runtime_change_and_echo_guard():
    eng, tts = make_engine()
    eng.threshold_dB = 99
    assert eng.process_audio_chunk(LOUD) == State.IDLE
    eng.threshold_dB = 68
    tts.is_busy = True                                # own speaker playing
    assert eng.process_audio_chunk(LOUD) == State.IDLE and not tts.spoken


def test_snapshot_shape():
    eng, _ = make_engine()
    eng.process_audio_chunk(LOUD)
    snap = eng.snapshot()
    assert snap["state"] == "GENERAL_WARNING" and snap["last_event"]["spoken"] is True
    assert snap["threshold_db"] == 68.0


def test_speech_reported_even_below_threshold():
    eng, tts = make_engine(speech=True)
    eng.threshold_dB = 90                                  # gate blocks any intervention
    assert eng.process_audio_chunk(LOUD) == State.IDLE and not tts.spoken
    snap = eng.snapshot()
    assert snap["speech_detected"] is True and snap["speech_ratio"] == 1.0


def test_digital_silence_skips_vad():
    eng, _ = make_engine()
    assert eng.process_audio_chunk(bytes(64000)) == State.IDLE
    assert eng.snapshot()["speech_detected"] is False
