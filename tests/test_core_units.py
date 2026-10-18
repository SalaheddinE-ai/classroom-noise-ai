"""Pure unit tests: config, models, vector store, audio helpers, templates, stream buffer."""
import numpy as np
import pytest
from pydantic import ValidationError

from config import Settings, get_settings
from core.audio_stream import AudioStream
from core.audio_utils import compute_db, pcm16_bytes_to_float
from database.db_session import init_db, session_scope
from database.models import EventType, InterventionLog, Student, VoiceProfile
from database.vector_store import VectorStore


def _vec(seed, dim=192):
    return np.random.default_rng(seed).normal(size=dim).astype(np.float32)


# ------------------------------------------------------------------ config
def test_settings_defaults():
    s = get_settings()
    assert s.threshold_db == 68.0 and s.speaker_match_threshold == 0.75
    assert s.chunk_samples == 32000 and s.hop_samples == 16000


def test_cooldown_range_enforced():
    s = Settings()
    with pytest.raises(ValidationError):
        s.cooldown_seconds = 5
    with pytest.raises(ValidationError):
        s.cooldown_seconds = 99
    s.cooldown_seconds = 30


# ------------------------------------------------------------- vector store
def test_cosine_similarity_known_values():
    a = np.array([1.0, 0.0]); b = np.array([0.0, 1.0])
    assert VectorStore.cosine_similarity(a, a) == pytest.approx(1.0)
    assert VectorStore.cosine_similarity(a, b) == pytest.approx(0.0)
    assert VectorStore.cosine_similarity(a, -a) == pytest.approx(-1.0)
    assert VectorStore.cosine_similarity(a, np.zeros(2)) == 0.0


def test_search_ranks_and_scales_invariant():
    vs = VectorStore(192)
    vs.add("A", _vec(1)); vs.add("B", _vec(2))
    best = vs.search(_vec(1) * 7.0, top_k=2)
    assert best[0][0] == "A" and best[0][1] == pytest.approx(1.0, abs=1e-5)
    assert best[1][0] == "B" and best[1][1] < 0.4


def test_search_keeps_best_sample_per_student():
    vs = VectorStore(192)
    vs.add("A", _vec(1)); vs.add("A", _vec(3))
    res = vs.search(_vec(3), top_k=5)
    assert len(res) == 1 and res[0][1] == pytest.approx(1.0, abs=1e-5)


def test_empty_store_and_wrong_dim_and_remove():
    vs = VectorStore(192)
    assert vs.search(_vec(1)) == []
    with pytest.raises(ValueError):
        vs.add("X", np.ones(10))
    vs.add("A", _vec(1)); vs.add("B", _vec(2))
    assert vs.remove_student("A") == 1 and len(vs) == 1
    assert vs.remove_student("nope") == 0


# ---------------------------------------------------------------- database
def test_db_roundtrip_cascade_and_embedding_blob():
    init_db()
    emb = _vec(5)
    with session_scope() as s:
        st = Student(name="Amine", age=12, student_code="T-1")
        s.add(st); s.flush()
        s.add(VoiceProfile(student_id=st.id, embedding_blob=VoiceProfile.pack_embedding(emb)))
        s.add(InterventionLog(event_type=EventType.SINGLE_STUDENT, student_id=st.id, dB_level=71.2,
                              message_spoken="hi", match_score=0.8, was_spoken=True))
        sid = st.id
    vs = VectorStore(192)
    with session_scope() as s:
        assert vs.load_from_db(s) >= 1
        prof = s.query(VoiceProfile).filter_by(student_id=sid).one()
        assert np.allclose(prof.embedding, emb)
        assert len(sid) == 36
    with session_scope() as s:  # delete -> profile cascades, log.student_id -> NULL
        s.delete(s.get(Student, sid))
    with session_scope() as s:
        assert s.query(VoiceProfile).filter_by(student_id=sid).count() == 0
        assert s.query(InterventionLog).filter_by(student_id=None).count() >= 1


def test_unique_student_code():
    from sqlalchemy.exc import IntegrityError
    init_db()
    with session_scope() as s:
        s.add(Student(name="A", student_code="DUP"))
    with pytest.raises(IntegrityError):
        with session_scope() as s:
            s.add(Student(name="B", student_code="DUP"))


# ------------------------------------------------------------ audio helpers
def test_compute_db_scale_and_monotonic():
    t = np.linspace(0, 200, 32000)
    quiet = (np.sin(t) * 500).astype(np.int16)
    loud = (np.sin(t) * 12000).astype(np.int16)
    dq = compute_db(pcm16_bytes_to_float(quiet.tobytes()))
    dl = compute_db(pcm16_bytes_to_float(loud.tobytes()))
    assert dl > dq
    assert dl - dq == pytest.approx(20 * np.log10(24), abs=0.2)
    assert compute_db(np.zeros(100, dtype=np.float32)) == 0.0
    assert compute_db(np.zeros(0, dtype=np.float32)) == 0.0
    assert compute_db(pcm16_bytes_to_float(loud.tobytes()), offset_db=5) == pytest.approx(dl + 5)


# ------------------------------------------------------------- audio stream
def test_stream_sliding_window_logic():
    st = AudioStream(sample_rate=16000, chunk_seconds=2.0, hop_seconds=1.0)
    st._closed = False
    assert st.read_chunk(timeout=0.05) is None            # not enough data
    st._ingest(np.ones(16000, dtype=np.int16))
    assert st.read_chunk(timeout=0.05) is None            # 1 s < 2 s window
    st._ingest(np.ones(16000, dtype=np.int16))
    chunk = st.read_chunk(timeout=0.05)
    assert chunk is not None and len(chunk) == 32000 * 2  # 2 s of int16
    assert st.read_chunk(timeout=0.05) is None            # hop not yet elapsed
    st._ingest(np.full(16000, 3, dtype=np.int16))
    nxt = np.frombuffer(st.read_chunk(timeout=0.05), dtype=np.int16)
    assert nxt[-1] == 3 and nxt[0] == 1                   # window slid by 1 s
    assert st.recent_db(0.5) > 0
    st.stop()
    assert st.read_chunk(timeout=0.05) is None


# ---------------------------------------------------------------- templates
def test_templates_five_variants_each_and_name_substitution():
    pytest.importorskip("soundfile")
    from core.tts_engine import TEMPLATES
    for lang, groups in TEMPLATES.items():
        for et in EventType:
            assert len(set(groups[et])) == 5, (lang, et)
    assert all("{name}" in t for t in TEMPLATES["en"][EventType.SINGLE_STUDENT])


def test_resample_helper():
    x = (np.arange(4800) % 100).astype(np.int16)
    assert len(AudioStream.resample_int16(x, 48000, 16000)) == 1600      # integer ratio
    assert len(AudioStream.resample_int16(x, 44100, 16000)) == round(4800 * 16000 / 44100)
    assert AudioStream.resample_int16(x, 16000, 16000) is x
    assert AudioStream.resample_int16(np.zeros(0, dtype=np.int16), 48000, 16000).size == 0
