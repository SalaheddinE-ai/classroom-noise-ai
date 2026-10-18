"""Student profiles over the REST API with a stubbed embedding model (no downloads, no torch)."""
import json
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import deps
from api.main import app
from api.routes import students as students_route
from config import Settings
from database.db_session import init_db, session_scope
from database.models import InterventionLog, Student, VoiceProfile
from database.vector_store import VectorStore
from tests.helpers import SR, fake_embed, tone

init_db()


def voice(freq, seconds=2.5, amp=0.3):
    """Speech-like clip: tone bursts separated by pauses (so it has clear dynamics)."""
    n = int(seconds * SR)
    out = np.zeros(n, dtype=np.float32)
    burst = tone(freq, 0.45, amp)
    for k in range(int(seconds / 0.7)):
        a = int(k * 0.7 * SR)
        out[a: a + len(burst)] = burst[: n - a]
    return out


def wav_bytes(wav):  # the stubbed decoder reads raw float32
    return np.asarray(wav, dtype=np.float32).tobytes()


class FakeService:
    def __init__(self, **settings):
        self.settings = Settings(**settings)
        self.vector_store = VectorStore(192)
        self.engine = None

    def load_models(self):
        pass

    def embed_audio(self, wav):
        return fake_embed(wav)


@pytest.fixture
def api(monkeypatch):
    service = FakeService()
    monkeypatch.setattr(students_route, "_decode_upload", lambda data, sr: np.frombuffer(data, dtype=np.float32))
    app.dependency_overrides[deps.get_service] = lambda: service
    yield TestClient(app), service
    app.dependency_overrides.clear()


def uniq(prefix="Stu"):
    return f"{prefix}{uuid.uuid4().hex[:6]}"


def register(client, name, freq, **extra):
    data = {"name": name, **extra}
    return client.post("/students/register", data=data, files={"audio": ("a.wav", wav_bytes(voice(freq)), "audio/wav")})


def add_sample(client, sid, freq, **extra):
    return client.post(f"/students/{sid}/voice", data=extra, files={"audio": ("a.wav", wav_bytes(voice(freq, amp=0.25)), "audio/wav")})


def test_register_original_request_format_and_profile_retrieval(api):
    client, service = api
    name, code = uniq(), uniq("C")
    r = client.post("/students/register", data={"name": name, "age": 12, "student_code": code},
                    files={"audio": ("a.wav", wav_bytes(voice(500)), "audio/wav")})
    assert r.status_code == 201
    out = r.json()
    assert {"id", "name", "age", "student_code", "created_at", "voice_samples"} <= set(out)     # old contract intact
    assert out["voice_samples"] == 1 and out["recognition_status"] == "ready"
    profile = client.get(f"/students/{out['id']}").json()
    assert profile["name"] == name and len(profile["samples"]) == 1 and profile["min_samples_required"] == 1
    assert profile["samples"][0]["duration_seconds"] == pytest.approx(2.5, abs=0.2)
    assert profile["samples"][0]["quality_weight"] is not None and "embedding" not in json.dumps(profile)
    assert any(s["id"] == out["id"] for s in client.get("/students").json())
    assert service.vector_store.name_of(out["id"]) == name and service.vector_store.sample_count(out["id"]) == 1


def test_enrollment_audio_is_not_stored_by_default(api):
    client, service = api
    sid = register(client, uniq(), 500).json()["id"]
    assert not list(service.settings.samples_dir.glob(f"{sid}_*.wav"))
    assert client.get(f"/students/{sid}").json()["samples"][0]["has_audio"] is False
    with session_scope() as s:
        assert s.query(VoiceProfile).filter_by(student_id=sid).one().audio_sample_path is None


def test_multiple_samples_and_min_samples_status(api):
    client, service = api
    service.settings.min_voice_samples = 2
    sid = register(client, uniq(), 700).json()["id"]
    assert client.get(f"/students/{sid}").json()["recognition_status"] == "needs_samples"
    r = add_sample(client, sid, 700)
    assert r.status_code == 200 and r.json()["voice_samples"] == 2 and r.json()["recognition_status"] == "ready"
    assert service.vector_store.sample_count(sid) == 2


def test_sample_that_does_not_resemble_the_student_is_refused_unless_forced(api):
    client, _ = api
    sid = register(client, uniq(), 500).json()["id"]
    bad = add_sample(client, sid, 4000)
    assert bad.status_code == 422 and "does not resemble" in bad.json()["detail"]
    assert add_sample(client, sid, 4000, force="true").status_code == 200


def test_voice_matching_another_student_is_never_silently_attached(api):
    client, _ = api
    a = register(client, uniq("A"), 900).json()
    dup = register(client, uniq("B"), 900)                  # same voice, new name -> probable duplicate
    assert dup.status_code == 409 and "very similar" in dup.json()["detail"]
    other = register(client, uniq("C"), 2500).json()
    r = add_sample(client, other["id"], 900)                # C's profile gets A's voice
    assert r.status_code in (409, 422)


def test_duplicate_name_and_code(api):
    client, _ = api
    name, code = uniq(), uniq("C")
    assert register(client, name, 600, student_code=code).status_code == 201
    r = register(client, name.upper(), 2000)
    assert r.status_code == 409 and "already exists" in r.json()["detail"]
    assert register(client, name, 2000, allow_duplicate_name="true").status_code == 201   # e.g. two Mohameds
    r = register(client, uniq(), 3000, student_code=code)
    assert r.status_code == 409 and "code" in r.json()["detail"]


def test_bad_recordings_are_rejected_with_reasons(api):
    client, _ = api
    def post(wav):
        return client.post("/students/register", data={"name": uniq()},
                           files={"audio": ("a.wav", wav_bytes(wav), "audio/wav")})
    silent = post(np.zeros(3 * SR))
    assert silent.status_code == 422 and "rejected" in silent.json()["detail"]
    clipped = post(np.clip(voice(500) * 20, -1, 1))
    assert clipped.status_code == 422 and "clipped" in clipped.json()["detail"]
    steady = post(tone(500, 3.0))
    assert steady.status_code == 422 and "pauses" in steady.json()["detail"]
    short = post(voice(500, 0.8))
    assert short.status_code == 422 and "short" in short.json()["detail"].lower()
    assert post(np.array([np.nan] * SR * 2, dtype=np.float32)).status_code == 422


def test_update_profile_fields(api):
    client, service = api
    sid = register(client, uniq(), 500, student_code=uniq("C")).json()["id"]
    other = register(client, uniq(), 2500, student_code=uniq("C")).json()
    new = uniq("New")
    r = client.patch(f"/students/{sid}", json={"name": new, "notes": "front row", "group_name": "6B", "age": 11})
    assert r.status_code == 200 and r.json()["name"] == new and r.json()["group_name"] == "6B"
    assert service.vector_store.name_of(sid) == new                                 # matching uses the new name
    assert client.patch(f"/students/{sid}", json={"name": "  "}).status_code == 422
    assert client.patch(f"/students/{sid}", json={"student_code": other["student_code"]}).status_code == 409
    assert client.patch("/students/unknown", json={"age": 10}).status_code == 404


def test_delete_one_sample_then_student_removes_everything(monkeypatch):
    service = FakeService(store_enrollment_audio=True)
    monkeypatch.setattr(students_route, "_decode_upload", lambda data, sr: np.frombuffer(data, dtype=np.float32))
    app.dependency_overrides[deps.get_service] = lambda: service
    try:
        client = TestClient(app)
        sid = register(client, uniq(), 500).json()["id"]
        add_sample(client, sid, 500)
        files = list(service.settings.samples_dir.glob(f"{sid}_*.wav"))
        assert len(files) == 2
        profile = client.get(f"/students/{sid}").json()
        first = profile["samples"][0]["id"]
        assert profile["samples"][0]["has_audio"] and client.get(f"/students/{sid}/voice/{first}/audio").status_code == 200
        assert client.delete(f"/students/{sid}/voice/{first}").status_code == 204
        assert len(list(service.settings.samples_dir.glob(f"{sid}_*.wav"))) == 1
        assert service.vector_store.sample_count(sid) == 1
        assert client.delete(f"/students/{sid}/voice/{first}").status_code == 404
        # a stored event mentioning the student must be anonymised on deletion
        with session_scope() as s:
            s.add(InterventionLog(event_type="GROUP", dB_level=70, message_spoken="x", event_id="evt-" + sid,
                                  details_json=json.dumps({"speakers": [{"student_id": sid, "student_name": "Zed",
                                                                         "is_identified": True}]})))
        assert client.delete(f"/students/{sid}").status_code == 204
        assert not list(service.settings.samples_dir.glob(f"{sid}_*.wav"))           # recordings gone
        assert service.vector_store.sample_count(sid) == 0 and service.vector_store.name_of(sid) is None
        assert client.get(f"/students/{sid}").status_code == 404
        with session_scope() as s:
            assert s.query(VoiceProfile).filter_by(student_id=sid).count() == 0
            row = s.query(InterventionLog).filter_by(event_id="evt-" + sid).one()
        assert "Zed" not in row.details_json and "Unknown Speaker" in row.details_json
    finally:
        app.dependency_overrides.clear()


def test_sample_audio_endpoint_cannot_read_arbitrary_files(api, tmp_path):
    client, service = api
    sid = register(client, uniq(), 500).json()["id"]
    secret = tmp_path / "secret.wav"
    secret.write_bytes(b"RIFFsecret")
    with session_scope() as s:
        prof = s.query(VoiceProfile).filter_by(student_id=sid).one()
        prof.audio_sample_path = str(secret)                      # path outside the samples directory
        pid = prof.id
    assert client.get(f"/students/{sid}/voice/{pid}/audio").status_code == 404
    other = register(client, uniq(), 2500).json()["id"]
    assert client.get(f"/students/{other}/voice/{pid}/audio").status_code == 404     # wrong owner


def test_diagnose_reports_similarities_and_stores_nothing(api):
    client, service = api
    sid = register(client, uniq(), 500).json()["id"]
    before = service.vector_store.sample_count(sid)
    r = client.post("/students/diagnose", files={"audio": ("p.wav", wav_bytes(voice(500)), "audio/wav")})
    assert r.status_code == 200
    out = r.json()
    assert out["results"][0]["student_id"] == sid and out["results"][0]["similarity"] > 0.9
    assert out["threshold"] == service.settings.speaker_match_threshold and "identified" in out["hint"]
    assert service.vector_store.sample_count(sid) == before  # nothing stored
    service.settings.speaker_match_threshold = 0.999999      # force a rejection -> actionable hint
    far = client.post("/students/diagnose", files={"audio": ("p.wav", wav_bytes(voice(1900)), "audio/wav")}).json()
    assert "Below the threshold" in far["hint"] or "identified" in far["hint"]
