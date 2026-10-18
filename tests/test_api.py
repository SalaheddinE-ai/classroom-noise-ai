"""REST API tests (no models loaded)."""
import io
import wave

import os

import pytest
from fastapi.testclient import TestClient

from api.main import app
from database.db_session import init_db

init_db()
client = TestClient(app)


def test_health_and_status_stopped():
    assert client.get("/health").json() == {"status": "ok"}
    st = client.get("/status").json()
    assert st["ui_state"] == "Stopped" and st["running"] is False


def test_settings_roundtrip_and_validation():
    r = client.put("/settings", json={"threshold_db": 72, "cooldown_seconds": 25, "language": "fr"})
    assert r.status_code == 200 and r.json()["threshold_db"] == 72 and r.json()["language"] == "fr"
    assert client.put("/settings", json={"cooldown_seconds": 5}).status_code == 422
    assert client.put("/settings", json={"language": "xx"}).status_code == 422
    client.put("/settings", json={"threshold_db": 68, "cooldown_seconds": 20, "language": "en"})


def test_logs_filters_and_empty_students():
    assert client.get("/students").json() == []
    assert client.get("/logs", params={"event_type": ["GROUP", "GENERAL"], "limit": 10}).status_code == 200
    assert client.get("/logs", params={"event_type": "BOGUS"}).status_code == 422


def test_register_rejects_bad_input():
    if os.environ.get("CNAI_TEST_REAL_TORCH") != "1":
        pytest.skip("needs real torch/torchaudio")
    assert client.post("/students/register", data={"name": "  "},
                       files={"audio": ("a.wav", b"x", "audio/wav")}).status_code in (422,)
    r = client.post("/students/register", data={"name": "X"}, files={"audio": ("a.wav", b"garbage", "audio/wav")})
    assert r.status_code == 422


def test_register_too_short_recording():
    if os.environ.get("CNAI_TEST_REAL_TORCH") != "1":
        pytest.skip("needs real torch/torchaudio")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\x00\x00" * 8000)
    r = client.post("/students/register", data={"name": "X"}, files={"audio": ("a.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 422 and "short" in r.json()["detail"].lower()


def test_delete_unknown_student_404():
    assert client.delete("/students/unknown").status_code == 404


def test_websocket_stream():
    with client.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json()["ui_state"] == "Stopped"


def test_threshold_above_physical_ceiling_rejected():
    assert client.put("/settings", json={"threshold_db": 100}).status_code == 422
