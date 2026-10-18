"""Dashboard rendering: figure builders for single / multi / unknown events + a Streamlit AppTest smoke run."""
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from api.deps import get_service
from engine.events import AnalysisResult, EventAggregator, SpeakerDetection
from tests.test_multispeaker import NOW, PAIR_AUDIO, SLOTS, make_engine
from ui import components as ui

APP = str(Path(__file__).resolve().parent.parent / "ui" / "app.py")


def _det(name, sid, start, end, score, identified=True):
    return SpeakerDetection("spk", sid, name, start, end, score, is_identified=identified, segments=[(start, end)],
                            waveform=[0.1, 0.8, 0.4] if identified else None,
                            waveform_kind="masked_mixture" if identified else "activity_only")


def _event(speakers, overlap=()):
    ov = list(overlap)
    res = AnalysisResult(speakers, bool(ov), ov, False, "pyannote_slots", NOW - 2, NOW)
    ev, _ = EventAggregator().ingest(res, 71.0)
    ev.message, ev.warning_triggered, ev.warning_status = "Please lower your voices.", True, "spoken"
    return ev.to_dict()


SINGLE = _event([_det("Salah", "id-salah", NOW - 2, NOW - 0.5, 0.87)])
MULTI = _event([_det("Salah", "id-salah", NOW - 2, NOW - 0.4, 0.86), _det("Sarah", "id-sarah", NOW - 1.5, NOW, 0.81)], [(NOW - 1.5, NOW - 0.4)])
UNKNOWN = _event([_det("Unknown Speaker", None, NOW - 2, NOW, 0.31, identified=False)])


# ------------------------------------------------------------- pure builders
def test_titles_and_rows_reflect_real_event_values():
    assert ui.event_title(SINGLE) == "Single speaker: Salah"
    assert ui.event_title(MULTI) == "Multiple speakers detected"
    assert ui.event_title(UNKNOWN) == "Unknown speaker detected"
    rows = ui.speaker_rows(MULTI)
    assert [r["Speaker"] for r in rows] == ["Salah", "Sarah"] and [r["Similarity"] for r in rows] == ["0.86", "0.81"]
    assert rows[0]["Status"] == "Identified" and ui.speaker_rows(UNKNOWN)[0]["Status"] == "Unknown Speaker"
    assert ui.overlap_text(MULTI) == "1.1 s" and ui.overlap_text(SINGLE) == "No overlap"
    assert "Multiple Speakers" in MULTI["labels"] and "Overlapping Speech" in MULTI["labels"]
    assert UNKNOWN["labels"] == ["Single Speaker", "Unknown Speaker"] or "Unknown Speaker" in UNKNOWN["labels"]


def test_colours_are_stable_per_identity_and_grey_for_unknown():
    assert ui.color_for("id-salah") == ui.color_for("id-salah")
    assert ui.color_for(None, False) == ui.UNKNOWN_COLOR == ui.color_for("id-salah", False)


def test_timeline_has_one_track_per_speaker_and_marks_overlap():
    eng, _ = make_engine(SLOTS)
    eng.process_audio_chunk(PAIR_AUDIO, end_time=NOW)
    tl = eng.snapshot()["timeline"]
    fig = ui.build_timeline_figure(tl)
    assert sorted(t.name for t in fig.data) == ["Salah", "Sarah"]
    assert len(fig.layout.shapes) == 1                                      # the shaded overlap band
    assert list(fig.layout.yaxis.categoryarray) in (["Salah", "Sarah"], ["Sarah", "Salah"])
    empty = ui.build_timeline_figure({"tracks": [], "overlaps": [], "window_start": 0, "window_end": 30})
    assert len(empty.data) == 0 and empty.layout.annotations


def test_several_unknown_speakers_get_distinct_tracks():
    tl = {"window_start": NOW - 30, "window_end": NOW, "overlaps": [],
          "tracks": [{"key": "a", "label": "Unknown Speaker", "student_id": None, "is_identified": False, "segments": [[NOW - 5, NOW - 3]]},
                     {"key": "b", "label": "Unknown Speaker", "student_id": None, "is_identified": False, "segments": [[NOW - 4, NOW - 2]]}]}
    assert sorted(t.name for t in ui.build_timeline_figure(tl).data) == ["Unknown Speaker 1", "Unknown Speaker 2"]


def test_speaker_waveform_figure_distinguishes_separated_masked_and_activity_only():
    eng, _ = make_engine(SLOTS)
    eng.process_audio_chunk(PAIR_AUDIO, end_time=NOW)
    view = eng.snapshot()["latest_analysis"]
    fig = ui.build_speaker_waveforms_figure(view)
    titles = [a.text for a in fig.layout.annotations]
    assert titles[0] == "Combined microphone" and len(titles) == 3 and all("masked mixture" in t for t in titles[1:])
    view["speakers"][0]["waveform"], view["speakers"][0]["waveform_kind"] = None, "activity_only"
    assert "activity only" in ui.build_speaker_waveforms_figure(view).layout.annotations[1].text
    assert not ui.build_speaker_waveforms_figure(None).data
    assert "NOT an isolated voice" in ui.KIND_CAPTIONS["masked_mixture"]


def test_event_preview_and_mic_waveform_figures():
    for ev in (SINGLE, MULTI, UNKNOWN):
        fig = ui.build_event_preview_figure(ev)
        assert len(fig.data) == ev["speaker_count"]
    assert len(ui.build_mic_waveform_figure([0.1, 0.5, 0.2], 4.0).data) == 2 and not ui.build_mic_waveform_figure([]).data
    assert "Salah (0.86)" in ui.speaker_badges_html([{"label": "Salah", "student_id": "x", "is_identified": True, "similarity_score": 0.86}])


# ------------------------------------------------------------- full app run
def _fake_backend():
    from tests.test_multispeaker import FakeTTS  # noqa: F401  (engine fixture reuses it)

    eng, _ = make_engine(SLOTS)
    eng.process_audio_chunk(PAIR_AUDIO, end_time=NOW)
    svc = get_service()
    svc.engine = eng
    status = svc.status()
    svc.engine = None
    events = [SINGLE, MULTI, UNKNOWN]
    student = {"id": "id-salah", "name": "Salah", "age": 12, "student_code": "S1", "created_at": "2026-10-01T10:00:00",
               "voice_samples": 2, "notes": None, "group_name": "6B", "updated_at": None, "recognition_status": "ready", "min_voice_samples": 1}
    detail = {**student, "samples": [{"id": 1, "created_at": "2026-10-01T10:00:00", "duration_seconds": 2.5, "rms_db": 55.0, "has_audio": False}]}
    routes = {
        "/health": {"status": "ok"}, "/settings": {"threshold_db": 68.0, "cooldown_seconds": 20, "speaker_match_threshold": 0.75, "language": "en"},
        "/students": [student], "/students/id-salah": detail, "/status": status, "/events": events, "/logs": [],
    }

    def fake_get(url, params=None, timeout=None):
        path = "/" + url.split("//", 1)[1].split("/", 1)[1]
        resp = mock.Mock(status_code=200)
        resp.json.return_value = routes[path]
        resp.raise_for_status.return_value = None
        return resp

    return fake_get


def test_dashboard_renders_single_multi_and_unknown_events():
    from streamlit.testing.v1 import AppTest

    fake_get = _fake_backend()
    with mock.patch("requests.get", fake_get), mock.patch("requests.request", mock.Mock()):
        at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert [t.label for t in at.tabs] == ["Student Registration", "Live Monitor", "Recent Alerts", "Analytics and Logs"]
    text = " ".join(m.value for m in at.markdown)
    assert "Single speaker: Salah" in text and "Multiple speakers detected" in text and "Unknown speaker detected" in text
    assert "identified, similarity 0.86" in text and "Overlap duration: 1.1 s" in text
    assert "Salah (0.86)" in text or "Salah" in text                         # profile / current speakers rendered
