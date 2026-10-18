"""Streamlit control dashboard for classroom-noise-ai.

The dashboard only talks to the FastAPI backend (CNAI_API_URL, default http://127.0.0.1:8000).
Tabs: Student Registration (profiles), Live Monitor (waveform + speaker timeline),
Recent Alerts (grouped, expandable events) and Analytics and Logs.
"""
from __future__ import annotations

import io
import os
import sys
import time
import wave
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # `streamlit run ui/app.py` from anywhere

from ui import components as ui  # noqa: E402

API_URL = os.getenv("CNAI_API_URL", "http://127.0.0.1:8000").rstrip("/")
EVENT_TYPES = ["SINGLE_STUDENT", "GROUP", "GENERAL"]
ENROLL_SENTENCE = "The quick brown fox jumps over the lazy dog, and today I am joining the class."
STATUS_BADGE = {"ready": "Ready", "needs_samples": "Needs more samples", "no_samples": "No voice sample"}

st.set_page_config(page_title="Classroom Noise AI", layout="wide")


# ------------------------------------------------------------------ API helpers
def api_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: float = 10) -> Any:
    try:
        resp = requests.get(f"{API_URL}{path}", params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def api_call(method: str, path: str, timeout: float = 300, **kwargs: Any) -> Optional[requests.Response]:
    try:
        return requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Backend unreachable at {API_URL}: {exc}")
        return None


def error_detail(resp: requests.Response) -> str:
    try:
        return str(resp.json().get("detail", resp.text))
    except Exception:
        return resp.text


def wav_duration(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / float(w.getframerate())


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("Classroom Noise AI")
    health = api_get("/health")
    if health:
        st.success(f"Backend online: {API_URL}")
    else:
        st.error(f"Backend offline: {API_URL}")
        st.caption("Start it with: python main.py api")
        st.stop()

    st.subheader("Settings")
    current = api_get("/settings") or {}
    threshold = st.slider("Loudness threshold (dB)", 20.0, 90.0, min(float(current.get("threshold_db", 68.0)), 90.0), 1.0)
    cooldown = st.slider("Cooldown (seconds)", 15, 30, int(current.get("cooldown_seconds", 20)))
    match = st.slider("Speaker match threshold", 0.20, 0.95, float(current.get("speaker_match_threshold", 0.75)), 0.01)
    language = st.selectbox("Warning language", ["en", "fr"], index=["en", "fr"].index(current.get("language", "en")))
    with st.expander("Multi-speaker settings"):
        spk_cooldown = st.slider("Per-speaker cooldown (s)", 0, 120, int(current.get("speaker_cooldown_seconds", 20)))
        min_ident = st.slider("Min. speech to identify (s)", 0.3, 5.0, float(current.get("min_identify_seconds", 1.0)), 0.1)
        max_spk = st.slider("Max simultaneous speakers", 1, 6, int(current.get("max_simultaneous_speakers", 3)))
        unknown_opts = ["report", "log_only", "ignore"]
        unknown_mode = st.selectbox(
            "Unknown speakers", unknown_opts,
            index=unknown_opts.index(current.get("unknown_speaker_handling", "report")),
            help="report: show and warn - log_only: show, never warn because of them - ignore: drop them",
        )
        st.caption("Overlap sensitivity applies when set: CNAI_OVERLAP_SENSITIVITY in .env.")
    if st.button("Apply settings", use_container_width=True):
        r = api_call(
            "PUT", "/settings", timeout=15,
            json={"threshold_db": threshold, "cooldown_seconds": cooldown,
                  "speaker_match_threshold": match, "language": language,
                  "speaker_cooldown_seconds": spk_cooldown, "min_identify_seconds": min_ident,
                  "max_simultaneous_speakers": max_spk, "unknown_speaker_handling": unknown_mode},
        )
        if r is not None and r.ok:
            st.success("Settings applied")
        elif r is not None:
            st.error(error_detail(r))

tab_register, tab_live, tab_alerts, tab_logs = st.tabs(
    ["Student Registration", "Live Monitor", "Recent Alerts", "Analytics and Logs"]
)


# ------------------------------------------------------ Tab 1: registration + profiles
def render_profile(student_id: str) -> None:
    """Profile of one student: info, editable fields, voice samples, add sample, delete."""
    profile = api_get(f"/students/{student_id}")
    if not profile:
        st.warning("Could not load this profile.")
        return
    st.markdown(f"#### {profile['name']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Voice samples", profile["voice_samples"], help=f"At least {profile['min_samples_required']} needed")
    m2.metric("Recognition", STATUS_BADGE.get(profile["recognition_status"], profile["recognition_status"]))
    m3.metric("Registered", str(profile["created_at"])[:10])
    m4.metric("Code", profile.get("student_code") or "-")

    with st.form(f"edit_{student_id}"):
        e1, e2, e3 = st.columns(3)
        new_name = e1.text_input("Full name", profile["name"])
        new_age = e2.number_input("Age", 3, 99, int(profile.get("age") or 12))
        new_code = e3.text_input("Student code", profile.get("student_code") or "")
        g1, g2 = st.columns([1, 2])
        new_group = g1.text_input("Class / group", profile.get("group_name") or "")
        new_notes = g2.text_area("Notes", profile.get("notes") or "", height=80)
        if st.form_submit_button("Save changes"):
            r = api_call("PATCH", f"/students/{student_id}", timeout=30, json={
                "name": new_name, "age": int(new_age), "student_code": new_code,
                "group_name": new_group, "notes": new_notes})
            if r is not None and r.ok:
                st.success("Profile updated.")
            elif r is not None:
                st.error(error_detail(r))

    st.markdown("**Registered voice samples**")
    samples = profile.get("samples", [])
    if not samples:
        st.caption("No voice sample yet.")
    for sample in samples:
        c1, c2, c3 = st.columns([3, 2, 1])
        c1.caption(f"Sample #{sample['id']} - {str(sample['created_at'])[:19]} - "
                   f"{sample.get('duration_seconds') or '?'} s")
        if sample["has_audio"]:
            audio = api_call("GET", f"/students/{student_id}/voice/{sample['id']}/audio", timeout=15)
            if audio is not None and audio.ok:
                c2.audio(audio.content, format="audio/wav")
        if c3.button("Delete", key=f"del_sample_{sample['id']}"):
            r = api_call("DELETE", f"/students/{student_id}/voice/{sample['id']}", timeout=30)
            if r is not None and r.status_code == 204:
                st.success("Sample deleted - reselect the student to refresh.")
            elif r is not None:
                st.error(error_detail(r))

    st.markdown("**Add another voice sample**")
    st.caption("Several samples (different sentences, a few seconds each) make recognition more robust.")
    more = st.audio_input("Record an additional sample", key=f"more_{student_id}")
    force = st.checkbox("Add even if it does not resemble this student's existing samples",
                        key=f"force_{student_id}")
    if st.button("Add sample", key=f"add_{student_id}", disabled=more is None):
        with st.spinner("Computing the embedding..."):
            r = api_call("POST", f"/students/{student_id}/voice", data={"force": str(force).lower()},
                         files={"audio": ("sample.wav", more.getvalue(), "audio/wav")})
        if r is not None and r.ok:
            st.success("Voice sample added.")
        elif r is not None:
            st.error(error_detail(r))

    with st.expander("Delete this student"):
        st.warning("Deletes the profile, every voice recording and embedding, and removes the "
                   "student's identity from stored alerts. This cannot be undone.")
        confirm = st.checkbox(f"Yes, permanently delete {profile['name']}", key=f"confirm_del_{student_id}")
        if st.button("Delete student", key=f"delete_{student_id}", disabled=not confirm, type="primary"):
            r = api_call("DELETE", f"/students/{student_id}", timeout=30)
            if r is not None and r.status_code == 204:
                st.success("Student deleted. Reload the list to update it.")
            elif r is not None:
                st.error(error_detail(r))


with tab_register:
    st.header("Student Registration")
    st.write("Record about 3 seconds of the student's natural voice in a quiet place, then save.")
    st.info(f"Suggested sentence: {ENROLL_SENTENCE}")
    st.caption("Voice samples and embeddings are stored locally in this installation only. "
               "Speaker recognition is an estimate, not proof of identity.")

    f1, f2, f3 = st.columns(3)
    name = f1.text_input("Student name")
    age = f2.number_input("Age", min_value=3, max_value=99, value=12, step=1)
    code = f3.text_input("Student code (optional)")
    f4, f5 = st.columns([1, 2])
    group = f4.text_input("Class / group (optional)")
    notes = f5.text_input("Notes (optional)")
    allow_dup = st.checkbox("A student with the same name already exists - register anyway")
    recording = st.audio_input("Record voice sample")

    if st.button("Compute embedding and save", type="primary"):
        if not name.strip():
            st.error("Please enter the student name.")
        elif recording is None:
            st.error("Please record a voice sample first.")
        else:
            data = recording.getvalue()
            try:
                duration = wav_duration(data)
            except Exception:
                duration = 0.0
            if duration and duration < 1.5:
                st.error(f"Recording too short ({duration:.1f}s). Please record at least 2 seconds.")
            else:
                with st.spinner("Computing the 192-D ECAPA-TDNN embedding..."):
                    r = api_call(
                        "POST", "/students/register",
                        data={"name": name.strip(), "age": int(age), "student_code": code.strip(),
                              "group_name": group.strip(), "notes": notes.strip(),
                              "allow_duplicate_name": str(allow_dup).lower()},
                        files={"audio": ("sample.wav", data, "audio/wav")},
                    )
                if r is not None and r.status_code == 201:
                    st.success(f"Student '{name.strip()}' registered.")
                elif r is not None:
                    st.error(error_detail(r))

    with st.expander("Test my voice (why am I not recognised?)"):
        st.caption("Records a few seconds and shows the raw similarity to every registered profile. Nothing is saved.")
        probe = st.audio_input("Record a test sample", key="diag_audio")
        if probe is not None and st.button("Compare with registered voices", key="diag_btn"):
            r = api_call("POST", "/students/diagnose", timeout=120,
                         files={"audio": ("probe.wav", probe.getvalue(), "audio/wav")})
            if r is not None and r.ok:
                out = r.json()
                st.dataframe(pd.DataFrame(out["results"])[["name", "similarity", "samples", "passes_threshold"]],
                             use_container_width=True, hide_index=True)
                st.info(f"Current threshold: {out['threshold']:.2f}. {out['hint']}")
            elif r is not None:
                st.error(error_detail(r))

    st.subheader("Registered students")
    students = api_get("/students") or []
    if students:
        sdf = pd.DataFrame(students)
        sdf["recognition"] = sdf["recognition_status"].map(lambda s: STATUS_BADGE.get(s, s))
        st.dataframe(
            sdf[["name", "student_code", "group_name", "age", "voice_samples", "recognition", "created_at"]],
            use_container_width=True, hide_index=True,
        )
        target = st.selectbox("Open a student profile", options=students, format_func=lambda s: s["name"],
                              key="profile_sel")
        if target:
            render_profile(target["id"])
    else:
        st.caption("No students registered yet.")


# --------------------------------------------------------------- Tab 2: live
with tab_live:
    st.header("Live Monitor")
    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("Start monitoring", type="primary", use_container_width=True):
        with st.spinner("Loading models and opening the microphone (first run downloads models)..."):
            r = api_call("POST", "/monitor/start")
        if r is not None and not r.ok:
            st.error(error_detail(r))
    if c2.button("Stop monitoring", use_container_width=True):
        api_call("POST", "/monitor/stop", timeout=30)

    if "db_history" not in st.session_state:
        st.session_state.db_history = []

    @st.fragment(run_every=1.0)
    def live_panel() -> None:
        status = api_get("/status", timeout=5)
        if not status:
            st.warning("Backend not reachable: showing nothing instead of stale data. Retrying...")
            return
        db = float(status["live_db"])
        thr = float(status["threshold_db"])
        hist: List[float] = st.session_state.db_history
        hist.append(db)
        del hist[:-120]

        left, right = st.columns([1, 1])
        with left:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=db,
                    number={"suffix": " dB"},
                    title={"text": "Classroom noise level"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#c0392b" if db >= thr else "#2e86c1"},
                        "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.8, "value": thr},
                    },
                )
            )
            gauge.update_layout(height=260, margin=dict(l=20, r=20, t=60, b=10))
            st.plotly_chart(gauge, use_container_width=True)
        with right:
            label = status["ui_state"]
            if label.startswith("Warning"):
                st.error(f"State: {label}")
            elif label in ("Analyzing", "Loading models"):
                st.warning(f"State: {label}")
            else:
                st.info(f"State: {label}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Threshold", f"{thr:.0f} dB")
            m2.metric("Cooldown", f"{status['cooldown_remaining']:.0f} s" if status["in_cooldown"] else "Ready")
            m3.metric("Registered voices", status["registered_voices"])
            if status.get("running") and not status.get("mic_alive", True):
                st.error("No audio received from the microphone. Check the input device (python main.py devices).")
            elif status.get("mic_silent", False):
                st.error("The microphone delivers digital silence (0 dB). Check the Windows microphone privacy "
                         "setting, the muted state, or the input device index.")
            s1, s2 = st.columns(2)
            s1.metric("Speech (VAD)", "Detected" if status.get("speech_detected") else "None")
            s2.metric("Speech ratio", f"{status.get('speech_ratio', 0.0):.0%}")
            st.caption(f"Overlap detector: {status.get('osd_backend') or 'not loaded'}")
            age = time.time() - float(status.get("server_time", time.time()))
            if abs(age) < 60:  # same host clock: flag a status that is not fresh
                st.caption(f"Live data age: {max(age, 0):.1f} s")
            caps = status.get("capabilities") or {}
            if caps:
                if caps.get("source_separation"):
                    st.caption("Overlapping voices: source separation active (isolated audio per speaker).")
                elif caps.get("diarization"):
                    st.caption("Overlapping voices: diarization active - speaker activity and mixed-signal "
                               "views, no isolated audio.")
                else:
                    st.caption("Overlapping voices: embedding window scan only - a speaker is named only if "
                               "they dominate part of the window. No isolated audio.")
            if status.get("separation_error"):
                st.warning(f"Source separation unavailable: {status['separation_error']}")
            if status.get("error"):
                st.error(status["error"])
            st.line_chart(pd.DataFrame({"dB": hist, "Threshold": [thr] * len(hist)}), height=140)

        with st.expander("Monitoring and model health"):
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Models loaded", "yes" if status.get("models_loaded") else "no")
            h2.metric("Microphone", "alive" if status.get("mic_alive") else "no audio")
            h3.metric("Dropped windows", status.get("dropped_windows", 0))
            h4.metric("Processing errors", status.get("processing_errors", 0))
            lat = status.get("latency") or {}
            rows = [{"stage": k, "median ms": v["median_ms"], "p95 ms": v["p95_ms"], "n": v["count"]}
                    for k, v in lat.items() if isinstance(v, dict) and "median_ms" in v]
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            rtf = lat.get("real_time_factor")
            if rtf:
                st.caption(f"Real-time factor (processing time / hop): median {rtf['median']}, p95 {rtf['p95']} "
                           "(must stay below 1.0 to keep up with the microphone).")

        # ---- current detected speakers
        current_event = status.get("current_event")
        st.subheader("Current speakers")
        if current_event and current_event.get("speakers"):
            cols = st.columns(min(len(current_event["speakers"]), 4))
            for col, sp in zip(cols, current_event["speakers"]):
                col.markdown(
                    f"<div style='border-left:6px solid {ui.speaker_color(sp)};padding-left:8px'>"
                    f"<b>{ui.track_label(sp)}</b><br>"
                    f"<small>{ui.speaker_line(sp).split(' - ', 1)[1]}"
                    f"</small></div>", unsafe_allow_html=True)
            if current_event.get("is_overlap"):
                st.caption(ui.overlap_text(current_event))
            st.caption(ui.SIMILARITY_HELP)
        else:
            st.caption("Nobody above the loudness threshold right now.")

        # ---- waveform + speaker timeline
        viz = api_get("/live/visualization", timeout=5) or {}
        events = viz.get("events", [])
        history = float(viz.get("history_seconds") or 60)
        now = ui.parse_dt(viz["now"]) if viz.get("now") else datetime.now()
        window = (now - timedelta(seconds=min(history, 60)), now)
        st.plotly_chart(ui.build_mic_waveform_figure(viz.get("waveform", {}), ui.overlap_spans(events), window),
                        use_container_width=True, key="mic_wave")
        st.plotly_chart(ui.build_timeline_figure(events, window), use_container_width=True, key="speaker_timeline")
        wf_fig = ui.build_speaker_waveforms_figure(events, window)
        if wf_fig is not None:
            with st.expander("Per-speaker waveforms", expanded=False):
                st.caption("Rows labelled 'separated audio' are isolated signals. Rows labelled 'mixed "
                           "microphone signal' show the combined audio during that speaker's activity - "
                           "they are NOT isolated voices.")
                st.plotly_chart(wf_fig, use_container_width=True, key="speaker_waves")

    live_panel()


# ------------------------------------------------------------ Tab 3: recent alerts
with tab_alerts:
    st.header("Recent Alerts")
    st.caption("Consecutive detections are grouped into one alert while the noise continues. "
               "Similarity is a cosine score, not a probability - treat identities as estimates.")
    shown = st.slider("Alerts to show", 5, 50, 15, key="alerts_n")
    cat_filter = st.multiselect("Event types", ["Single Speaker", "Multiple Speakers", "Unknown Speaker",
                                                "Overlapping Speech"],
                                default=["Single Speaker", "Multiple Speakers", "Unknown Speaker", "Overlapping Speech"],
                                key="alert_cats")
    st.caption(ui.SIMILARITY_HELP)

    @st.fragment(run_every=3.0)
    def alerts_panel() -> None:
        events = api_get("/events", params={"limit": 100}, timeout=10)
        if events is None:
            st.warning("Backend not reachable.")
            return
        events = [e for e in events if set(e.get("labels", [e.get("category")])) & set(cat_filter)][:shown]
        if not events:
            st.caption("No alerts for this selection.")
            return
        for event in events:
            vm = ui.alert_view_model(event)
            with st.container(border=True):
                top, side = st.columns([3, 1])
                top.markdown(f"**{vm['title']}**")
                top.caption(" | ".join(vm["badges"]))
                for line in vm["speaker_lines"]:
                    top.markdown(f"- {line}")
                side.markdown(f"`{vm['time_text']}`")
                side.caption(vm["overlap_text"])
                side.caption(vm["status_text"])
                side.caption(vm["warning_text"])
                side.caption(f"Decision: {vm['decision_text']}")
                with st.expander("Details", expanded=False):
                    st.dataframe(pd.DataFrame(vm["speaker_rows"]), use_container_width=True, hide_index=True)
                    st.caption(f"Identification method: {vm['method_text']} - {vm['detector_text']} - {vm['latency_text']}")
                    for note in vm["notes"]:
                        st.caption(f"Note: {note}")
                    st.plotly_chart(ui.build_timeline_figure([event]), use_container_width=True,
                                    key=f"tl_{vm['event_id']}")
                    fig = ui.build_speaker_waveforms_figure([event])
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True, key=f"wf_{vm['event_id']}")

    alerts_panel()


# ------------------------------------------------------------ Tab 4: logs
with tab_logs:
    st.header("Analytics and Logs")
    a1, a2 = st.columns([2, 2])
    today = date.today()
    rng = a1.date_input("Date range", value=(today - timedelta(days=7), today))
    types = a2.multiselect("Event types", EVENT_TYPES, default=EVENT_TYPES)
    t1, t2 = st.columns(2)
    t_from = t1.time_input("From time", value=dtime(0, 0))
    t_to = t2.time_input("To time", value=dtime(23, 59))

    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        start_dt = datetime.combine(rng[0], t_from)
        end_dt = datetime.combine(rng[1], t_to)
        params = {"start": start_dt.isoformat(), "end": end_dt.isoformat(), "event_type": types, "limit": 5000}
        logs = api_get("/logs", params=params, timeout=30)
        if logs:
            df = pd.DataFrame(logs)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            k1, k2, k3 = st.columns(3)
            k1.metric("Interventions", len(df))
            k2.metric("Spoken", int(df["was_spoken"].sum()))
            k3.metric("Average level", f"{df['dB_level'].mean():.1f} dB")
            st.bar_chart(df["event_type"].value_counts())
            columns = ["timestamp", "event_type", "category", "student_name", "speaker_count",
                       "overlap_duration", "dB_level", "match_score", "was_spoken", "message_spoken"]
            st.dataframe(df[[c for c in columns if c in df.columns]], use_container_width=True, hide_index=True)
            export = df.drop(columns=[c for c in ("speakers", "notes") if c in df.columns])
            st.download_button("Download CSV", export.to_csv(index=False).encode("utf-8"), "interventions.csv", "text/csv")
        else:
            st.caption("No interventions for this selection.")
    else:
        st.caption("Select an end date to load the logs.")
