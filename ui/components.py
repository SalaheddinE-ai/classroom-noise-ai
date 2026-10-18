"""Pure helpers and Plotly figure builders for the Streamlit dashboard.

Everything that turns backend events into rows, labels or figures lives here so it can be tested
without Streamlit. ``plotly`` is imported lazily inside the figure builders.

Honest labelling rules (see README, "Diarization vs identification vs separation"):

* ``separated``      -> "separated audio": a genuinely isolated waveform of that speaker.
* ``masked_mixture`` -> "mixed microphone signal during this speaker's activity": NOT isolated audio.
* ``activity_only``  -> timeline bars only: we know *when* the speaker was active, not their audio.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

UNKNOWN_LABEL = "Unknown Speaker"

# Colour-blind-friendly categorical palette; a speaker keeps the same colour in every chart.
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#9D755D", "#EECA3B"]
UNKNOWN_COLORS = ["#8D939B", "#B0B5BC", "#6B7078", "#C9CDD2"]
OVERLAP_FILL = "rgba(214, 69, 65, 0.16)"

SIGNAL_LABELS = {
    "separated": "separated audio",
    "masked_mixture": "mixed microphone signal during this speaker's activity (not isolated audio)",
    "activity_only": "activity timeline only (no audio signal)",
}
METHOD_LABELS = {
    "single_stream": "single stream (no overlap)",
    "sepformer": "sepformer - source separation (estimated sources)",
    "pyannote_slots": "pyannote_slots - speaker activity + clean-region embeddings (not isolated audio)",
    "mixture_embedding": "mixture_embedding - approximate fallback on the mixed signal",
}
SIMILARITY_HELP = ("Similarity is the cosine similarity between voice embeddings. It is NOT a probability or a "
                   "confidence percentage; a higher value only means a closer match to the stored profile.")
DECISION_LABELS = {"confirmed": "confirmed", "tentative": "tentative (not enough evidence yet)",
                   "unknown": "unknown", "mixed": "mixed"}
STATUS_LABELS = {
    "matched": "identified",
    "below_threshold": "similarity below threshold",
    "ambiguous": "ambiguous between students",
    "taken": "best match already attributed to another speaker",
    "no_students": "no student registered",
    "no_embedding": "segment too short to identify",
    "unresolved_overlap": "overlapping voice that could not be resolved",
    "deleted_profile": "profile deleted",
    "unidentified": "not identified",
}


# ----------------------------------------------------------------------------- basics
def parse_dt(value: Any) -> datetime:
    """ISO-8601 string (or datetime) -> naive datetime in the dashboard machine's local time.

    Plotly draws dates without time zones; converting first keeps every chart on the same clock.
    """
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


def speaker_key(speaker: Dict[str, Any]) -> str:
    """Stable identity of a speaker across events (student id, else name, else per-event id)."""
    if speaker.get("decision_status", "confirmed" if speaker.get("is_identified") else "unknown") != "unknown" \
            and (speaker.get("student_id") or speaker.get("student_name")):
        return f"student:{speaker.get('student_id') or speaker['student_name']}"
    return f"unknown:{speaker.get('speaker_id', 'S?')}"


def speaker_color(speaker: Dict[str, Any]) -> str:
    """Consistent colour: hash of the student identity; unknown voices share the grey family."""
    key = speaker_key(speaker)
    if key.startswith("unknown:"):
        digits = "".join(ch for ch in key if ch.isdigit()) or "0"
        return UNKNOWN_COLORS[int(digits) % len(UNKNOWN_COLORS)]
    return PALETTE[int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % len(PALETTE)]


def track_label(speaker: Dict[str, Any]) -> str:
    """Row label: the student's name, or ``Unknown Speaker`` (numbered when several are present)."""
    if speaker.get("is_identified"):
        return speaker.get("student_name") or UNKNOWN_LABEL
    if speaker.get("decision_status") == "tentative" and speaker.get("student_name"):
        return f"{speaker['student_name']} (tentative)"
    return f"{UNKNOWN_LABEL} ({speaker.get('speaker_id', '?')})"


def fmt_similarity(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def overlap_text(event: Dict[str, Any]) -> str:
    """Overlap duration, or an explicit statement when it cannot be measured (never a fake zero)."""
    if not event.get("is_overlap"):
        return "No overlap detected"
    ms = event.get("overlap_duration_ms")
    estimate = " (approximate)" if (event.get("metadata") or {}).get("overlap_estimate") == "approximate" else ""
    if ms is None:
        return "Overlap detected - duration not measurable with the active method"
    return f"Overlap duration: {ms / 1000:.1f} s{estimate}"


# ---------------------------------------------------------------------- alert view model
def event_headline(event: Dict[str, Any]) -> str:
    n = event.get("speaker_count", len(event.get("speakers", [])))
    if event.get("is_overlap") and n >= 2:
        return "Multiple speakers detected (overlapping speech)"
    if n >= 2:
        return "Multiple speakers detected"
    if event.get("is_overlap"):
        return "Overlapping speech detected"
    speakers = event.get("speakers", [])
    if speakers and not speakers[0].get("is_identified"):
        return "Unknown speaker detected"
    return "Single speaker detected"


def speaker_line(speaker: Dict[str, Any]) -> str:
    """One readable line per speaker, built only from measured values."""
    if speaker.get("is_identified"):
        return f"{speaker['student_name']} - identified, similarity: {fmt_similarity(speaker.get('similarity_score'))}"
    if speaker.get("decision_status") == "tentative":
        return (f"{speaker.get('student_name')} - tentative match, similarity: "
                f"{fmt_similarity(speaker.get('similarity_score'))} (not confirmed yet)")
    reason = STATUS_LABELS.get(speaker.get("rejection_reason") or speaker.get("identification_status", "unidentified"),
                               "not identified")
    best = speaker.get("similarity_score")
    extra = f", best similarity: {fmt_similarity(best)}" if best is not None else ""
    return f"{track_label(speaker)} - not identified ({reason}){extra}"


def alert_view_model(event: Dict[str, Any]) -> Dict[str, Any]:
    """Everything the Recent Alerts card displays, as plain data."""
    start, end = parse_dt(event["start_time"]), parse_dt(event["end_time"])
    speakers = event.get("speakers", [])
    warning = ("Warning spoken" if event.get("warning_triggered")
               else "No warning spoken (cooldown or unknown-speaker policy)")
    return {
        "event_id": event["event_id"],
        "title": event_headline(event),
        "badges": list(event.get("labels", [])) or [event.get("category", "")],
        "speaker_lines": [speaker_line(s) for s in speakers],
        "time_text": f"{start:%H:%M:%S} - {end:%H:%M:%S}",
        "overlap_text": overlap_text(event),
        "status_text": f"Detection status: {event.get('status', 'completed')}",
        "warning_text": warning,
        "method_text": METHOD_LABELS.get(event.get("identification_method", ""),
                                         event.get("identification_method", "")),
        "detector_text": f"overlap flagged by: {event.get('detection_method', 'n/a')}",
        "decision_text": DECISION_LABELS.get(event.get("decision_status", ""), event.get("decision_status", "")),
        "latency_text": ("latency: n/a" if event.get("processing_latency_ms") is None
                         else f"latency: {event['processing_latency_ms']:.0f} ms"),
        "notes": list(event.get("notes", [])),
        "speaker_rows": [
            {
                "Speaker": track_label(s),
                "Student": s.get("student_name") if s.get("is_identified") else "-",
                "Decision": s.get("decision_status", "unknown"),
                "Why not identified": STATUS_LABELS.get(s.get("rejection_reason") or "", "-") if s.get("rejection_reason") else "-",
                "Similarity (cosine)": fmt_similarity(s.get("similarity_score")),
                "Runner-up": fmt_similarity(s.get("runner_up_similarity")),
                "Start": f"{parse_dt(s['start_time']):%H:%M:%S}",
                "End": f"{parse_dt(s['end_time']):%H:%M:%S}",
                "Signal": SIGNAL_LABELS.get(s.get("signal_type", "activity_only"), s.get("signal_type", "")),
            }
            for s in speakers
        ],
    }


# --------------------------------------------------------------------- timeline data
def track_order(events: Sequence[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """Ordered ``(row label, speaker key, colour)`` of every speaker seen: students first (A-Z), unknown last."""
    seen: Dict[str, Tuple[str, str, str]] = {}
    for event in events:
        for sp in event.get("speakers", []):
            key = speaker_key(sp)
            if key not in seen:
                seen[key] = (track_label(sp), key, speaker_color(sp))
    # an unknown voice is only unique within its event: disambiguate labels across events
    labels: Dict[str, int] = {}
    ordered = sorted(seen.values(), key=lambda t: (t[1].startswith("unknown:"), t[0].lower()))
    out = []
    for label, key, color in ordered:
        labels[label] = labels.get(label, 0) + 1
        out.append((label if labels[label] == 1 else f"{label} #{labels[label]}", key, color))
    return out


def timeline_rows(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per speech segment: ``track, start, end, color, similarity, event_id``."""
    labels = {key: label for label, key, _ in track_order(events)}
    rows: List[Dict[str, Any]] = []
    for event in events:
        for sp in event.get("speakers", []):
            key = speaker_key(sp)
            segments = sp.get("segments") or [{"start": sp["start_time"], "end": sp["end_time"]}]
            for seg in segments:
                rows.append({
                    "track": labels[key], "key": key, "start": parse_dt(seg["start"]), "end": parse_dt(seg["end"]),
                    "color": speaker_color(sp), "similarity": sp.get("similarity_score"),
                    "event_id": event["event_id"], "signal_type": sp.get("signal_type", "activity_only"),
                })
    return rows


def overlap_spans(events: Sequence[Dict[str, Any]]) -> List[Tuple[datetime, datetime]]:
    return [(parse_dt(s["start"]), parse_dt(s["end"]))
            for e in events for s in e.get("overlap_segments", [])]


def waveform_xy(waveform: Dict[str, Any]) -> Tuple[List[datetime], List[float]]:
    """Dense backend series (``t0_epoch`` + fixed step + peaks) -> x datetimes, y peaks."""
    peaks = waveform.get("peaks") or []
    t0 = waveform.get("t0_epoch")
    if t0 is None or not peaks:
        return [], []
    dt = float(waveform["dt_seconds"])
    return [datetime.fromtimestamp(t0 + i * dt) for i in range(len(peaks))], list(peaks)


def speaker_waveform_tracks(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-speaker waveform series (merged over events) with an honest ``signal_type``."""
    labels = {key: label for label, key, _ in track_order(events)}
    tracks: Dict[str, Dict[str, Any]] = {}
    for event in events:
        for sp in event.get("speakers", []):
            wf = sp.get("waveform")
            if not wf or not wf.get("peaks"):
                continue
            key = speaker_key(sp)
            t0 = parse_dt(wf["t0"]).timestamp()
            track = tracks.setdefault(key, {"label": labels[key], "color": speaker_color(sp),
                                            "signal_type": sp.get("signal_type"), "points": {}})
            if sp.get("signal_type") == "separated":
                track["signal_type"] = "separated"
            step = float(wf["dt_seconds"])
            for i, value in enumerate(wf["peaks"]):
                track["points"][round(t0 + i * step, 3)] = value
    result = []
    for track in tracks.values():
        times = sorted(track["points"])
        result.append({"label": track["label"], "color": track["color"], "signal_type": track["signal_type"],
                       "x": [datetime.fromtimestamp(t) for t in times], "y": [track["points"][t] for t in times]})
    return result


# ------------------------------------------------------------------------------ figures
def _base_layout(fig, height: int, title: Optional[str] = None) -> None:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=36 if title else 10, b=10), title=title,
                      showlegend=False, bargap=0.35)


def build_mic_waveform_figure(waveform: Dict[str, Any], overlaps: Sequence[Tuple[datetime, datetime]] = (),
                              window: Optional[Tuple[datetime, datetime]] = None):
    """Combined microphone waveform (peak envelope), with overlap spans shaded."""
    import plotly.graph_objects as go

    xs, ys = waveform_xy(waveform)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=0.5, color="#4C78A8"),
                             fill="tozeroy", fillcolor="rgba(76,120,168,0.45)", name="mic", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[-v for v in ys], mode="lines", line=dict(width=0.5, color="#4C78A8"),
                             fill="tozeroy", fillcolor="rgba(76,120,168,0.45)", name="mic", hoverinfo="skip"))
    for a, b in overlaps:
        fig.add_shape(type="rect", xref="x", yref="paper", x0=a, x1=b, y0=0, y1=1, fillcolor=OVERLAP_FILL, line_width=0)
    fig.update_yaxes(range=[-1, 1], title="amplitude", zeroline=True)
    if window:
        fig.update_xaxes(range=list(window))
    _base_layout(fig, 200, "Combined microphone waveform (all voices mixed)")
    return fig


def build_timeline_figure(events: Sequence[Dict[str, Any]], window: Optional[Tuple[datetime, datetime]] = None,
                          height: Optional[int] = None):
    """One horizontal track per speaker on a shared time axis; overlap spans shaded."""
    import plotly.graph_objects as go

    rows = timeline_rows(events)
    order = track_order(events)
    fig = go.Figure()
    for label, key, color in order:
        mine = [r for r in rows if r["key"] == key]
        fig.add_trace(go.Bar(
            base=[r["start"] for r in mine], x=[(r["end"] - r["start"]).total_seconds() * 1000 for r in mine],
            y=[label] * len(mine), orientation="h", marker_color=color,
            customdata=[[f"{r['start']:%H:%M:%S}", f"{r['end']:%H:%M:%S}", fmt_similarity(r["similarity"])] for r in mine],
            hovertemplate="%{y}<br>%{customdata[0]} - %{customdata[1]}<br>similarity: %{customdata[2]}<extra></extra>",
            name=label,
        ))
    for a, b in overlap_spans(events):
        fig.add_shape(type="rect", xref="x", yref="paper", x0=a, x1=b, y0=0, y1=1, fillcolor=OVERLAP_FILL, line_width=0)
    fig.update_xaxes(type="date", range=list(window) if window else None)
    fig.update_yaxes(categoryorder="array", categoryarray=[t[0] for t in order][::-1], autorange=True)
    if not order:
        fig.add_annotation(text="No speech detected yet", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
    _base_layout(fig, height or max(180, 60 + 46 * len(order)), "Speaker activity timeline (shaded = overlapping speech)")
    return fig


def build_speaker_waveforms_figure(events: Sequence[Dict[str, Any]], window: Optional[Tuple[datetime, datetime]] = None):
    """One waveform row per speaker that has an audio-derived series; rows are labelled by signal type."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    tracks = speaker_waveform_tracks(events)
    if not tracks:
        return None
    titles = [f"{t['label']} - {SIGNAL_LABELS.get(t['signal_type'], t['signal_type'])}" for t in tracks]
    fig = make_subplots(rows=len(tracks), cols=1, shared_xaxes=True, subplot_titles=titles, vertical_spacing=0.12)
    for i, track in enumerate(tracks, start=1):
        for sign in (1, -1):
            fig.add_trace(go.Scatter(x=track["x"], y=[sign * v for v in track["y"]], mode="lines",
                                     line=dict(width=0.5, color=track["color"]), fill="tozeroy",
                                     hoverinfo="skip"), row=i, col=1)
        fig.update_yaxes(range=[-1, 1], row=i, col=1, showticklabels=False)
    if window:
        fig.update_xaxes(range=list(window))
    _base_layout(fig, 130 * len(tracks) + 30)
    return fig
