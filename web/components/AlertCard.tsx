"use client";
import type { DetectionEvent } from "@/lib/types";
import { METHOD_LABELS, DECISION_LABELS, alertRows, eventHeadline, hms, ms, overlapText, speakerLine } from "@/lib/format";
import { SpeakerWaveforms, Timeline } from "./Charts";

export default function AlertCard({ event }: { event: DetectionEvent }) {
  const rows = alertRows(event);
  return (
    <div className="card">
      <div className="row" style={{ alignItems: "flex-start" }}>
        <div style={{ flex: 3, minWidth: 260 }}>
          <strong>{eventHeadline(event)}</strong>
          <div style={{ margin: "4px 0" }}>{(event.labels?.length ? event.labels : [event.category ?? ""]).map((b) => <span className="badge" key={b}>{b}</span>)}</div>
          <ul style={{ margin: 0, paddingLeft: 18 }}>{event.speakers.map((s, i) => <li key={i}>{speakerLine(s)}</li>)}</ul>
        </div>
        <div style={{ flex: 1, minWidth: 200 }} className="small">
          <code>{hms(ms(event.start_time))} - {hms(ms(event.end_time))}</code><br />
          {overlapText(event)}<br />
          Detection status: {event.status ?? "completed"}<br />
          {event.warning_triggered ? "Warning spoken" : "No warning spoken (cooldown or unknown-speaker policy)"}<br />
          Decision: {DECISION_LABELS[event.decision_status ?? ""] ?? event.decision_status ?? "n/a"}
        </div>
      </div>
      <details>
        <summary>Details</summary>
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead><tr><th>Speaker</th><th>Student</th><th>Decision</th><th>Why not identified</th><th>Similarity (cosine)</th><th>Runner-up</th><th>Start</th><th>End</th><th>Signal</th></tr></thead>
            <tbody>{rows.map((r, i) => <tr key={i}><td>{r.speaker}</td><td>{r.student}</td><td>{r.decision}</td><td>{r.why}</td><td>{r.similarity}</td><td>{r.runnerUp}</td><td>{r.start}</td><td>{r.end}</td><td>{r.signal}</td></tr>)}</tbody>
          </table>
        </div>
        <p className="small">
          Identification method: {METHOD_LABELS[event.identification_method ?? ""] ?? event.identification_method ?? "n/a"} - overlap flagged by: {event.detection_method ?? "n/a"} -{" "}
          {event.processing_latency_ms == null ? "latency: n/a" : `latency: ${event.processing_latency_ms.toFixed(0)} ms`}
        </p>
        {(event.notes ?? []).map((n, i) => <p className="small" key={i}>Note: {n}</p>)}
        <Timeline events={[event]} />
        <SpeakerWaveforms events={[event]} />
      </details>
    </div>
  );
}
