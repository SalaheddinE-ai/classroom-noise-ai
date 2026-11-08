// Port of ui/components.py: pure helpers turning backend events into labels, rows and chart data.
// Honest labelling: "separated" = isolated audio; "masked_mixture" = mixed signal; "activity_only" = timeline only.
import type { DetectionEvent, SpeakerEntry, Waveform } from "./types";

export const UNKNOWN_LABEL = "Unknown Speaker";
export const PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#9D755D", "#EECA3B"];
export const UNKNOWN_COLORS = ["#8D939B", "#B0B5BC", "#6B7078", "#C9CDD2"];
export const OVERLAP_FILL = "rgba(214, 69, 65, 0.18)";

export const SIGNAL_LABELS: Record<string, string> = {
  separated: "separated audio",
  masked_mixture: "mixed microphone signal during this speaker's activity (not isolated audio)",
  activity_only: "activity timeline only (no audio signal)",
};
export const METHOD_LABELS: Record<string, string> = {
  single_stream: "single stream (no overlap)",
  sepformer: "sepformer - source separation (estimated sources)",
  pyannote_slots: "pyannote_slots - speaker activity + clean-region embeddings (not isolated audio)",
  mixture_embedding: "mixture_embedding - approximate fallback on the mixed signal",
};
export const SIMILARITY_HELP =
  "Similarity is the cosine similarity between voice embeddings. It is NOT a probability or a confidence percentage; a higher value only means a closer match to the stored profile.";
export const DECISION_LABELS: Record<string, string> = {
  confirmed: "confirmed", tentative: "tentative (not enough evidence yet)", unknown: "unknown", mixed: "mixed",
};
export const STATUS_LABELS: Record<string, string> = {
  matched: "identified", below_threshold: "similarity below threshold", ambiguous: "ambiguous between students",
  taken: "best match already attributed to another speaker", no_students: "no student registered",
  no_embedding: "segment too short to identify", unresolved_overlap: "overlapping voice that could not be resolved",
  deleted_profile: "profile deleted", unidentified: "not identified",
};
export const RECOGNITION_LABELS: Record<string, string> = { ready: "Ready", needs_samples: "Needs more samples", no_samples: "No voice sample" };

// ------------------------------------------------------------------ time
/** ISO string -> epoch ms. Strings without a zone are the server's local time (same machine in practice). */
export const ms = (iso: string): number => new Date(iso).getTime();
export const hms = (t: number): string => new Date(t).toLocaleTimeString([], { hour12: false });
export const fmtSimilarity = (v?: number | null): string => (v === null || v === undefined ? "n/a" : v.toFixed(2));

// -------------------------------------------------------------- speakers
export function speakerKey(s: SpeakerEntry): string {
  const status = s.decision_status ?? (s.is_identified ? "confirmed" : "unknown");
  if (status !== "unknown" && (s.student_id || s.student_name)) return `student:${s.student_id || s.student_name}`;
  return `unknown:${s.speaker_id ?? "S?"}`;
}
function hash(str: string): number {
  let h = 5381;
  for (let i = 0; i < str.length; i++) h = ((h << 5) + h + str.charCodeAt(i)) >>> 0;
  return h;
}
export function speakerColor(s: SpeakerEntry): string {
  const key = speakerKey(s);
  if (key.startsWith("unknown:")) return UNKNOWN_COLORS[parseInt(key.replace(/\D/g, "") || "0", 10) % UNKNOWN_COLORS.length];
  return PALETTE[hash(key) % PALETTE.length];
}
export function trackLabel(s: SpeakerEntry): string {
  if (s.is_identified) return s.student_name || UNKNOWN_LABEL;
  if (s.decision_status === "tentative" && s.student_name) return `${s.student_name} (tentative)`;
  return `${UNKNOWN_LABEL} (${s.speaker_id ?? "?"})`;
}
export function speakerLine(s: SpeakerEntry): string {
  if (s.is_identified) return `${s.student_name} - identified, similarity: ${fmtSimilarity(s.similarity_score)}`;
  if (s.decision_status === "tentative")
    return `${s.student_name} - tentative match, similarity: ${fmtSimilarity(s.similarity_score)} (not confirmed yet)`;
  const reason = STATUS_LABELS[s.rejection_reason || s.identification_status || "unidentified"] ?? "not identified";
  const extra = s.similarity_score != null ? `, best similarity: ${fmtSimilarity(s.similarity_score)}` : "";
  return `${trackLabel(s)} - not identified (${reason})${extra}`;
}
export function overlapText(e: DetectionEvent): string {
  if (!e.is_overlap) return "No overlap detected";
  if (e.overlap_duration_ms == null) return "Overlap detected - duration not measurable with the active method";
  const approx = e.metadata && e.metadata["overlap_estimate"] === "approximate" ? " (approximate)" : "";
  return `Overlap duration: ${(e.overlap_duration_ms / 1000).toFixed(1)} s${approx}`;
}
export function eventHeadline(e: DetectionEvent): string {
  const n = e.speaker_count ?? e.speakers.length;
  if (e.is_overlap && n >= 2) return "Multiple speakers detected (overlapping speech)";
  if (n >= 2) return "Multiple speakers detected";
  if (e.is_overlap) return "Overlapping speech detected";
  if (e.speakers.length && !e.speakers[0].is_identified) return "Unknown speaker detected";
  return "Single speaker detected";
}

// -------------------------------------------------------------- tracks
export interface Track { label: string; key: string; color: string }
export function trackOrder(events: DetectionEvent[]): Track[] {
  const seen = new Map<string, Track>();
  for (const e of events) for (const sp of e.speakers) {
    const key = speakerKey(sp);
    if (!seen.has(key)) seen.set(key, { label: trackLabel(sp), key, color: speakerColor(sp) });
  }
  const ordered = [...seen.values()].sort((a, b) =>
    Number(a.key.startsWith("unknown:")) - Number(b.key.startsWith("unknown:")) || a.label.toLowerCase().localeCompare(b.label.toLowerCase()));
  const count: Record<string, number> = {};
  return ordered.map((t) => {
    count[t.label] = (count[t.label] ?? 0) + 1;
    return count[t.label] === 1 ? t : { ...t, label: `${t.label} #${count[t.label]}` };
  });
}
export interface Bar { key: string; start: number; end: number; color: string; similarity?: number | null }
export function timelineBars(events: DetectionEvent[]): Bar[] {
  const bars: Bar[] = [];
  for (const e of events) for (const sp of e.speakers) {
    const segs = sp.segments?.length ? sp.segments : [{ start: sp.start_time, end: sp.end_time }];
    for (const g of segs) bars.push({ key: speakerKey(sp), start: ms(g.start), end: ms(g.end), color: speakerColor(sp), similarity: sp.similarity_score });
  }
  return bars;
}
export const overlapSpans = (events: DetectionEvent[]): [number, number][] =>
  events.flatMap((e) => (e.overlap_segments ?? []).map((s) => [ms(s.start), ms(s.end)] as [number, number]));

export function waveformPoints(w: Waveform | undefined | null): { t: number; v: number }[] {
  if (!w || !w.peaks?.length || w.t0_epoch == null) return [];
  return w.peaks.map((v, i) => ({ t: (w.t0_epoch as number + i * w.dt_seconds) * 1000, v }));
}
export interface WaveTrack { label: string; color: string; signal: string; points: { t: number; v: number }[] }
export const MAX_WAVEFORM_ROWS = 6;
export function speakerWaveTracks(events: DetectionEvent[]): WaveTrack[] {
  const labels = new Map(trackOrder(events).map((t) => [t.key, t.label]));
  const tracks = new Map<string, { label: string; color: string; signal: string; pts: Map<number, number> }>();
  for (const e of events) for (const sp of e.speakers) {
    const wf = sp.waveform;
    if (!wf?.peaks?.length || !wf.t0) continue;
    const key = speakerKey(sp);
    let tr = tracks.get(key);
    if (!tr) { tr = { label: labels.get(key) ?? key, color: speakerColor(sp), signal: sp.signal_type ?? "activity_only", pts: new Map() }; tracks.set(key, tr); }
    if (sp.signal_type === "separated") tr.signal = "separated";
    const t0 = ms(wf.t0);
    wf.peaks.forEach((v, i) => tr!.pts.set(Math.round(t0 + i * wf.dt_seconds * 1000), v));
  }
  const out = [...tracks.values()].map((t) => ({
    label: t.label, color: t.color, signal: t.signal,
    points: [...t.pts.entries()].sort((a, b) => a[0] - b[0]).map(([t2, v]) => ({ t: t2, v })),
  }));
  // identified students first, then unknown voices; capped so a crowded event never floods the page
  return out.sort((a, b) => Number(a.label.startsWith("Unknown")) - Number(b.label.startsWith("Unknown"))).slice(0, MAX_WAVEFORM_ROWS);
}

export function alertRows(e: DetectionEvent) {
  return e.speakers.map((s) => ({
    speaker: trackLabel(s), student: s.is_identified ? s.student_name ?? "-" : "-", decision: s.decision_status ?? "unknown",
    why: s.rejection_reason ? STATUS_LABELS[s.rejection_reason] ?? s.rejection_reason : "-",
    similarity: fmtSimilarity(s.similarity_score), runnerUp: fmtSimilarity(s.runner_up_similarity),
    start: hms(ms(s.start_time)), end: hms(ms(s.end_time)), signal: SIGNAL_LABELS[s.signal_type ?? "activity_only"] ?? s.signal_type ?? "",
  }));
}
