"use client";
// Dependency-free SVG charts. Every chart renders only data the backend returned.
import type { DetectionEvent, Waveform } from "@/lib/types";
import { OVERLAP_FILL, SIGNAL_LABELS, fmtSimilarity, hms, overlapSpans, speakerWaveTracks, timelineBars, trackOrder, waveformPoints } from "@/lib/format";

const W = 1000;
const PAD_L = 140, PAD_R = 12;

function ticks(t0: number, t1: number, n = 5): number[] { return Array.from({ length: n + 1 }, (_, i) => t0 + ((t1 - t0) * i) / n); }
function scaler(t0: number, t1: number, left: number, right: number) { return (t: number) => left + ((t - t0) / Math.max(t1 - t0, 1)) * (right - left); }

function Axis({ t0, t1, y, left, right }: { t0: number; t1: number; y: number; left: number; right: number }) {
  const x = scaler(t0, t1, left, right);
  return <g>{ticks(t0, t1).map((t) => <text key={t} x={x(t)} y={y} textAnchor="middle">{hms(t)}</text>)}</g>;
}
function Overlaps({ spans, t0, t1, left, right, top, h }: { spans: [number, number][]; t0: number; t1: number; left: number; right: number; top: number; h: number }) {
  const x = scaler(t0, t1, left, right);
  return <g>{spans.map(([a, b], i) => {
    const xa = Math.max(left, x(a)), xb = Math.min(right, x(b));
    return xb > xa ? <rect key={i} x={xa} y={top} width={xb - xa} height={h} fill={OVERLAP_FILL} /> : null;
  })}</g>;
}
function envelope(points: { t: number; v: number }[], x: (t: number) => number, mid: number, amp: number, t0: number, t1: number): string {
  const vis = points.filter((p) => p.t >= t0 && p.t <= t1);
  if (!vis.length) return "";
  const top = vis.map((p) => `${x(p.t).toFixed(1)},${(mid - Math.min(p.v, 1) * amp).toFixed(1)}`);
  const bot = [...vis].reverse().map((p) => `${x(p.t).toFixed(1)},${(mid + Math.min(p.v, 1) * amp).toFixed(1)}`);
  return `M${top.join("L")}L${bot.join("L")}Z`;
}

export function Gauge({ db, threshold }: { db: number; threshold: number }) {
  const max = 100, r = 90, cx = 120, cy = 110;
  const pt = (v: number, rad = r) => { const a = Math.PI * (1 - Math.min(Math.max(v, 0), max) / max); return [cx + rad * Math.cos(a), cy - rad * Math.sin(a)]; };
  const [sx, sy] = pt(0), [ex, ey] = pt(db), [fx, fy] = pt(max), [tx1, ty1] = pt(threshold, r - 14), [tx2, ty2] = pt(threshold, r + 8);
  const over = db >= threshold;
  return (
    <svg viewBox="0 0 240 140" width="100%" role="img" aria-label={`Noise level ${db.toFixed(1)} dB, threshold ${threshold} dB`}>
      <path d={`M${sx},${sy} A${r},${r} 0 0 1 ${fx},${fy}`} fill="none" stroke="var(--border)" strokeWidth="16" />
      <path d={`M${sx},${sy} A${r},${r} 0 0 1 ${ex},${ey}`} fill="none" stroke={over ? "var(--danger)" : "var(--accent)"} strokeWidth="16" />
      <line x1={tx1} y1={ty1} x2={tx2} y2={ty2} stroke="var(--text)" strokeWidth="3" />
      <text x={cx} y={cy - 8} textAnchor="middle" style={{ fontSize: 26, fontWeight: 600, fill: "var(--text)" }}>{db.toFixed(1)} dB</text>
      <text x={cx} y={cy + 12} textAnchor="middle">threshold {threshold.toFixed(0)} dB</text>
    </svg>
  );
}

export function DbHistory({ values, threshold, height = 140 }: { values: number[]; threshold: number; height?: number }) {
  if (values.length < 2) return <p className="small">Collecting data…</p>;
  const lo = 0, hi = 100, H = height, top = 8, bot = H - 18;
  const y = (v: number) => bot - ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * (bot - top);
  const x = (i: number) => PAD_L / 4 + (i / (values.length - 1)) * (W - PAD_L / 4 - PAD_R);
  const d = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Noise level history">
      <line x1={PAD_L / 4} x2={W - PAD_R} y1={y(threshold)} y2={y(threshold)} stroke="var(--danger)" strokeDasharray="5 4" />
      <path d={d} fill="none" stroke="var(--accent)" strokeWidth="2" />
      <text x={W - PAD_R} y={y(threshold) - 4} textAnchor="end">threshold {threshold.toFixed(0)} dB</text>
      <text x={PAD_L / 4} y={H - 3}>last {values.length} s</text>
    </svg>
  );
}

export function MicWaveform({ waveform, events, window }: { waveform: Waveform; events: DetectionEvent[]; window: [number, number] }) {
  const H = 170, mid = 80, amp = 70, left = 40, right = W - PAD_R;
  const x = scaler(window[0], window[1], left, right);
  const pts = waveformPoints(waveform);
  const d = envelope(pts, x, mid, amp, window[0], window[1]);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Combined microphone waveform">
      <text x={left} y={12}>Combined microphone waveform (all voices mixed) - shaded = overlapping speech</text>
      <Overlaps spans={overlapSpans(events)} t0={window[0]} t1={window[1]} left={left} right={right} top={18} h={mid * 2 - 10} />
      <line x1={left} x2={right} y1={mid} y2={mid} stroke="var(--border)" />
      {d ? <path d={d} fill="var(--wave)" fillOpacity="0.55" stroke="var(--wave)" strokeWidth="0.5" /> : <text x={W / 2} y={mid} textAnchor="middle">No audio data yet</text>}
      <Axis t0={window[0]} t1={window[1]} y={H - 6} left={left} right={right} />
    </svg>
  );
}

export function Timeline({ events, window, title = true }: { events: DetectionEvent[]; window?: [number, number]; title?: boolean }) {
  const order = trackOrder(events);
  const bars = timelineBars(events);
  const starts = bars.map((b) => b.start), ends = bars.map((b) => b.end);
  const t0 = window?.[0] ?? (starts.length ? Math.min(...starts) - 500 : Date.now() - 60000);
  const t1 = window?.[1] ?? (ends.length ? Math.max(...ends) + 500 : Date.now());
  const rowH = 34, top = title ? 22 : 6, H = top + Math.max(order.length, 1) * rowH + 22;
  const x = scaler(t0, t1, PAD_L, W - PAD_R);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Speaker activity timeline">
      {title && <text x={PAD_L} y={12}>Speaker activity timeline (shaded = overlapping speech)</text>}
      <Overlaps spans={overlapSpans(events)} t0={t0} t1={t1} left={PAD_L} right={W - PAD_R} top={top} h={Math.max(order.length, 1) * rowH} />
      {order.length === 0 && <text x={W / 2} y={top + 18} textAnchor="middle">No speech detected yet</text>}
      {order.map((tr, i) => (
        <g key={tr.key}>
          <text x={PAD_L - 8} y={top + i * rowH + rowH / 2 + 4} textAnchor="end" style={{ fill: "var(--text)" }}>{tr.label.length > 22 ? tr.label.slice(0, 21) + "…" : tr.label}</text>
          <line x1={PAD_L} x2={W - PAD_R} y1={top + (i + 1) * rowH} y2={top + (i + 1) * rowH} stroke="var(--border)" />
          {bars.filter((b) => b.key === tr.key).map((b, j) => {
            const xa = Math.max(PAD_L, x(b.start)), xb = Math.min(W - PAD_R, x(b.end));
            return xb > xa ? (
              <rect key={j} x={xa} y={top + i * rowH + 6} width={Math.max(xb - xa, 2)} height={rowH - 12} rx="3" fill={b.color}>
                <title>{`${tr.label}\n${hms(b.start)} - ${hms(b.end)}\nsimilarity: ${fmtSimilarity(b.similarity)}`}</title>
              </rect>
            ) : null;
          })}
        </g>
      ))}
      <Axis t0={t0} t1={t1} y={H - 6} left={PAD_L} right={W - PAD_R} />
    </svg>
  );
}

export function SpeakerWaveforms({ events, window }: { events: DetectionEvent[]; window?: [number, number] }) {
  const tracks = speakerWaveTracks(events);
  if (!tracks.length) return null;
  const all = tracks.flatMap((t) => t.points.map((p) => p.t));
  const t0 = window?.[0] ?? Math.min(...all), t1 = window?.[1] ?? Math.max(...all) + 1;
  const rowH = 90, H = tracks.length * rowH + 20, left = 40, right = W - PAD_R;
  const x = scaler(t0, t1, left, right);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Per-speaker waveforms">
      {tracks.map((tr, i) => {
        const mid = i * rowH + 52;
        return (
          <g key={tr.label}>
            <text x={left} y={i * rowH + 14}>{`${tr.label} - ${SIGNAL_LABELS[tr.signal] ?? tr.signal}`}</text>
            <line x1={left} x2={right} y1={mid} y2={mid} stroke="var(--border)" />
            <path d={envelope(tr.points, x, mid, 30, t0, t1)} fill={tr.color} fillOpacity="0.55" stroke={tr.color} strokeWidth="0.5" />
          </g>
        );
      })}
      <Axis t0={t0} t1={t1} y={H - 4} left={left} right={right} />
    </svg>
  );
}
