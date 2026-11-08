"use client";
import { useEffect, useMemo, useState } from "react";
import { Metric, Notice } from "@/components/ui";
import { api } from "@/lib/api";
import type { LogRow } from "@/lib/types";

const TYPES = ["SINGLE_STUDENT", "GROUP", "GENERAL"];
const day = (d: Date) => d.toISOString().slice(0, 10);

function csv(rows: LogRow[]): string {
  const cols: (keyof LogRow)[] = ["id", "timestamp", "event_type", "category", "student_name", "speaker_count", "overlap_duration", "dB_level", "match_score", "was_spoken", "message_spoken"];
  const esc = (v: unknown) => { const s = v == null ? "" : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  return [cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c])).join(","))].join("\n");
}

export default function LogsPage() {
  const today = new Date();
  const [from, setFrom] = useState(day(new Date(today.getTime() - 7 * 864e5)));
  const [to, setTo] = useState(day(today));
  const [types, setTypes] = useState<string[]>(TYPES);
  const [rows, setRows] = useState<LogRow[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    api<LogRow[]>("/logs", { params: { start: `${from}T00:00:00`, end: `${to}T23:59:59`, event_type: types, limit: 5000 } })
      .then((r) => { if (alive) { setRows(r); setErr(null); } })
      .catch((e) => { if (alive) setErr(e instanceof Error ? e.message : String(e)); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [from, to, types]);

  const stats = useMemo(() => {
    const r = rows ?? [];
    const counts: Record<string, number> = {};
    r.forEach((x) => { counts[x.event_type] = (counts[x.event_type] ?? 0) + 1; });
    return { n: r.length, spoken: r.filter((x) => x.was_spoken).length, avg: r.length ? r.reduce((s, x) => s + x.dB_level, 0) / r.length : 0, counts };
  }, [rows]);
  const maxCount = Math.max(1, ...Object.values(stats.counts));

  function download() {
    const url = URL.createObjectURL(new Blob([csv(rows ?? [])], { type: "text/csv" }));
    const a = document.createElement("a"); a.href = url; a.download = "interventions.csv"; a.click(); URL.revokeObjectURL(url);
  }

  return (
    <>
      <h1>Analytics and Logs</h1>
      <div className="card row">
        <label className="f">From<input type="date" value={from} onChange={(e) => setFrom(e.target.value)} /></label>
        <label className="f">To<input type="date" value={to} onChange={(e) => setTo(e.target.value)} /></label>
        {TYPES.map((t) => <label key={t}><input type="checkbox" checked={types.includes(t)} onChange={() => setTypes((x) => x.includes(t) ? x.filter((y) => y !== t) : [...x, t])} /> {t}</label>)}
      </div>
      {err && <Notice kind="err">{err}</Notice>}
      {loading && !rows && <p className="small">Loading…</p>}
      {rows && rows.length === 0 && <p className="small">No interventions for this selection.</p>}
      {rows && rows.length > 0 && (
        <>
          <div className="card grid g3">
            <Metric label="Interventions" value={stats.n} />
            <Metric label="Spoken" value={stats.spoken} />
            <Metric label="Average level" value={`${stats.avg.toFixed(1)} dB`} />
          </div>
          <div className="card">
            <h2>Events by type</h2>
            {Object.entries(stats.counts).map(([k, n]) => (
              <div className="row" key={k} style={{ marginBottom: 4 }}>
                <span style={{ width: 150 }}>{k}</span>
                <div style={{ width: `${(n / maxCount) * 60}%`, minWidth: 4, height: 14, background: "var(--accent)", borderRadius: 3 }} /><span className="small">{n}</span>
              </div>
            ))}
          </div>
          <div className="row" style={{ marginBottom: 8 }}><button onClick={download}>Download CSV</button><span className="small">{rows.length} rows (max 5000)</span></div>
          <div className="card" style={{ overflowX: "auto" }}>
            <table>
              <thead><tr><th>timestamp</th><th>event_type</th><th>category</th><th>student</th><th>speakers</th><th>overlap (s)</th><th>dB</th><th>match</th><th>spoken</th><th>message</th></tr></thead>
              <tbody>
                {rows.slice(0, 500).map((r) => (
                  <tr key={r.id}><td>{r.timestamp.replace("T", " ").slice(0, 19)}</td><td>{r.event_type}</td><td>{r.category ?? "-"}</td><td>{r.student_name ?? "-"}</td>
                    <td>{r.speaker_count ?? "-"}</td><td>{r.overlap_duration ?? "-"}</td><td>{r.dB_level.toFixed(1)}</td><td>{r.match_score?.toFixed(2) ?? "-"}</td><td>{r.was_spoken ? "yes" : "no"}</td><td>{r.message_spoken}</td></tr>
                ))}
              </tbody>
            </table>
            {rows.length > 500 && <p className="small">Showing the first 500 rows; the CSV contains all {rows.length}.</p>}
          </div>
        </>
      )}
    </>
  );
}
