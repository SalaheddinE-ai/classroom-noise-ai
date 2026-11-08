"use client";
import { useEffect, useRef, useState } from "react";
import { DbHistory, Gauge, MicWaveform, SpeakerWaveforms, Timeline } from "@/components/Charts";
import { Metric, Notice } from "@/components/ui";
import { api } from "@/lib/api";
import { SIMILARITY_HELP, overlapText, speakerColor, speakerLine, trackLabel, ms } from "@/lib/format";
import { usePoll, useStatusStream } from "@/lib/hooks";
import type { MonitorStatus, Visualization } from "@/lib/types";

const isStage = (v: unknown): v is { median_ms: number; p95_ms: number; count: number } =>
  typeof v === "object" && v !== null && "median_ms" in v;

export default function LivePage() {
  const { status, connected, error } = useStatusStream();
  const viz = usePoll(() => api<Visualization>("/live/visualization"), 1000);
  const [history, setHistory] = useState<number[]>([]);
  const [busy, setBusy] = useState(false);
  const [ctlError, setCtlError] = useState<string | null>(null);
  const last = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (!status || status.server_time === last.current) return;
    last.current = status.server_time;
    setHistory((h) => [...h, status.live_db].slice(-120));
  }, [status]);

  async function control(path: "/monitor/start" | "/monitor/stop") {
    setBusy(true); setCtlError(null);
    try { await api<MonitorStatus>(path, { method: "POST" }); } catch (e) { setCtlError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  const v = viz.data;
  const events = v?.events ?? [];
  const now = v?.now ? ms(v.now) : Date.now();
  const win: [number, number] = [now - Math.min(v?.history_seconds ?? 60, 60) * 1000, now];
  const cur = status?.current_event;
  const caps = status?.capabilities ?? {};
  const lat = (status?.latency ?? {}) as Record<string, unknown>;
  const stages = Object.entries(lat).filter(([, x]) => isStage(x)) as [string, { median_ms: number; p95_ms: number; count: number }][];
  const rtf = lat["real_time_factor"] as { median: number; p95: number } | undefined;

  return (
    <>
      <h1>Live Monitor</h1>
      <div className="row" style={{ marginBottom: 12 }}>
        <button className="primary" disabled={busy} onClick={() => control("/monitor/start")}>{busy ? "Loading models and opening the microphone…" : "Start monitoring"}</button>
        <button disabled={busy} onClick={() => control("/monitor/stop")}>Stop monitoring</button>
        <span className="small"><span className={`dot ${connected ? "on" : ""}`} />{connected ? "live stream connected" : "reconnecting (REST fallback)…"}</span>
      </div>
      {ctlError && <Notice kind="err">{ctlError}</Notice>}
      {!status && <Notice kind="warn">{error ?? "Waiting for the backend…"} - showing nothing instead of stale data.</Notice>}

      {status && (
        <>
          <div className="grid g2">
            <div className="card">
              <Gauge db={status.live_db} threshold={status.threshold_db} />
              <DbHistory values={history} threshold={status.threshold_db} />
            </div>
            <div className="card">
              <Notice kind={status.ui_state.startsWith("Warning") ? "err" : ["Analyzing", "Loading models"].includes(status.ui_state) ? "warn" : "info"}>State: {status.ui_state}</Notice>
              <div className="grid g3">
                <Metric label="Threshold" value={`${status.threshold_db.toFixed(0)} dB`} />
                <Metric label="Cooldown" value={status.in_cooldown ? `${status.cooldown_remaining.toFixed(0)} s` : "Ready"} />
                <Metric label="Registered voices" value={status.registered_voices} />
                <Metric label="Speech (VAD)" value={status.speech_detected ? "Detected" : "None"} />
                <Metric label="Speech ratio" value={`${Math.round((status.speech_ratio ?? 0) * 100)}%`} />
              </div>
              {status.running && status.mic_alive === false && <Notice kind="err">No audio received from the microphone. Check the input device (python main.py devices).</Notice>}
              {status.mic_silent && <Notice kind="err">The microphone delivers digital silence (0 dB). Check the Windows microphone privacy setting, the muted state, or the input device index.</Notice>}
              <p className="small">Overlap detector: {status.osd_backend ?? "not loaded"}</p>
              <p className="small">
                {caps.source_separation ? "Overlapping voices: source separation active (isolated audio per speaker)."
                  : caps.diarization ? "Overlapping voices: diarization active - speaker activity and mixed-signal views, no isolated audio."
                  : "Overlapping voices: embedding window scan only - a speaker is named only if they dominate part of the window. No isolated audio."}
              </p>
              {status.separation_error && <Notice kind="warn">Source separation unavailable: {status.separation_error}</Notice>}
              {status.error && <Notice kind="err">{status.error}</Notice>}
            </div>
          </div>

          <details className="card">
            <summary>Monitoring and model health</summary>
            <div className="grid g4" style={{ marginTop: 8 }}>
              <Metric label="Models loaded" value={status.models_loaded ? "yes" : "no"} />
              <Metric label="Microphone" value={status.mic_alive ? "alive" : "no audio"} />
              <Metric label="Dropped windows" value={status.dropped_windows ?? 0} />
              <Metric label="Processing errors" value={status.processing_errors ?? 0} />
            </div>
            {status.compute && <p className="small">Compute: {JSON.stringify(status.compute)}</p>}
            {stages.length > 0 && (
              <table><thead><tr><th>stage</th><th>median ms</th><th>p95 ms</th><th>n</th></tr></thead>
                <tbody>{stages.map(([k, x]) => <tr key={k}><td>{k}</td><td>{x.median_ms}</td><td>{x.p95_ms}</td><td>{x.count}</td></tr>)}</tbody></table>
            )}
            {rtf && <p className="small">Real-time factor (processing time / hop): median {rtf.median}, p95 {rtf.p95} (must stay below 1.0 to keep up with the microphone).</p>}
          </details>

          <div className="card">
            <h2>Current speakers</h2>
            {cur && cur.speakers?.length ? (
              <>
                <div className="grid g4">
                  {cur.speakers.slice(0, 4).map((sp, i) => (
                    <div key={i} className="spk" style={{ borderLeftColor: speakerColor(sp) }}>
                      <b>{trackLabel(sp)}</b><br /><span className="small">{speakerLine(sp).split(" - ").slice(1).join(" - ")}</span>
                    </div>
                  ))}
                </div>
                {cur.is_overlap && <p className="small">{overlapText(cur)}</p>}
                <p className="small">{SIMILARITY_HELP}</p>
              </>
            ) : <p className="small">Nobody above the loudness threshold right now.</p>}
          </div>
        </>
      )}

      <div className="card">
        {viz.error && <Notice kind="warn">Visualisation unavailable: {viz.error}</Notice>}
        {!v && !viz.error && <p className="small">Loading…</p>}
        {v && <MicWaveform waveform={v.waveform} events={events} window={win} />}
        {v && <Timeline events={events} window={win} />}
        {events.some((e) => e.speakers.some((s) => s.waveform?.peaks?.length)) && (
          <details>
            <summary>Per-speaker waveforms</summary>
            <p className="small">Rows labelled &quot;separated audio&quot; are isolated signals. Rows labelled &quot;mixed microphone signal&quot; show the combined audio during that speaker&apos;s activity - they are NOT isolated voices.</p>
            <SpeakerWaveforms events={events} window={win} />
          </details>
        )}
      </div>
    </>
  );
}
