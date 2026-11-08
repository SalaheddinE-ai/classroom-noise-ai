"use client";
import { useEffect, useState } from "react";
import { Field, Notice } from "@/components/ui";
import { api } from "@/lib/api";
import type { Settings } from "@/lib/types";

export default function SettingsPage() {
  const [s, setS] = useState<Settings | null>(null);
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  useEffect(() => { api<Settings>("/settings").then(setS).catch((e) => setMsg({ kind: "err", text: String(e.message ?? e) })); }, []);
  if (!s) return <>{msg ? <Notice kind="err">{msg.text}</Notice> : <p className="small">Loading…</p>}</>;
  const num = (k: keyof Settings, label: string, min: number, max: number, step: number) => (
    <Field label={`${label}: ${s[k]}`}>
      <input type="range" min={min} max={max} step={step} value={Number(s[k])} onChange={(e) => setS({ ...s, [k]: Number(e.target.value) })} />
    </Field>
  );
  async function apply() {
    if (!s) return;
    try {
      const body = {
        threshold_db: s.threshold_db, cooldown_seconds: s.cooldown_seconds, speaker_match_threshold: s.speaker_match_threshold, language: s.language,
        speaker_cooldown_seconds: s.speaker_cooldown_seconds, min_identify_seconds: s.min_identify_seconds,
        max_simultaneous_speakers: s.max_simultaneous_speakers, unknown_speaker_handling: s.unknown_speaker_handling,
      };
      setS(await api<Settings>("/settings", { method: "PUT", json: body }));
      setMsg({ kind: "ok", text: "Settings applied (live; not persisted across restarts - put them in .env to keep them)." });
    } catch (e) { setMsg({ kind: "err", text: e instanceof Error ? e.message : String(e) }); }
  }
  return (
    <>
      <h1>Settings</h1>
      <p className="sub">Applied to the running backend immediately. They are not saved across restarts: copy the values to <code>.env</code> (CNAI_*) to keep them.</p>
      <div className="card grid g2">
        {num("threshold_db", "Loudness threshold (dB)", 20, 90, 1)}
        {num("cooldown_seconds", "Cooldown (seconds)", 15, 30, 1)}
        {num("speaker_match_threshold", "Speaker match threshold (cosine)", 0.2, 0.95, 0.01)}
        <Field label="Warning language">
          <select value={s.language} onChange={(e) => setS({ ...s, language: e.target.value })}><option value="en">en</option><option value="fr">fr</option></select>
        </Field>
      </div>
      <div className="card">
        <h2>Multi-speaker</h2>
        <div className="grid g2">
          {num("speaker_cooldown_seconds", "Per-speaker cooldown (s)", 0, 120, 1)}
          {num("min_identify_seconds", "Min. speech to identify (s)", 0.3, 5, 0.1)}
          {num("max_simultaneous_speakers", "Max simultaneous speakers", 1, 6, 1)}
          <Field label="Unknown speakers">
            <select value={s.unknown_speaker_handling} onChange={(e) => setS({ ...s, unknown_speaker_handling: e.target.value })}>
              <option value="report">report - show and warn</option><option value="log_only">log_only - show, never warn because of them</option><option value="ignore">ignore - drop them</option>
            </select>
          </Field>
        </div>
        <p className="small">Overlap sensitivity: set CNAI_OVERLAP_SENSITIVITY in .env.</p>
      </div>
      <button className="primary" onClick={apply}>Apply settings</button>
      {msg && <Notice kind={msg.kind}>{msg.text}</Notice>}
    </>
  );
}
