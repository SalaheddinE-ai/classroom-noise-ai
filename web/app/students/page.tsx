"use client";
import Link from "next/link";
import { useState } from "react";
import Recorder, { Recording } from "@/components/Recorder";
import { Field, Notice } from "@/components/ui";
import { api } from "@/lib/api";
import { RECOGNITION_LABELS } from "@/lib/format";
import { usePoll } from "@/lib/hooks";
import type { DiagnoseResult, Student } from "@/lib/types";

const SENTENCE = "The quick brown fox jumps over the lazy dog, and today I am joining the class.";

export default function StudentsPage() {
  const students = usePoll(() => api<Student[]>("/students"), 8000);
  const [f, setF] = useState({ name: "", age: "12", code: "", group: "", notes: "" });
  const [allowDup, setAllowDup] = useState(false);
  const [rec, setRec] = useState<Recording | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const [probe, setProbe] = useState<Recording | null>(null);
  const [diag, setDiag] = useState<DiagnoseResult | null>(null);
  const [diagErr, setDiagErr] = useState<string | null>(null);
  const set = (k: keyof typeof f) => (e: React.ChangeEvent<HTMLInputElement>) => setF({ ...f, [k]: e.target.value });

  async function save() {
    setMsg(null);
    if (!f.name.trim()) return setMsg({ kind: "err", text: "Please enter the student name." });
    if (!rec) return setMsg({ kind: "err", text: "Please record a voice sample first." });
    if (rec.seconds < 1.5) return setMsg({ kind: "err", text: `Recording too short (${rec.seconds.toFixed(1)}s). Record at least 2 seconds.` });
    const fd = new FormData();
    fd.append("name", f.name.trim()); fd.append("age", f.age || "12"); fd.append("student_code", f.code.trim());
    fd.append("group_name", f.group.trim()); fd.append("notes", f.notes.trim());
    fd.append("allow_duplicate_name", String(allowDup)); fd.append("audio", rec.blob, "sample.wav");
    setBusy(true);
    try {
      await api("/students/register", { method: "POST", body: fd });
      setMsg({ kind: "ok", text: `Student '${f.name.trim()}' registered.` });
      setF({ name: "", age: "12", code: "", group: "", notes: "" }); setRec(null); void students.refresh();
    } catch (e) { setMsg({ kind: "err", text: e instanceof Error ? e.message : String(e) }); }
    finally { setBusy(false); }
  }

  async function runDiagnose() {
    if (!probe) return; setDiag(null); setDiagErr(null); setBusy(true);
    const fd = new FormData(); fd.append("audio", probe.blob, "probe.wav");
    try { setDiag(await api<DiagnoseResult>("/students/diagnose", { method: "POST", body: fd })); }
    catch (e) { setDiagErr(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  return (
    <>
      <h1>Student Registration</h1>
      <p className="sub">Record about 3 seconds of the student&apos;s natural voice in a quiet place, then save.</p>
      <Notice kind="info">Suggested sentence: {SENTENCE}</Notice>
      <p className="small">Voice samples and embeddings are stored locally in this installation only. Speaker recognition is an estimate, not proof of identity.</p>

      <div className="card">
        <div className="grid g3">
          <Field label="Student name"><input value={f.name} onChange={set("name")} /></Field>
          <Field label="Age"><input type="number" min={3} max={99} value={f.age} onChange={set("age")} /></Field>
          <Field label="Student code (optional)"><input value={f.code} onChange={set("code")} /></Field>
          <Field label="Class / group (optional)"><input value={f.group} onChange={set("group")} /></Field>
          <Field label="Notes (optional)"><input value={f.notes} onChange={set("notes")} /></Field>
        </div>
        <p><label><input type="checkbox" checked={allowDup} onChange={(e) => setAllowDup(e.target.checked)} /> A student with the same name already exists - register anyway</label></p>
        <Recorder onChange={setRec} label="Record voice sample" />
        <p><button className="primary" disabled={busy} onClick={save}>{busy ? "Computing the ECAPA-TDNN embedding…" : "Compute embedding and save"}</button></p>
        {msg && <Notice kind={msg.kind}>{msg.text}</Notice>}
      </div>

      <details className="card">
        <summary>Test my voice (why am I not recognised?)</summary>
        <p className="small">Records a few seconds and shows the raw similarity to every registered profile. Nothing is saved.</p>
        <Recorder onChange={setProbe} label="Record a test sample" />
        <p><button disabled={!probe || busy} onClick={runDiagnose}>Compare with registered voices</button></p>
        {diagErr && <Notice kind="err">{diagErr}</Notice>}
        {diag && (
          <>
            <table>
              <thead><tr><th>Name</th><th>Similarity</th><th>Samples</th><th>Passes threshold</th></tr></thead>
              <tbody>{diag.results.map((r) => <tr key={r.student_id}><td>{r.name}</td><td>{r.similarity.toFixed(3)}</td><td>{r.samples}</td><td>{r.passes_threshold ? "yes" : "no"}</td></tr>)}</tbody>
            </table>
            <Notice kind="info">Current threshold: {diag.threshold.toFixed(2)}. {diag.hint}</Notice>
          </>
        )}
      </details>

      <h2>Registered students</h2>
      {students.error && <Notice kind="err">{students.error}</Notice>}
      {students.data && students.data.length === 0 && <p className="small">No students registered yet.</p>}
      {students.data && students.data.length > 0 && (
        <div className="card" style={{ overflowX: "auto" }}>
          <table>
            <thead><tr><th>Name</th><th>Code</th><th>Group</th><th>Age</th><th>Voice samples</th><th>Recognition</th><th>Registered</th></tr></thead>
            <tbody>
              {students.data.map((s) => (
                <tr key={s.id}>
                  <td><Link href={`/students/${s.id}`}>{s.name}</Link></td><td>{s.student_code ?? "-"}</td><td>{s.group_name ?? "-"}</td><td>{s.age ?? "-"}</td>
                  <td>{s.voice_samples}</td>
                  <td><span className={`badge ${s.recognition_status === "ready" ? "ok" : "warn"}`}>{RECOGNITION_LABELS[s.recognition_status] ?? s.recognition_status}</span></td>
                  <td>{s.created_at.slice(0, 10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
