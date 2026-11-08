"use client";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import Recorder, { Recording } from "@/components/Recorder";
import { Field, Metric, Notice } from "@/components/ui";
import { api, audioUrl } from "@/lib/api";
import { RECOGNITION_LABELS } from "@/lib/format";
import { usePoll } from "@/lib/hooks";
import type { StudentProfile } from "@/lib/types";

export default function StudentPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const p = usePoll(() => api<StudentProfile>(`/students/${id}`), 15000, [id]);
  const [form, setForm] = useState({ name: "", age: "", code: "", group: "", notes: "" });
  const [rec, setRec] = useState<Recording | null>(null);
  const [force, setForce] = useState(false);
  const [confirm, setConfirm] = useState(false);
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const loaded = p.data?.id;

  useEffect(() => {
    if (p.data) setForm({ name: p.data.name, age: String(p.data.age ?? ""), code: p.data.student_code ?? "", group: p.data.group_name ?? "", notes: p.data.notes ?? "" });
    // fill the form once per student, not on every refresh (that would overwrite the user's typing)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loaded]);

  const run = async (fn: () => Promise<void>, ok: string) => {
    setMsg(null); setBusy(true);
    try { await fn(); setMsg({ kind: "ok", text: ok }); void p.refresh(); }
    catch (e) { setMsg({ kind: "err", text: e instanceof Error ? e.message : String(e) }); }
    finally { setBusy(false); }
  };

  if (p.error && !p.data) return <><Link href="/students">← Students</Link><Notice kind="err">{p.error}</Notice></>;
  if (!p.data) return <p className="small">Loading…</p>;
  const s = p.data;
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => setForm({ ...form, [k]: e.target.value });

  return (
    <>
      <Link href="/students">← Students</Link>
      <h1>{s.name}</h1>
      <div className="card grid g4">
        <Metric label="Voice samples" value={s.voice_samples} hint={`At least ${s.min_samples_required} needed`} />
        <Metric label="Recognition" value={RECOGNITION_LABELS[s.recognition_status] ?? s.recognition_status} />
        <Metric label="Registered" value={s.created_at.slice(0, 10)} />
        <Metric label="Code" value={s.student_code ?? "-"} />
      </div>
      {msg && <Notice kind={msg.kind}>{msg.text}</Notice>}

      <div className="card">
        <h2>Profile</h2>
        <div className="grid g3">
          <Field label="Full name"><input value={form.name} onChange={set("name")} /></Field>
          <Field label="Age"><input type="number" min={3} max={99} value={form.age} onChange={set("age")} /></Field>
          <Field label="Student code"><input value={form.code} onChange={set("code")} /></Field>
          <Field label="Class / group"><input value={form.group} onChange={set("group")} /></Field>
        </div>
        <p><Field label="Notes"><textarea rows={3} value={form.notes} onChange={set("notes")} /></Field></p>
        <button className="primary" disabled={busy} onClick={() => run(() => api("/students/" + id, { method: "PATCH", json: { name: form.name, age: form.age ? Number(form.age) : null, student_code: form.code, group_name: form.group, notes: form.notes } }), "Profile updated.")}>Save changes</button>
      </div>

      <div className="card">
        <h2>Registered voice samples</h2>
        {s.samples.length === 0 && <p className="small">No voice sample yet.</p>}
        {s.samples.map((x) => (
          <div className="row" key={x.id} style={{ marginBottom: 8 }}>
            <span className="small">Sample #{x.id} - {x.created_at.slice(0, 19).replace("T", " ")} - {x.duration_seconds ?? "?"} s{x.quality_weight != null ? ` - weight ${x.quality_weight.toFixed(2)}` : ""}</span>
            {x.has_audio && <audio controls preload="none" src={audioUrl(id, x.id)} />}
            <span className="spacer" />
            <button disabled={busy} onClick={() => run(() => api(`/students/${id}/voice/${x.id}`, { method: "DELETE" }), "Sample deleted.")}>Delete</button>
          </div>
        ))}
        <h3>Add another voice sample</h3>
        <p className="small">Several samples (different sentences, a few seconds each) make recognition more robust.</p>
        <Recorder onChange={setRec} label="Record an additional sample" />
        <p><label><input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} /> Add even if it does not resemble this student&apos;s existing samples</label></p>
        <button className="primary" disabled={!rec || busy} onClick={() => run(async () => {
          const fd = new FormData(); fd.append("force", String(force)); fd.append("audio", rec!.blob, "sample.wav");
          await api(`/students/${id}/voice`, { method: "POST", body: fd }); setRec(null);
        }, "Voice sample added.")}>Add sample</button>
      </div>

      <details className="card">
        <summary>Delete this student</summary>
        <Notice kind="warn">Deletes the profile, every voice recording and embedding, and removes the student&apos;s identity from stored alerts. This cannot be undone.</Notice>
        <p><label><input type="checkbox" checked={confirm} onChange={(e) => setConfirm(e.target.checked)} /> Yes, permanently delete {s.name}</label></p>
        <button className="danger" disabled={!confirm || busy} onClick={() => run(async () => { await api(`/students/${id}`, { method: "DELETE" }); router.push("/students"); }, "Student deleted.")}>Delete student</button>
      </details>
    </>
  );
}
