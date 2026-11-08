"use client";
import { useEffect, useRef, useState } from "react";
import { encodeWav, fileToWav } from "@/lib/wav";

export interface Recording { blob: Blob; seconds: number; url: string }

/** Record from the browser microphone (or upload a file) and hand back a mono WAV the backend can read. */
export default function Recorder({ onChange, label = "Voice sample" }: { onChange: (r: Recording | null) => void; label?: string }) {
  const [recording, setRecording] = useState(false);
  const [rec, setRec] = useState<Recording | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const state = useRef<{ ctx: AudioContext; stream: MediaStream; node: ScriptProcessorNode; chunks: Float32Array[] } | null>(null);

  useEffect(() => () => { state.current?.stream.getTracks().forEach((t) => t.stop()); void state.current?.ctx.close(); }, []);
  useEffect(() => () => { if (rec) URL.revokeObjectURL(rec.url); }, [rec]);

  const publish = (r: Recording | null) => { setRec(r); onChange(r); };

  async function start() {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) { setError("This browser cannot access the microphone (use https or localhost)."); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
      const ctx = new AudioContext();
      const src = ctx.createMediaStreamSource(stream);
      const node = ctx.createScriptProcessor(4096, 1, 1);
      const chunks: Float32Array[] = [];
      node.onaudioprocess = (e) => {
        const data = new Float32Array(e.inputBuffer.getChannelData(0));
        chunks.push(data);
        let peak = 0; for (let i = 0; i < data.length; i++) peak = Math.max(peak, Math.abs(data[i]));
        setLevel(peak);
      };
      src.connect(node); node.connect(ctx.destination);
      state.current = { ctx, stream, node, chunks };
      publish(null); setRecording(true);
    } catch (e) { setError(`Microphone unavailable: ${e instanceof Error ? e.message : String(e)}`); }
  }

  async function stop() {
    const s = state.current; if (!s) return;
    s.node.disconnect(); s.stream.getTracks().forEach((t) => t.stop());
    const total = s.chunks.reduce((n, c) => n + c.length, 0);
    const all = new Float32Array(total); let o = 0;
    for (const c of s.chunks) { all.set(c, o); o += c.length; }
    const rate = s.ctx.sampleRate; void s.ctx.close(); state.current = null;
    setRecording(false); setLevel(0);
    const blob = encodeWav(all, rate);
    publish({ blob, seconds: total / rate, url: URL.createObjectURL(blob) });
  }

  async function onFile(file: File | undefined) {
    if (!file) return; setError(null);
    try { const { blob, seconds } = await fileToWav(file); publish({ blob, seconds, url: URL.createObjectURL(blob) }); }
    catch (e) { setError(`Could not read this audio file: ${e instanceof Error ? e.message : String(e)}`); }
  }

  return (
    <div>
      <div className="small" style={{ marginBottom: 6 }}>{label}</div>
      <div className="row">
        {!recording
          ? <button type="button" className="primary" onClick={start}>● Record</button>
          : <button type="button" className="danger" onClick={stop}>■ Stop</button>}
        <label className="small">or upload a file: <input type="file" accept="audio/*" onChange={(e) => onFile(e.target.files?.[0])} style={{ width: "auto" }} /></label>
        {recording && <div style={{ width: 120, height: 8, background: "var(--border)", borderRadius: 4 }}>
          <div style={{ width: `${Math.min(level * 100, 100)}%`, height: 8, background: "var(--ok)", borderRadius: 4 }} /></div>}
        {rec && <button type="button" onClick={() => publish(null)}>Discard</button>}
      </div>
      {rec && <div className="row" style={{ marginTop: 8 }}><audio controls src={rec.url} /><span className="small">{rec.seconds.toFixed(1)} s</span></div>}
      {error && <div className="notice err">{error}</div>}
    </div>
  );
}
