"use client";
import { useState } from "react";
import AlertCard from "@/components/AlertCard";
import { Notice } from "@/components/ui";
import { api } from "@/lib/api";
import { SIMILARITY_HELP } from "@/lib/format";
import { usePoll } from "@/lib/hooks";
import type { DetectionEvent } from "@/lib/types";

const CATS = ["Single Speaker", "Multiple Speakers", "Unknown Speaker", "Overlapping Speech"];

export default function AlertsPage() {
  const [shown, setShown] = useState(15);
  const [cats, setCats] = useState<string[]>(CATS);
  const ev = usePoll(() => api<DetectionEvent[]>("/events", { params: { limit: 100 } }), 3000);
  const toggle = (c: string) => setCats((x) => (x.includes(c) ? x.filter((y) => y !== c) : [...x, c]));
  const list = (ev.data ?? [])
    .filter((e) => (e.labels?.length ? e.labels : [e.category ?? ""]).some((l) => cats.includes(l)))
    .slice(0, shown);
  return (
    <>
      <h1>Recent Alerts</h1>
      <p className="sub">Consecutive detections are grouped into one alert while the noise continues. Similarity is a cosine score, not a probability - treat identities as estimates.</p>
      <div className="card row">
        {CATS.map((c) => <label key={c}><input type="checkbox" checked={cats.includes(c)} onChange={() => toggle(c)} /> {c}</label>)}
        <span className="spacer" />
        <label className="row">Alerts to show: {shown}<input type="range" min={5} max={50} value={shown} onChange={(e) => setShown(Number(e.target.value))} style={{ width: 140 }} /></label>
      </div>
      <p className="small">{SIMILARITY_HELP}</p>
      {ev.error && <Notice kind="warn">Backend not reachable: {ev.error}</Notice>}
      {ev.data && list.length === 0 && <p className="small">No alerts for this selection.</p>}
      {list.map((e) => <AlertCard key={e.event_id} event={e} />)}
    </>
  );
}
