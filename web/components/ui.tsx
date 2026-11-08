"use client";
import type { ReactNode } from "react";

export function Notice({ kind, children }: { kind: "info" | "warn" | "err" | "ok"; children: ReactNode }) {
  return <div className={`notice ${kind}`}>{children}</div>;
}
export function Metric({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return <div className="metric" title={hint}><span className="v">{value}</span><span className="l">{label}</span></div>;
}
export function Field({ label, children }: { label: string; children: ReactNode }) {
  return <label className="f">{label}{children}</label>;
}
