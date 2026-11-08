"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, WS_URL } from "./api";
import type { MonitorStatus } from "./types";

/** Poll an async loader. Keeps the last good value, exposes the error, never shows data as fresher than it is. */
export function usePoll<T>(load: () => Promise<T>, intervalMs: number, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const loadRef = useRef(load);
  loadRef.current = load;
  const refresh = useCallback(async () => {
    try { setData(await loadRef.current()); setError(null); setUpdatedAt(Date.now()); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  }, []);
  useEffect(() => {
    let alive = true;
    const tick = async () => { if (alive) await refresh(); };
    void tick();
    const id = window.setInterval(tick, intervalMs);
    return () => { alive = false; window.clearInterval(id); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, refresh, ...deps]);
  return { data, error, updatedAt, refresh };
}

/** Live status over WebSocket (/ws/stream, 2 Hz) with automatic reconnection and a REST fallback. */
export function useStatusStream() {
  const [status, setStatus] = useState<MonitorStatus | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let ws: WebSocket | null = null; let timer: number | undefined; let closed = false; let poll: number | undefined;
    const startPolling = () => {
      if (poll) return;
      poll = window.setInterval(async () => {
        try { setStatus(await api<MonitorStatus>("/status")); setError(null); } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
      }, 1000);
    };
    const connect = () => {
      if (closed) return;
      try { ws = new WebSocket(`${WS_URL}/ws/stream`); } catch { startPolling(); return; }
      ws.onopen = () => { setConnected(true); setError(null); if (poll) { window.clearInterval(poll); poll = undefined; } };
      ws.onmessage = (m) => { try { setStatus(JSON.parse(m.data) as MonitorStatus); } catch { /* ignore a malformed frame */ } };
      ws.onerror = () => setError("WebSocket error");
      ws.onclose = () => { setConnected(false); startPolling(); if (!closed) timer = window.setTimeout(connect, 2000); };
    };
    connect();
    return () => { closed = true; if (timer) window.clearTimeout(timer); if (poll) window.clearInterval(poll); ws?.close(); };
  }, []);
  return { status, connected, error };
}
