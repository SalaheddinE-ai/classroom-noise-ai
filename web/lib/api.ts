export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
export const WS_URL = API_URL.replace(/^http/, "ws");

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) { super(message); this.status = status; }
}

async function detail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    const d = body?.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) return d.map((x: { msg?: string }) => x.msg ?? JSON.stringify(x)).join("; ");
    return JSON.stringify(body);
  } catch { return `${res.status} ${res.statusText}`; }
}

/** JSON request. Throws ApiError with the backend's `detail` message. */
export async function api<T>(path: string, init?: RequestInit & { json?: unknown; params?: Record<string, string | number | string[] | undefined> }): Promise<T> {
  const url = new URL(API_URL + path);
  for (const [k, v] of Object.entries(init?.params ?? {})) {
    if (Array.isArray(v)) v.forEach((x) => url.searchParams.append(k, x));
    else if (v !== undefined) url.searchParams.set(k, String(v));
  }
  const headers: Record<string, string> = {};
  let body = init?.body;
  if (init?.json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(init.json); }
  let res: Response;
  try { res = await fetch(url.toString(), { ...init, headers: { ...headers, ...(init?.headers as Record<string, string>) }, body, cache: "no-store" }); }
  catch (e) { throw new ApiError(0, `Backend unreachable at ${API_URL}`); }
  if (!res.ok) throw new ApiError(res.status, await detail(res));
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export function audioUrl(studentId: string, sampleId: number): string {
  return `${API_URL}/students/${studentId}/voice/${sampleId}/audio`;
}
