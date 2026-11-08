"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { api, API_URL } from "@/lib/api";
import { usePoll } from "@/lib/hooks";
import ThemeToggle from "./ThemeToggle";

const LINKS = [
  { href: "/students", label: "Student Registration" },
  { href: "/live", label: "Live Monitor" },
  { href: "/alerts", label: "Recent Alerts" },
  { href: "/logs", label: "Analytics and Logs" },
  { href: "/settings", label: "Settings" },
];

export default function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const health = usePoll(() => api<{ status: string }>("/health"), 5000);
  const online = !!health.data && !health.error;
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">Classroom Noise AI</div>
        <nav className="nav">
          {LINKS.map((l) => (
            <Link key={l.href} href={l.href} className={path.startsWith(l.href) ? "active" : ""}>{l.label}</Link>
          ))}
        </nav>
        <div style={{ height: 12 }} />
        <ThemeToggle />
        <p className="small" style={{ padding: "12px 8px" }}>
          <span className={`dot ${online ? "on" : ""}`} />
          {online ? "Backend online" : "Backend offline"}
          <br />{API_URL}
          {!online && <><br />Start it with: <code>python main.py api</code></>}
        </p>
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}
