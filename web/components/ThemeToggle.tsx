"use client";
import { useEffect, useState } from "react";

type Theme = "system" | "light" | "dark";
const KEY = "cnai-theme";

function apply(t: Theme) {
  const root = document.documentElement;
  if (t === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", t);
}

/** Light / dark / follow-the-system switch. The choice is kept in this browser only. */
export default function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");
  useEffect(() => {
    try { const v = localStorage.getItem(KEY); if (v === "light" || v === "dark") setTheme(v); } catch { /* storage blocked */ }
  }, []);
  const choose = (t: Theme) => {
    setTheme(t); apply(t);
    try { if (t === "system") localStorage.removeItem(KEY); else localStorage.setItem(KEY, t); } catch { /* storage blocked */ }
  };
  const items: [Theme, string][] = [["light", "Light"], ["dark", "Dark"], ["system", "Auto"]];
  return (
    <div className="theme" role="group" aria-label="Theme">
      {items.map(([t, label]) => (
        <button key={t} type="button" className={theme === t ? "on" : ""} aria-pressed={theme === t} onClick={() => choose(t)}>{label}</button>
      ))}
    </div>
  );
}
