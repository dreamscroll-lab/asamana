import { useEffect, useRef, useState } from "react";
import { useHref } from "react-router-dom";

import { useDevTools } from "../lib/deployment";
import type { MapTemplate } from "../types";

// The list's key for "let the backend choose". Not a template name: those are directory
// names, and no directory is called "".
const AUTO = "";

/**
 * Which map a new world is built on, defaulting to 自动选图 (the backend's TemplateSelector
 * picks from the theme). The hovered or focused entry is described in a side panel: a native
 * tooltip would arrive late, truncate, and never show for the keyboard.
 */
export default function MapPicker({
  templates,
  value,
  onChange,
}: {
  templates: MapTemplate[];
  value: string | null;
  onChange: (template: string | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const [peek, setPeek] = useState<string | null>(null);
  const root = useRef<HTMLDivElement>(null);
  const devTools = useDevTools();

  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false);
    };
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", away);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);

  const chosen = templates.find((t) => t.template === value) ?? null;
  const shownKey = peek ?? value ?? AUTO;
  const shown = templates.find((t) => t.template === shownKey) ?? null;
  const labHref = useHref(`/lab?t=${encodeURIComponent(shown?.template ?? "")}`);

  const pick = (key: string) => {
    onChange(key === AUTO ? null : key);
    setOpen(false);
  };

  const row = (key: string, label: string, sub: string) => {
    const on = (value ?? AUTO) === key;
    return (
      <button
        key={key}
        onClick={() => pick(key)}
        onMouseEnter={() => setPeek(key)}
        onFocus={() => setPeek(key)}
        className={`w-full text-left px-3 py-2 rounded-lg transition flex items-center gap-2 focus:outline-none ${
          on ? "bg-indigo-600/20 text-indigo-100" : "text-slate-300 hover:bg-slate-800/70 focus:bg-slate-800/70"
        }`}
      >
        <span className="flex-1 min-w-0">
          <span className="block text-xs truncate">{label}</span>
          <span className="block text-[10px] text-slate-500 truncate">{sub}</span>
        </span>
        {on && <span className="text-indigo-300 text-xs">✓</span>}
      </button>
    );
  };

  return (
    <div ref={root} className="relative">
      <button
        onClick={() => {
          setPeek(null);
          setOpen((o) => !o);
        }}
        aria-expanded={open}
        title="选择这个世界建在哪张地图上"
        className={`px-3 py-1.5 rounded-lg text-xs border transition flex items-center gap-1.5 ${
          chosen
            ? "bg-indigo-950/50 border-indigo-500/50 text-indigo-100"
            : "bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200"
        }`}
      >
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"
          />
        </svg>
        {chosen ? chosen.world_name || chosen.template : "自动选图"}
      </button>

      {open && (
        <div
          onMouseLeave={() => setPeek(null)}
          className="absolute left-0 top-full mt-2 z-30 w-[min(34rem,calc(100vw-2rem))] flex flex-col sm:flex-row rounded-xl border border-slate-800 bg-slate-950 shadow-2xl shadow-black/50 overflow-hidden"
        >
          <div className="sm:w-48 shrink-0 p-1.5 space-y-0.5 sm:border-r border-b sm:border-b-0 border-slate-800 max-h-72 overflow-y-auto">
            {row(AUTO, "自动选图", "按主题挑选")}
            {templates.map((t) =>
              row(t.template, t.world_name || t.template, `${t.era_name} · ${t.location_count} 处地点`),
            )}
          </div>
          <div className="flex-1 min-w-0 p-4 space-y-2">
            {shown ? (
              <>
                <div className="flex items-baseline gap-2">
                  <span className="text-sm font-medium text-slate-100">{shown.world_name || shown.template}</span>
                  <span className="text-[10px] text-slate-500">
                    {shown.era_name} · {shown.location_count} 处地点
                  </span>
                </div>
                <p className="text-xs leading-relaxed text-slate-400">{shown.description}</p>
                {/* The workbench is a developer screen, gated with the rest of them. */}
                {devTools && (
                  <a
                    href={labHref}
                    target="_blank"
                    rel="noopener"
                    className="inline-block text-[11px] text-indigo-300 hover:text-indigo-200 transition"
                  >
                    Open in map workbench ↗
                  </a>
                )}
              </>
            ) : (
              <>
                <div className="text-sm font-medium text-slate-100">自动选图</div>
                <p className="text-xs leading-relaxed text-slate-400">
                  根据你写下的主题，从左侧的地图中挑一张最能承载它的。
                </p>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
