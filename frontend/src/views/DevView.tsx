// Developer tools: four tabs, see TABS below.
//
// Debug and Director are separate: Debug tunes agent cognition, Director a human-facing product
// surface (free text in, validation decides). Their failure modes differ in kind.
//
// The world selector sits above the tabs (the one axis all four share). One control per axis:
// tabs only pick the tool; how to view the data belongs to the trace page's own Scope / Group by.

import { lazy, Suspense, useEffect, useState } from "react";
import { Link, useHref, useSearchParams } from "react-router-dom";

import { AlphaTag } from "../dev/ui";
import { useWorlds } from "../lib/worldsContext";

const TraceTab = lazy(() => import("../dev/TraceTab"));
const AuditTab = lazy(() => import("../dev/AuditTab"));
const PromptReplay = lazy(() => import("../dev/PromptReplay"));
const StageRunner = lazy(() => import("../dev/StageRunner"));
const DirectorConsole = lazy(() => import("../dev/DirectorConsole"));
const RecallInspector = lazy(() => import("../dev/RecallInspector"));

type Tab = "trace" | "audit" | "bench" | "director";
type Instrument = "stage" | "replay" | "recall";

const TABS: { id: Tab; label: string; hint: string }[] = [
  { id: "trace", label: "Trace", hint: "every LLM call, and the totals" },
  { id: "audit", label: "Audit", hint: "score the narrative" },
  { id: "bench", label: "Debug", hint: "run one cognition path on its own" },
  { id: "director", label: "Director", hint: "assemble → call → validate" },
];

const INSTRUMENTS: { id: Instrument; label: string; hint: string; alpha?: boolean }[] = [
  { id: "stage", label: "Stage suite", hint: "re-run a stage with the current code", alpha: true },
  { id: "replay", label: "Prompt replay", hint: "edit the text of a call that happened" },
  { id: "recall", label: "Recall", hint: "the whole retrieval funnel" },
];

const isTab = (v: string | null): v is Tab => TABS.some((t) => t.id === v);
const isInstrument = (v: string | null): v is Instrument => INSTRUMENTS.some((i) => i.id === v);

export default function DevView() {
  const { worlds, status } = useWorlds();
  const [params, setParams] = useSearchParams();
  // tab / instrument / call are read from the URL once, on open: clicking a #call_id on the trace
  // page opens Prompt replay in a new browser tab, leaving the original trace page (filters,
  // expansion, scroll) intact. Switching tabs afterwards doesn't write back to the URL.
  const [tab, setTab] = useState<Tab>(() => (isTab(params.get("tab")) ? (params.get("tab") as Tab) : "trace"));
  const [instrument, setInstrument] = useState<Instrument>(() =>
    isInstrument(params.get("instrument")) ? (params.get("instrument") as Instrument) : "stage",
  );
  const [seedCallId] = useState(() => params.get("call") ?? "");

  const world = params.get("world") ?? "";
  useEffect(() => {
    if (!world && worlds.length) setParams({ world: worlds[0].world_id }, { replace: true });
  }, [world, worlds, setParams]);

  const replayHref = useHref("/dev");
  const pickCall = (id: string) => {
    const q = new URLSearchParams({ world, tab: "bench", instrument: "replay", call: id });
    window.open(`${replayHref}?${q}`, "_blank", "noopener");
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* The header draws no bottom rule; the nav owns it. With both drawn, the selected tab's
          underline sits on the header's line and it no longer reads as tabs vs. a row of labels. */}
      <header className="sticky top-0 z-10 bg-slate-950/95 px-6 pt-3 backdrop-blur">
        <div className="flex flex-wrap items-center gap-4">
          {/* The instruments don't live in the product shell (see main.tsx), so they carry their own way
              back: there's no sidebar here. */}
          <Link to="/" title="Back to Asamana" className="text-slate-500 hover:text-slate-200 transition leading-none">
            ‹
          </Link>
          <h1 className="text-sm font-semibold">Developer tools</h1>
          <select
            value={world}
            onChange={(e) => setParams({ world: e.target.value })}
            className="rounded-lg border border-slate-800 bg-slate-900 px-2.5 py-1.5 text-xs text-slate-200 focus:border-indigo-500/70 focus:outline-none"
          >
            {worlds.map((w) => (
              <option key={w.world_id} value={w.world_id}>
                {w.world_name || w.world_id}
              </option>
            ))}
          </select>
          <span className="font-mono text-[10px] text-slate-600">{world}</span>
        </div>
        {/* Folder-style tabs: a baseline + the selected tab's panel background + a top highlight
            bar. Don't draw these lines with Tailwind `border-*`: preflight is disabled (see
            tailwind.config.cjs), so borders render at width 0. Hence an inset box-shadow and a
            solid child element. */}
        <nav className="mt-3 flex items-end gap-1 shadow-[inset_0_-1px_0_theme(colors.slate.700)]">
          {TABS.map((t) => {
            const on = tab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                aria-current={on ? "page" : undefined}
                className={`relative rounded-t-lg px-4 pb-2 pt-2.5 text-sm transition ${
                  on ? "bg-slate-900" : "hover:bg-slate-900/40"
                }`}
              >
                {on && (
                  <span className="absolute inset-x-0 top-0 h-[2px] rounded-t bg-indigo-500" />
                )}
                <span className={on ? "font-semibold text-slate-100" : "text-slate-400"}>
                  {t.label}
                </span>
                <span className={`ml-2 text-[10px] ${on ? "text-slate-400" : "text-slate-600"}`}>
                  {t.hint}
                </span>
              </button>
            );
          })}
        </nav>
      </header>

      <main className="px-6 py-5">
        {status === "offline" && (
          <div className="mb-3 text-xs text-amber-400">
            Backend not responding — the world list and data may be stale.
          </div>
        )}
        {!world ? (
          <div className="py-16 text-center text-xs text-slate-500">No worlds yet.</div>
        ) : (
          <Suspense fallback={<div className="py-16 text-center text-xs text-slate-500">Loading…</div>}>
            {tab === "trace" && <TraceTab key={world} world={world} onPickCallId={pickCall} />}
            {tab === "audit" && <AuditTab key={world} world={world} />}
            {tab === "director" && <DirectorConsole key={world} world={world} />}
            {tab === "bench" && (
              <div className="flex gap-5">
                <aside className="w-44 shrink-0 space-y-1">
                  {INSTRUMENTS.map((i) => (
                    <button
                      key={i.id}
                      onClick={() => setInstrument(i.id)}
                      className={`w-full rounded-lg border px-3 py-2 text-left transition ${
                        instrument === i.id
                          ? "border-indigo-500/40 bg-indigo-600/15 text-indigo-100"
                          : "border-slate-800 bg-slate-900/40 text-slate-400 hover:border-slate-700 hover:text-slate-200"
                      }`}
                    >
                      <div className="flex items-center gap-1.5 text-xs font-medium">
                        {i.label}
                        {i.alpha && <AlphaTag />}
                      </div>
                      <div className="mt-0.5 text-[10px] leading-snug text-slate-500">{i.hint}</div>
                    </button>
                  ))}
                </aside>
                <div className="min-w-0 flex-1">
                  {instrument === "stage" && <StageRunner key={world} world={world} />}
                  {instrument === "replay" && (
                    <PromptReplay key={world} world={world} seedCallId={seedCallId} />
                  )}
                  {instrument === "recall" && <RecallInspector key={world} world={world} />}
                </div>
              </div>
            )}
          </Suspense>
        )}
      </main>
    </div>
  );
}
