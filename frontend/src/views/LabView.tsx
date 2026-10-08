import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { api, API_BASE } from "../api/client";
import CastBench from "../lab/CastBench";
import TemplateImport from "../lab/TemplateImport";
import { deleteTemplate } from "../lab/importTemplate";
import { loadMapSource, type MapSource } from "../phaser/mapSource";
import type { MapTemplate } from "../types";

const DEFAULT_TEMPLATE = "changan_iso";

// Deferred because both pull in MapStage and with it Phaser (~1.5 MB), which the cast
// bench — plain DOM — has no use for.
const MapBench = lazy(() => import("../lab/MapBench"));
const SceneBench = lazy(() => import("../lab/SceneBench"));

const BENCHES = [
  ["map", "Map"],
  ["cast", "Cast"],
  ["scenes", "Scenes"],
] as const;
type Bench = (typeof BENCHES)[number][0];

/**
 * Map workbench: the shell around the workbenches, and the one thing they share.
 *
 * That shared thing is the axis: a template and the artifact loaded from it, picked once
 * above everything (as the world selector sits above the developer tools' tabs) so no
 * bench holds a second answer. The artifact is fetched here, once, and a bench mounts
 * only with a real `MapSource`: loading and load failure belong to the template's owner.
 */
export default function LabView() {
  const [params, setParams] = useSearchParams();
  const template = params.get("t") || DEFAULT_TEMPLATE;
  const mode: Bench = BENCHES.find(([id]) => id === params.get("m"))?.[0] ?? "map";
  // Replace, never push: the back button should leave the workbench, not walk back through
  // every scene looked at on the way here.
  const setUrl = useCallback(
    (next: Record<string, string>) =>
      setParams(
        (prev) => {
          const p = new URLSearchParams(prev);
          for (const [k, v] of Object.entries(next)) p.set(k, v);
          return p;
        },
        { replace: true },
      ),
    [setParams],
  );

  const pickScene = useCallback((id: string) => setUrl({ s: id }), [setUrl]);

  const [templates, setTemplates] = useState<MapTemplate[]>([]);
  // null = loading, string = the reason it failed, MapSource = ready to render.
  const [source, setSource] = useState<MapSource | string | null>(null);
  const [importing, setImporting] = useState(false);
  // Two-step, in place: deleting a map is not undoable from here, so the button asks
  // before it acts rather than a dialog interrupting the bench behind it.
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  // Re-importing over the template already on screen changes no state, so the load
  // below needs something else to move before it will fetch the new files.
  const [reload, setReload] = useState(0);

  const refreshTemplates = useCallback(
    () => api.listTemplates().then(setTemplates).catch(() => setTemplates([])),
    [],
  );

  useEffect(() => {
    refreshTemplates();
  }, [refreshTemplates]);

  // The list has to hold the new name before the picker is pointed at it, or the
  // <select> falls back to showing its first option while the address says another.
  const imported = useCallback(
    async (name: string) => {
      await refreshTemplates();
      setUrl({ t: name });
      setReload((n) => n + 1);
    },
    [refreshTemplates, setUrl],
  );

  const removeMap = useCallback(async () => {
    const result = await deleteTemplate(template);
    setConfirmDelete(false);
    if (!result.ok) {
      setDeleteError(result.message);
      return;
    }
    setDeleteError(null);
    // Point the picker at whatever is left before the list is re-read, or it shows its
    // first option while the address still names the map that is gone.
    const left = await api.listTemplates().catch(() => []);
    setTemplates(left);
    setUrl({ t: left[0]?.template ?? "" });
    setReload((n) => n + 1);
  }, [template, setUrl]);

  // The template's map artifact, through the same loader the observation view uses (only
  // the address differs), so the lab renders what ships.
  useEffect(() => {
    let live = true;
    setSource(null);
    loadMapSource(`${API_BASE}/api/templates/${template}`)
      .then((src) => live && setSource(src))
      .catch((err: Error) =>
        live &&
        setSource(`Could not load the ground or character assets of map "${template}" (${err.message}).`),
      );
    return () => {
      live = false;
    };
  }, [template, reload]);

  return (
    <div className="h-screen flex flex-col min-h-0">
      {/* ---- the axis ------------------------------------------------------
          Three bands: who you are (left), what you are looking through (middle), what
          you are looking at and may do to it (right), so a destructive action never
          sits beside a mode switch at the same weight. */}
      <header className="relative shrink-0 h-12 flex items-center gap-4 px-4 border-b border-slate-800/70 bg-slate-950">
        {/* The instrument stands outside the product shell, so it carries its own way
            back — there is no sidebar here to click. */}
        <Link
          to="/"
          title="Back to Asamana"
          className="text-slate-500 hover:text-slate-200 transition text-lg leading-none -mt-0.5"
        >
          ‹
        </Link>
        <h1 className="text-sm font-semibold text-slate-100 tracking-wide whitespace-nowrap">
          Map workbench
        </h1>

        {/* Three benches on one axis, as siblings so switching unmounts the others: the
            scene bench holds a running Phaser game and a window-wide keyboard map. */}
        <nav className="flex items-center gap-0.5 rounded-lg border border-slate-800 bg-slate-900/60 p-0.5">
          {BENCHES.map(([id, label]) => (
            <button
              key={id}
              onClick={() => setUrl({ m: id })}
              className={`text-[12px] px-3 py-1 rounded-md transition ${
                mode === id
                  ? "bg-slate-700/80 text-slate-50 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {label}
            </button>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-2">
          {/* The map is the axis every bench sits on, so it gets the weight: labeled by
              world_name, then the directory name the API, the URL and this panel take. */}
          <div className="relative">
            <select
              value={template}
              onChange={(e) => setUrl({ t: e.target.value })}
              className="appearance-none w-64 text-[12px] bg-slate-900 border border-slate-700 rounded-lg pl-3 pr-8 py-1.5 text-slate-100 hover:border-slate-600 focus:outline-none focus:border-indigo-500/70 transition"
            >
              {templates.length === 0 && <option value={template}>{template}</option>}
              {templates.map((t) => (
                <option key={t.template} value={t.template}>
                  {t.world_name ? `${t.world_name} · ${t.template}` : t.template}
                </option>
              ))}
            </select>
            <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 text-[10px]">
              ▾
            </span>
          </div>

          {confirmDelete ? (
            <>
              <span className="text-[12px] text-amber-300 whitespace-nowrap">
                Delete "{template}"? Worlds already built are not affected.
              </span>
              <button
                onClick={removeMap}
                className="text-[12px] px-3 py-1.5 rounded-lg bg-rose-700 border border-rose-600 text-white hover:bg-rose-600 transition"
              >
                Delete
              </button>
              <button
                onClick={() => setConfirmDelete(false)}
                className="text-[12px] px-3 py-1.5 rounded-lg text-slate-400 hover:text-slate-200 transition"
              >
                Cancel
              </button>
            </>
          ) : (
            <>
              <button
                data-import-trigger
                onClick={() => setImporting((on) => !on)}
                className={`text-[12px] px-3 py-1.5 rounded-lg border transition ${
                  importing
                    ? "bg-indigo-600 border-indigo-500 text-white"
                    : "bg-slate-900 border-slate-700 text-slate-300 hover:border-slate-600 hover:text-slate-100"
                }`}
              >
                Import map
              </button>
              {/* Quiet until you reach for it: it cannot be undone from here, so it has
                  no business competing for attention with the button beside it. */}
              <button
                onClick={() => {
                  setDeleteError(null);
                  setConfirmDelete(true);
                }}
                disabled={!templates.length}
                title="Delete this map"
                className="text-[12px] px-2 py-1.5 rounded-lg text-slate-500 hover:text-rose-300 hover:bg-rose-950/30 transition disabled:opacity-30 disabled:hover:text-slate-500 disabled:hover:bg-transparent"
              >
                Delete
              </button>
            </>
          )}
        </div>
        {/* Hangs off the header so it floats over the bench instead of pushing it down. */}
        {importing && (
          <TemplateImport onClose={() => setImporting(false)} onImported={imported} />
        )}
      </header>

      {deleteError && (
        <p className="shrink-0 px-4 py-2 text-[12px] text-rose-400 border-b border-slate-800/70 bg-rose-950/20">
          {deleteError}
        </p>
      )}

      <div className="flex-1 min-h-0 flex">
        {source === null && (
          <div className="flex-1 grid place-items-center text-slate-500 text-sm">Loading map assets…</div>
        )}
        {typeof source === "string" && (
          <div className="flex-1 grid place-items-center text-rose-300/80 text-sm px-8 text-center">
            {source}
          </div>
        )}
        {source !== null && typeof source !== "string" && mode === "map" && (
          <Suspense fallback={<div className="flex-1" />}>
            <MapBench source={source} />
          </Suspense>
        )}
        {source !== null && typeof source !== "string" && mode === "cast" && (
          <CastBench cast={source.characters} />
        )}
        {source !== null && typeof source !== "string" && mode === "scenes" && (
          <Suspense fallback={<div className="flex-1" />}>
            <SceneBench
              template={template}
              source={source}
              pickId={params.get("s")}
              onPick={pickScene}
            />
          </Suspense>
        )}
      </div>
    </div>
  );
}
