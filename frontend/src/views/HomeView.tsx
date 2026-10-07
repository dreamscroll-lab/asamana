// Home / "new world": a centred theme prompt (Claude-style). The world LIST lives
// in the sidebar; this view is purely the create-a-world experience + its build
// progress. On success it hands off to the review/confirm stage.
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { api } from "../api/client";
import BuildingScreen from "../components/BuildingScreen";
import MapPicker from "../components/MapPicker";
import { NO_MODEL_KEYS_HINT, useDeployment } from "../lib/deployment";
import { useWorlds } from "../lib/worldsContext";
import type { BuildJob, MapTemplate, ThemePreset } from "../types";

/** How tall the theme box may grow before it scrolls inside (about six lines), so it doesn't
 *  push the sample themes off the fold. Same technique as DirectorBar's INPUT_MAX_PX. */
const THEME_MAX_PX = 400;

/** …and how tall it is before a word is typed: a one-line box reads as a search field, and
 *  this screen asks for a few sentences. */
const THEME_MIN_PX = 120;

const BUILD_POLL_GIVE_UP = 5;

export default function HomeView() {
  const navigate = useNavigate();
  const { refresh } = useWorlds();
  const deployment = useDeployment();
  const modelKeys = deployment?.model_keys ?? false;
  const [theme, setTheme] = useState("");
  const [building, setBuilding] = useState<BuildJob | null>(null);
  // Consecutive failed polls of a build. One is a network blip, not the build failing: it is still
  // running on the server, so only a run of them gives up on it.
  const [pollFailures, setPollFailures] = useState(0);
  const [error, setError] = useState("");
  // Sample themes come from the backend's content file, so changing them is an
  // edit + a reload. No local copy: a second source would drift from the first.
  const [presets, setPresets] = useState<ThemePreset[]>([]);
  const [templates, setTemplates] = useState<MapTemplate[]>([]);
  // null = let the backend read the theme and pick.
  const [template, setTemplate] = useState<string | null>(null);
  const box = useRef<HTMLTextAreaElement>(null);

  // Grow to fit what has been typed, capped. Reset height to 0 before measuring, or
  // scrollHeight reports the taller of (content, current box) and the field never shrinks.
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(Math.max(el.scrollHeight, THEME_MIN_PX), THEME_MAX_PX)}px`;
  }, [theme]);

  useEffect(() => {
    // Either failure costs that section of the form, not the ability to create
    // a world — the theme box alone is enough to build one.
    api
      .listPresets()
      .then(setPresets)
      .catch(() => setPresets([]));
    api
      .listTemplates()
      .then(setTemplates)
      .catch(() => setTemplates([]));
  }, []);

  // Poll the build job; on success go straight to the review/confirm stage.
  useEffect(() => {
    if (!building || building.status !== "building") return;
    // Left the page while a poll was out: its answer must not navigate the user away.
    let alive = true;
    const t = setTimeout(async () => {
      try {
        const job = await api.getBuildJob(building.job_id);
        if (!alive) return;
        setPollFailures(0);
        if (job.status === "building") {
          setBuilding(job);
        } else {
          setBuilding(null);
          await refresh();
          if (job.status === "failed") setError(job.error ?? "构建失败");
          else if (job.world_id) navigate(`/worlds/${job.world_id}/review`);
        }
      } catch (e) {
        if (!alive) return;
        if (pollFailures + 1 < BUILD_POLL_GIVE_UP) {
          setPollFailures((n) => n + 1);
          return;
        }
        setPollFailures(0);
        setError(String(e));
        setBuilding(null);
      }
    }, 1500);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [building, pollFailures, refresh, navigate]);

  async function create() {
    if (!modelKeys) return;
    setError("");
    const t = theme.trim();
    if (!t) {
      setError("请先输入一个世界主题！");
      return;
    }
    try {
      setBuilding(await api.createWorld(t, template));
    } catch (e) {
      setError(String(e));
    }
  }

  if (building) return <BuildingScreen />;

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 py-12 animate-[fadeIn_0.4s_ease-out]">
      <div className="w-full max-w-2xl">
        <div className="text-center mb-8">
          <img src="/logo.png" alt="" className="w-16 h-16 mx-auto mb-2 object-contain" />
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-violet-300 via-indigo-200 to-cyan-300">
              AI AS A HUMAN
            </span>
          </h1>
          <p className="text-slate-500 text-sm mt-3">
          如果有一天 AI 你变成了人，你最想做什么？
          </p>
        </div>

        <div className="relative">
          <textarea
            ref={box}
            rows={1}
            value={theme}
            onChange={(e) => setTheme(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") create();
            }}
            placeholder="输入叙事主题，例如：嘉靖时期的内阁…"
            className="w-full bg-slate-900 border border-slate-800 focus:border-indigo-500/80 focus:ring-1 focus:ring-indigo-500/40 rounded-2xl p-4 pr-4 pb-14 text-slate-200 placeholder-slate-600 text-sm focus:outline-none transition resize-none shadow-2xl"
          />
          {templates.length > 0 && (
            <div className="absolute bottom-3 left-3">
              <MapPicker templates={templates} value={template} onChange={setTemplate} />
            </div>
          )}
          <button
            onClick={create}
            disabled={!modelKeys || !theme.trim()}
            title={modelKeys ? undefined : NO_MODEL_KEYS_HINT}
            className="absolute bottom-3 right-3 px-5 py-2 rounded-xl text-xs font-semibold tracking-wider bg-indigo-600 text-white hover:bg-indigo-500 shadow-lg shadow-indigo-600/30 border border-indigo-500 transition flex items-center gap-2 disabled:bg-slate-800 disabled:text-slate-500 disabled:shadow-none disabled:border-slate-800 disabled:cursor-not-allowed"
          >
            <svg
              className="w-3.5 h-3.5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
            构建
          </button>
        </div>

        {/* Only once known: the line would otherwise flash on every load. */}
        {deployment?.model_keys === false && (
          <p className="mt-3 text-xs text-slate-500">{NO_MODEL_KEYS_HINT}</p>
        )}

        {/* Pills, not bare text, so they read as clickable; wrapped into a row rather
            than stacked so they read as a compact set of options. */}
        {presets.length > 0 && (
          <div className="mt-5">
            <p className="text-xs text-slate-600 mb-2 font-medium">试试这些</p>
            <div className="flex flex-wrap gap-2">
              {presets.map((p) => (
                <button
                  key={p.title}
                  onClick={() => setTheme(p.theme)}
                  className="px-3.5 py-1.5 rounded-full text-xs bg-slate-900/70 border border-slate-800 text-slate-300 hover:border-indigo-500/60 hover:bg-indigo-950/40 hover:text-indigo-100 transition focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400"
                >
                  {p.title}
                </button>
              ))}
            </div>
          </div>
        )}

        {error && (
          <div className="mt-4 p-3 bg-rose-950/20 border border-rose-900/40 rounded-xl text-rose-400 text-xs">
            ⚠️ {error}
          </div>
        )}
      </div>
    </div>
  );
}
