import { useEffect, useMemo, useRef, useState } from "react";

import { api, API_BASE } from "../api/client";
import type { TiledWorldScene } from "../phaser/TiledWorldScene";
import { renderStepInto } from "../phaser/driveStep";
import { loadMapSource, type MapSource } from "../phaser/mapSource";
import type { MapFocus, MapSelect, StepEvent } from "../types";
import MapStage from "./MapStage";
import WorldClock from "./WorldClock";

interface Props {
  worldId: string;
  step: StepEvent | null;
  // Fired when a step's render TIMELINE fully finishes. The driver waits for this
  // before advancing, so step N is fully rendered before step N+1 begins.
  onRendered?: (stepNo: number) => void;
  focus: MapFocus;
  onSelect: (t: MapSelect) => void;
  onFocusChange: (next: MapFocus) => void; // toolbar: toggle follow / clear / drop a chip
  // Immersive is owned by WorldView, not the map, so the whole surface (playback bar
  // included) goes immersive together; the map only reflects the flag in its button.
  immersive: boolean;
  onToggleImmersive: () => void;
}

/**
 * The live 2D map: a Phaser scene rendering the world's native Tiled map (the same .tmj
 * the engine built on) with agents overlaid from the live step stream. Movement follows
 * the engine transit contract; no client pathfinding or bundled map copy — the backend
 * owns the one map artifact.
 */
export default function LiveWorldMap({
  worldId,
  step,
  onRendered,
  focus,
  onSelect,
  onFocusChange,
  immersive,
  onToggleImmersive,
}: Props) {
  const [scene, setScene] = useState<TiledWorldScene | null>(null);
  const sceneRef = useRef<TiledWorldScene | null>(null);
  sceneRef.current = scene;
  const onRenderedRef = useRef(onRendered);
  onRenderedRef.current = onRendered;
  // id → name / identity colour for the focus chips, from the current step's figures.
  // Both tiers: NPC bodies can be clicked into focus too, and reading only `agent_states`
  // would leave their chip showing a bare id.
  const nameOf = useMemo(() => {
    const m = new Map<string, string>();
    if (step) {
      for (const s of Object.values(step.agent_states)) m.set(s.agent_id, s.agent_name);
      for (const n of step.npcs ?? []) if (n.name) m.set(n.npc_id, n.name);
    }
    return (id: string) => m.get(id) ?? "某人";
  }, [step]);
  const colorOf = useMemo(() => {
    const m = new Map<string, string>();
    if (step) for (const s of Object.values(step.agent_states)) if (s.color) m.set(s.agent_id, s.color);
    return (id: string) => m.get(id) ?? null;
  }, [step]);
  const placeName = focus.place ? sceneRef.current?.locationName(focus.place.id) ?? focus.place.id : null;
  const ready = scene !== null;
  const [animSpeed, setAnimSpeed] = useState(1); // animation-speed multiplier
  // null = loading, false = no map, MapSource = map + its art, ready to render.
  const [mapSource, setMapSource] = useState<MapSource | null | false>(null);

  // Built off API_BASE (not BASE_URL) so a remote-backend deploy hits the backend. The
  // routes and their degradation are loadMapSource's business.
  useEffect(() => {
    let live = true;
    setMapSource(null);
    loadMapSource(`${API_BASE}/api/worlds/${worldId}`)
      .then((src) => live && setMapSource(src))
      .catch(() => live && setMapSource(false));
    return () => {
      live = false;
    };
  }, [worldId]);

  useEffect(() => {
    if (ready) sceneRef.current?.setSpeed(animSpeed);
  }, [ready, animSpeed]);

  // Feed the scene each agent's gender + age (from the step-0 /agents profiles) so it can
  // pick the right body. Keyed on the scene, not the world id: a scene rebuilt under the
  // same id would otherwise get no cast and every figure falls back to the default body.
  const castFed = useRef<TiledWorldScene | null>(null);
  useEffect(() => {
    const target = sceneRef.current;
    if (!ready || !target || castFed.current === target) return;
    castFed.current = target;
    api
      .agents(worldId)
      .then((cast) => target.setCast(
        Object.fromEntries(cast.map((p) => [p.agent_id, { gender: p.gender, age: p.age }])),
      ))
      .catch(() => {}); // no roster → figures stay on the default adult body
  }, [ready, worldId]);

  useEffect(() => {
    if (ready) sceneRef.current?.setFocus(focus);
  }, [ready, focus]);

  useEffect(() => {
    if (!ready || !sceneRef.current || !step) return;
    // A new step supersedes any in-flight render (scrub); its onRendered is
    // suppressed via `cancelled` so only the current step's completion advances.
    let cancelled = false;
    const stepNo = step.step;
    // A render that threw still finished (renderStepInto hands the error back). Report it
    // anyway: the driver's gate is only reopened here, so an unreported failure stops
    // playback for good.
    renderStepInto(sceneRef.current, step).then(() => {
      if (!cancelled) onRenderedRef.current?.(stepNo);
    });
    return () => {
      cancelled = true;
    };
  }, [ready, step]);

  // With no map the scene never mounts and onRendered never fires, freezing the driver's
  // gate (clock, roster, cards, DirectorBar all stall). No render timeline to wait for, so
  // this step counts as already drawn.
  useEffect(() => {
    if (mapSource !== false || !step) return;
    onRenderedRef.current?.(step.step);
  }, [mapSource, step]);

  if (mapSource === false) {
    return (
      <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6">
        <h3 className="text-sm font-semibold tracking-wider text-slate-400 uppercase mb-2">
          🗺️ 世界地图
        </h3>
        {/* The two fetches loadMapSource refuses to carry on without — a missing
            tileset or no derived ground still draws, so neither lands here. */}
        <p className="text-slate-500 text-sm">取不到这个世界的地图文件或人物资产。</p>
      </div>
    );
  }

  return (
    // The map fills its container and never escapes it (no fixed overlay); WorldView
    // handles immersive by collapsing its own chrome.
    <div
      className={
        immersive
          ? "bg-[#150f24] h-full min-h-0 flex flex-col"
          : "bg-[#150f24] border border-slate-800 rounded-2xl overflow-hidden h-full min-h-[300px] flex flex-col"
      }
    >
      {/* overflow-hidden: a transiently oversized canvas (mid-refit) must not bleed over
          the footer. */}
      <div className="relative flex-1 min-h-0 overflow-hidden">
        {mapSource && (
          <MapStage
            source={mapSource}
            onScene={setScene}
            onSelect={onSelect}
            // A hand on the camera drops follow, keeps the rest of the focus.
            onGrabCamera={() => onFocusChange({ ...focus, follow: false })}
            // Immersive collapses the shell's chrome, which resizes this box through a
            // containing block the observer can straddle.
            refit={immersive}
          />
        )}
        {!step && (
          <div className="absolute inset-0 grid place-items-center pointer-events-none">
            <span className="text-slate-500 text-sm">运行世界以在地图上看到角色活动。</span>
          </div>
        )}
        {/* Focus toolbar: subject chips and follow / clear as one left-aligned cluster.
            pr-[27rem] reserves the top-right for the director console overlay, or the chips
            wrap under it; below `sm` the console is near full-width, so no reserve. */}
        {(focus.agents.length > 0 || focus.place) && (
          <div className="absolute top-2 left-2 right-2 sm:pr-[27rem] flex items-center flex-wrap gap-1.5 pointer-events-none">
            <span className="text-[10px] text-slate-400 bg-slate-950/80 px-1.5 py-0.5 rounded pointer-events-auto">🎯 焦点</span>
            {focus.agents.map((id) => {
              const c = colorOf(id);
              return (
                <button
                  key={id}
                  onClick={() => onFocusChange({ ...focus, agents: focus.agents.filter((a) => a !== id) })}
                  className={`pointer-events-auto text-[11px] px-2 py-0.5 rounded-md border font-medium transition hover:brightness-125 ${
                    c ? "" : "border-violet-500/60 bg-violet-950/70 text-violet-200"
                  }`}
                  // Identity colour, with a faint wash behind (8-digit hex = colour + alpha);
                  // violet only when the step carries no colour for this id.
                  style={c ? { color: c, borderColor: c, backgroundColor: `${c}22` } : undefined}
                  title="移出焦点"
                >
                  {nameOf(id)} ✕
                </button>
              );
            })}
            {placeName && (
              <button
                onClick={() => onFocusChange({ ...focus, place: null })}
                className="pointer-events-auto text-[11px] px-2 py-0.5 rounded-md border border-violet-500/60 bg-violet-950/70 text-violet-200 hover:border-violet-400 transition"
                title="取消地点聚焦"
              >
                📍 {placeName} ✕
              </button>
            )}
            <button
              onClick={() => onFocusChange({ ...focus, follow: !focus.follow })}
              className={`pointer-events-auto text-[11px] px-2 py-0.5 rounded-md border transition ${
                focus.follow
                  ? "bg-indigo-600 border-indigo-500 text-white"
                  : "bg-slate-950/80 border-slate-800 text-slate-300 hover:border-slate-700"
              }`}
            >
              {focus.follow ? "🎥 跟随中" : "🎥 跟随"}
            </button>
            <button
              onClick={() => onFocusChange({ agents: [], place: null, follow: focus.follow })}
              className="pointer-events-auto text-[11px] px-2 py-0.5 rounded-md border bg-slate-950/80 border-slate-800 text-slate-300 hover:border-slate-700 transition"
            >
              清除焦点
            </button>
          </div>
        )}
      </div>
      <div className="shrink-0 flex items-center justify-end gap-1 px-3 py-2 border-t border-slate-800/70 bg-slate-950/50">
        {/* The clock shows here only in immersive, when the header that carries it is collapsed. */}
        {immersive && step && <WorldClock step={step} size="sm" className="mr-auto" />}
        <span className="text-[10px] text-slate-500 mr-1">动画速度</span>
        {[0.25, 0.5, 1, 2].map((mult) => (
          <button
            key={mult}
            onClick={() => setAnimSpeed(mult)}
            className={`text-[11px] px-2 py-0.5 rounded-md border transition ${
              animSpeed === mult
                ? "bg-indigo-600 border-indigo-500 text-white"
                : "bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700"
            }`}
          >
            {mult}×
          </button>
        ))}
        <button
          onClick={() => sceneRef.current?.resetView()}
          title="复位视角（回到整图）"
          className="ml-1 text-[11px] px-2 py-0.5 rounded-md border bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700 hover:text-white transition"
        >
          ⟳ 复位
        </button>
        <button
          onClick={onToggleImmersive}
          title={immersive ? "退出沉浸 (Esc)" : "沉浸模式（地图铺满）"}
          className="ml-1 text-[11px] px-2 py-0.5 rounded-md border bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700 hover:text-white transition"
        >
          {immersive ? "⤢ 退出" : "⛶ 沉浸"}
        </button>
      </div>
    </div>
  );
}
