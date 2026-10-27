import { type ReactNode, useEffect, useRef, useState } from "react";

import { identityOf } from "../lib/worldTable";
import { API_BASE } from "../api/client";
import type { TiledWorldScene } from "../phaser/TiledWorldScene";
import { renderStepInto } from "../phaser/driveStep";
import { loadMapSource, type MapSource } from "../phaser/mapSource";
import { useCast, loadCast } from "../lib/cast";
import { placeName, usePlaces, loadPlaces } from "../lib/places";
import { identityColor, toHex } from "../lib/color";
import { useNpcs, loadNpcs } from "../lib/npcs";
import type { Viewer } from "../lib/viewer";
import type { MapFocus, MapSelect, MapOperationTarget, StepView } from "../types";
import MapStage from "./MapStage";
import WorldClock from "./WorldClock";

interface Props {
  worldId: string;
  step: StepView | null;
  // Fired when a step's render TIMELINE fully finishes. The driver waits for this
  // before advancing, so step N is fully rendered before step N+1 begins.
  onRendered?: (stepNo: number) => void;
  focus: MapFocus;
  onSelect: (t: MapSelect) => void;
  onOperation?: (t: MapOperationTarget) => void;
  onFocusChange: (next: MapFocus) => void; // toolbar: toggle follow / clear / drop a chip
  // Immersive is owned by WorldView, not the map, so the whole surface (playback bar
  // included) goes immersive together; the map only reflects the flag in its button.
  immersive: boolean;
  onToggleImmersive: () => void;
  // Whose view the steps are filtered to; null for the director.
  viewer?: Viewer;
  // What floats over the map (the director's or the actor's console), kept inside the map
  // itself so it never covers the controls under it.
  children?: ReactNode;
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
  onOperation,
  onFocusChange,
  immersive,
  onToggleImmersive,
  viewer = null,
  children,
}: Props) {
  const { cast, status: castStatus, error: castError } = useCast(worldId);
  const { places, error: placeError } = usePlaces(worldId);
  const { npcs, status: npcStatus, error: npcError } = useNpcs(worldId);
  const [scene, setScene] = useState<TiledWorldScene | null>(null);
  const sceneRef = useRef<TiledWorldScene | null>(null);
  sceneRef.current = scene;
  const onRenderedRef = useRef(onRendered);
  onRenderedRef.current = onRendered;
  const nameOf = (id: string) => identityOf(id, cast, npcs)?.name ?? "某人";
  const colorOf = (id: string) => toHex(identityColor(identityOf(id, cast, npcs)?.color, id));
  const focusPlace = focus.place ? placeName(places, focus.place.id) : null;
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

  useEffect(() => {
    if (ready) sceneRef.current?.setFocus(focus);
  }, [ready, focus]);

  useEffect(() => {
    if (scene && castStatus === "ready") scene.setCast(cast);
  }, [scene, castStatus, cast]);

  useEffect(() => {
    if (scene && npcStatus === "ready") scene.setNpcs(npcs);
  }, [scene, npcStatus, npcs]);

  useEffect(() => {
    scene?.setPlaces(places);
  }, [scene, places]);

  useEffect(() => {
    if (!ready || !sceneRef.current || !step || castStatus !== "ready" || npcStatus !== "ready") return;
    // A new step supersedes any in-flight render (scrub); its onRendered is
    // suppressed via `cancelled` so only the current step's completion advances.
    let cancelled = false;
    const stepNo = step.step;
    sceneRef.current.setViewer(viewer);
    // A render that threw still finished (renderStepInto hands the error back). Report it
    // anyway: the driver's gate is only reopened here, so an unreported failure stops
    // playback for good.
    renderStepInto(sceneRef.current, step).then(() => {
      if (!cancelled) onRenderedRef.current?.(stepNo);
    });
    return () => {
      cancelled = true;
    };
  }, [scene, ready, step, viewer, castStatus, cast, npcStatus, npcs]);

  // With no map the scene never mounts and onRendered never fires, freezing the driver's
  // gate (clock, roster, cards, DirectorBar all stall). No render timeline to wait for, so
  // this step counts as already drawn.
  useEffect(() => {
    if ((mapSource !== false && castStatus !== "error" && npcStatus !== "error") || !step) return;
    onRenderedRef.current?.(step.step);
  }, [mapSource, step, castStatus, npcStatus]);

  if (mapSource === false) {
    return (
      <div className="relative h-full bg-slate-900/40 border border-slate-800 rounded-2xl p-6">
        <h3 className="text-sm font-semibold tracking-wider text-slate-400 uppercase mb-2">
          🗺️ 世界地图
        </h3>
        {/* The two fetches loadMapSource refuses to carry on without — a missing
            tileset or no derived ground still draws, so neither lands here. */}
        <p className="text-slate-500 text-sm">地图或角色形象加载失败。</p>
        {placeError && <button className="absolute bottom-20 left-2 text-xs text-rose-300 bg-slate-950/90 rounded px-3 py-2" onClick={() => { void loadPlaces(worldId).catch(() => {}); }}>地点资料加载失败，点击重试</button>}
        {npcError && <button className="absolute bottom-10 left-2 text-xs text-rose-300 bg-slate-950/90 rounded px-3 py-2" onClick={() => { void loadNpcs(worldId).catch(() => {}); }}>NPC 资料加载失败，点击重试</button>}
        {castError && <button className="absolute bottom-2 left-2 text-xs text-rose-300 bg-slate-950/90 rounded px-3 py-2" onClick={() => { void loadCast(worldId).catch(() => {}); }}>角色资料加载失败，点击重试</button>}
        {children}
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
            onOperation={onOperation}
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
            pr-[27rem] reserves the top-right for the director console or the actor's console, or
            the chips wrap under it; below `sm` either is near full-width, so no reserve. */}
        {(focus.agents.length > 0 || focus.place) && (
          <div className="absolute top-2 left-2 right-2 sm:pr-[27rem] flex items-center flex-wrap gap-1.5 pointer-events-none">
            <span className="text-[10px] text-slate-400 bg-slate-950/80 px-1.5 py-0.5 rounded pointer-events-auto">🎯 焦点</span>
            {focus.agents.map((id) => {
              const c = colorOf(id);
              return (
                <button
                  key={id}
                  onClick={() => onFocusChange({ ...focus, agents: focus.agents.filter((a) => a !== id) })}
                  className="pointer-events-auto text-[11px] px-2 py-0.5 rounded-md border font-medium transition hover:brightness-125"
                  style={{ color: c, borderColor: c, backgroundColor: `${c}22` }}
                  title="移出焦点"
                >
                  {nameOf(id)} ✕
                </button>
              );
            })}
            {focusPlace && (
              <button
                onClick={() => onFocusChange({ ...focus, place: null })}
                className="pointer-events-auto text-[11px] px-2 py-0.5 rounded-md border border-violet-500/60 bg-violet-950/70 text-violet-200 hover:border-violet-400 transition"
                title="取消地点聚焦"
              >
                📍 {focusPlace} ✕
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
        {placeError && <button className="absolute bottom-20 left-2 text-xs text-rose-300 bg-slate-950/90 rounded px-3 py-2" onClick={() => { void loadPlaces(worldId).catch(() => {}); }}>地点资料加载失败，点击重试</button>}
        {npcError && <button className="absolute bottom-10 left-2 text-xs text-rose-300 bg-slate-950/90 rounded px-3 py-2" onClick={() => { void loadNpcs(worldId).catch(() => {}); }}>NPC 资料加载失败，点击重试</button>}
        {castError && <button className="absolute bottom-2 left-2 text-xs text-rose-300 bg-slate-950/90 rounded px-3 py-2" onClick={() => { void loadCast(worldId).catch(() => {}); }}>角色资料加载失败，点击重试</button>}
        {children}
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
          title="重置地图视角（显示完整地图）"
          className="ml-1 text-[11px] px-2 py-0.5 rounded-md border bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700 hover:text-white transition"
        >
          ⟳ 重置视角
        </button>
        <button
          onClick={onToggleImmersive}
          title={immersive ? "退出沉浸（Esc）" : "沉浸模式（地图铺满）"}
          className="ml-1 text-[11px] px-2 py-0.5 rounded-md border bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700 hover:text-white transition"
        >
          {immersive ? "⤢ 退出沉浸" : "⛶ 沉浸模式"}
        </button>
      </div>
    </div>
  );
}
