import { lazy, Suspense, useEffect, useState } from "react";
import { useNavigate, useOutletContext, useParams } from "react-router-dom";

import type { ShellContext } from "../App";
import { api } from "../api/client";
import { WorldSocket } from "../api/ws";
import CharacterBody from "../components/CharacterBody";
import CharacterCard, { type CardAgent } from "../components/CharacterCard";
import DirectorBar from "../components/DirectorBar";
import EditableWorldName from "../components/EditableWorldName";
import NarrativeStep from "../components/narrative/NarrativeStep";
import RelationshipGraph from "../components/RelationshipGraph";
import WorldClock from "../components/WorldClock";
import { useStepPlayer } from "../hooks/useStepPlayer";
import { NO_MODEL_KEYS_HINT, useModelKeys } from "../lib/deployment";
import { useWorlds } from "../lib/worldsContext";
import type { MapFocus, MapSelect, RunState, WorldMeta, WsFrame } from "../types";

// Phaser is ~1.4 MB — load the 2D map only when its view is opened.
const LiveWorldMap = lazy(() => import("../components/LiveWorldMap"));

/**
 * A world is an IDENTITY, not a parameter — so switching worlds is a REMOUNT.
 *
 * Everything below is world-scoped (step buffer, playback cursor, caches, focus, mode, the
 * Phaser scene). React Router reuses this element across `/worlds/A` → `/worlds/B`, so
 * without a key it would all survive the switch: world A's step would render onto world B's
 * map, and the step cache, keyed by step number, could serve world A's step 3 as world B's.
 *
 * Keyed rather than reset on an `id` effect: every `useState` added later would have to join
 * a reset list, and a remount cannot forget.
 */
export default function WorldView() {
  const { id = "" } = useParams();
  // A reset renumbers the run from step 1, so everything keyed by step number is stale too.
  const [resets, setResets] = useState(0);
  return (
    <WorldObservation key={`${id}:${resets}`} id={id} onReset={() => setResets((n) => n + 1)} />
  );
}

function WorldObservation({ id, onReset }: { id: string; onReset: () => void }) {
  const navigate = useNavigate();
  const { setImmersive: setShellImmersive } = useOutletContext<ShellContext>();
  const { worlds, refresh: refreshWorlds } = useWorlds();
  const modelKeys = useModelKeys();

  const [meta, setMeta] = useState<WorldMeta | null>(null);
  const [runState, setRunState] = useState<RunState>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState("");
  const [stepsInput, setStepsInput] = useState("50");

  const [mode, setMode] = useState<"live" | "playback">("live");
  // The 2D map is the permanent stage. Everything else is a selectively-summoned
  // lens onto it: the roster and narrative dock as push-drawers beside the map
  // (the map yields width), the relationship graph opens as an overlay over the
  // map, and immersive collapses all chrome so the map fills the shell.
  const [rosterOpen, setRosterOpen] = useState(true); // left character drawer
  const [logOpen, setLogOpen] = useState(true); // right narrative drawer
  const [graphOpen, setGraphOpen] = useState(false); // relationship-graph overlay
  // Body-chart overlay, of the selected agent. It shares the graph's place over the map, so
  // opening one closes the other.
  const [bodyOpen, setBodyOpen] = useState(false);
  const [immersive, setImmersive] = useState(false); // map fills the shell
  // Read one step of the feed alone: a temporary lens, not a mode. Other steps are hidden,
  // not dropped (the feed stays "the story so far", see useStepPlayer), and it is independent
  // of the playback scrubber, which says how far the world has got.
  const [isolatedStep, setIsolatedStep] = useState<number | null>(null);
  // Unified map focus: highlight (spotlight/dim/ring/card) + OPTIONAL camera follow.
  // Shared by the map, the character card, and the graph (one selection across all views).
  //
  // follow starts off: selecting is emphasis, not a camera act, and following by default
  // would re-centre every frame and undo any drag. Follow is an opt-in via the 🎥 button,
  // and a manual pan/zoom releases it (the scene calls onFocusChange).
  const [focus, setFocus] = useState<MapFocus>({ agents: [], place: null, follow: false });
  const selectOne = (agentId: string) => setFocus((f) => ({ ...f, agents: [agentId], place: null }));
  const handleMapSelect = (t: MapSelect) => {
    setFocus((f) => {
      if (t.kind === "clear") return { agents: [], place: null, follow: f.follow };
      if (t.kind === "location") return { ...f, place: { kind: "location", id: t.id }, agents: [] };
      // t.kind === "agent"
      const agents = f.agents.includes(t.id) ? f.agents.filter((a) => a !== t.id) : [...f.agents, t.id];
      return { ...f, agents, place: null };
    });
  };

  // --- Step playback state machine (live buffer + playback + graph) ---
  const {
    displayStep,
    feed,
    steps,
    pbIndex,
    playing,
    graph,
    onLiveStep,
    doneRendering,
    goTo,
    seek,
    togglePlay,
  } = useStepPlayer(id, mode, graphOpen || bodyOpen); // the chart reads ties off the graph

  const agents = displayStep ? Object.values(displayStep.agent_states) : [];
  const isBusy = ["running", "paused", "stopping"].includes(runState ?? "");

  // Roster for the shared character card (main characters first). Live per-step
  // state is passed separately so the card shows the same rich layout as review.
  const roster: CardAgent[] = [...agents]
    .sort((a, b) => Number(b.is_main_character) - Number(a.is_main_character))
    .map((a) => ({ id: a.agent_id, name: a.agent_name, is_main_character: a.is_main_character, color: a.color }));
  const activeAgentId = focus.agents[focus.agents.length - 1] ?? roster[0]?.id ?? null;
  const activeAgent = roster.find((a) => a.id === activeAgentId);

  // --- WebSocket: connect on mount, stream live status/snapshot/step ------
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const m = await api.getWorld(id);
        if (!mounted) return;
        // Unconfirmed worlds belong in the review/confirm stage, not observation.
        if (!m.confirmed) {
          navigate(`/worlds/${id}/review`);
          return;
        }
        setMeta(m);
        setRunState(m.run_state);
      } catch (e) {
        if (mounted) setError(String(e));
      }
    })();

    const onFrame = (frame: WsFrame) => {
      if (frame.type === "status") setRunState(frame.data.status);
      else onLiveStep(frame.data);
    };
    const socket = new WorldSocket(
      id,
      onFrame,
      () => setConnected(true),
      () => setConnected(false),
    );
    socket.connect();
    return () => {
      mounted = false;
      socket.close();
    };
  }, [id]);

  // Esc backs out of whatever narrowed the view, INNERMOST FIRST — the isolated step is a
  // lens inside the drawer, immersive is the whole shell, so one Esc should not throw away
  // both. (Owned here, not in the map — see LiveWorldMap header.)
  useEffect(() => {
    if (!immersive && isolatedStep === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (isolatedStep !== null) setIsolatedStep(null);
      else setImmersive(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [immersive, isolatedStep]);

  // The sidebar is the shell's, so immersive goes up rather than reaching across.
  // Cleared on unmount too: leaving the world must not leave the rail collapsed.
  useEffect(() => {
    setShellImmersive(immersive);
    return () => setShellImmersive(false);
  }, [immersive, setShellImmersive]);

  async function control(fn: () => Promise<unknown>) {
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(String(e));
    }
  }

  // Blank (or junk) means one step — see api.run. Never "until you remember to stop it".
  const runSteps = () => {
    const n = parseInt(stepsInput, 10);
    return Number.isFinite(n) && n > 0 ? n : undefined;
  };

  async function remove() {
    if (!confirm("删除该世界？此操作不可恢复。")) return;
    await control(() => api.deleteWorld(id));
    await refreshWorlds();
    navigate("/");
  }

  const BTN =
    "px-3 py-1.5 rounded-lg text-xs bg-slate-950 border border-slate-800 text-slate-300 hover:border-slate-700 hover:text-white transition disabled:opacity-40 disabled:cursor-not-allowed";
  const BTN_PRIMARY =
    "px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 border border-indigo-500 text-white hover:bg-indigo-500 transition disabled:opacity-40 disabled:cursor-not-allowed";
  const toggle = (on: boolean) => (on ? BTN_PRIMARY : BTN);
  // Panel collapse affordance — a plain text link, no boxed button chrome.
  const COLLAPSE = "text-xs text-slate-500 hover:text-slate-200 transition";
  const stateBadge =
    runState === "running"
      ? "text-teal-400 border-teal-700/60"
      : runState === "paused"
        ? "text-amber-400 border-amber-700/60"
        : runState === "failed"
          ? "text-rose-400 border-rose-700/60"
          : "text-slate-400 border-slate-700";

  const divider = <span className="w-px self-stretch bg-slate-800 mx-1 hidden sm:block" />;

  // The narrative feed body, the single narrative stream, in the right drawer. One list in
  // both modes ("the story so far"); only the empty-state copy differs, because the reason
  // differs.
  //
  // The feed honours `focus` (the multi-select, not just activeAgentId) like every other
  // panel, so one selection really spans all views.
  //
  // An isolated step survives new steps arriving (the feed shouldn't decide you're done
  // reading). It closes, saying so, when the step leaves the window (FEED_CAP, or scrubbing
  // back before it).
  const isolatedFeed = isolatedStep === null ? feed : feed.filter((ev) => ev.step === isolatedStep);
  const feedList = !feed.length ? (
    <p className="text-slate-500 text-sm">
      {mode === "live" ? "运行世界以生成叙事，每步实时追加。" : "该世界暂无历史步可回放。"}
    </p>
  ) : !isolatedFeed.length ? (
    <p className="text-slate-500 text-sm">
      第 {isolatedStep} 步已不在叙事流中。
      <button className="ml-2 text-indigo-300 hover:text-indigo-200" onClick={() => setIsolatedStep(null)}>
        看全部 ›
      </button>
    </p>
  ) : (
    <div className="space-y-4">
      {isolatedFeed.map((ev) => (
        <NarrativeStep
          key={ev.step}
          step={ev}
          focusAgents={focus.agents}
          isolated={isolatedStep === ev.step}
          onToggleIsolate={(s) => setIsolatedStep((cur) => (cur === s ? null : s))}
        />
      ))}
    </div>
  );

  // Push-drawers: the map yields width when a drawer is open. Immersive hides both
  // drawers so the map goes edge-to-edge. Each combination is a LITERAL class string
  // so Tailwind's JIT scanner picks it up (a computed template would be dropped).
  const showRoster = rosterOpen && !immersive;
  const showLog = logOpen && !immersive;
  const stageCols =
    showRoster && showLog
      ? "lg:grid-cols-[340px_minmax(0,1fr)_360px]"
      : showRoster
        ? "lg:grid-cols-[340px_minmax(0,1fr)]"
        : showLog
          ? "lg:grid-cols-[minmax(0,1fr)_360px]"
          : "lg:grid-cols-1";

  return (
    <div
      className={`flex flex-col h-screen overflow-hidden animate-[fadeIn_0.3s_ease-out] ${
        immersive ? "gap-2 p-2" : "gap-3 p-4 lg:p-5"
      }`}
    >
      {/* Top control bar: identity · run controls · panel toggles. Hidden in
          immersive so the map owns the shell — the map's own strip has the exit. */}
      {!immersive && (
        <section className="bg-slate-900 border border-slate-800/80 rounded-2xl p-4 space-y-3 shrink-0">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div className="flex items-center gap-3 flex-wrap">
              {meta ? (
                <EditableWorldName
                  worldId={id}
                  // Shown in TWO places at once — here and in the sidebar row — and either
                  // can rename it. Read from the shared list so both always agree, with
                  // `meta` covering the moment before that list has loaded.
                  name={
                    worlds.find((w) => w.world_id === id)?.world_name || meta.world_name || id
                  }
                  className="text-slate-100 text-base font-bold"
                  onRenamed={(m) => {
                    setMeta(m);
                    void refreshWorlds();
                  }}
                />
              ) : (
                <strong className="text-slate-100 text-base">{id}</strong>
              )}
              {/* Only surface an ACTIVE run state — idle/null is the resting default, no badge. */}
              {runState && runState !== "idle" && runState !== "completed" && (
                <span className={`text-[11px] px-2 py-0.5 rounded-full border ${stateBadge}`}>
                  {runState}
                </span>
              )}
              <span
                className={`text-[11px] px-2 py-0.5 rounded-full border ${
                  connected ? "text-teal-400 border-teal-700/60" : "text-rose-400 border-rose-700/60"
                }`}
              >
                {connected ? "● 实时" : "○ 断开"}
              </span>
            </div>
            {/* Both layers of "when", side by side — see WorldClock for why they are two
                things and not one sentence. */}
            {displayStep && <WorldClock step={displayStep} />}
          </div>

          {/* Run controls + source + panel toggles — one row. */}
          <div className="flex items-center gap-2 flex-wrap border-t border-slate-800/60 pt-3">
            <input
              value={stepsInput}
              onChange={(e) => setStepsInput(e.target.value)}
              // No spinner arrows (they'd eat half a narrow box for no use); type=number stays
              // for the numeric keypad and the min guard.
              className="w-24 bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500/70 [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
              placeholder="步数(默认1)"
              type="number"
              min={1}
            />
            <button
              className={BTN_PRIMARY}
              disabled={isBusy || !modelKeys}
              title={modelKeys ? undefined : NO_MODEL_KEYS_HINT}
              onClick={() => control(() => api.run(id, runSteps()))}
            >
              ▶ 运行
            </button>
            <button className={BTN} disabled={runState !== "running"} onClick={() => control(() => api.pause(id))}>
              ⏸ 暂停
            </button>
            <button className={BTN} disabled={runState !== "paused"} onClick={() => control(() => api.resume(id))}>
              ⏵ 继续
            </button>
            <button
              className={BTN}
              disabled={runState !== "running" && runState !== "paused"}
              onClick={() => control(() => api.stop(id))}
            >
              ⏹ 停止
            </button>
            {/* A control panel, so the button greys out rather than hiding: positions stay put
                and the row shows what exists. The map overlay, a work surface, hides instead. */}
            <button
              className={BTN}
              disabled={isBusy || !modelKeys}
              onClick={() => control(() => api.stepOnce(id))}
              title={modelKeys ? "推进一步后停下" : NO_MODEL_KEYS_HINT}
            >
              ⏭ 推一步
            </button>
            <button
              className={BTN}
              disabled={isBusy || !modelKeys}
              title={modelKeys ? undefined : NO_MODEL_KEYS_HINT}
              onClick={() =>
                control(async () => {
                  await api.reset(id);
                  onReset();
                })
              }
            >
              ↺ 重置
            </button>
            {divider}
            <span className="text-xs text-slate-500">模式</span>
            <button className={toggle(mode === "live")} onClick={() => setMode("live")}>
              实时
            </button>
            <button className={toggle(mode === "playback")} onClick={() => setMode("playback")}>
              回放
            </button>
            {divider}
            {/* Lens toggles — each opens a lens onto the always-present map stage. */}
            <span className="text-xs text-slate-500">视角</span>
            <button
              className={`${toggle(rosterOpen)} hidden lg:inline-block`}
              onClick={() => setRosterOpen((o) => !o)}
              title={rosterOpen ? "收起角色栏" : "展开角色栏"}
            >
              👥 角色
            </button>
            <button
              className={`${toggle(logOpen)} hidden lg:inline-block`}
              onClick={() => setLogOpen((o) => !o)}
              title={logOpen ? "收起叙事流" : "展开叙事流"}
            >
              📜 叙事流
            </button>
            <button
              className={toggle(graphOpen)}
              onClick={() => {
                setGraphOpen((o) => !o);
                setBodyOpen(false);
              }}
              title={graphOpen ? "关闭关系图谱" : "打开关系图谱"}
            >
              🕸️ 关系图谱
            </button>
            <button
              className={`${toggle(immersive)} hidden lg:inline-block`}
              onClick={() => setImmersive(true)}
              title="沉浸模式（地图铺满，Esc 退出）"
            >
              ⛶ 沉浸
            </button>
            <button
              className="ml-auto px-3 py-1.5 rounded-lg text-xs bg-slate-950 border border-rose-900/60 text-rose-400 hover:bg-rose-950/30 transition"
              onClick={remove}
            >
              删除
            </button>
          </div>

          {error && <p className="text-rose-400 text-xs">{error}</p>}
        </section>
      )}

      {/* Stage: the 2D map is the permanent center; roster/narrative push in as
          side drawers, the graph opens as an overlay over the map. */}
      <div className={`grid grid-cols-1 gap-4 min-h-0 flex-1 items-stretch ${stageCols}`}>
        {showRoster && (
          <aside className="hidden lg:flex flex-col bg-slate-900 border border-slate-800/80 rounded-2xl p-4 min-h-0">
            <div className="flex items-center justify-between mb-3 shrink-0">
              <h3 className="text-sm font-semibold tracking-wider text-slate-400 uppercase inline-flex items-center rounded-full bg-slate-800/60 border border-slate-700/60 px-4 py-1.5">
                👥 角色 ({roster.length})
              </h3>
              <button className={COLLAPSE} onClick={() => setRosterOpen(false)} title="收起角色栏">
                ‹ 收起
              </button>
            </div>
            <div className="min-h-0 overflow-y-auto pr-1">
              <CharacterCard
                worldId={id}
                roster={roster}
                activeId={activeAgentId}
                onActiveChange={selectOne}
                liveStates={displayStep?.agent_states}
                onExamine={(agentId) => {
                  selectOne(agentId);
                  setBodyOpen(true);
                  setGraphOpen(false);
                }}
              />
            </div>
          </aside>
        )}

        <main className="relative min-w-0 min-h-0 h-full">
          <Suspense
            fallback={
              <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 text-slate-400 text-sm animate-pulse h-full">
                加载 2D 渲染器…
              </div>
            }
          >
            <LiveWorldMap
              worldId={id}
              step={displayStep}
              onRendered={doneRendering}
              focus={focus}
              onSelect={handleMapSelect}
              onFocusChange={setFocus}
              immersive={immersive}
              onToggleImmersive={() => setImmersive((o) => !o)}
            />
          </Suspense>

          {/* The director's console over the map's top-right, live mode only (playback is
              history, with no next step to land on). A child of the map, so it survives
              immersive; under the graph overlay (z-30 vs z-40). */}
          {/* Only a worldId: whether to advance a step for this command is the backend's
              call (only it knows whether the loop is alive); don't duplicate run-state rules
              here. */}
          {/* step is only a refresh trigger: an intervention's receipt exists only once the
              step it lands in has run. */}
          {mode === "live" && <DirectorBar worldId={id} step={displayStep?.step ?? null} />}

          {/* Relationship graph — a summoned overlay over the map, dismissible. */}
          {graphOpen && (
            <div className="absolute inset-0 z-40 flex flex-col rounded-2xl overflow-hidden border border-slate-800 bg-slate-950/95 backdrop-blur-sm">
              <div className="flex items-center justify-between px-3 py-2 border-b border-slate-800/70 shrink-0">
                <h3 className="text-sm font-semibold tracking-wider text-slate-400 uppercase inline-flex items-center rounded-full bg-slate-800/60 border border-slate-700/60 px-4 py-1.5">🕸️ 关系图谱</h3>
                <button className={BTN} onClick={() => setGraphOpen(false)} title="关闭关系图谱">
                  关闭 ✕
                </button>
              </div>
              <div className="flex-1 min-h-0">
                <RelationshipGraph graph={graph} activeId={activeAgentId} onNodeClick={selectOne} />
              </div>
            </div>
          )}

          {/* Body chart — the same kind of overlay, for the selected agent. */}
          {bodyOpen && activeAgent && (
            <div className="absolute inset-0 z-40 flex flex-col rounded-2xl overflow-hidden border border-slate-800 bg-slate-950/90 backdrop-blur-sm">
              <div className="flex items-center justify-between px-3 py-2 border-b border-slate-800/70 shrink-0">
                <h3 className="text-sm font-semibold tracking-wider text-slate-400 uppercase inline-flex items-center rounded-full bg-slate-800/60 border border-slate-700/60 px-4 py-1.5">🩺 认知体检</h3>
                <button className={BTN} onClick={() => setBodyOpen(false)} title="关闭认知体检">
                  关闭 ✕
                </button>
              </div>
              <div className="flex-1 min-h-0">
                <CharacterBody
                  worldId={id}
                  agent={activeAgent}
                  live={displayStep?.agent_states[activeAgent.id]}
                  graph={graph}
                />
              </div>
            </div>
          )}
        </main>

        {showLog && (
          <aside className="hidden lg:flex flex-col bg-slate-900 border border-slate-800/80 rounded-2xl p-4 min-h-0">
            <div className="flex items-center justify-between mb-3 shrink-0">
              <h3 className="text-sm font-semibold tracking-wider text-slate-400 uppercase inline-flex items-center rounded-full bg-slate-800/60 border border-slate-700/60 px-4 py-1.5">📜 叙事流</h3>
              <button className={COLLAPSE} onClick={() => setLogOpen(false)} title="收起叙事流">
                收起 ›
              </button>
            </div>
            <div className="min-h-0 overflow-y-auto pr-1">{feedList}</div>
          </aside>
        )}
      </div>

      {/* Playback transport — pinned below the stage. It is a normal flex sibling
          (the map never escapes into a fixed overlay), so it stays visible in
          immersive too. */}
      {mode === "playback" && (
        <section className="bg-slate-900 border border-slate-800/80 rounded-2xl px-4 py-3 shrink-0">
          <div className="flex items-center gap-2 flex-wrap">
            <button className={BTN} onClick={togglePlay} disabled={!steps.length}>
              {playing ? "⏸ 暂停" : "▶ 播放"}
            </button>
            {/* Greyed out at the ends of the tape, because there a step button genuinely
                does nothing — better to look spent than to look broken. */}
            <button className={BTN} onClick={() => seek(-1)} disabled={pbIndex <= 0}>
              ⏮
            </button>
            <button className={BTN} onClick={() => seek(1)} disabled={pbIndex >= steps.length - 1}>
              ⏭
            </button>
            <input
              type="range"
              className="flex-1 min-w-[200px] accent-indigo-500"
              min={0}
              max={Math.max(steps.length - 1, 0)}
              value={pbIndex}
              onChange={(e) => goTo(Number(e.target.value))}
              disabled={!steps.length}
            />
            {steps.length > 0 && (
              <span className="text-xs text-slate-400 tabular-nums">
               {steps[pbIndex]} / {steps.length}
              </span>
            )}
            {/* In immersive the top bar is gone — offer a quick exit here too. */}
            {immersive && (
              <button className={BTN} onClick={() => setImmersive(false)} title="退出沉浸 (Esc)">
                ⤢ 退出沉浸
              </button>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
