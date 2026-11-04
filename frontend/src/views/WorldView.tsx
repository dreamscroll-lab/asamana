import { lazy, Suspense, useEffect, useState } from "react";
import { useNavigate, useOutletContext, useParams } from "react-router-dom";

import type { ShellContext } from "../App";
import { api } from "../api/client";
import { WorldSocket } from "../api/ws";
import ActorIcon from "../components/ActorIcon";
import MapOperations from "../components/actor/MapOperations";
import { ActorSubmission } from "../components/actor/submission";
import ActorConsole from "../components/actor/ActorConsole";
import { mergeEntries } from "../components/actor/thread";
import { BTN, BTN_PRIMARY } from "../components/buttonStyles";
import CharacterBody from "../components/CharacterBody";
import { ObservedCharacterCard, type CardAgent } from "../components/CharacterCard";
import DirectorBar from "../components/DirectorBar";
import EditableWorldName from "../components/EditableWorldName";
import NarrativeStep from "../components/narrative/NarrativeStep";
import RelationshipGraph from "../components/RelationshipGraph";
import WorldClock from "../components/WorldClock";
import { useNpcs, loadNpcs } from "../lib/npcs";
import { useCast } from "../lib/cast";
import { usePlaces } from "../lib/places";
import { indexStep } from "../lib/step";
import { viewerOf } from "../lib/viewer";
import { useStepPlayer } from "../hooks/useStepPlayer";
import { NO_MODEL_KEYS_HINT, useModelKeys } from "../lib/deployment";
import { useWorlds } from "../lib/worldsContext";
import type { ActorEntry, ActorTurn, MapFocus, MapOperationTarget, MapSelect, RunState, WorldMeta, WsFrame } from "../types";

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
  return <WorldEntry key={id} id={id} />;
}

function WorldEntry({ id }: { id: string }) {
  const navigate = useNavigate();
  const [meta, setMeta] = useState<WorldMeta | null>(null);
  const [error, setError] = useState("");
  const [resets, setResets] = useState(0);
  useEffect(() => {
    let current = true;
    api.getWorld(id).then((world) => {
      if (!current) return;
      if (!world.confirmed) navigate(`/worlds/${id}/review`, { replace: true });
      else setMeta(world);
    }).catch((failure) => { if (current) setError(String(failure)); });
    return () => { current = false; };
  }, [id, navigate, resets]);
  if (!meta?.confirmed) return <p className="p-6 text-sm text-slate-500">{error || "正在加载世界…"}</p>;
  return <WorldObservation key={resets} id={id} initialMeta={meta} onReset={() => { setMeta(null); setResets((n) => n + 1); }} />;
}

function WorldObservation({ id, initialMeta, onReset }: { id: string; initialMeta: WorldMeta; onReset: () => void }) {
  const navigate = useNavigate();
  const { cast } = useCast(id);
  const { npcs, status: npcStatus } = useNpcs(id);
  const { places } = usePlaces(id);
  const { setImmersive: setShellImmersive } = useOutletContext<ShellContext>();
  const { worlds, refresh: refreshWorlds } = useWorlds();
  const modelKeys = useModelKeys();

  const [meta, setMeta] = useState<WorldMeta>(initialMeta);
  // The character an actor performs here, whose view every payload is filtered to; null for the
  // director. Every panel below reads the world through it (see lib/viewer).
  const viewer = viewerOf(meta);
  // The turn waiting for the actor, pushed over the socket whenever one opens or closes.
  const [turn, setTurn] = useState<ActorTurn | null>(null);
  // The actor's thread: its history read once the world's viewer is known, each new entry pushed
  // over the socket.
  const [thread, setThread] = useState<ActorEntry[]>([]);
  const [runState, setRunState] = useState<RunState>(initialMeta.run_state);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState("");
  const [stepsInput, setStepsInput] = useState("50");

  const [operationTarget, setOperationTarget] = useState<(MapOperationTarget & { turnId: string }) | null>(null);
  const [mode, setMode] = useState<"live" | "playback">("live");
  useEffect(() => setOperationTarget(null), [id, turn?.turn_id, mode]);
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
  // would re-center every frame and undo any drag. Follow is an opt-in via the 🎥 button,
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

  // Seen through a character, the camera starts on it and follows it.
  useEffect(() => {
    if (viewer) setFocus({ agents: [viewer], place: null, follow: true });
  }, [viewer]);

  // The actor's thread from the backend: on arrival, on every (re)connect, since entries pushed
  // while the socket was down are lost, and whenever the run settles. A run stopped mid-step
  // discards that step, so the thread read then replaces what was pushed live; while the world
  // runs, the read only adds to what arrives meanwhile.
  const settled = runState !== "running" && runState !== "stopping";
  useEffect(() => {
    if (!viewer) return;
    let current = true;
    api
      .actorThread(id)
      .then((r) => current && setThread((t) => (settled ? r.entries : mergeEntries(r.entries, t))))
      // Unreadable now: keep what is shown.
      .catch(() => {});
    return () => {
      current = false;
    };
  }, [id, viewer, settled, connected]);

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

  const liveAgents = displayStep ? indexStep(displayStep).agents : undefined;
  const agents = liveAgents ? Object.values(liveAgents) : [];
  const isBusy = ["running", "paused", "stopping"].includes(runState ?? "");

  // Roster for the shared character card (main characters first). Live per-step
  // state is passed separately so the card shows the same rich layout as review.
  // The performed character leads, so its own state is always at hand.
  const roster: CardAgent[] = [...agents]
    .sort((a, b) =>
      Number(b.agent_id === viewer) - Number(a.agent_id === viewer)
      || Number(b.is_main_character) - Number(a.is_main_character))
    .map((a) => ({ id: a.agent_id, is_main_character: a.is_main_character ?? false }));
  const activeAgentId = focus.agents[focus.agents.length - 1] ?? viewer ?? roster[0]?.id ?? null;
  const activeAgent = roster.find((a) => a.id === activeAgentId);

  // --- WebSocket: connect on mount, stream live status/latest_step/step ------
  useEffect(() => {
    const onFrame = (frame: WsFrame) => {
      if (frame.type === "status") setRunState(frame.data.status);
      else if (frame.type === "actor_turn") setTurn(frame.data);
      else if (frame.type === "actor_entry") setThread((t) => mergeEntries(t, [frame.data]));
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

  // The actor steps out for good. Remounted afterwards: every step held here was filtered to the
  // character.
  async function exitActor() {
    if (!confirm("退出演员模式？退出后这个角色交还给 AI，你回到导演模式，且不能再扮演。")) return;
    await control(async () => {
      await api.exitActor(id);
      await refreshWorlds();
      onReset();
    });
  }

  async function remove() {
    if (!confirm("删除该世界？此操作不可恢复。")) return;
    await control(() => api.deleteWorld(id));
    await refreshWorlds();
    navigate("/");
  }

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
  // The feed honors `focus` (the multi-select, not just activeAgentId) like every other
  // panel, so one selection really spans all views.
  //
  // An isolated step survives new steps arriving (the feed shouldn't decide you're done
  // reading). It closes, saying so, when the step leaves the window (FEED_CAP, or scrubbing
  // back before it).
  const isolatedFeed = isolatedStep === null ? feed : feed.filter((ev) => ev.step === isolatedStep);
  const feedList = npcStatus === "error" ? (
    <button className="text-sm text-rose-300" onClick={() => { void loadNpcs(id).catch(() => {}); }}>NPC 资料加载失败，点击重试</button>
  ) : npcStatus === "loading" ? (
    <p className="text-sm text-slate-500">正在加载 NPC 资料…</p>
  ) : !feed.length ? (
    <p className="text-slate-500 text-sm">
      {mode === "live" ? "运行世界以生成叙事，每步实时追加。" : "该世界暂无历史记录可回放。"}
    </p>
  ) : !isolatedFeed.length ? (
    <p className="text-slate-500 text-sm">
      第 {isolatedStep} 步已不在叙事流中。
      <button className="ml-2 text-indigo-300 hover:text-indigo-200" onClick={() => setIsolatedStep(null)}>
        查看全部 ›
      </button>
    </p>
  ) : (
    <div className="space-y-4">
      {isolatedFeed.map((ev) => (
        <NarrativeStep
          cast={cast}
          npcs={npcs}
          places={places}
          key={ev.step}
          step={ev}
          viewer={viewer}
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
                {connected ? "● 已连接" : "○ 未连接"}
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
              placeholder="运行步数（默认 1）"
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
            {/* A control panel, so the button grays out rather than hiding: positions stay put
                and the row shows what exists. The map overlay, a work surface, hides instead. */}
            <button
              className={BTN}
              disabled={isBusy || !modelKeys}
              onClick={() => control(() => api.stepOnce(id))}
              title={modelKeys ? "推进一步后停下" : NO_MODEL_KEYS_HINT}
            >
              ⏭ 推进一步
            </button>
            {/* Resetting rewrites the story: the director's power, not the actor's. */}
            {!viewer && (
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
            )}
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
              ⛶ 沉浸模式
            </button>
            {viewer ? (
              // Like reset, only on a stopped world: the backend takes no exit mid-run.
              <button
                className="ml-auto px-3 py-1.5 rounded-lg text-xs bg-slate-950 border border-amber-900/60 text-amber-300 hover:bg-amber-950/30 transition disabled:opacity-40 disabled:cursor-not-allowed"
                disabled={isBusy || !modelKeys}
                onClick={exitActor}
                title={isBusy ? "先停止运行再退出" : modelKeys ? "把角色交还给 AI，回到导演模式" : NO_MODEL_KEYS_HINT}
              >
                <ActorIcon /> 退出演员模式
              </button>
            ) : (
              <button
                className="ml-auto px-3 py-1.5 rounded-lg text-xs bg-slate-950 border border-rose-900/60 text-rose-400 hover:bg-rose-950/30 transition"
                onClick={remove}
              >
                删除
              </button>
            )}
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
              <ObservedCharacterCard
                worldId={id}
                roster={roster}
                activeId={activeAgentId}
                onActiveChange={selectOne}
                liveStates={liveAgents}
                viewer={viewer}
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
                正在加载地图…
              </div>
            }
          >
            {/* Held until the world's meta says whose view the steps are: drawn as the director's
                first, a character's step would show its blanks as the god view. */}
            <ActorSubmission scope={`${id}:${mode}:${turn?.turn_id ?? ""}`}>
            <LiveWorldMap
              worldId={id}
              step={meta ? displayStep : null}
              viewer={viewer}
              onRendered={doneRendering}
              focus={focus}
              onSelect={handleMapSelect}
              onOperation={viewer && mode === "live" && turn?.kind === "decide" && turn.briefing ? (target) => setOperationTarget({ ...target, turnId: turn.turn_id }) : undefined}
              onFocusChange={setFocus}
              immersive={immersive}
              onToggleImmersive={() => setImmersive((o) => !o)}
            >
              {/* The director's console over the map's top-right, live mode only (playback is
                  history, with no next step to land on). Inside the map, so it survives immersive
                  and never covers the map's own controls; under the graph overlay (z-30 vs z-40). */}
              {/* Only a worldId: whether to advance a step for this command is the backend's
                  call (only it knows whether the loop is alive); don't duplicate run-state rules
                  here. */}
              {/* step is only a refresh trigger: an intervention's receipt exists only once the
                  step it lands in has run. */}
              {mode === "live" && !viewer && <DirectorBar worldId={id} step={displayStep?.step ?? null} />}

              {/* The actor's console takes the director console's corner: a viewer has no director
                  console, and deciding over the map keeps what they decide next to where it happens. */}
              {viewer && mode === "live" && (
                <ActorConsole
                  worldId={id}
                  selfId={viewer}
                  entries={thread}
                  turn={turn}
                  runState={runState}
                  step={feed[0]?.step ?? 0}
                  pace={meta?.actor_pace ?? "every_step"}
                  immersive={immersive}
                />
              )}
              {viewer && mode === "live" && turn?.kind === "decide" && turn.briefing && operationTarget?.turnId === turn.turn_id && (
                <MapOperations key={`${turn.turn_id}:${operationTarget.target.kind}:${operationTarget.target.id}:${operationTarget.x}:${operationTarget.y}`}
                  worldId={id} turn={turn} selection={operationTarget} onClose={() => setOperationTarget(null)} />
              )}
            </LiveWorldMap>
            </ActorSubmission>
          </Suspense>

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
                <RelationshipGraph cast={cast} graph={graph} activeId={activeAgentId} onNodeClick={selectOne} />
              </div>
            </div>
          )}

          {/* Body chart — the same kind of overlay, for the selected agent. */}
          {bodyOpen && activeAgent && (
            <div className="absolute inset-0 z-40 flex flex-col rounded-2xl overflow-hidden border border-slate-800 bg-slate-950/90 backdrop-blur-sm">
              <div className="flex items-center justify-between px-3 py-2 border-b border-slate-800/70 shrink-0">
                <h3 className="text-sm font-semibold tracking-wider text-slate-400 uppercase inline-flex items-center rounded-full bg-slate-800/60 border border-slate-700/60 px-4 py-1.5">🩺 认知状态</h3>
                <button className={BTN} onClick={() => setBodyOpen(false)} title="关闭认知体检">
                  关闭 ✕
                </button>
              </div>
              <div className="flex-1 min-h-0">
                <CharacterBody
                  viewer={viewer}
                  worldId={id}
                  agent={activeAgent}
                  live={liveAgents?.[activeAgent.id]}
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
            {/* Grayed out at the ends of the tape, because there a step button genuinely
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
              <button className={BTN} onClick={() => setImmersive(false)} title="退出沉浸模式（Esc）">
                ⤢ 退出沉浸
              </button>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
