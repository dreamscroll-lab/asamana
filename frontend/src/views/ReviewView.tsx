// Creation-review dashboard: after a world is built the user reviews the cast
// (CharacterCard) and their relationship web (RelationshipGraph), then confirms
// to lock initialization and enter the narrative phase. Until confirmed a world
// cannot start its run.
import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { api } from "../api/client";
import CharacterCard, { type CardAgent } from "../components/CharacterCard";
import EditableWorldName from "../components/EditableWorldName";
import RelationshipGraph from "../components/RelationshipGraph";
import { useWorlds } from "../lib/worldsContext";
import type { WorldGraph, WorldMeta } from "../types";

/**
 * A step's world duration in the units a reader thinks in — a full day reads as
 * "约一天", not "24 小时". Mirrors the backend's core.duration.describe_duration:
 * the wire carries seconds, and this is the one place that turns them into words.
 */
function stepDurationLabel(seconds: number): string {
  const total = Math.floor(seconds / 60);
  const parts: [number, string][] = [[Math.floor(total / 1440), "天"], [Math.floor((total % 1440) / 60), "小时"], [total % 60, "分钟"]];
  return `约 ${parts.filter(([n]) => n).map(([n, unit]) => `${n} ${unit}`).join(" ") || "0 分钟"}`;
}

/**
 * Keyed on the world for the same reason WorldView is (see its header): this screen holds
 * world-scoped state too — `activeId` is an agent id, and carried into another world it
 * selects a character that does not exist there.
 */
export default function ReviewView() {
  const { id = "" } = useParams();
  return <WorldReview key={id} id={id} />;
}

function WorldReview({ id }: { id: string }) {
  const navigate = useNavigate();
  // A rename here has to reach the sidebar list too — it is the same world, shown
  // twice. Reading the name back out of that shared list is also what keeps the two
  // in agreement whichever one was edited.
  const { worlds, refresh: refreshWorlds } = useWorlds();

  const [meta, setMeta] = useState<WorldMeta | null>(null);
  const [graph, setGraph] = useState<WorldGraph | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  // Full-screen is this page's call, not the graph's: the view owns the screen.
  const [graphExpanded, setGraphExpanded] = useState(false);

  useEffect(() => {
    if (!graphExpanded) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setGraphExpanded(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [graphExpanded]);

  const load = useCallback(async () => {
    try {
      const [m, g] = await Promise.all([api.getWorld(id), api.graph(id)]);
      setMeta(m);
      setGraph(g);
    } catch (e) {
      setError(String(e));
    }
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  // Roster: main characters first.
  const roster = useMemo<CardAgent[]>(
    () =>
      [...(graph?.nodes ?? [])]
        .sort((a, b) => Number(b.is_main_character) - Number(a.is_main_character))
        .map((n) => ({
          id: n.id,
          name: n.name,
          is_main_character: n.is_main_character,
          role: n.role,
          color: n.color, // identity colour → avatar matches the map token + graph node
        })),
    [graph],
  );
  const effectiveId = activeId ?? roster[0]?.id ?? null;

  async function confirm() {
    setBusy(true);
    setError("");
    try {
      await api.confirmWorld(id);
      navigate(`/worlds/${id}`);
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  }

  async function discard() {
    if (!window.confirm(`丢弃世界「${meta?.world_name || id}」？此操作不可恢复。`)) return;
    setBusy(true);
    try {
      await api.deleteWorld(id);
      navigate("/");
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  }

  return (
    <div className="text-slate-100 space-y-6 p-4 lg:p-6 animate-[fadeIn_0.4s_ease-out]">
      {/* Action bar */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <button
          onClick={() => navigate("/")}
          className="text-xs text-slate-400 hover:text-white px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 hover:border-slate-700 transition"
        >
          ← 返回世界列表
        </button>
        <div className="flex items-center space-x-2">
          {meta?.confirmed && (
            <span className="text-[11px] text-emerald-400 bg-emerald-950/40 border border-emerald-800/60 px-2.5 py-1 rounded-full">
              已确认
            </span>
          )}
          <button
            onClick={discard}
            disabled={busy}
            className="text-xs px-4 py-2 rounded-lg bg-slate-950 border border-rose-900/60 text-rose-400 hover:bg-rose-950/30 transition disabled:opacity-40"
          >
            丢弃
          </button>
          <button
            onClick={confirm}
            disabled={busy}
            className="text-xs font-semibold px-5 py-2 rounded-lg bg-indigo-600 text-white hover:bg-indigo-500 border border-indigo-500 shadow-lg shadow-indigo-600/30 transition disabled:opacity-40"
          >
            {meta?.confirmed ? "进入叙事 →" : "确认并进入叙事 →"}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-rose-950/20 border border-rose-900/40 rounded-xl text-rose-400 text-xs">
          {error}
        </div>
      )}

      {/* World lore card */}
      <div className="bg-slate-900 border border-slate-800/80 rounded-2xl p-6 relative overflow-hidden shadow-lg">
        <div className="absolute top-0 right-0 w-64 h-64 bg-violet-600/5 rounded-full blur-3xl pointer-events-none" />
        <div className="flex items-start md:items-center justify-between flex-col md:flex-row gap-4 border-b border-slate-800 pb-4 mb-4">
          <div>
            <h3 className="text-2xl font-black text-slate-100">
              {meta ? (
                <EditableWorldName
                  worldId={id}
                  name={
                    worlds.find((w) => w.world_id === id)?.world_name || meta.world_name || id
                  }
                  onRenamed={(m) => {
                    setMeta(m);
                    void refreshWorlds();
                  }}
                />
              ) : (
                id
              )}
            </h3>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <div className="flex items-center space-x-2 text-xs bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 text-slate-400">
              <span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
              <span>
                {roster.length} 名角色 &amp; {graph?.edges.length ?? 0} 个关系连接
              </span>
            </div>
            {/* The story's own pace, chosen for it at build and frozen by confirming —
                a story told a day at a time is a different story from one told hour by
                hour, so it belongs here, next to the cast, before the user locks it. */}
            {meta?.seconds_per_step ? (
              <div className="flex items-center space-x-2 text-xs bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 text-slate-400">
                <span>⏳</span>
                <span>叙事节奏 · 每步 {stepDurationLabel(meta.seconds_per_step)}</span>
              </div>
            ) : null}
          </div>
        </div>
        {(meta?.description || meta?.theme) && (
          <p className="text-slate-300 text-sm leading-relaxed max-w-4xl italic">
             {meta?.description || meta?.theme} 
          </p>
        )}
      </div>

      {/* Cast + graph — each a unified card (title + content in one bordered container),
          matching the lore card above and the narrative page's drawers. */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
        <div className="lg:col-span-5">
          <div className="h-full flex flex-col bg-slate-900 border border-slate-800/80 rounded-2xl p-4 shadow-lg">
            <div className="flex items-center justify-between mb-3 shrink-0">
              <h4 className="text-sm font-semibold tracking-wider text-slate-400 uppercase">
                <span className="inline-flex items-center rounded-full bg-slate-800/60 border border-slate-700/60 px-4 py-1.5">👥 角色档案</span>
              </h4>
              <span className="text-xs text-slate-500">
                {roster.length ? roster.findIndex((r) => r.id === effectiveId) + 1 : 0} / {roster.length}
              </span>
            </div>
            <CharacterCard
              worldId={id}
              roster={roster}
              activeId={effectiveId}
              onActiveChange={setActiveId}
            />
          </div>
        </div>

        <div className="lg:col-span-7">
          <div
            className={
              graphExpanded
                ? "fixed inset-0 z-[60] flex flex-col bg-slate-950 p-4"
                : "h-full flex flex-col bg-slate-900 border border-slate-800/80 rounded-2xl p-4 shadow-lg"
            }
          >
            <div className="flex items-center justify-between mb-3 shrink-0">
              <h4 className="text-sm font-semibold tracking-wider text-slate-400 uppercase">
                <span className="inline-flex items-center rounded-full bg-slate-800/60 border border-slate-700/60 px-4 py-1.5">🕸️ 角色关系网图谱</span>
              </h4>
              <div className="flex items-center gap-3">
                <span className="text-xs text-slate-500">提示: 点击节点可高亮该角色关系链</span>
                <button
                  onClick={() => setGraphExpanded((v) => !v)}
                  title={graphExpanded ? "退出全屏 (Esc)" : "全屏"}
                  className="shrink-0 text-[11px] px-2.5 py-1.5 rounded-lg bg-slate-950/90 border border-slate-700 text-slate-300 hover:text-white hover:border-slate-600 transition"
                >
                  {graphExpanded ? "⤢ 退出" : "⛶ 全屏"}
                </button>
              </div>
            </div>
            <div className="flex-1 min-h-0">
              <RelationshipGraph graph={graph} activeId={effectiveId} onNodeClick={setActiveId} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
