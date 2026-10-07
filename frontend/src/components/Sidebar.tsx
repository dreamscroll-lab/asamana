// Persistent left sidebar (Claude-style): brand, a "new world" action, and the
// collapsible world list. Collapses to a slim icon rail. Reads the world list from
// the shared context so a build/delete elsewhere reflects here immediately.
import { Link, useHref, useMatch, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import EditableWorldName from "./EditableWorldName";
import { avatarGradient } from "../lib/avatar";
import { useDevTools } from "../lib/deployment";
import { useWorlds } from "../lib/worldsContext";
import type { RunState, WorldMeta } from "../types";

function statusDot(s: RunState): string {
  return s === "running"
    ? "bg-teal-400"
    : s === "paused"
      ? "bg-amber-400"
      : s === "failed"
        ? "bg-rose-400"
        : "bg-slate-600";
}

// No gradient tile behind the mark: the world avatars below are gradient squares, so the
// brand would read as the first world in the list.
const Brand = () => (
  <img src="/logo.png" alt="" className="w-8 h-8 object-contain shrink-0" />
);

export default function Sidebar({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  const { worlds, status, refresh } = useWorlds();
  const devTools = useDevTools();
  const navigate = useNavigate();
  // Call BOTH matches unconditionally — a `?? useMatch(...)` would short-circuit and
  // skip the second hook on world routes, changing the hook count (React error #300).
  const mObserve = useMatch("/worlds/:id");
  const mReview = useMatch("/worlds/:id/review");
  const activeId = mObserve?.params.id ?? mReview?.params.id ?? null;

  const destOf = (w: WorldMeta) =>
    w.confirmed ? `/worlds/${w.world_id}` : `/worlds/${w.world_id}/review`;
  // Under hash routing useHref("/") is "#/", so appending gives an address window.open accepts.
  const hrefRoot = useHref("/");

  // A world row can't be an <a>: its rename input and delete button would trigger navigation.
  // So Cmd/Ctrl+click and middle-click are routed to a new tab by hand.
  function openRow(w: WorldMeta, e: React.MouseEvent) {
    if (e.metaKey || e.ctrlKey || e.button === 1) {
      window.open(hrefRoot + destOf(w).slice(1), "_blank", "noopener");
      return;
    }
    navigate(destOf(w));
  }

  async function remove(id: string, name: string, e: React.MouseEvent) {
    e.stopPropagation();
    if (!window.confirm(`删除世界「${name}」？此操作不可恢复。`)) return;
    try {
      await api.deleteWorld(id);
      if (activeId === id) navigate("/");
      await refresh();
    } catch {
      /* surfaced elsewhere; sidebar stays quiet */
    }
  }

  if (collapsed) {
    return (
      <aside className="w-14 shrink-0 h-screen border-r border-slate-800/70 bg-slate-950 flex flex-col items-center py-3 gap-2">
        <button
          onClick={onToggle}
          title="展开世界栏"
          className="w-9 h-9 rounded-lg text-slate-400 hover:text-white hover:bg-slate-900 grid place-items-center transition"
        >
          ☰
        </button>
        <button
          onClick={() => navigate("/")}
          title="新世界"
          className="w-9 h-9 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-lg leading-none grid place-items-center border border-indigo-500 transition"
        >
          ＋
        </button>
        <div className="flex-1 w-full overflow-y-auto flex flex-col items-center gap-2 pt-2 no-scrollbar">
          {worlds.map((w) => {
            const name = w.world_name || w.world_id;
            return (
              <Link
                key={w.world_id}
                to={destOf(w)}
                title={name}
                className={`relative w-9 h-9 rounded-lg bg-gradient-to-tr ${avatarGradient(w.world_id)} text-white text-sm font-bold grid place-items-center transition ${
                  activeId === w.world_id ? "ring-2 ring-indigo-400" : "opacity-80 hover:opacity-100"
                }`}
              >
                {name[0]}
                <span
                  className={`absolute -bottom-0.5 -right-0.5 w-2 h-2 rounded-full ${statusDot(w.run_state)} ring-2 ring-slate-950`}
                />
              </Link>
            );
          })}
        </div>
      </aside>
    );
  }

  return (
    <aside className="w-64 shrink-0 h-screen border-r border-slate-800/70 bg-slate-950 flex flex-col">
      <div className="flex items-center gap-2 px-3 py-3">
        <Link to="/" className="flex items-center gap-2 flex-1 min-w-0 group">
          <Brand />
          <span className="font-bold bg-clip-text text-transparent bg-gradient-to-r from-violet-400 via-indigo-200 to-cyan-300">
            Asamana
          </span>
        </Link>
        <button
          onClick={onToggle}
          title="收起世界栏"
          className="shrink-0 w-7 h-7 rounded-lg text-slate-400 hover:text-white hover:bg-slate-900 grid place-items-center transition"
        >
          «
        </button>
      </div>

      <div className="px-3">
        <button
          onClick={() => navigate("/")}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold border border-indigo-500 shadow-lg shadow-indigo-600/20 transition"
        >
          <span className="text-base leading-none">＋</span> 新世界
        </button>
      </div>

      <div className="mt-4 px-2 flex-1 overflow-y-auto">
        <div className="px-2 mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-600">
          世界{worlds.length > 0 && ` · ${worlds.length}`}
        </div>
        {worlds.length === 0 && status === "offline" ? (
          // Not the empty state: drawn the same, a backend that won't boot reads as
          // "all your worlds are gone".
          <div className="px-2 py-6 text-xs leading-relaxed space-x-2">
            <span className="text-amber-400/80">加载失败</span>
            <button
              onClick={() => void refresh()}
              className="text-slate-400 hover:text-slate-200 underline underline-offset-2 transition"
            >
              重试
            </button>
          </div>
        ) : worlds.length === 0 ? (
          <div className="px-2 py-6 text-xs text-slate-600 leading-relaxed">
            {status === "loading" ? "载入中…" : "什么都没有~"}
          </div>
        ) : (
          <div className="space-y-0.5">
            {worlds.map((w) => {
              const name = w.world_name || w.world_id;
              const active = activeId === w.world_id;
              return (
                <div
                  key={w.world_id}
                  onClick={(e) => openRow(w, e)}
                  onAuxClick={(e) => e.button === 1 && openRow(w, e)}
                  className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer border transition ${
                    active
                      ? "bg-indigo-950/50 border-indigo-800/60"
                      : "border-transparent hover:bg-slate-900"
                  }`}
                >
                  <div className="flex-1 min-w-0 flex">
                    <EditableWorldName
                      worldId={w.world_id}
                      name={name}
                      className={`text-sm ${active ? "text-slate-100" : "text-slate-300"}`}
                      onRenamed={() => void refresh()}
                    />
                  </div>
                  {!w.confirmed && <span className="shrink-0 text-[9px] text-amber-400">待确认</span>}
                  <button
                    onClick={(e) => remove(w.world_id, name, e)}
                    title="删除"
                    className="shrink-0 opacity-0 group-hover:opacity-100 text-slate-500 hover:text-rose-400 transition px-0.5 text-xs"
                  >
                    ✕
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {devTools && (
        <div className="p-3 border-t border-slate-800/70 space-y-1.5">
          {/* Map workbench: a map, its cast and the renderer, off its live files. Sits
              with the other developer instruments — it needs the same dev routes. */}
          <Link
            to="/lab"
            target="_blank"
            rel="noopener"
            className="text-xs text-slate-500 hover:text-slate-300 transition flex items-center gap-1"
          >
            Map workbench
          </Link>
          <Link
            to="/dev"
            target="_blank"
            rel="noopener"
            className="text-xs text-slate-500 hover:text-slate-300 transition flex items-center gap-1"
          >
            Developer tools
          </Link>
        </div>
      )}
    </aside>
  );
}
