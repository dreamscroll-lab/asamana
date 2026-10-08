// Character panel: a roster of every agent above the selected agent's detail. Shared by
// observation (liveStates present: evolving state first, static dossier collapsed) and
// creation review (no liveStates: dossier open).
import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { reportedDead, unfinishedGoals } from "../lib/agentState";
import { avatarStyle, emotionColor } from "../lib/avatar";
import type { AgentProfile, AgentStateSummary } from "../types";

export interface CardAgent {
  id: string;
  name: string;
  is_main_character: boolean;
  role?: string;
  color?: string; // fixed identity color (#RRGGBB); empty when the world gave none
}

function Badge({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-[10px] font-bold uppercase bg-slate-950/90 px-2 py-0.5 rounded-full border border-slate-800 text-slate-300">
      {children}
    </span>
  );
}

function DossierBlock({
  icon,
  title,
  color,
  value,
}: {
  icon: string;
  title: string;
  color: string;
  value?: string;
}) {
  return (
    <div className="bg-slate-950/50 border border-slate-800/70 rounded-lg px-2.5 py-2">
      <span className={`text-[10px] font-bold ${color}`}>
        {icon} {title}
      </span>
      <p className="text-xs text-slate-300 leading-relaxed mt-0.5">{value || "—"}</p>
    </div>
  );
}

// Goals are ordered, so a numbered, hanging-indented line each. Don't join them with 「；」:
// that blurs where one goal ends and the next begins.
function GoalList({ label, color, goals }: { label: string; color: string; goals: string[] }) {
  return (
    <div className="text-xs leading-relaxed">
      <span className={color}>{label}</span>
      <ol className="mt-0.5 space-y-0.5">
        {goals.map((g, i) => (
          <li key={i} className="flex gap-1.5">
            <span className="shrink-0 font-mono tabular-nums text-slate-500">{i + 1}.</span>
            <span className="text-slate-300">{g}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

export default function CharacterCard({
  worldId,
  roster,
  activeId,
  onActiveChange,
  liveStates,
  onExamine,
}: {
  worldId: string;
  roster: CardAgent[];
  activeId: string | null;
  onActiveChange: (id: string) => void;
  liveStates?: Record<string, AgentStateSummary>;
  /** Clicking the portrait opens the body chart (observation only). */
  onExamine?: (id: string) => void;
}) {
  const [profiles, setProfiles] = useState<Record<string, AgentProfile>>({});
  const [dossierOpen, setDossierOpen] = useState(false);

  const observing = !!liveStates;
  const active = roster.find((r) => r.id === activeId) ?? roster[0];
  const profile = active ? profiles[active.id] : undefined;
  const live = active && liveStates ? liveStates[active.id] : undefined;

  const { planned, owed } = useMemo(() => unfinishedGoals(live), [live]);

  useEffect(() => {
    const aid = active?.id;
    if (!aid || profiles[aid]) return;
    let alive = true;
    api
      .agentProfile(worldId, aid)
      .then((p) => alive && setProfiles((prev) => ({ ...prev, [aid]: p })))
      .catch(() => {
        if (!alive) return;
        // Degrade to an empty profile so fields show "—" instead of spinning.
        setProfiles((prev) => ({
          ...prev,
          [aid]: {
            agent_id: aid,
            name: active?.name ?? aid,
            role: active?.role ?? "",
            age: null,
            gender: "",
            background: "",
            appearance: "",
            color: active?.color ?? "",
            core_traits: [],
            core_values: [],
            self_image: "",
            life_goal: "",
            secret: "",
            is_main_character: active?.is_main_character ?? false,
          },
        }));
      });
    return () => {
      alive = false;
    };
  }, [active?.id, worldId]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!active) {
    return (
      <div className="bg-slate-900 border border-slate-800/80 rounded-2xl p-6 text-slate-500 text-sm">
        暂无角色。运行世界后角色状态会实时刷新。
      </div>
    );
  }

  const role = active.role || profile?.role;

  return (
    <div className="space-y-3">
      <div className="bg-slate-900/40 border border-slate-800/70 rounded-2xl p-2">
        <div className="space-y-0.5 max-h-[38vh] overflow-y-auto">
          {roster.map((a) => {
            const ls = liveStates?.[a.id];
            const on = a.id === active.id;
            const dead = ls ? reportedDead(ls) : false;
            return (
              <button
                key={a.id}
                onClick={() => onActiveChange(a.id)}
                className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-left transition ${
                  on ? "bg-indigo-950/50 border border-indigo-700/50" : "border border-transparent hover:bg-slate-900"
                } ${dead ? "opacity-60" : ""}`}
              >
                <div
                  className={`w-7 h-7 rounded-lg grid place-items-center text-white text-xs font-bold shrink-0`}
                  style={avatarStyle(a.color, a.id)}
                >
                  {a.name[0]}
                </div>
                <div className="min-w-0 flex-1">
                  <div title={a.name} className={`text-sm truncate ${on ? "text-slate-100" : "text-slate-300"}`}>
                    {a.name}
                    {a.is_main_character && <span className="text-indigo-400"> ★</span>}
                  </div>
                  {ls && (
                    <div
                      title={[ls.emotion_label, ls.location, ls.condition].filter(Boolean).join(" · ")}
                      className="text-[10px] text-slate-500 truncate"
                    >
                      <span className={emotionColor(ls.emotion_valence)}>{ls.emotion_label}</span>
                      <span> · 📍 {ls.location}</span>
                      {/* A standing condition is context, not an event: it rides the secondary
                          line. Amber, because rose is death's. */}
                      {ls.condition && <span className="text-amber-500/70"> · {ls.condition}</span>}
                    </div>
                  )}
                </div>
                {/* Same badge the map wears on a fallen token — one fact, one look. */}
                {dead && <span className="shrink-0 text-[9px] font-semibold text-rose-400">OVER</span>}
              </button>
            );
          })}
        </div>
      </div>

      <div className="bg-gradient-to-b from-slate-900/95 to-slate-950/95 border border-slate-800 rounded-2xl p-4">
        <div className="flex items-center gap-3">
          {onExamine ? (
            <button
              onClick={() => onExamine(active.id)}
              title="查看认知体检"
              className={`w-12 h-12 rounded-2xl grid place-items-center text-white text-2xl font-black shrink-0 border border-slate-100/10 shadow-[0_6px_20px_rgb(0,0,0,0.4)] cursor-zoom-in hover:ring-2 hover:ring-slate-100/40 transition`}
              style={avatarStyle(active.color, active.id)}
            >
              {active.name[0]}
            </button>
          ) : (
            <div
              className={`w-12 h-12 rounded-2xl grid place-items-center text-white text-2xl font-black shrink-0 border border-slate-100/10 shadow-[0_6px_20px_rgb(0,0,0,0.4)]`}
              style={avatarStyle(active.color, active.id)}
            >
              {active.name[0]}
            </div>
          )}
          <div className="min-w-0">
            <div title={active.name} className="text-lg font-extrabold text-slate-100 truncate">
              {active.name}
            </div>
            <div className="flex items-center gap-1 mt-1 flex-wrap">
              {active.is_main_character && (
                <span className="text-[10px] uppercase font-bold bg-indigo-950/80 px-2 py-0.5 rounded-full border border-indigo-800 text-indigo-300">
                  ★ 主角
                </span>
              )}
              {role && <Badge>{role}</Badge>}
              {profile?.gender && <Badge>{profile.gender}</Badge>}
              {profile?.age != null && <Badge>{profile.age} 岁</Badge>}
            </div>
          </div>
        </div>

        {profile?.background && (
          // In full: the builder already caps `background`. Clamping here would push the rest
          // into a hover-only tooltip a touch screen can't reach.
          <p className="mt-3 text-xs text-slate-400 leading-relaxed">
            {profile.background}
          </p>
        )}

        {/* Evolving state (observation only) — the interesting, per-step data, first. */}
        {observing && live && (
          <div className="mt-3 border-t border-slate-800/70 pt-3 space-y-2.5">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
              <span className={`font-medium ${emotionColor(live.emotion_valence)}`}>{live.emotion_label}</span>
              <span className="text-slate-500">
                📍 {live.location} · {live.activity_label}
              </span>
              {live.dominant_need && <span className="text-slate-500">需求 {live.dominant_need_label}</span>}
              {live.vitality != null && <span className="text-slate-500">生命值 {Math.round(live.vitality * 100)}</span>}
              {live.condition && <span className="text-amber-500/70">处境 {live.condition}</span>}
              {live.is_active === false && <span className="text-rose-400">· OVER</span>}
            </div>

            {(planned.length > 0 || owed.length > 0 || live.long_term_goals.length > 0) && (
              <div className="space-y-1.5">
                {planned.length > 0 && (
                  <GoalList label="短期目标" color="text-emerald-500/70" goals={planned} />
                )}
                {owed.length > 0 && (
                  <GoalList label="待办事项" color="text-amber-500/70" goals={owed} />
                )}
                {live.long_term_goals.length > 0 && (
                  <GoalList label="长期目标" color="text-cyan-500/70" goals={live.long_term_goals} />
                )}
              </div>
            )}

            {/* No "最新活动" here: deeds are events and the feed narrates them in full. The card
                keeps what only it can show — the inner state — and activity_status above. */}
          </div>
        )}

        {/* Static dossier — reference. Collapsed by default while observing; open in review. */}
        <div className="mt-3 border-t border-slate-800/70 pt-3">
          {observing && (
            <button
              onClick={() => setDossierOpen((o) => !o)}
              className="text-xs text-slate-400 hover:text-slate-200 transition flex items-center gap-1"
            >
              <span className="text-slate-600">{dossierOpen ? "▾" : "▸"}</span> 完整档案
            </button>
          )}
          {(!observing || dossierOpen) && (
            <div className={`space-y-2 ${observing ? "mt-2.5" : ""}`}>
              {/* Most people have none, so no "—" placeholder block. */}
              {profile?.secret && (
                <DossierBlock icon="🤫" title="秘密" color="text-rose-400" value={profile.secret} />
              )}
              <DossierBlock icon="🪞" title="自我认知" color="text-amber-400" value={profile?.self_image} />
              <DossierBlock icon="🧠" title="性格特点" color="text-violet-400" value={profile?.core_traits.join("、")} />
              <DossierBlock icon="💎" title="价值观" color="text-purple-400" value={profile?.core_values.join("、")} />
              <DossierBlock icon="🎯" title="人生目标" color="text-cyan-400" value={profile?.life_goal} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
