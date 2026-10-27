// Character panel: a roster of every agent above the selected agent's detail. Shared by
// observation (liveStates present: evolving state first, static dossier collapsed) and
// creation review (no liveStates: dossier open, and the cast revisable until the world is
// confirmed).
import { useEffect, useMemo, useState } from "react";
import { identityOf } from "../lib/worldTable";
import ActorIcon from "./ActorIcon";
import { api } from "../api/client";
import { useCast, loadCast, type Cast } from "../lib/cast";
import { placeName, usePlaces, type Places } from "../lib/places";
import { reportedDead, unfinishedGoals } from "../lib/agentState";
import { avatarStyle, emotionColor } from "../lib/avatar";
import type { Placed } from "../lib/step";
import type { Viewer } from "../lib/viewer";
import type { AgentProfile, AgentView } from "../types";

export interface CardAgent {
  id: string;
  is_main_character: boolean;
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

export interface CharacterCardProps {
  cast: Cast;
  places: Places;
  profile?: AgentProfile;
  roster: CardAgent[];
  activeId: string | null;
  onActiveChange: (id: string) => void;
  liveStates?: Record<string, Placed<AgentView>>;
  /** Clicking the portrait opens the body chart (observation only). */
  onExamine?: (id: string) => void;
  // Public identity is shared; private dossiers belong to the director or the character.
  viewer?: Viewer;
  // Revising the cast in the world's setup (creation review, before confirming); the editing
  // itself is the caller's. `onAdd` is left out once the cast is full.
  onEdit?: (id: string) => void;
  onRemove?: (id: string) => void;
  onAdd?: () => void;
}

export function ObservedCharacterCard({ worldId, ...props }: Omit<CharacterCardProps, "cast" | "places" | "profile"> & { worldId: string }) {
  const { roster, activeId, viewer = null } = props;
  const { cast, error: castError } = useCast(worldId);
  const { places } = usePlaces(worldId);
  const scope = JSON.stringify([worldId, viewer]);
  const [loaded, setLoaded] = useState<{ scope: string; profiles: Record<string, AgentProfile>; failed: string[] }>({ scope, profiles: {}, failed: [] });
  const profiles = loaded.scope === scope ? loaded.profiles : {};
  const failed = loaded.scope === scope ? loaded.failed : [];
  const active = roster.find((r) => r.id === activeId) ?? roster[0];
  const privateKnown = !!active && (viewer === null || active.id === viewer);

  useEffect(() => {
    const aid = active?.id;
    if (!aid || !privateKnown || profiles[aid] || failed.includes(aid)) return;
    let alive = true;
    api.agentProfile(worldId, aid)
      .then((p) => {
        if (!alive) return;
        setLoaded((prev) => ({ scope, profiles: { ...(prev.scope === scope ? prev.profiles : {}), [aid]: p }, failed: prev.scope === scope ? prev.failed : [] }));
      })
      .catch(() => {
        if (!alive) return;
        setLoaded((prev) => ({ scope, profiles: prev.scope === scope ? prev.profiles : {}, failed: [...(prev.scope === scope ? prev.failed : []), aid] }));
      });
    return () => { alive = false; };
  }, [active?.id, worldId, scope, privateKnown, failed.includes(active?.id ?? "")]);

  return (
    <div className="space-y-3">
      {castError && <button className="text-xs text-rose-300" onClick={() => { void loadCast(worldId).catch(() => {}); }}>角色资料加载失败，点击重试</button>}
      {active && privateKnown && failed.includes(active.id) && <button className="text-xs text-rose-300" onClick={() => setLoaded((prev) => ({ ...prev, failed: prev.failed.filter((id) => id !== active.id) }))}>角色资料加载失败，点击重试</button>}
      <CharacterCard {...props} cast={cast} places={places} profile={active && privateKnown ? profiles[active.id] : undefined} />
    </div>
  );
}

export default function CharacterCard({
  cast,
  places,
  profile,
  roster,
  activeId,
  onActiveChange,
  liveStates,
  onExamine,
  viewer = null,
  onEdit,
  onRemove,
  onAdd,
}: CharacterCardProps) {
  const [dossierOpen, setDossierOpen] = useState(false);

  const observing = !!liveStates;
  const active = roster.find((r) => r.id === activeId) ?? roster[0];
  const live = active && liveStates ? liveStates[active.id] : undefined;
  const knowsPrivate = (id: string) => viewer === null || id === viewer;
  const privateKnown = active ? knowsPrivate(active.id) : false;

  const shortTermGoals = useMemo(() => unfinishedGoals(live?.private), [live]);

  if (!active) {
    return (
      <div className="bg-slate-900 border border-slate-800/80 rounded-2xl p-6 text-slate-500 text-sm">
        暂无角色。运行世界后角色状态会实时刷新。
      </div>
    );
  }

  const identity = identityOf(active.id, cast);
  const role = identity?.role;

  return (
    <div className="space-y-3">
      <div className="bg-slate-900/40 border border-slate-800/70 rounded-2xl p-2">
        {onAdd && (
          <button
            onClick={onAdd}
            className="w-full mb-1 px-2 py-1.5 rounded-lg border border-dashed border-slate-600 text-xs text-slate-300 hover:border-indigo-400 hover:text-white transition"
          >
            ＋ 添加角色
          </button>
        )}
        <div className="space-y-0.5 max-h-[38vh] overflow-y-auto">
          {roster.map((a) => {
            const ls = liveStates?.[a.id];
            const on = a.id === active.id;
            const dead = ls ? reportedDead(ls.is_active, ls.private.vitality) : false;
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
                  style={avatarStyle(identityOf(a.id, cast)?.color, a.id)}
                >
                  {(identityOf(a.id, cast)?.name ?? "某人")[0]}
                </div>
                <div className="min-w-0 flex-1">
                  <div title={identityOf(a.id, cast)?.name} className={`text-sm truncate ${on ? "text-slate-100" : "text-slate-300"}`}>
                    {identityOf(a.id, cast)?.name ?? "某人"}
                    {a.is_main_character && <span className="text-indigo-400"> ★</span>}
                    {a.id === viewer && <span className="ml-1 text-[10px] text-amber-300"><ActorIcon /> 我</span>}
                  </div>
                  {ls && (
                    <div
                      title={[knowsPrivate(a.id) ? ls.private.emotion_label : "", placeName(places, ls.location_id), ls.public.condition].filter(Boolean).join(" · ")}
                      className="text-[10px] text-slate-500 truncate"
                    >
                      {knowsPrivate(a.id) && (
                        <span className={emotionColor(ls.private.emotion_valence)}>{ls.private.emotion_label} · </span>
                      )}
                      <span>📍 {placeName(places, ls.location_id)}</span>
                      {/* A standing condition is context, not an event: it rides the secondary
                          line. Amber, because rose is death's. */}
                      {ls.public.condition && <span className="text-amber-500/70"> · {ls.public.condition}</span>}
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
          {onExamine && privateKnown ? (
            <button
              onClick={() => onExamine(active.id)}
              title="查看认知体检"
              className={`w-12 h-12 rounded-2xl grid place-items-center text-white text-2xl font-black shrink-0 border border-slate-100/10 shadow-[0_6px_20px_rgb(0,0,0,0.4)] cursor-zoom-in hover:ring-2 hover:ring-slate-100/40 transition`}
              style={avatarStyle(identity?.color, active.id)}
            >
              {(identity?.name ?? "某人")[0]}
            </button>
          ) : (
            <div
              className={`w-12 h-12 rounded-2xl grid place-items-center text-white text-2xl font-black shrink-0 border border-slate-100/10 shadow-[0_6px_20px_rgb(0,0,0,0.4)]`}
              style={avatarStyle(identity?.color, active.id)}
            >
              {(identity?.name ?? "某人")[0]}
            </div>
          )}
          <div className="min-w-0">
            <div title={identity?.name} className="text-lg font-extrabold text-slate-100 truncate">
              {identity?.name ?? "某人"}
              {active.id === viewer && <span className="ml-1.5 text-xs font-semibold text-amber-300"><ActorIcon /> 我</span>}
            </div>
            <div className="flex items-center gap-1 mt-1 flex-wrap">
              {active.is_main_character && (
                <span className="text-[10px] uppercase font-bold bg-indigo-950/80 px-2 py-0.5 rounded-full border border-indigo-800 text-indigo-300">
                  ★ 主角
                </span>
              )}
              {role && <Badge>{role}</Badge>}
              {identity?.gender && <Badge>{identity.gender}</Badge>}
              {identity?.age != null && <Badge>{identity.age} 岁</Badge>}
            </div>
          </div>
          {(onEdit || onRemove) && (
            <div className="ml-auto shrink-0 self-start flex gap-1.5">
              {onEdit && (
                <button
                  onClick={() => onEdit(active.id)}
                  className="text-[11px] px-2.5 py-1 rounded-lg bg-slate-950 border border-slate-700 text-slate-300 hover:text-white hover:border-slate-600 transition"
                >
                  ✎ 编辑
                </button>
              )}
              {onRemove && (
                <button
                  onClick={() => onRemove(active.id)}
                  className="text-[11px] px-2.5 py-1 rounded-lg bg-slate-950 border border-rose-900/60 text-rose-400 hover:bg-rose-950/30 transition"
                >
                  删除
                </button>
              )}
            </div>
          )}
        </div>

        {identity?.appearance && (
          <p className="mt-3 text-xs text-slate-400 leading-relaxed">{identity.appearance}</p>
        )}

        {identity?.background && (
          // In full: the builder already caps `background`. Clamping here would push the rest
          // into a hover-only tooltip a touch screen can't reach.
          <p className="mt-3 text-xs text-slate-400 leading-relaxed">
            {identity.background}
          </p>
        )}

        {/* Evolving state (observation only) — the interesting, per-step data, first. */}
        {observing && live && !privateKnown && (
          <div className="mt-3 border-t border-slate-800/70 pt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
            <span className="text-slate-500">
              📍 {placeName(places, live.location_id)}
              {live.public.activity_label && ` · ${live.public.activity_label}`}
            </span>
            {live.public.condition && <span className="text-amber-500/70">处境 {live.public.condition}</span>}
            {live.is_active === false && <span className="text-rose-400">· OVER</span>}
          </div>
        )}
        {observing && live && privateKnown && (
          <div className="mt-3 border-t border-slate-800/70 pt-3 space-y-2.5">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
              <span className={`font-medium ${emotionColor(live.private.emotion_valence)}`}>{live.private.emotion_label}</span>
              <span className="text-slate-500">
                📍 {placeName(places, live.location_id)} · {live.private.activity_label}
              </span>
              {live.private.dominant_need && <span className="text-slate-500">需求 {live.private.dominant_need_label}</span>}
              {live.private.vitality != null && <span className="text-slate-500">生命值 {Math.round(live.private.vitality * 100)}</span>}
              {live.public.condition && <span className="text-amber-500/70">处境 {live.public.condition}</span>}
              {live.is_active === false && <span className="text-rose-400">· OVER</span>}
            </div>

            {(shortTermGoals.length > 0 || live.private.long_term_goals.length > 0) && (
              <div className="space-y-1.5">
                {shortTermGoals.length > 0 && (
                  <GoalList label="短期目标" color="text-emerald-500/70" goals={shortTermGoals} />
                )}
                {live.private.long_term_goals.length > 0 && (
                  <GoalList label="长期目标" color="text-cyan-500/70" goals={live.private.long_term_goals} />
                )}
              </div>
            )}

            {/* No "最新活动" here: deeds are events and the feed narrates them in full. The card
                keeps what only it can show — the inner state — and activity_status above. */}
          </div>
        )}

        {/* Static dossier — reference. Collapsed by default while observing; open in review.
            Nobody's but the viewer's own when the world is seen through a character. */}
        {privateKnown && <div className="mt-3 border-t border-slate-800/70 pt-3">
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
        </div>}
      </div>
    </div>
  );
}
