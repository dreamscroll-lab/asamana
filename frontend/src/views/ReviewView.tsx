// Creation-review dashboard: after a world is built the user chooses how to enter it, then
// confirms to lock initialization and enter the narrative phase. Until confirmed a world
// cannot start its run.
//
// The choice comes first because the director's review shows everything: every character's
// dossier, secrets and relations. Someone who will live in the world as one character must not
// see that, so the page shows only the world's premise until the user picks:
//   - as the director: the whole cast, their relationship web and the world's historical events, then
//     confirm. The setup shown here (dossiers, historical events, the relations the world starts
//     with, the story's pace) is revisable until confirming. One way: the world takes no actor afterwards (the
//     backend records it).
//   - as a character: the cast as anyone could know them, one picked, then confirm. Their own
//     full dossier waits until they are in the world, or browsing candidates would reveal
//     everyone's.
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { identityOf } from "../lib/worldTable";
import { api } from "../api/client";
import ActorIcon from "../components/ActorIcon";
import CharacterCard, { ObservedCharacterCard, type CardAgent } from "../components/CharacterCard";
import EditableWorldName from "../components/EditableWorldName";
import RelationshipGraph from "../components/RelationshipGraph";
import SeedPanel from "../components/setup/SeedPanel";
import HistoryPanel from "../components/setup/HistoryPanel";
import RelationPanel from "../components/setup/RelationPanel";
import StepLengthEditor from "../components/setup/StepLengthEditor";
import CharacterEditor, {
  blankDraft, draftOf, newCharacterOf, revisionOf, type Draft,
} from "../components/setup/CharacterEditor";
import { reloadNpcs } from "../lib/npcs";
import { useCast, loadCast, reloadCast, type Cast } from "../lib/cast";
import { MAX_CAST_SIZE } from "../lib/contract";
import { usePlaces, loadPlaces, reloadPlaces, type Places } from "../lib/places";
import { setupGraph } from "../lib/setupGraph";
import { useVocabulary } from "../lib/vocabulary";
import { avatarStyle } from "../lib/avatar";
import { viewerOf } from "../lib/viewer";
import { useWorlds } from "../lib/worldsContext";
import type { CharacterIdentity, SetupView, WorldGraph, WorldMeta } from "../types";

/**
 * Keyed on the world for the same reason WorldView is (see its header): this screen holds
 * world-scoped state too — `activeId` is an agent id, and carried into another world it
 * selects a character that does not exist there.
 */
export default function ReviewView() {
  const { id = "" } = useParams();
  return <ReviewEntry key={id} id={id} />;
}

function ReviewEntry({ id }: { id: string }) {
  const navigate = useNavigate();
  const [meta, setMeta] = useState<WorldMeta | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let current = true;
    api.getWorld(id).then((world) => {
      if (!current) return;
      if (viewerOf(world)) navigate(`/worlds/${id}`, { replace: true });
      else setMeta(world);
    }).catch((failure) => { if (current) setError(String(failure)); });
    return () => { current = false; };
  }, [id, navigate]);
  if (!meta || viewerOf(meta)) return <p className="p-6 text-sm text-slate-500">{error || "正在加载世界…"}</p>;
  return meta.confirmed
    ? <ConfirmedReview id={id} initialMeta={meta} />
    : <DraftReview id={id} initialMeta={meta} />;
}

interface ReviewData {
  id: string;
  initialMeta: WorldMeta;
  identities: Cast;
  places: Places;
  dataError: Error | null;
  retryData: () => void;
}

function DraftReview({ id, initialMeta }: { id: string; initialMeta: WorldMeta }) {
  const [data, setData] = useState<{ identities: Cast; places: Places; error: Error | null }>({
    identities: {}, places: {}, error: null,
  });
  const load = useCallback(async () => {
    try {
      const [characters, locations] = await Promise.all([api.agents(id), api.places(id)]);
      setData({
        identities: Object.fromEntries(characters.map((row) => [row.agent_id, row])),
        places: Object.fromEntries(locations.map((row) => [row.location_id, row])),
        error: null,
      });
    } catch (failure) {
      setData((current) => ({ ...current, error: failure instanceof Error ? failure : new Error(String(failure)) }));
    }
  }, [id]);
  useEffect(() => { void load(); }, [load]);
  return <WorldReview id={id} initialMeta={initialMeta} identities={data.identities} places={data.places}
    dataError={data.error} retryData={() => { void load(); }} />;
}

function ConfirmedReview({ id, initialMeta }: { id: string; initialMeta: WorldMeta }) {
  const { cast, error } = useCast(id);
  const { places, error: placesError } = usePlaces(id);
  return <WorldReview id={id} initialMeta={initialMeta} identities={cast} places={places}
    dataError={error ?? placesError} retryData={() => { void Promise.allSettled([loadCast(id), loadPlaces(id)]); }} />;
}

function WorldReview({ id, initialMeta, identities, places, dataError, retryData }: ReviewData) {
  const navigate = useNavigate();
  // A rename here has to reach the sidebar list too — it is the same world, shown
  // twice. Reading the name back out of that shared list is also what keeps the two
  // in agreement whichever one was edited.
  const { worlds, refresh: refreshWorlds } = useWorlds();

  const [meta, setMeta] = useState<WorldMeta>(initialMeta);
  // A confirmed world's relations as its latest step holds them; before confirming there is no
  // step, and the graph is drawn from the setup instead.
  const [storedGraph, setStoredGraph] = useState<WorldGraph | null>(null);
  const [setup, setSetup] = useState<SetupView | null>(null);
  const vocabulary = useVocabulary();
  // The character being edited (`id` null: one being added), with the form it started from.
  const [editing, setEditing] = useState<{ id: string | null; draft: Draft } | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  // Full-screen is this page's call, not the graph's: the view owns the screen.
  const [graphExpanded, setGraphExpanded] = useState(false);
  // Picking whom to perform; chosen with the confirmation and locked by it.
  const [casting, setCasting] = useState(false);
  const [seedEditing, setSeedEditing] = useState(false);
  const cast: Cast = useMemo(() => !meta.confirmed && setup
    ? Object.fromEntries(setup.characters.map((c) => [c.agent_id, {
      agent_id: c.agent_id, name: c.name, gender: c.gender, age: c.age, role: c.role,
      background: c.background, appearance: c.appearance, color: c.color,
    }]))
    : identities, [meta.confirmed, setup, identities]);
  const candidates = Object.values(cast);
  const [castId, setCastId] = useState("");

  // The whole cast is the director's: once reviewed (or once the world is confirmed without an
  // actor) nobody can be cast, and only then is it shown.
  const directing = meta.director_reviewed || meta.confirmed;
  const stage = directing ? "director" : casting ? "casting" : "choose";
  // Confirming locks the setup; an unconfirmed world can't have run.
  const revisable = stage === "director" && !meta.confirmed;

  useEffect(() => {
    if (!graphExpanded) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setGraphExpanded(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [graphExpanded]);

  // Private relations and the historical events are loaded only after the director enters review.
  useEffect(() => {
    if (stage !== "director") return;
    api.getSetup(id).then(setSetup).catch((e) => setError(String(e)));
    if (meta.confirmed) api.graph(id).then(setStoredGraph).catch((e) => setError(String(e)));
  }, [id, stage, meta.confirmed]);

  const graph = useMemo(
    () => (meta.confirmed ? storedGraph : setup ? setupGraph(setup) : null),
    [meta.confirmed, storedGraph, setup],
  );

  function setupChanged(next: SetupView, activeAfter: string | null) {
    setSetup(next);
    setEditing(null);
    setActiveId(activeAfter);
  }

  function startEdit(agentId: string) {
    const entry = setup?.characters.find((c) => c.agent_id === agentId);
    if (!entry || !vocabulary) return;
    setEditing({ id: agentId, draft: draftOf(entry, vocabulary) });
  }

  async function saveCharacter(draft: Draft) {
    if (!editing || !vocabulary) return;
    if (editing.id === null) {
      const added = await api.addCharacter(id, newCharacterOf(draft, blankDraft(vocabulary)));
      return setupChanged(added.setup, added.agent_id);
    }
    const revision = revisionOf(editing.draft, draft);
    if (!Object.keys(revision).length) return setEditing(null);
    setupChanged(await api.reviseCharacter(id, editing.id, revision), editing.id);
  }

  async function removeCharacter(agentId: string) {
    if (!setup) return;
    const name = identityOf(agentId, cast)?.name ?? "某人";
    const relations = setup.relations.filter((r) => r.source_id === agentId || r.target_id === agentId).length;
    const involved = setup.historical_events.filter((e) => e.figure_ids.includes(agentId));
    const dropped = involved.filter((e) => e.figure_ids.length === 1).length;
    const notes = [
      relations && `${relations} 条关系`,
      dropped && `${dropped} 条只有${name}参与的过往事件`,
    ].filter(Boolean);
    const kept = involved.length - dropped;
    const message = [
      `删除角色「${name}」？`,
      notes.length ? `会一起删除：${notes.join("、")}。` : "",
      kept ? `另有 ${kept} 条过往事件会保留，只是不再把${name}列为相关角色。` : "",
    ].filter(Boolean).join("\n");
    if (!window.confirm(message)) return;
    try {
      setupChanged(await api.removeCharacter(id, agentId), null);
    } catch (e) {
      setError(String(e));
    }
  }

  async function openDirectorReview() {
    setBusy(true);
    setError("");
    try {
      setMeta(await api.openReview(id));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  // Roster: main characters first.
  const roster = useMemo<CardAgent[]>(
    () =>
      [...(graph?.nodes ?? [])]
        .sort((a, b) => Number(b.is_main_character) - Number(a.is_main_character))
        .map((n) => ({
          id: n.id,
          is_main_character: n.is_main_character ?? false,
        })),
    [graph],
  );
  const effectiveId = activeId ?? roster[0]?.id ?? null;

  async function confirm() {
    setBusy(true);
    setError("");
    try {
      await api.confirmWorld(id, stage === "casting" ? castId : undefined);
      await Promise.allSettled([reloadCast(id), reloadPlaces(id), reloadNpcs(id)]);
      void refreshWorlds();
      navigate(`/worlds/${id}`);
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  }

  async function discard() {
    if (!window.confirm(`丢弃世界「${meta.world_name || id}」？此操作不可恢复。`)) return;
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
          {meta.confirmed && (
            <span className="text-[11px] text-emerald-400 bg-emerald-950/40 border border-emerald-800/60 px-2.5 py-1 rounded-full">
              已确认
            </span>
          )}
          {stage === "casting" && (
            <button
              onClick={() => setCasting(false)}
              disabled={busy}
              className="text-xs text-slate-400 hover:text-white px-3 py-2 rounded-lg bg-slate-900 border border-slate-800 hover:border-slate-700 transition disabled:opacity-40"
            >
              ← 换个方式
            </button>
          )}
          <button
            onClick={discard}
            disabled={busy}
            className="text-xs px-4 py-2 rounded-lg bg-slate-950 border border-rose-900/60 text-rose-400 hover:bg-rose-950/30 transition disabled:opacity-40"
          >
            丢弃
          </button>
          {(stage === "director" || stage === "casting") && (
            <button
              onClick={confirm}
              disabled={busy || seedEditing || (stage === "casting" && !castId)}
              className="text-xs font-semibold px-5 py-2 rounded-lg bg-indigo-600 text-white hover:bg-indigo-500 border border-indigo-500 shadow-lg shadow-indigo-600/30 transition disabled:opacity-40"
            >
              {/* Confirming materializes the world (its backstory memories embedded), which takes a moment. */}
              {busy && !meta.confirmed
                ? "正在生成世界…"
                : stage === "casting"
                  ? <><ActorIcon /> 我是{candidates.find((c) => c.agent_id === castId)?.name ?? "所选角色"} →</>
                  : meta.confirmed ? "进入叙事 →" : "确认并进入叙事 →"}
            </button>
          )}
        </div>
      </div>

      {dataError && <button className="text-sm text-rose-300" onClick={retryData}>世界资料加载失败，点击重试</button>}

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
            {stage === "director" && (
              <div className="flex items-center space-x-2 text-xs bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 text-slate-400">
                <span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
                <span>
                  {roster.length} 名角色 &amp; {graph?.edges.length ?? 0} 个关系连接
                </span>
              </div>
            )}
            {/* The story's own pace, chosen for it at build and frozen by confirming —
                a story told a day at a time is a different story from one told hour by
                hour, so it belongs here, next to the cast, before the user locks it. */}
            {(setup?.seconds_per_step ?? meta.seconds_per_step) ? (
              <StepLengthEditor
                secondsPerStep={setup?.seconds_per_step ?? meta.seconds_per_step}
                editable={revisable}
                onSave={async (seconds) => {
                  setSetup(await api.reviseSecondsPerStep(id, seconds));
                }}
              />
            ) : null}
          </div>
        </div>
        {(meta.description || meta.theme) && (
          <p className="text-slate-300 text-sm leading-relaxed max-w-4xl italic">
             {meta.description || meta.theme}
          </p>
        )}
      </div>

      {stage === "choose" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <EntryChoice
            icon="🎬"
            title="导演模式"
            description="可以以导演的视角去观察世界，操控世界，可以看到完整的世界。"
            disabled={busy}
            onClick={openDirectorReview}
          />
          <EntryChoice
            icon={<ActorIcon />}
            title="演员模式"
            description="扮演世界中的一个角色，以第一视角与世界中的其他角色共同叙事。"
            disabled={busy}
            onClick={() => setCasting(true)}
          />
        </div>
      )}

      {stage === "casting" && (
        <div className="space-y-3">
          <p className="text-xs text-slate-500">选择一个想要扮演的角色</p>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            {candidates.map((c) => (
              <CandidateCard key={c.agent_id} c={c} picked={c.agent_id === castId} onPick={() => setCastId(c.agent_id)} />
            ))}
          </div>
        </div>
      )}

      {/* Cast + graph — each a unified card (title + content in one bordered container),
          matching the lore card above and the narrative page's drawers. */}
      {stage === "director" && (
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
              {editing && vocabulary ? (
                <div className="bg-gradient-to-b from-slate-900/95 to-slate-950/95 border border-slate-800 rounded-2xl p-4 space-y-3 overflow-y-auto">
                  <p className="text-sm font-bold text-slate-100">
                    {editing.id === null ? "添加角色" : `编辑 · ${identityOf(editing.id, cast)?.name ?? "某人"}`}
                  </p>
                  <CharacterEditor
                    key={editing.id ?? "new"}
                    initial={editing.draft}
                    isNew={editing.id === null}
                    places={places}
                    vocabulary={vocabulary}
                    onSave={saveCharacter}
                    onCancel={() => setEditing(null)}
                  />
                </div>
              ) : (
                meta.confirmed ? (
                  <ObservedCharacterCard worldId={id}
                    roster={roster} activeId={effectiveId} onActiveChange={setActiveId} />
                ) : (
                  <CharacterCard cast={cast} places={places}
                    profile={setup?.characters.find((c) => c.agent_id === effectiveId)}
                    roster={roster}
                    activeId={effectiveId}
                    onActiveChange={setActiveId}
                    {...(revisable && vocabulary && setup ? {
                      onEdit: (agentId: string) => startEdit(agentId),
                      onRemove: (agentId: string) => void removeCharacter(agentId),
                      ...(roster.length < MAX_CAST_SIZE && {
                        onAdd: () => setEditing({ id: null, draft: blankDraft(vocabulary) }),
                      }),
                    } : {})}
                  />
                )
              )}
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
                <RelationshipGraph cast={cast} graph={graph} activeId={effectiveId} onNodeClick={setActiveId} />
              </div>
            </div>
          </div>
        </div>
      )}

      {stage === "director" && setup && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <div className="lg:col-span-5 space-y-8">
            <HistoryPanel
              events={setup.historical_events}
              cast={cast}
              editable={revisable}
              onSave={async (events) => setSetup(await api.replaceHistory(id, events))}
            />
            <SeedPanel kind="entity" entities={setup.entities} places={places}
              editable={revisable && !busy && !seedEditing} onEditingChange={setSeedEditing}
              onSave={async (entityId, revision) => setSetup(await api.reviseEntity(id, entityId, revision))}
              onAdd={async (entity) => setSetup((await api.addEntity(id, entity)).setup)}
              onRemove={async (entityId) => setSetup(await api.removeEntity(id, entityId))} />
          </div>
          <div className="lg:col-span-7 space-y-8">
            <RelationPanel
              relations={setup.relations}
              cast={cast}
              editable={revisable}
              onSave={async (relations) => setSetup(await api.replaceRelations(id, relations))}
            />
            <SeedPanel kind="npc" npcs={setup.npcs} places={places}
              editable={revisable && !busy && !seedEditing} onEditingChange={setSeedEditing}
              onSave={async (npcId, revision) => setSetup(await api.reviseNpc(id, npcId, revision))}
              onAdd={async (npc) => setSetup((await api.addNpc(id, npc)).setup)}
              onRemove={async (npcId) => setSetup(await api.removeNpc(id, npcId))} />
          </div>
        </div>
      )}
    </div>
  );
}

/** One way into the world, before anything of the cast is shown. */
function EntryChoice({
  icon,
  title,
  description,
  disabled,
  onClick,
}: {
  icon: ReactNode;
  title: string;
  description: string;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="group text-left flex flex-col gap-4 bg-gradient-to-br from-slate-900 to-slate-900/60 border border-slate-700/80 rounded-2xl p-6 shadow-lg cursor-pointer transition duration-200 hover:-translate-y-0.5 hover:border-indigo-500/80 hover:shadow-indigo-900/40 hover:shadow-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:translate-y-0 disabled:hover:border-slate-700/80"
    >
      <div className="flex items-center gap-3">
        <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-indigo-500/10 border border-indigo-500/30 text-2xl transition group-hover:bg-indigo-500/20">
          {icon}
        </span>
        <span className="text-lg font-bold text-slate-100">{title}</span>
      </div>
      <p className="text-sm leading-relaxed text-slate-400">{description}</p>
    </button>
  );
}

/** A character as anyone could know them; the identity color is their face. */
function CandidateCard({ c, picked, onPick }: { c: CharacterIdentity; picked: boolean; onPick: () => void }) {
  return (
    <button
      onClick={onPick}
      className={`text-left rounded-2xl border p-4 cursor-pointer transition duration-200 hover:-translate-y-0.5 hover:shadow-xl hover:shadow-indigo-900/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
        picked
          ? "bg-indigo-950/50 border-indigo-500/80 ring-1 ring-indigo-500/60"
          : "bg-slate-900 border-slate-800/80 hover:bg-indigo-950/30 hover:border-indigo-500/60"
      }`}
    >
      <div className="flex items-center gap-3">
        <div
          className="w-10 h-10 rounded-xl grid place-items-center text-white text-lg font-black shrink-0"
          style={avatarStyle(c.color, c.agent_id)}
        >
          {c.name[0]}
        </div>
        <div className="min-w-0">
          <div className="text-sm font-bold text-slate-100 truncate">{c.name}</div>
          <div className="text-[11px] text-slate-400">
            {[c.role, c.gender, c.age != null ? `${c.age} 岁` : ""].filter(Boolean).join(" · ")}
          </div>
        </div>
      </div>
      {c.appearance && <p className="mt-2 text-xs text-slate-300 leading-relaxed">{c.appearance}</p>}
      {c.background && <p className="mt-1.5 text-xs text-slate-500 leading-relaxed">{c.background}</p>}
    </button>
  );
}
