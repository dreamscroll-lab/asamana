/**
 * MindPanel — the performed character's mind, always at hand atop the actor's console: feeling,
 * goals and relations, and an insight the actor may add to it.
 *
 * Each is a current value: the actor overwrites it or leaves it be; an insight is only added, one
 * at a time, and kept among the character's memories. Edits gather in one draft and
 * are saved together; DECIDE and INTERRUPT saves land at once, other saves at the
 * start of the next step (a write between steps belongs to no snapshot). Saved but not landed,
 * a value shows the actor's version, marked.
 *
 * The engine keeps moving the same values. A tab the engine changed since the last look carries
 * a dot, and a moved relation shows by how much, so the actor feels what happened to them.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "../../api/client";
import { AFFECTION_RANGE, EMOTION_INTENSITY_RANGE, EMOTION_VALENCE_RANGE, TRUST_RANGE } from "../../lib/contract";
import { useVocabulary } from "../../lib/vocabulary";
import type { ActorAmendments, ActorMind, BriefRelation, EmotionAmendment, GoalAmendment, VocabularyEntry } from "../../types";
import { BTN, BTN_PRIMARY } from "../buttonStyles";
import AutoGrowTextarea from "../AutoGrowTextarea";
import { CONSOLE_EDIT_BOX, CONSOLE_INPUT_MAX_PX } from "../consoleStyles";
import { reasonText } from "./decisionSummary";
import GoalPanel from "./GoalPanel";
import { goalRows, goalsChanged, shortTermAmendments } from "./goals";

type Tab = "emotion" | "goals" | "relations" | "insight";

const TABS: { key: Tab; label: string }[] = [
  { key: "emotion", label: "情绪" },
  { key: "goals", label: "目标" },
  { key: "relations", label: "关系" },
  { key: "insight", label: "洞察" },
];

// The backend's cap on one insight (interaction/api/actor.py).
const MAX_INSIGHT = 500;

interface Draft {
  emotion: EmotionAmendment | null;
  relations: Record<string, { trust: number | null; affection: number | null }>;
  long: string[] | null;
  short: GoalAmendment[] | null;
  shortRemoved: string[];
  insight: string;
}

const EMPTY_DRAFT: Draft = { emotion: null, relations: {}, long: null, short: null, shortRemoved: [], insight: "" };

const clean = (goals: string[]) => goals.map((g) => g.trim()).filter(Boolean);

function editCount(d: Draft): number {
  return (
    (d.emotion ? 1 : 0) +
    Object.keys(d.relations).length +
    (d.long ? 1 : 0) +
    (d.short ? 1 : 0) +
    (d.insight.trim() ? 1 : 0)
  );
}

function amendmentsOf(d: Draft, mind: ActorMind): ActorAmendments {
  return {
    emotion: d.emotion,
    relations: Object.entries(d.relations).map(([agent_id, v]) => ({ agent_id, ...v })),
    long_term_goals: d.long ? clean(d.long) : null,
    ...shortTermAmendments(mind, d.short, d.shortRemoved),
    insights: d.insight.trim() ? [d.insight.trim()] : [],
  };
}

/** The mind as it will stand once what's saved lands: the engine's values under `pending`. */
function withPending(m: ActorMind) {
  const p = m.pending;
  const relations = new Map((p?.relations ?? []).map((r) => [r.agent_id, r]));
  return {
    emotion: p?.emotion ?? { emotion: m.emotion, intensity: m.emotion_intensity, valence: m.emotion_valence },
    long: p?.long_term_goals ?? m.long_term_goals,
    relation: (r: BriefRelation) => ({
      trust: relations.get(r.agent_id)?.trust ?? r.trust,
      affection: relations.get(r.agent_id)?.affection ?? r.affection,
      pending: relations.has(r.agent_id),
    }),
    insights: p?.insights ?? [],
  };
}

/** What the engine changed between two looks at the mind. */
interface Moved {
  emotion: boolean;
  goals: boolean;
  relations: Record<string, { trust: number; affection: number }>;
}

const NOTHING_MOVED: Moved = { emotion: false, goals: false, relations: {} };

function movedBetween(before: ActorMind, after: ActorMind): Moved {
  const old = new Map(before.relations.map((r) => [r.agent_id, r]));
  const relations: Moved["relations"] = {};
  for (const r of after.relations) {
    const o = old.get(r.agent_id);
    const trust = r.trust - (o?.trust ?? r.trust);
    const affection = r.affection - (o?.affection ?? r.affection);
    if (!o || Math.abs(trust) >= 0.005 || Math.abs(affection) >= 0.005) relations[r.agent_id] = { trust, affection };
  }
  return {
    emotion:
      before.emotion !== after.emotion ||
      Math.abs(before.emotion_intensity - after.emotion_intensity) >= 0.005 ||
      Math.abs(before.emotion_valence - after.emotion_valence) >= 0.005,
    goals: goalsChanged(before, after),
    relations,
  };
}

function hasMoved(m: Moved, tab: Tab): boolean {
  if (tab === "emotion") return m.emotion;
  if (tab === "goals") return m.goals;
  if (tab === "relations") return Object.keys(m.relations).length > 0;
  return false;
}

export default function MindPanel({
  worldId,
  refreshKey,
  turnId,
  running,
  immediate,
  onMindChange,
  mind,
}: {
  worldId: string;
  // Changes whenever the world may have moved the mind (a step landed, a turn opened or closed).
  refreshKey: string;
  turnId: string | null;
  running: boolean;
  immediate: boolean;
  onMindChange: (mind: ActorMind) => void;
  mind: ActorMind | null;
}) {
  const [moved, setMoved] = useState<Moved>(NOTHING_MOVED);
  // Tabs looked at since the engine last moved something there.
  const [looked, setLooked] = useState<Set<Tab>>(new Set());
  const [tab, setTab] = useState<Tab | null>(null);
  const [draft, setDraft] = useState<Draft>(EMPTY_DRAFT);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<{ text: string; error: boolean } | null>(null);
  const emotions = useVocabulary()?.emotions ?? [];
  const last = useRef<ActorMind | null>(null);
  const loadSequence = useRef(0);

  // `ownSave`: the change is the actor's own, not something that happened to them.
  const load = useCallback(
    async (ownSave = false) => {
      const sequence = ++loadSequence.current;
      try {
        const next = await api.actorMind(worldId);
        if (sequence !== loadSequence.current) return;
        if (last.current && !ownSave) {
          setMoved(movedBetween(last.current, next));
          setLooked(new Set());
        }
        last.current = next;
        onMindChange(next);
      } catch {
        // Unreadable now (no session yet): keep what is shown.
      }
    },
    [worldId, onMindChange],
  );

  useEffect(() => {
    void load();
    return () => { loadSequence.current++; };
  }, [load, refreshKey]);

  // A new turn puts the moment first: the mind folds down to its tab bar.
  useEffect(() => {
    if (turnId) setTab(null);
  }, [turnId]);

  if (!mind) return null;
  const shown = withPending(mind);
  const edits = editCount(draft);
  const goalAlerts = goalRows(mind).filter((g) => g.detail?.attention).length;

  const open = (t: Tab) => {
    setTab((cur) => (cur === t ? null : t));
    setLooked((s) => new Set(s).add(t));
  };

  const save = async () => {
    setBusy(true);
    setNotice(null);
    try {
      const r = await api.amendActor(worldId, amendmentsOf(draft, mind));
      if (!r.accepted) {
        setNotice({ text: reasonText(r.reason), error: true });
        return;
      }
      setDraft(EMPTY_DRAFT);
      setNotice({ text: r.landed ? "已生效" : "下一步生效", error: false });
      await load(true);
    } catch (e) {
      setNotice({ text: String(e), error: true });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1">
        {TABS.map(({ key, label }) => {
          const dot = hasMoved(moved, key) && !looked.has(key);
          return (
            <button
              key={key}
              onClick={() => open(key)}
              className={`relative rounded-full px-3 py-1 text-[12px] transition ${
                tab === key ? "bg-indigo-600/80 text-white" : "text-slate-400 hover:bg-slate-800/70 hover:text-slate-200"
              }`}
            >
              {label}
              {key === "goals" && goalAlerts > 0 && (
                <span className="ml-1 text-amber-300">{goalAlerts}</span>
              )}
              {dot && <span className="absolute -top-0.5 -right-0.5 h-2 w-2 rounded-full bg-amber-400" />}
            </button>
          );
        })}
      </div>

      {tab === "emotion" && (
        <EmotionTab
          value={draft.emotion ?? shown.emotion}
          pending={!!mind.pending?.emotion}
          edited={!!draft.emotion}
          emotions={emotions}
          onChange={(e) => setDraft({ ...draft, emotion: e })}
        />
      )}
      {tab === "goals" && (
        <div className="space-y-2">
          <GoalList
            label="长期目标"
            goals={draft.long ?? shown.long}
            pending={!!mind.pending?.long_term_goals}
            edited={!!draft.long}
            onChange={(g) => setDraft({ ...draft, long: g })}
          />
          <GoalPanel mind={mind} draft={draft.short} removedIds={draft.shortRemoved}
            onChange={(goals, removedId) => setDraft((current) => ({
              ...current, short: goals,
              shortRemoved: goals === null ? [] : removedId
                ? [...current.shortRemoved, removedId] : current.shortRemoved,
            }))} />
        </div>
      )}
      {tab === "relations" && (
        <div className="space-y-1.5">
          {mind.relations.length === 0 && <p className="text-[12px] text-slate-600">暂无关系。</p>}
          {mind.relations.map((r) => (
            <RelationLine
              key={r.agent_id}
              relation={r}
              current={shown.relation(r)}
              moved={moved.relations[r.agent_id]}
              edit={draft.relations[r.agent_id]}
              onChange={(v) => {
                const relations = { ...draft.relations };
                if (v === undefined) delete relations[r.agent_id];
                else relations[r.agent_id] = v;
                setDraft({ ...draft, relations });
              }}
            />
          ))}
        </div>
      )}
      {tab === "insight" && (
        <div className="space-y-1.5">
          {shown.insights.map((text, i) => (
            <div key={i} className="flex items-start gap-2 text-[12px] text-slate-300">
              <span className="flex-1">{text}</span>
              <Mark pending edited={false} />
            </div>
          ))}
          <AutoGrowTextarea
            maxPx={CONSOLE_INPUT_MAX_PX}
            value={draft.insight}
            onChange={(e) => setDraft({ ...draft, insight: e.target.value })}
            rows={2}
            maxLength={MAX_INSIGHT}
            placeholder="以你的视角记录一条新的认识或理解…"
            className={CONSOLE_EDIT_BOX}
          />
        </div>
      )}

      {tab !== null && (edits > 0 || notice) && (
        <div className="flex items-center gap-2 border-t border-slate-800/60 pt-2">
          {edits > 0 ? (
            <span className="text-[11px] text-slate-400">
              {edits} 处修改 · {immediate ? "保存后立即生效" : running ? "下一步生效" : "运行后的下一步生效"}
            </span>
          ) : (
            <span className={`text-[11px] ${notice!.error ? "text-rose-400" : "text-emerald-400"}`}>{notice!.text}</span>
          )}
          {edits > 0 && notice?.error && <span className="text-[11px] text-rose-400">{notice.text}</span>}
          {edits > 0 && (
            <div className="ml-auto flex gap-2">
              <button className={BTN} disabled={busy} onClick={() => setDraft(EMPTY_DRAFT)}>
                撤销
              </button>
              <button className={BTN_PRIMARY} disabled={busy} onClick={save}>
                {busy ? "保存中…" : "保存"}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Mark({ pending, edited }: { pending: boolean; edited: boolean }) {
  if (edited) return <span className="text-[10px] text-indigo-300">未保存</span>;
  if (pending) return <span className="text-[10px] text-amber-300/80">待生效</span>;
  return null;
}

function EmotionTab({
  value,
  pending,
  edited,
  emotions,
  onChange,
}: {
  value: EmotionAmendment;
  pending: boolean;
  edited: boolean;
  emotions: VocabularyEntry[];
  onChange: (e: EmotionAmendment | null) => void;
}) {
  const set = (e: Partial<EmotionAmendment>) => onChange({ ...value, ...e });
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-2">
        <select
          value={value.emotion}
          onChange={(e) => set({ emotion: e.target.value })}
          className="bg-slate-950/80 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-200"
        >
          {!emotions.some((e) => e.value === value.emotion) && <option value={value.emotion}>…</option>}
          {emotions.map((e) => (
            <option key={e.value} value={e.value}>
              {e.label}
            </option>
          ))}
        </select>
        <Mark pending={pending} edited={edited} />
        {edited && (
          <button className="ml-auto text-[11px] text-slate-500 hover:text-slate-300" onClick={() => onChange(null)}>
            撤销
          </button>
        )}
      </div>
      <div className="grid grid-cols-2 gap-3">
        <Slider label="强度" range={EMOTION_INTENSITY_RANGE} value={value.intensity} onChange={(v) => set({ intensity: v })} />
        <Slider label="正负值" range={EMOTION_VALENCE_RANGE} value={value.valence} onChange={(v) => set({ valence: v })} />
      </div>
    </div>
  );
}

function GoalList({
  label,
  goals,
  pending,
  edited,
  onChange,
}: {
  label: string;
  goals: string[];
  pending: boolean;
  edited: boolean;
  onChange: (g: string[] | null) => void;
}) {
  const [editing, setEditing] = useState(false);
  const set = (i: number, text: string) => onChange(goals.map((g, j) => (j === i ? text : g)));
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2 text-[11px] text-slate-500">
        {label}
        <button className="ml-auto hover:text-white" onClick={() => setEditing(!editing)}>{editing ? "收起编辑" : "编辑"}</button>
        <Mark pending={pending} edited={edited} />
        {edited && (
          <button className="ml-auto text-slate-500 hover:text-slate-300" onClick={() => onChange(null)}>
            撤销
          </button>
        )}
      </div>
      {goals.length === 0 && <p className="text-[12px] text-slate-600">暂无</p>}
      {goals.map((g, i) => (
        <div key={i} className="flex items-start gap-1.5">
          <span className="shrink-0 text-[12px] text-slate-500 tabular-nums">{i + 1}.</span>
          {editing ? <AutoGrowTextarea maxPx={CONSOLE_INPUT_MAX_PX} value={g} rows={2} onChange={(e) => set(i, e.target.value)} className={CONSOLE_EDIT_BOX} />
            : <p className="m-0 flex-1 whitespace-pre-wrap break-words text-[12px] text-slate-200">{g}</p>}
          {editing && (
            <button
              className="shrink-0 px-1 text-[12px] text-slate-600 hover:text-rose-300"
              title="删除目标"
              onClick={() => onChange(goals.filter((_, j) => j !== i))}
            >
              ×
            </button>
          )}
        </div>
      ))}
      <button className="text-[11px] text-slate-500 hover:text-slate-200" onClick={() => { setEditing(true); onChange([...goals, ""]); }}>
        ＋ 添加
      </button>
    </div>
  );
}

function RelationLine({
  relation,
  current,
  moved,
  edit,
  onChange,
}: {
  relation: BriefRelation;
  current: { trust: number; affection: number; pending: boolean };
  moved: { trust: number; affection: number } | undefined;
  edit: { trust: number | null; affection: number | null } | undefined;
  onChange: (v: { trust: number | null; affection: number | null } | undefined) => void;
}) {
  const trust = edit?.trust ?? current.trust;
  const affection = edit?.affection ?? current.affection;
  return (
    <div className="space-y-0.5">
      <div className="flex items-center gap-2 text-[12px] text-slate-300">
        <span>{relation.name}</span>
        {relation.labels.length > 0 && <span className="text-[11px] text-slate-500">{relation.labels.join("、")}</span>}
        <Mark pending={current.pending} edited={!!edit} />
        {edit && (
          <button className="ml-auto text-[11px] text-slate-500 hover:text-slate-300" onClick={() => onChange(undefined)}>
            撤销
          </button>
        )}
      </div>
      <div className="grid grid-cols-2 gap-3">
        <Slider
          label="信任度"
          range={TRUST_RANGE}
          value={trust}
          delta={moved?.trust}
          onChange={(v) => onChange({ trust: v, affection: edit?.affection ?? null })}
        />
        <Slider
          label="好感度"
          range={AFFECTION_RANGE}
          value={affection}
          delta={moved?.affection}
          onChange={(v) => onChange({ trust: edit?.trust ?? null, affection: v })}
        />
      </div>
    </div>
  );
}

function Slider({
  label,
  range,
  value,
  delta,
  onChange,
}: {
  label: string;
  range: [number, number];
  value: number;
  delta?: number;
  onChange: (v: number) => void;
}) {
  const moved = delta !== undefined && Math.abs(delta) >= 0.005;
  return (
    <label className="flex min-w-0 items-center gap-1 text-[11px] text-slate-500">
      <span className="shrink-0 whitespace-nowrap">{label}</span>
      <input
        type="range"
        min={range[0]}
        max={range[1]}
        step={0.05}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-0 min-w-0 flex-1 border-0 bg-transparent p-0 accent-indigo-500"
      />
      <span className="w-8 shrink-0 tabular-nums text-slate-400">{value.toFixed(2)}</span>
      {moved && (
        <span className={`shrink-0 tabular-nums ${delta! > 0 ? "text-emerald-400/80" : "text-rose-400/80"}`}>
          {delta! > 0 ? "↑" : "↓"}
          {Math.abs(delta!).toFixed(2)}
        </span>
      )}
    </label>
  );
}
