// The editor for one character of the world's setup, before the world is confirmed: a new
// character's whole entry, or a revision of an existing one. The name is given once, when the
// character is added: historical events name people in their text as it was written.
import { useState } from "react";

import { BTN, BTN_PRIMARY } from "../buttonStyles";
import type { Places } from "../../lib/places";
import { joinList, splitList } from "../../lib/textList";
import type {
  CharacterRevision, CharacterSetupView, InnateNeedView, NewCharacter, Vocabulary,
} from "../../types";

const INPUT =
  "w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500/70 disabled:opacity-50";
const LABEL = "text-[10px] font-bold text-slate-400";

type TextField = "role" | "appearance" | "background" | "secret" | "self_image" | "life_goal";

const TEXT_FIELDS: { key: TextField; label: string; rows: number }[] = [
  { key: "role", label: "身份", rows: 1 },
  { key: "appearance", label: "外貌", rows: 2 },
  { key: "background", label: "背景", rows: 4 },
  { key: "secret", label: "秘密", rows: 2 },
  { key: "self_image", label: "自我认知", rows: 2 },
  { key: "life_goal", label: "人生目标", rows: 2 },
];

export interface Draft {
  name: string;
  role: string;
  appearance: string;
  background: string;
  self_image: string;
  life_goal: string;
  secret: string;
  gender: string;
  age: string;
  core_traits: string;
  core_values: string;
  is_main_character: boolean;
  initial_location: string;
  long_term_goals: string;
  hard_constraints: string;
  emotion: string;
  emotion_intensity: number;
  emotion_valence: number;
  // One row per need type, in the vocabulary's order; a blank label means "the generic one".
  needs: InnateNeedView[];
}

const lines = (text: string) => text.split("\n").map((l) => l.trim()).filter(Boolean);

function needRows(vocabulary: Vocabulary, needs: readonly InnateNeedView[]): InnateNeedView[] {
  return vocabulary.needs.map(({ value }) =>
    needs.find((n) => n.type === value) ?? { type: value, label: "", intensity: 0.5, weight: 1, is_hidden: false },
  );
}

/** A new character's empty form. */
export function blankDraft(vocabulary: Vocabulary): Draft {
  return {
    name: "", role: "", appearance: "", background: "", self_image: "", life_goal: "", secret: "",
    gender: "", age: "", core_traits: "", core_values: "", is_main_character: false,
    initial_location: "", long_term_goals: "", hard_constraints: "",
    emotion: "neutral", emotion_intensity: 0.2, emotion_valence: 0,
    needs: needRows(vocabulary, []),
  };
}

export function draftOf(s: CharacterSetupView, vocabulary: Vocabulary): Draft {
  return {
    name: s.name,
    role: s.role,
    appearance: s.appearance,
    background: s.background,
    self_image: s.self_image,
    life_goal: s.life_goal ?? "",
    secret: s.secret,
    gender: s.gender,
    age: s.age != null ? String(s.age) : "",
    core_traits: joinList(s.core_traits),
    core_values: joinList(s.core_values),
    is_main_character: s.is_main_character,
    initial_location: s.initial_location,
    long_term_goals: s.long_term_goals.join("\n"),
    hard_constraints: s.hard_constraints.join("\n"),
    emotion: s.initial_emotion.emotion,
    emotion_intensity: s.initial_emotion.intensity,
    emotion_valence: s.initial_emotion.valence,
    needs: needRows(vocabulary, s.needs),
  };
}

/** Only what changed from `before`, so an untouched field is never rewritten. */
export function revisionOf(before: Draft, draft: Draft): CharacterRevision {
  const revision: CharacterRevision = {};
  for (const { key } of TEXT_FIELDS) {
    if (draft[key].trim() !== before[key].trim()) revision[key] = draft[key].trim();
  }
  if (draft.gender !== before.gender && draft.gender) revision.gender = draft.gender;
  if (draft.age.trim() !== before.age && draft.age.trim() !== "") revision.age = Number(draft.age);
  if (draft.core_traits !== before.core_traits) revision.core_traits = splitList(draft.core_traits);
  if (draft.core_values !== before.core_values) revision.core_values = splitList(draft.core_values);
  if (draft.is_main_character !== before.is_main_character) revision.is_main_character = draft.is_main_character;
  if (draft.initial_location !== before.initial_location && draft.initial_location) {
    revision.initial_location = draft.initial_location;
  }
  if (draft.long_term_goals !== before.long_term_goals) revision.long_term_goals = lines(draft.long_term_goals);
  if (draft.hard_constraints !== before.hard_constraints) revision.hard_constraints = lines(draft.hard_constraints);
  if (
    draft.emotion !== before.emotion
    || draft.emotion_intensity !== before.emotion_intensity
    || draft.emotion_valence !== before.emotion_valence
  ) {
    revision.initial_emotion = { emotion: draft.emotion, intensity: draft.emotion_intensity, valence: draft.emotion_valence };
  }
  if (JSON.stringify(draft.needs) !== JSON.stringify(before.needs)) {
    revision.needs = draft.needs.filter((n) => n.label.trim()).map((n) => ({ ...n, label: n.label.trim() }));
  }
  return revision;
}

/** What a new character still lacks before it can be added; null when nothing. */
export function missingForNew(draft: Draft): string | null {
  const missing = [
    !draft.name.trim() && "名字",
    !draft.gender && "性别",
    !draft.age.trim() && "年龄",
    !draft.role.trim() && "身份",
    !draft.initial_location && "初始地点",
  ].filter(Boolean);
  return missing.length ? `还需填写：${missing.join("、")}` : null;
}

/** A new character's whole setup: the required fields always, the rest when filled. */
export function newCharacterOf(draft: Draft, blank: Draft): NewCharacter {
  return {
    ...revisionOf(blank, draft),
    name: draft.name.trim(),
    role: draft.role.trim(),
    gender: draft.gender,
    age: Number(draft.age),
    is_main_character: draft.is_main_character,
    initial_location: draft.initial_location,
  };
}

function Slider({ label, value, min, max, step, disabled, onChange }: {
  label: string; value: number; min: number; max: number; step: number; disabled: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <label className="flex items-center gap-1.5 text-[11px] text-slate-400">
      <span className="shrink-0 whitespace-nowrap">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled} onChange={(e) => onChange(Number(e.target.value))} className="min-w-0 flex-1" />
      <span className="tabular-nums w-8 shrink-0">{Math.round(value * 100)}</span>
    </label>
  );
}

export default function CharacterEditor({
  initial,
  isNew,
  places,
  vocabulary,
  onSave,
  onCancel,
}: {
  initial: Draft;
  isNew: boolean;
  places: Places;
  vocabulary: Vocabulary;
  // Rejects with the backend's reason; the editor stays open so the form can be fixed.
  onSave: (draft: Draft) => Promise<void>;
  onCancel: () => void;
}) {
  const [draft, setDraft] = useState<Draft>(initial);
  const [motivesOpen, setMotivesOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const set = <K extends keyof Draft>(key: K) => (value: Draft[K]) => setDraft((d) => ({ ...d, [key]: value }));
  const setNeed = (i: number, patch: Partial<InnateNeedView>) =>
    setDraft((d) => ({ ...d, needs: d.needs.map((n, j) => (j === i ? { ...n, ...patch } : n)) }));
  const missing = isNew ? missingForNew(draft) : null;
  const needLabel = (value: string) => vocabulary.needs.find((n) => n.value === value)?.label ?? value;

  async function save() {
    if (missing) return;
    setBusy(true);
    setError("");
    try {
      await onSave(draft);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }

  return (
    <div className="space-y-2.5">
      {isNew && (
        <label className="block space-y-1">
          <span className={LABEL}>名字 <span className="font-normal text-slate-500">· 添加后不可修改</span></span>
          <input value={draft.name} disabled={busy} onChange={(e) => set("name")(e.target.value)} className={INPUT} />
        </label>
      )}
      <div className="grid grid-cols-2 gap-2">
        <label className="space-y-1">
          <span className={LABEL}>性别</span>
          <select value={draft.gender} disabled={busy} onChange={(e) => set("gender")(e.target.value)} className={INPUT}>
            {!draft.gender && <option value="" disabled>请选择</option>}
            <option value="男">男</option>
            <option value="女">女</option>
          </select>
        </label>
        <label className="space-y-1">
          <span className={LABEL}>年龄</span>
          <input type="number" min={0} value={draft.age} disabled={busy} onChange={(e) => set("age")(e.target.value)} className={INPUT} />
        </label>
        <label className="space-y-1">
          <span className={LABEL}>初始地点</span>
          <select value={draft.initial_location} disabled={busy} onChange={(e) => set("initial_location")(e.target.value)} className={INPUT}>
            {!draft.initial_location && <option value="" disabled>请选择</option>}
            {Object.values(places).map((p) => <option key={p.location_id} value={p.location_id}>{p.name}</option>)}
          </select>
        </label>
        <label className="flex items-center gap-2 self-end pb-1.5 text-xs text-slate-300">
          <input type="checkbox" checked={draft.is_main_character} disabled={busy} onChange={(e) => set("is_main_character")(e.target.checked)} />
          主角
        </label>
      </div>
      {TEXT_FIELDS.map(({ key, label, rows }) => (
        <label key={key} className="block space-y-1">
          <span className={LABEL}>{label}</span>
          {rows === 1 ? (
            <input value={draft[key]} disabled={busy} onChange={(e) => set(key)(e.target.value)} className={INPUT} />
          ) : (
            <textarea rows={rows} value={draft[key]} disabled={busy} onChange={(e) => set(key)(e.target.value)} className={`${INPUT} resize-y`} />
          )}
        </label>
      ))}
      {(["core_traits", "core_values"] as const).map((key) => (
        <label key={key} className="block space-y-1">
          <span className={LABEL}>
            {key === "core_traits" ? "性格特点" : "价值观"}
            <span className="font-normal text-slate-500"> · 以「、」分隔</span>
          </span>
          <input value={draft[key]} disabled={busy} onChange={(e) => set(key)(e.target.value)} className={INPUT} />
        </label>
      ))}

      <div className="border-t border-slate-800/70 pt-2.5">
        <button onClick={() => setMotivesOpen((o) => !o)} className="text-xs text-slate-400 hover:text-slate-200 transition flex items-center gap-1">
          <span className="text-slate-600">{motivesOpen ? "▾" : "▸"}</span> 动机
        </button>
        {motivesOpen && (
          <div className="mt-2.5 space-y-2.5">
            {(["long_term_goals", "hard_constraints"] as const).map((key) => (
              <label key={key} className="block space-y-1">
                <span className={LABEL}>
                  {key === "long_term_goals" ? "长期目标" : "底线"}
                  <span className="font-normal text-slate-500"> · 每行一条</span>
                </span>
                <textarea rows={3} value={draft[key]} disabled={busy} onChange={(e) => set(key)(e.target.value)} className={`${INPUT} resize-y`} />
              </label>
            ))}
            <div className="space-y-1">
              <span className={LABEL}>初始情绪</span>
              <select value={draft.emotion} disabled={busy} onChange={(e) => set("emotion")(e.target.value)} className={INPUT}>
                {vocabulary.emotions.map((e) => <option key={e.value} value={e.value}>{e.label}</option>)}
              </select>
              <Slider label="强度" value={draft.emotion_intensity} min={0} max={1} step={0.05} disabled={busy} onChange={set("emotion_intensity")} />
              <Slider label="基调" value={draft.emotion_valence} min={-1} max={1} step={0.05} disabled={busy} onChange={set("emotion_valence")} />
            </div>
            <div className="space-y-2">
              <span className={LABEL}>内在需求 <span className="font-normal text-slate-500">· 描述留空则用通用描述</span></span>
              {draft.needs.map((n, i) => (
                <div key={n.type} className="bg-slate-950/50 border border-slate-800/70 rounded-lg px-2.5 py-2 space-y-1.5">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-slate-300 font-semibold">{needLabel(n.type)}</span>
                    <label className="flex items-center gap-1.5 text-slate-400">
                      <input type="checkbox" checked={n.is_hidden} disabled={busy} onChange={(e) => setNeed(i, { is_hidden: e.target.checked })} />
                      潜意识
                    </label>
                  </div>
                  <input
                    value={n.label}
                    disabled={busy}
                    placeholder="这个需求对这个角色意味着什么"
                    onChange={(e) => setNeed(i, { label: e.target.value })}
                    className={INPUT}
                  />
                  <Slider label="在意程度" value={n.weight} min={0.1} max={2} step={0.05} disabled={busy} onChange={(weight) => setNeed(i, { weight })} />
                  <Slider label="初始强度" value={n.intensity} min={0} max={1} step={0.05} disabled={busy} onChange={(intensity) => setNeed(i, { intensity })} />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {(missing || error) && <p className="text-[11px] text-rose-400">{missing ?? error}</p>}
      <div className="flex justify-end gap-2">
        <button onClick={onCancel} disabled={busy} className={BTN}>取消</button>
        <button onClick={() => void save()} disabled={busy || !!missing} className={BTN_PRIMARY}>{busy ? "保存中…" : "保存"}</button>
      </div>
    </div>
  );
}
