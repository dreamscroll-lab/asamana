// The relations a world starts with, a pair at a time with both directions side by side: an
// asymmetric bond is only right when each side names the other correctly.
// Revisable until the world is confirmed; a save replaces the list.
import { useState } from "react";

import { identityOf } from "../../lib/worldTable";
import { BTN, BTN_PRIMARY } from "../buttonStyles";
import { SETUP_INPUT } from "./styles";
import type { Cast } from "../../lib/cast";
import { joinList, splitList } from "../../lib/textList";
import type { SetupRelationView } from "../../types";

const pct = (x: number) => Math.round(x * 100);

/** A relation as edited: labels stay text until saved, so a half-typed 「、」 isn't eaten. */
interface Draft extends Omit<SetupRelationView, "labels" | "reverse_labels"> {
  labels: string;
  reverse_labels: string;
}

const draftOf = (r: SetupRelationView): Draft => ({
  ...r, labels: joinList(r.labels), reverse_labels: joinList(r.reverse_labels),
});

const relationOf = (d: Draft): SetupRelationView => ({
  ...d, labels: splitList(d.labels), reverse_labels: splitList(d.reverse_labels),
});

/** A label's parts, either colon taken: one is 「关系类型」, two are 「关系类型:角色」. */
const labelParts = (label: string) => label.split(/[:：]/).map((part) => part.trim());

/** Why a set of drafts can't be saved, in words for whoever edits them; null when it can. */
export function draftProblem(
  drafts: readonly Pick<Draft, "source_id" | "target_id" | "labels" | "reverse_labels">[],
): string | null {
  const pairs = new Set<string>();
  for (const d of drafts) {
    if (!d.source_id || !d.target_id) return "每条关系都要选定两个人。";
    if (d.source_id === d.target_id) return "关系的两端不能是同一个人。";
    const pair = [d.source_id, d.target_id].sort().join("|");
    if (pairs.has(pair)) return "同一对人只能有一条关系。";
    pairs.add(pair);
    for (const labels of [d.labels, d.reverse_labels]) {
      const parts = splitList(labels).map(labelParts);
      if (parts.some((p) => p.length > 2 || p.some((part) => !part))) {
        return "标签写作「关系类型」或「关系类型:对方身份」。";
      }
      const canonical = parts.map((p) => p.join(":"));
      if (new Set(canonical).size !== canonical.length) return "同一方向的标签不能重复。";
    }
  }
  return null;
}

function Direction({
  holder,
  other,
  trust,
  affection,
  labels,
  editing,
  busy,
  onChange,
}: {
  holder: string;
  other: string;
  trust: number;
  affection: number;
  labels: string;
  editing: boolean;
  busy: boolean;
  onChange: (patch: { trust?: number; affection?: number; labels?: string }) => void;
}) {
  return (
    <div className="bg-slate-950/50 border border-slate-800/70 rounded-lg px-3 py-2 space-y-1.5">
      <p className="text-[11px] text-slate-400">
        <span className="text-slate-200">{holder}</span> 眼中的 <span className="text-slate-200">{other}</span>
      </p>
      {editing ? (
        <>
          <input
            value={labels}
            disabled={busy}
            placeholder="关系类型:对方身份、关系类型"
            onChange={(e) => onChange({ labels: e.target.value })}
            className={`${SETUP_INPUT} w-full`}
          />
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-400">
            <label className="flex items-center gap-1.5">
              <span className="shrink-0 whitespace-nowrap">信任</span>
              <input type="range" min={0} max={1} step={0.05} value={trust} disabled={busy} onChange={(e) => onChange({ trust: Number(e.target.value) })} />
              <span className="tabular-nums w-7">{pct(trust)}</span>
            </label>
            <label className="flex items-center gap-1.5">
              <span className="shrink-0 whitespace-nowrap">好感</span>
              <input type="range" min={-1} max={1} step={0.05} value={affection} disabled={busy} onChange={(e) => onChange({ affection: Number(e.target.value) })} />
              <span className="tabular-nums w-8">{pct(affection)}</span>
            </label>
          </div>
        </>
      ) : (
        <p className="text-[11px] text-slate-500">
          {labels || "无标签"} · 信任 {pct(trust)} · 好感 {pct(affection)}
        </p>
      )}
    </div>
  );
}

export default function RelationPanel({
  relations,
  cast,
  editable,
  onSave,
}: {
  relations: SetupRelationView[];
  cast: Cast;
  editable: boolean;
  // Rejects with the backend's reason; the drafts stay so they can be fixed.
  onSave: (relations: SetupRelationView[]) => Promise<void>;
}) {
  const [drafts, setDrafts] = useState<Draft[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const members = Object.values(cast);
  const nameOf = (id: string) => identityOf(id, cast)?.name ?? "某人";
  const shown = drafts ?? relations.map(draftOf);
  const problem = drafts ? draftProblem(drafts) : null;

  const update = (i: number, patch: Partial<Draft>) =>
    setDrafts((d) => d && d.map((r, j) => (j === i ? { ...r, ...patch } : r)));

  async function save() {
    if (!drafts || problem) return;
    setBusy(true);
    setError("");
    try {
      await onSave(drafts.map(relationOf));
      setDrafts(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const personSelect = (value: string, onChange: (id: string) => void) => (
    <select value={value} disabled={busy} onChange={(e) => onChange(e.target.value)} className={SETUP_INPUT}>
      <option value="">选择角色</option>
      {members.map((m) => <option key={m.agent_id} value={m.agent_id}>{m.name}</option>)}
    </select>
  );

  return (
    <div className="bg-slate-900 border border-slate-800/80 rounded-2xl p-4 shadow-lg space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold tracking-wider text-slate-400 uppercase">
          <span className="inline-flex items-center rounded-full bg-slate-800/60 border border-slate-700/60 px-4 py-1.5">🤝 初始关系</span>
        </h4>
        {editable && !drafts && (
          <button onClick={() => setDrafts(relations.map(draftOf))} className={BTN}>✎ 编辑</button>
        )}
      </div>

      {!shown.length && !drafts && <p className="text-xs text-slate-500">还没有初始关系。</p>}
      {drafts && (
        <p className="text-[11px] text-slate-500 leading-relaxed">
          标签写作「关系类型:对方身份」，没有具体身份时只写关系类型。多个标签用「、」或逗号隔开，标签里不要出现顿号或逗号。
        </p>
      )}

      <div className="space-y-3">
        {shown.map((r, i) => (
          <div key={i} className="space-y-1.5">
            <div className="flex items-center gap-2 text-xs text-slate-300">
              {drafts ? (
                <>
                  {personSelect(r.source_id, (id) => update(i, { source_id: id }))}
                  <span className="text-slate-500">⇄</span>
                  {personSelect(r.target_id, (id) => update(i, { target_id: id }))}
                  <button
                    disabled={busy}
                    onClick={() => setDrafts((d) => d && d.filter((_, j) => j !== i))}
                    className="ml-auto text-[11px] text-rose-400 hover:text-rose-300"
                  >
                    删除
                  </button>
                </>
              ) : (
                <span className="font-semibold">{nameOf(r.source_id)} ⇄ {nameOf(r.target_id)}</span>
              )}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <Direction
                holder={nameOf(r.source_id)}
                other={nameOf(r.target_id)}
                trust={r.trust}
                affection={r.affection}
                labels={r.labels}
                editing={!!drafts}
                busy={busy}
                onChange={(p) => update(i, p)}
              />
              <Direction
                holder={nameOf(r.target_id)}
                other={nameOf(r.source_id)}
                trust={r.reverse_trust}
                affection={r.reverse_affection}
                labels={r.reverse_labels}
                editing={!!drafts}
                busy={busy}
                onChange={(p) => update(i, {
                  ...(p.trust != null && { reverse_trust: p.trust }),
                  ...(p.affection != null && { reverse_affection: p.affection }),
                  ...(p.labels != null && { reverse_labels: p.labels }),
                })}
              />
            </div>
          </div>
        ))}
      </div>

      {drafts && (
        <>
          <button
            disabled={busy}
            onClick={() => setDrafts((d) => d && [...d, {
              source_id: "", target_id: "", trust: 0.5, affection: 0, labels: "",
              reverse_trust: 0.5, reverse_affection: 0, reverse_labels: "",
            }])}
            className={`${BTN} w-full`}
          >
            ＋ 添加关系
          </button>
          {(problem || error) && <p className="text-[11px] text-rose-400">{problem ?? error}</p>}
          <div className="flex justify-end gap-2">
            <button onClick={() => { setDrafts(null); setError(""); }} disabled={busy} className={BTN}>取消</button>
            <button onClick={() => void save()} disabled={busy || !!problem} className={BTN_PRIMARY}>{busy ? "保存中…" : "保存"}</button>
          </div>
        </>
      )}
    </div>
  );
}
