// The world's historical events: what happened before the story starts, each remembered by the
// characters who took part. Part of the world's setup: revisable until the world is confirmed,
// read-only after; a save replaces the whole list.
import { useState } from "react";

import { identityOf } from "../../lib/worldTable";
import { BTN, BTN_PRIMARY } from "../buttonStyles";
import { durationLabel } from "../../lib/duration";
import type { Cast } from "../../lib/cast";
import type { SetupHistoricalEventView } from "../../types";

const INPUT =
  "bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500/70 disabled:opacity-50";

const pct = (x: number) => Math.round(x * 100);

/** Earliest first: historical events read in the order they happened. */
export function chronological(events: readonly SetupHistoricalEventView[]): SetupHistoricalEventView[] {
  return [...events].sort((a, b) => b.hours_before_start - a.hours_before_start);
}

export default function HistoryPanel({
  events,
  cast,
  editable,
  onSave,
}: {
  events: SetupHistoricalEventView[];
  cast: Cast;
  editable: boolean;
  // Rejects with the backend's reason; the drafts stay so they can be fixed.
  onSave: (events: SetupHistoricalEventView[]) => Promise<void>;
}) {
  const [drafts, setDrafts] = useState<SetupHistoricalEventView[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const editing = drafts !== null;
  const members = Object.values(cast);
  const nameOf = (id: string) => identityOf(id, cast)?.name ?? "某人";
  const when = (hours: number) => `故事开始${durationLabel(hours * 3600)}前`;

  const update = (i: number, patch: Partial<SetupHistoricalEventView>) =>
    setDrafts((d) => d && d.map((e, j) => (j === i ? { ...e, ...patch } : e)));

  async function save() {
    if (!drafts) return;
    setBusy(true);
    setError("");
    try {
      await onSave(drafts.map((e) => ({ ...e, event: e.event.trim() })));
      setDrafts(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="bg-slate-900 border border-slate-800/80 rounded-2xl p-4 shadow-lg space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold tracking-wider text-slate-400 uppercase">
          <span className="inline-flex items-center rounded-full bg-slate-800/60 border border-slate-700/60 px-4 py-1.5">📜 过往事件</span>
        </h4>
        {editable && !editing && (
          <button onClick={() => setDrafts(chronological(events))} className={BTN}>✎ 编辑</button>
        )}
      </div>

      {!editing && (
        events.length ? (
          <ol className="space-y-2">
            {chronological(events).map((e, i) => (
              <li key={i} className="bg-slate-950/50 border border-slate-800/70 rounded-lg px-3 py-2">
                <p className="text-xs text-slate-200 leading-relaxed">{e.event}</p>
                <p className="mt-1 text-[10px] text-slate-500">
                  {when(e.hours_before_start)} · {e.figure_ids.map(nameOf).join("、") || "无相关角色"} · 重要度 {pct(e.importance)}
                </p>
              </li>
            ))}
          </ol>
        ) : (
          <p className="text-xs text-slate-500">还没有过往事件。</p>
        )
      )}

      {drafts && (
        <div className="space-y-2">
          {drafts.map((e, i) => (
            <div key={i} className="bg-slate-950/50 border border-slate-800/70 rounded-lg p-3 space-y-2">
              <textarea
                rows={2}
                value={e.event}
                disabled={busy}
                placeholder="发生了什么"
                onChange={(ev) => update(i, { event: ev.target.value })}
                className={`${INPUT} w-full resize-y`}
              />
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[11px] text-slate-400 mr-1">相关角色</span>
                {members.map((m) => {
                  const on = e.figure_ids.includes(m.agent_id);
                  return (
                    <button
                      key={m.agent_id}
                      disabled={busy}
                      aria-pressed={on}
                      onClick={() => update(i, {
                        figure_ids: on ? e.figure_ids.filter((id) => id !== m.agent_id) : [...e.figure_ids, m.agent_id],
                      })}
                      className={`text-[11px] px-2.5 py-0.5 rounded-full border transition cursor-pointer ${
                        on
                          ? "bg-indigo-600 border-indigo-400 text-white shadow-sm shadow-indigo-900/50"
                          : "border-dashed border-slate-500 text-slate-300 hover:border-indigo-400 hover:text-white"
                      }`}
                    >
                      {on ? "✓ " : "＋ "}{m.name}
                    </button>
                  );
                })}
              </div>
              {!e.figure_ids.length && <p className="text-[10px] text-amber-400/80">至少选一位相关角色：没有人经历过的事，没有人记得。</p>}
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px] text-slate-400">
                <label className="flex items-center gap-1.5">
                  故事开始前
                  <input
                    type="number"
                    min={1}
                    value={e.hours_before_start}
                    disabled={busy}
                    onChange={(ev) => update(i, { hours_before_start: Math.max(1, Math.round(Number(ev.target.value) || 1)) })}
                    className={`${INPUT} w-20`}
                  />
                  小时（{durationLabel(e.hours_before_start * 3600)}）
                </label>
                <label className="flex items-center gap-1.5">
                  <span className="shrink-0 whitespace-nowrap">重要度</span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={e.importance}
                    disabled={busy}
                    onChange={(ev) => update(i, { importance: Number(ev.target.value) })}
                  />
                  <span className="tabular-nums w-7">{pct(e.importance)}</span>
                </label>
                <button
                  disabled={busy}
                  onClick={() => setDrafts((d) => d && d.filter((_, j) => j !== i))}
                  className="ml-auto text-rose-400 hover:text-rose-300"
                >
                  删除
                </button>
              </div>
            </div>
          ))}
          <button
            disabled={busy}
            onClick={() => setDrafts((d) => d && [...d, { event: "", hours_before_start: 1, figure_ids: [], importance: 0.5 }])}
            className={`${BTN} w-full`}
          >
            ＋ 添加事件
          </button>
          {error && <p className="text-[11px] text-rose-400">{error}</p>}
          <div className="flex justify-end gap-2">
            <button onClick={() => { setDrafts(null); setError(""); }} disabled={busy} className={BTN}>取消</button>
            <button onClick={() => void save()} disabled={busy} className={BTN_PRIMARY}>{busy ? "保存中…" : "保存"}</button>
          </div>
        </div>
      )}
    </div>
  );
}
