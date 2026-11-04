import { useState, type FormEvent, type ReactNode } from "react";

import type { SetupView } from "../../types";
import { BTN, BTN_PRIMARY } from "../buttonStyles";
import { SETUP_INPUT } from "./styles";
import { MIN_HOURS_PER_STEP, MAX_HOURS_PER_STEP } from "../../lib/contract";
import { durationLabel } from "../../lib/duration";

const HOURS = Array.from(
  { length: MAX_HOURS_PER_STEP - MIN_HOURS_PER_STEP + 1 },
  (_, i) => MIN_HOURS_PER_STEP + i,
);

type Lore = Pick<SetupView, "narrative_theme" | "opening_time" | "seconds_per_step">;

export default function LorePanel({ setup, editable, onSave, onEditingChange, renderHeader }: {
  renderHeader?: (pace: ReactNode) => ReactNode;
  setup: SetupView;
  editable: boolean;
  onSave: (lore: Lore) => Promise<void>;
  onEditingChange: (editing: boolean) => void;
}) {
  const [draft, setDraft] = useState<Lore | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  function close() {
    setDraft(null);
    setError("");
    onEditingChange(false);
  }
  async function save(event: FormEvent) {
    event.preventDefault();
    if (!draft || busy) return;
    setBusy(true);
    setError("");
    try {
      await onSave({
        ...draft, narrative_theme: draft.narrative_theme.trim(),
        opening_time: { ...draft.opening_time, era_name: draft.opening_time.era_name.trim() },
      });
      close();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }
  const time = setup.opening_time;
  const pace = <div className="flex items-center gap-2 text-xs bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 text-slate-400">
    <span>⏳</span>
    {draft ? <label className="flex items-center gap-2">
      <span>每步时长 ·</span>
      <select aria-label="每步时长" className={SETUP_INPUT} disabled={busy}
        value={draft.seconds_per_step / 3600}
        onChange={(e) => setDraft({ ...draft, seconds_per_step: Number(e.target.value) * 3600 })}>
        {HOURS.map((hours) => <option key={hours} value={hours}>{durationLabel(hours * 3600)}</option>)}
      </select>
    </label> : <span>每步时长 · {durationLabel(setup.seconds_per_step)}</span>}
  </div>;
  return <>
    {renderHeader ? renderHeader(pace) : pace}
    <div className="space-y-3">
      {editable && !draft && <div className="flex items-center justify-end">
        <button className={BTN} onClick={() => {
          setDraft({ narrative_theme: setup.narrative_theme,
            opening_time: { ...time }, seconds_per_step: setup.seconds_per_step });
          onEditingChange(true);
        }}>✎ 编辑</button>
      </div>}
      {draft ? <form onSubmit={(event) => void save(event)} className="space-y-3">
        <label className="block text-xs text-slate-400 space-y-1">
          <span>叙事主题</span>
          <textarea className={SETUP_INPUT + " block w-full mt-1"} rows={3} required value={draft.narrative_theme} disabled={busy}
            onChange={(e) => setDraft({ ...draft, narrative_theme: e.target.value })} />
        </label>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
          <label className="text-xs text-slate-400 space-y-1"><span>纪年名</span>
            <input placeholder="留空使用地图纪年" className={SETUP_INPUT + " block w-full mt-1"} value={draft.opening_time.era_name} disabled={busy}
              onChange={(e) => setDraft({ ...draft, opening_time: { ...draft.opening_time, era_name: e.target.value } })} />
          </label>
          {([
            ["start_year", "年", 1, undefined], ["start_month", "月", 1, 12],
            ["start_day", "日", 1, 30], ["start_hour", "时", 0, 23],
          ] as const).map(([key, label, min, max]) => <label key={key} className="text-xs text-slate-400 space-y-1">
            <span>{label}</span><input type="number" required min={min} max={max} step={1} disabled={busy}
              className={SETUP_INPUT + " w-full"} value={draft.opening_time[key]}
              onChange={(e) => setDraft({ ...draft, opening_time: { ...draft.opening_time, [key]: Number(e.target.value) } })} />
          </label>)}
        </div>
        <p className="text-xs text-slate-500">叙事主题供后续叙事参考。</p>
        {error && <p className="text-xs text-rose-400">{error}</p>}
        <div className="flex justify-end gap-2">
          <button type="button" className={BTN} disabled={busy} onClick={close}>取消</button>
          <button type="submit" className={BTN_PRIMARY} disabled={busy || !draft.narrative_theme.trim()}>{busy ? "保存中…" : "保存"}</button>
        </div>
      </form> : <dl className="text-sm space-y-3">
        <div><dt className="text-xs text-slate-500">叙事主题</dt><dd className="text-slate-300 whitespace-pre-wrap">{setup.narrative_theme || "未设置"}</dd></div>
        <div>
          <dt className="text-xs text-slate-500">开局时间</dt>
          <dd className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-slate-200">
            <span>{`${time.era_name ? time.era_name + " " : ""}${time.start_year}年${time.start_month}月${time.start_day}日`}</span>
            <span className="rounded-md border border-slate-800 bg-slate-950 px-2 py-0.5 font-mono text-xs tabular-nums text-indigo-300">
              {`${String(time.start_hour).padStart(2, "0")}:00`}
            </span>
          </dd>
        </div>
      </dl>}
    </div>
  </>;
}
