// The story's pace: how much world time one step spans. Chosen at build and revisable until
// the world is confirmed.
import { useState } from "react";

import { BTN, BTN_PRIMARY } from "../buttonStyles";
import { SETUP_INPUT } from "./styles";
import { MAX_HOURS_PER_STEP, MIN_HOURS_PER_STEP } from "../../lib/contract";
import { durationLabel } from "../../lib/duration";

const HOURS = Array.from(
  { length: MAX_HOURS_PER_STEP - MIN_HOURS_PER_STEP + 1 },
  (_, i) => MIN_HOURS_PER_STEP + i,
);

export default function StepLengthEditor({
  secondsPerStep,
  editable,
  onSave,
}: {
  secondsPerStep: number;
  editable: boolean;
  // Rejects with the backend's reason; the choice stays so it can be retried.
  onSave: (secondsPerStep: number) => Promise<void>;
}) {
  const [draft, setDraft] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function save() {
    if (draft == null) return;
    if (draft === secondsPerStep) return setDraft(null);
    setBusy(true);
    setError("");
    try {
      await onSave(draft);
      setDraft(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex items-center gap-2 flex-wrap text-xs bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 text-slate-400">
      <span>⏳</span>
      {draft == null ? (
        <>
          <span>叙事节奏 · 每步 {durationLabel(secondsPerStep)}</span>
          {editable && (
            <button onClick={() => setDraft(secondsPerStep)} className="text-slate-500 hover:text-slate-200 transition">✎ 编辑</button>
          )}
        </>
      ) : (
        <>
          <span>每步</span>
          <select
            value={Math.round(draft / 3600)}
            disabled={busy}
            onChange={(e) => setDraft(Number(e.target.value) * 3600)}
            className={SETUP_INPUT}
          >
            {HOURS.map((h) => <option key={h} value={h}>{durationLabel(h * 3600)}</option>)}
          </select>
          <button onClick={() => { setDraft(null); setError(""); }} disabled={busy} className={BTN}>取消</button>
          <button onClick={() => void save()} disabled={busy} className={BTN_PRIMARY}>{busy ? "保存中…" : "保存"}</button>
          {error && <span className="text-rose-400">{error}</span>}
        </>
      )}
    </div>
  );
}
