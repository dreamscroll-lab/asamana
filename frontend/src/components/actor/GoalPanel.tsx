import { useState } from "react";

import type { ActorMind, BriefGoal, GoalAmendment } from "../../types";
import { GOAL_STATUS, goalRows, liveGoal } from "./goals";

const EDIT_BOX = "w-full resize-y bg-slate-950/80 border border-slate-800 rounded-lg px-2 py-1 text-[12px] text-slate-200 focus:outline-none focus:border-indigo-500/70";

function GoalDetail({ goal: g }: { goal: BriefGoal | null }) {
  if (!g) return null;
  return (
    <span className="text-[11px] text-slate-500">
      {" · "}{GOAL_STATUS[g.status]}
      {g.time_hint && <span className={g.attention ? "text-amber-300" : ""}>{" "}{g.time_hint}</span>}
      {g.attention && g.due_step === null && <span className="text-amber-300"> · 这件事已经很久未完成，是否还要继续？</span>}
      {g.progress_summary && <span className="text-slate-400"> · {g.progress_summary}</span>}
    </span>
  );
}

export default function GoalPanel({ mind, draft, removedIds, onChange }: {
  mind: ActorMind;
  draft: GoalAmendment[] | null;
  removedIds: string[];
  onChange: (goals: GoalAmendment[] | null, removedId?: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const rows = goalRows(mind, draft, removedIds);
  const values = rows.map(({ id, text }) => ({ id, text }));
  const history = mind.short_term_goals.filter((g) => !liveGoal(g));
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-[11px] text-slate-400">
        <span>短期目标</span>
        {draft !== null ? <span className="text-indigo-300">未保存</span>
          : mind.pending?.short_term_goals && <span className="text-amber-300">待生效</span>}
        {rows.length > 0 && <button className="ml-auto hover:text-white" onClick={() => setEditing(!editing)}>
          {editing ? "收起编辑" : "编辑"}
        </button>}
        {draft !== null && <button onClick={() => onChange(null)}>撤销</button>}
      </div>
      {rows.map((row, index) => (
        <div key={row.id ?? `new-${index}`} className="rounded-lg border border-slate-800/70 p-2 space-y-1">
          {editing ? (
            <div className="flex items-start gap-1.5">
              <span className="shrink-0 text-[12px] text-slate-500 tabular-nums">{index + 1}.</span>
              <textarea aria-label="目标文本" value={row.text} rows={2} className={EDIT_BOX}
                onChange={(e) => onChange(values.map((g, i) => i === index ? { ...g, text: e.target.value } : g))} />
              <button title="删除目标（放下）" className="text-slate-500 hover:text-rose-300"
                onClick={() => onChange(values.filter((_, i) => i !== index), row.id ?? undefined)}>×</button>
            </div>
          ) : (
            <p className="m-0 text-[12px] whitespace-pre-wrap break-words text-slate-200">
              <span className="text-slate-500 tabular-nums">{index + 1}. </span>
              {row.text}<GoalDetail goal={row.detail} />
            </p>
          )}
          {editing && <GoalDetail goal={row.detail} />}
        </div>
      ))}
      <button className="text-[11px] text-slate-400 hover:text-white" onClick={() => {
        setEditing(true);
        onChange([...values, { id: null, text: "" }]);
      }}>＋ 添加</button>
      {history.length > 0 && (
        <details className="text-[11px] text-slate-500">
          <summary className="cursor-pointer">近期记录</summary>
          {history.map((g, index) => (
            <div key={g.id} className="mt-1 rounded-lg border border-slate-800/50 p-2">
              <p className="m-0 whitespace-pre-wrap break-words">
                <span className="text-slate-500 tabular-nums">{index + 1}. </span>
                {g.text}<GoalDetail goal={g} />
              </p>
            </div>
          ))}
        </details>
      )}
    </div>
  );
}
