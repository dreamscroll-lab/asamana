import { Fragment } from "react";

import { URGENCY_LABEL } from "../../lib/urgency";
import type { InjectionReceipt, InjectionSummary } from "../../types";
import type { StepCast } from "./cast";

// What a director's intervention actually DID. An injection can be delivered to the right
// inbox and still move nobody (the pressure evaluator judges it irrelevant), which from
// outside looks like a bug. So the key line is the negative one: delivered, and nobody
// looked up.
//
// One row per column, not one sentence: the four readings are independent and each can hold
// a whole cast. As a label column and a name column, the interrupt row is found at a glance and a name
// in "送达" but not in "压力" shows as a gap.
function ReceiptLine({ receipt, cast }: { receipt: InjectionReceipt; cast: StepCast }) {
  const { delivered_to, pressure, decided, interrupted } = receipt;
  if (!delivered_to.length && !pressure.length) {
    return (
      <div className="mt-1.5 text-[11px] text-slate-500">
        无角色感知到此事件
      </div>
    );
  }
  const nameOf = (id: string) => cast.nameOf(id) || "某人";
  const names = (ids: string[]) => ids.map(nameOf).join("、");
  const rows: [string, string][] = [];
  if (delivered_to.length) rows.push(["送达", names(delivered_to)]);
  if (pressure.length) {
    rows.push([
      "压力",
      pressure.map((p) => `${nameOf(p.agent_id)}（${URGENCY_LABEL[p.urgency] ?? p.urgency}）`).join("、"),
    ]);
  }
  if (decided.length) rows.push(["触发决策", names(decided)]);
  if (interrupted.length) rows.push(["中断", names(interrupted)]);
  return (
    <div className="mt-1.5 grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 text-[11px] leading-snug">
      {rows.map(([label, value]) => (
        <Fragment key={label}>
          <span className="whitespace-nowrap text-indigo-300/45">{label}</span>
          <span className="text-indigo-300/75">{value}</span>
        </Fragment>
      ))}
      {/* Delivered, but the world shrugged. Its own row, unlabeled — it is the verdict on
          the rows above, not a fifth column of names. */}
      {!pressure.length && (
        <span className="col-span-2 text-indigo-300/50">无角色因此改变行动</span>
      )}
    </div>
  );
}

export default function InjectionRow({ injection, cast }: { injection: InjectionSummary; cast: StepCast }) {
  // The editor's injections and the director's share this card but must never look alike.
  const byDirector = injection.authored_by === "director";
  return (
    <div
      className={
        byDirector
          ? "rounded-lg border border-indigo-800/60 bg-gradient-to-r from-indigo-950/40 to-transparent px-3 py-2 text-sm text-indigo-100/90"
          : "rounded-lg border border-amber-900/50 bg-gradient-to-r from-amber-950/30 to-transparent px-3 py-2 text-sm text-amber-200/90"
      }
    >
      <span className="mr-1.5">{byDirector ? "🎬" : "⚡"}</span>
      {byDirector && (
        <span className="mr-1.5 rounded px-1.5 py-0.5 text-[10px] align-middle bg-indigo-900/60 border border-indigo-700/60 text-indigo-200">
          导演
        </span>
      )}
      {injection.narrative}
      {/* What was actually typed, under what the engine made of it: only the comparison
          tells a badly worded intervention from a world that doesn't care. */}
      {injection.directive_text && (
        <div className="mt-1.5 text-[11px] text-indigo-300/60 italic">
          导演指令：{injection.directive_text}
        </div>
      )}
      {injection.receipt && <ReceiptLine receipt={injection.receipt} cast={cast} />}
    </div>
  );
}
