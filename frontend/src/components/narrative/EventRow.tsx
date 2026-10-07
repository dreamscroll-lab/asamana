import { Fragment } from "react";

import type { EventReceipt, WorldEventSummary } from "../../types";

// What a director's intervention actually DID. An injection can be delivered to the right
// inbox and still move nobody (the pressure evaluator judges it irrelevant), which from
// outside looks like a bug. So the key line is the negative one: delivered, and nobody
// looked up.
//
// One row per column, not one sentence: the four readings are independent and each can hold
// a whole cast. As a label column and a name column, 「打断」 is found at a glance and a name
// in "送达" but not in "压力" shows as a gap.
function EventReceiptLine({ receipt }: { receipt: EventReceipt }) {
  const { delivered_to, pressure, decided, interrupted } = receipt;
  if (!delivered_to.length && !pressure.length) {
    return (
      <div className="mt-1.5 text-[11px] text-slate-500">
        未落到任何人身上（无人可感知）
      </div>
    );
  }
  const names = (xs: { name: string }[]) => xs.map((x) => x.name).join("、");
  const rows: [string, string][] = [];
  if (delivered_to.length) rows.push(["送达", names(delivered_to)]);
  if (pressure.length) {
    rows.push([
      "压力",
      pressure.map((p: { name: string; urgency: string }) => `${p.name}（${p.urgency}）`).join("、"),
    ]);
  }
  if (decided.length) rows.push(["因此决策", names(decided)]);
  if (interrupted.length) rows.push(["打断", names(interrupted)]);
  return (
    <div className="mt-1.5 grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 text-[11px] leading-snug">
      {rows.map(([label, value]) => (
        <Fragment key={label}>
          <span className="whitespace-nowrap text-indigo-300/45">{label}</span>
          <span className="text-indigo-300/75">{value}</span>
        </Fragment>
      ))}
      {/* Delivered, but the world shrugged. Its own row, unlabelled — it is the verdict on
          the rows above, not a fifth column of names. */}
      {!pressure.length && (
        <span className="col-span-2 text-indigo-300/50">无人为此改变行止</span>
      )}
    </div>
  );
}

export default function EventRow({ ev }: { ev: WorldEventSummary }) {
  // World events and director injections share this card but must never look alike.
  const byDirector = ev.authored_by === "director";
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
      {ev.narrative}
      {/* What was actually typed, under what the engine made of it: only the comparison
          tells a badly worded intervention from a world that doesn't care. */}
      {ev.directive_text && (
        <div className="mt-1.5 text-[11px] text-indigo-300/60 italic">
          导演说：{ev.directive_text}
        </div>
      )}
      {ev.receipt && <EventReceiptLine receipt={ev.receipt} />}
    </div>
  );
}
