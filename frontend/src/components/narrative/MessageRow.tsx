import type { MessageSummary } from "../../types";
import type { StepCast } from "./cast";
import Who from "./Who";

/**
 * A letter LANDING: step.messages is delivery, not sending (the "传信" action card already
 * told the sending), so don't lead with the sender.
 *
 * WHO the subject is depends on how it was addressed (MessageSummary.scope):
 *   direct → the RECIPIENTS. A named letter's news is that it reached them.
 *   place/world → the SENDER. Listing every hearer would shred one public act into private
 *     letters.
 */
export default function MessageRow({ m, cast }: { m: MessageSummary; cast: StepCast }) {
  const to = m.receiver_ids.map(cast.nameOf).filter(Boolean);
  const sender = <Who cast={cast} id={m.sender_id} name={m.sender_name || "不知来源"} />;
  const heard = m.receiver_ids.length;
  return (
    <div className="flex items-start gap-1.5 text-[13px] text-slate-400">
      <span className="text-emerald-400 shrink-0">✉</span>
      <span>
        {m.scope === "world" || m.scope === "place" ? (
          <>
            {sender}
            <span className="text-slate-600">
              {m.scope === "world" ? " 世界广播" : ` 在${m.place}广播`}
              {heard > 0 ? `（${heard} 人听见）` : "（无人在侧）"} ·{" "}
            </span>
            {m.perceived_summary}
          </>
        ) : to.length > 0 ? (
          <>
            {/* List every named recipient: one letter, several names, never many. */}
            {to.map((name, k) => (
              <span key={m.receiver_ids[k]}>
                {k > 0 && <span className="text-slate-600"> · </span>}
                <Who cast={cast} id={m.receiver_ids[k]} name={name} />
              </span>
            ))}
            <span className="text-slate-600"> 收到 </span>
            {sender}
            <span className="text-slate-600"> 的传讯 · </span>
            {m.perceived_summary}
          </>
        ) : (
          // Addressed to someone, delivered to nobody (the recipient was gone); don't dress
          // it up as read.
          <span className="text-slate-500">
            {sender}
            <span className="text-slate-600"> 的传讯无人收到 · </span>
            {m.perceived_summary}
          </span>
        )}
      </span>
    </div>
  );
}
