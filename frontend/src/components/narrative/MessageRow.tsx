import type { Scoped } from "../../lib/step";
import type { MessageSummary } from "../../types";
import type { StepCast } from "./cast";
import Who from "./Who";

/**
 * A message landing: a message event is delivery, not sending (the send-message action card already
 * told the sending), so don't lead with the sender.
 *
 * WHO the subject is depends on how it was addressed (MessageSummary.scope):
 *   direct → the RECIPIENTS. A named letter's news is that it reached them.
 *   place/world → the SENDER. Listing every hearer would shred one public act into private
 *     letters.
 */
export default function MessageRow({ m, cast }: { m: Scoped<MessageSummary>; cast: StepCast }) {
  const to = m.receiver_ids.map(cast.nameOf).filter(Boolean);
  const sender = <Who cast={cast} id={m.anonymous ? undefined : m.sender_id ?? undefined} name={m.sender_name} />;
  const body = <span className="whitespace-pre-wrap break-words">{m.content}</span>;
  const heard = m.receiver_ids.length;
  return (
    <div className="flex items-start gap-1.5 text-[13px] text-slate-400">
      <span className="text-emerald-400 shrink-0">✉</span>
      <span className="min-w-0 flex-1">
        {m.scope === "world" || m.scope === "place" ? (
          <>
            {sender}
            <span className="text-slate-600">
              {m.scope === "world" ? " 世界广播" : ` 在${cast.placeOf(m.location_id)}广播`}
              {heard > 0 ? `（${heard} 人接收）` : "（无人接收）"} ·{" "}
            </span>
            {body}
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
            <span className="text-slate-600"> 的消息 · </span>
            {body}
          </>
        ) : (
          // Addressed to someone, delivered to nobody (the recipient was gone); don't dress
          // it up as read.
          <span className="text-slate-500">
            {sender}
            <span className="text-slate-600"> 的消息无人收到 · </span>
            {body}
          </span>
        )}
      </span>
    </div>
  );
}
