import type { BroadcastSummary } from "../../types";

/**
 * The world's own voice: senderless, so no name and no avatar.
 *
 * The row states its reach because scope decides WHO PERCEIVED IT
 * (BroadcastChannel.for_location() reaches only the agents standing there). `severity` is the
 * producer's mark (a death is HIGH), so HIGH burns amber.
 */
export default function BroadcastRow({ b, place }: { b: BroadcastSummary; place: string }) {
  const high = b.severity === "high";
  const reach = place || "世界广播"; // "" = the world scope
  return (
    <div
      className={`flex items-start gap-1.5 rounded-lg border px-2.5 py-1.5 text-[13px] ${
        high
          ? "border-amber-800/60 bg-amber-950/25 text-amber-100/90"
          : "border-teal-900/40 bg-teal-950/20 text-teal-100/80"
      }`}
    >
      <span className={`shrink-0 ${high ? "text-amber-400" : "text-teal-400"}`}>📣</span>
      <span>
        <span className={high ? "text-amber-500/80" : "text-slate-500"}>{reach} · </span>
        {b.content}
      </span>
    </div>
  );
}
