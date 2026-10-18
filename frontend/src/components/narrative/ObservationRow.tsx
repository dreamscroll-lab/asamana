import type { Placed } from "../../lib/step";
import type { ObservationSummary } from "../../types";
import { type StepCast, initiatorFirst } from "./cast";
import Who from "./Who";

/**
 * Something seen at a place whose deed happened out of the viewer's sight: a man leaving, a
 * change an injection made. Only reaches the feed when the action itself isn't in the step
 * (see buildFeed), so it never retells a deed already on the page.
 */
export default function ObservationRow({ o, cast }: { o: Placed<ObservationSummary>; cast: StepCast }) {
  const who = initiatorFirst(o.initiator_id, o.participant_ids)
    .map((id) => ({ id, name: cast.nameOf(id) }))
    .filter((p) => p.name);
  return (
    <div className="flex items-start gap-1.5 rounded-lg border border-slate-800/50 bg-slate-900/30 px-2.5 py-1.5 text-[13px] text-slate-300">
      <span className="shrink-0 text-slate-500">👁</span>
      <span>
        <span className="text-slate-500">📍 {cast.placeOf(o.location_id)} · </span>
        {who.map((p, i) => (
          <span key={p.id}>
            {i > 0 && " · "}
            <Who cast={cast} id={p.id} name={p.name} />
          </span>
        ))}
        {who.length > 0 && <span className="text-slate-600"> · </span>}
        {o.text}
      </span>
    </div>
  );
}
