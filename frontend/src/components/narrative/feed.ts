/**
 * The narrative feed's DATA MODEL: how a step's scopes become one ordered stream, and what a
 * focus is matched against. No JSX: this file is about the wire, not pixels.
 */

import { actedOnAgents } from "../../lib/contract";
import { type Placed, type Scoped, indexStep } from "../../lib/step";
import { type Viewer, actionDescriptionOf, partsOf, sharedOf } from "../../lib/viewer";
import type {
  ActionPrivateView,
  ActionSharedView,
  ActionSummary,
  BroadcastSummary,
  MessageSummary,
  ObservationSummary,
  StepView,
  InjectionSummary,
} from "../../types";

// Broadcasts (a death, an injected change) belong in the feed too, not only on the map.
export type FeedItem =
  | { kind: "injection"; seq: number; data: InjectionSummary }
  // What the viewer knows of an act beyond what it saw: its shared side (null from outside), the
  // doer's `action_description` ("" unless the viewer is the doer or the director) and the
  // participants' private parts it may read.
  | {
      kind: "action"; seq: number; data: Placed<ActionSummary>;
      shared: ActionSharedView | null; action_description: string; private_parts: ActionPrivateView[];
    }
  | { kind: "observation"; seq: number; data: Placed<ObservationSummary> }
  | { kind: "message"; seq: number; data: Scoped<MessageSummary> }
  | { kind: "broadcast"; seq: number; data: Scoped<BroadcastSummary> };

/**
 * One seq-ordered stream of what happened, as the viewer knows it.
 *
 * Each event once: an action is told where it is recorded, so a place that only saw it (an
 * observation under the same seq) adds nothing while the action is in the step. It is not in a
 * viewer's step when it happened out of their sight; then what they saw of it is the observation.
 * A letter to several recipients arrives as one delivery per receiver under one seq; it is one
 * letter, so they merge back into one row (a viewer's step holds only its own delivery).
 *
 * seq is used ONLY for sorting and is never rendered — it is an engine-layer ordinal, not
 * something the story knows about itself.
 */
export function buildFeed(step: StepView, viewer: Viewer = null): FeedItem[] {
  const index = indexStep(step);
  const acted = new Set(index.actions.map((a) => a.seq));
  const letters = new Map<number, Scoped<MessageSummary>>();
  for (const m of index.messages) {
    const seen = letters.get(m.seq);
    letters.set(m.seq, seen ? { ...seen, receiver_ids: [...seen.receiver_ids, ...m.receiver_ids] } : m);
  }
  return [
    ...step.observer.injections.map((inj) => ({ kind: "injection" as const, seq: inj.seq, data: inj })),
    ...index.actions.map((act) => ({
      kind: "action" as const, seq: act.seq, data: act,
      shared: sharedOf(act, viewer), action_description: actionDescriptionOf(act, viewer),
      private_parts: partsOf(act, viewer),
    })),
    ...index.observations
      .filter((o) => !acted.has(o.seq))
      .map((o) => ({ kind: "observation" as const, seq: o.seq, data: o })),
    ...[...letters.values()].map((m) => ({ kind: "message" as const, seq: m.seq, data: m })),
    ...index.broadcasts.map((b) => ({ kind: "broadcast" as const, seq: b.seq, data: b })),
  ].sort((a, b) => a.seq - b.seq);
}

/** A stable React key for a feed row. */
export function itemKey(item: FeedItem): string {
  if (item.kind === "injection") return `injection-${item.data.id}`;
  // One seq can be seen at several places; each is its own row.
  if (item.kind === "observation") return `observation-${item.seq}-${item.data.location_id}`;
  return `${item.kind}-${item.seq}`;
}

// Every agent a focus can match this beat against (ids, a join key, never rendered). A deed
// counts its whole cast plus who it was AIMED at and who CUT it.
function itemCast(item: FeedItem): string[] {
  if (item.kind === "action") {
    const act = item.data;
    return [...act.participant_ids, act.initiator_id, ...actedOnAgents(act.target), act.interrupted_by]
      .filter(Boolean);
  }
  if (item.kind === "observation") return [item.data.initiator_id, ...item.data.participant_ids].filter(Boolean);
  if (item.kind === "message") return [item.data.sender_id, ...item.data.receiver_ids];
  // An injection's only id-level cast is a closed director injection's receipt (who it was
  // delivered to); affected_names is narrative-layer, not ids.
  if (item.kind === "injection") return item.data.receipt?.delivered_to ?? [];
  return []; // broadcast — senderless by construction; see `involves`.
}

// Is this beat about anyone in the focus? Two "yes" answers aren't cast matches:
//   - A BROADCAST is the world's own voice with no agent ids; a death announced to the realm
//     is never filtered out by a character focus. They are rare, so this costs no scrolling.
//   - An INJECTION that names nobody lands on everybody.
export function involves(item: FeedItem, focus: Set<string>): boolean {
  if (item.kind === "broadcast") return true;
  const cast = itemCast(item);
  if (!cast.length) return true;
  return cast.some((id) => focus.has(id));
}


/**
 * Which things THIS step brought into the world, read off its own entity table.
 *
 * Don't diff against the previous step instead of reading `created_step`: the feed is a
 * sliding window (FEED_CAP), so its earliest step has no predecessor and every thing in it
 * would be credited to whichever deed touched it.
 *
 * Callers intersect with the action's `affected_entity_ids` to say which deed made it.
 */
export function bornInStep(step: StepView): Set<string> {
  const table = indexStep(step).entities;
  return new Set(Object.keys(table).filter((eid) => table[eid]?.created_step === step.step));
}
