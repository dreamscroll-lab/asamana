/**
 * The narrative feed's DATA MODEL: how a step's four channels become one ordered stream, and
 * what a focus is matched against. No JSX: this file is about the wire, not pixels.
 */

import { actedOnAgents } from "../../lib/contract";
import type {
  ActionSummary,
  BroadcastSummary,
  MessageSummary,
  StepEvent,
  WorldEventSummary,
} from "../../types";

// An action item carries EVERY record of its execution, not one: see foldExecutions.
// Broadcasts (a death, an injected event) belong in the feed too, not only on the map.
export type FeedItem =
  | { kind: "event"; seq: number; data: WorldEventSummary }
  | { kind: "action"; seq: number; data: ActionSummary; parts: ActionSummary[]; led: boolean }
  | { kind: "message"; seq: number; data: MessageSummary }
  | { kind: "broadcast"; seq: number; data: BroadcastSummary; key: string };

// One folded execution. `parts` is ordered initiator-first when `led`; `led` says whether the
// records actually told us who started it — see foldExecutions.
type Fold = { lead: ActionSummary; parts: ActionSummary[]; led: boolean };

/**
 * Fold an execution's records back into the one deed they came from.
 *
 * A joint action emits ONE RECORD PER PARTICIPANT (each agent learns from its own), but the
 * feed is a stream of events; a conscripted participant's outcome only restates the
 * initiator's intent back at them.
 *
 * Records are joined on execution_id; the initiator's record (agent_id == initiator_id) speaks
 * for the deed and its seq sets the slot. `parts` comes back initiator-first and `led` says the
 * direction is known; when records lack initiator_id, `led` is false and the card stays
 * symmetric rather than inventing a direction from arrival order.
 */
function foldExecutions(actions: ActionSummary[]): Fold[] {
  const groups = new Map<string, Fold>();
  const out: Fold[] = [];
  for (const act of actions) {
    const leads = !!act.initiator_id && act.agent_id === act.initiator_id;
    // No execution_id (step-0 placement records, ticks) → nothing to join on; stands alone.
    if (!act.execution_id) {
      out.push({ lead: act, parts: [act], led: leads });
      continue;
    }
    const g = groups.get(act.execution_id);
    if (!g) {
      const fresh: Fold = { lead: act, parts: [act], led: leads };
      groups.set(act.execution_id, fresh);
      out.push(fresh);
      continue;
    }
    // The initiator's record speaks for the deed — and it may arrive second, so it takes both
    // the lead slot and the head of the cast.
    if (leads) {
      g.lead = act;
      g.led = true;
      g.parts.unshift(act);
    } else {
      g.parts.push(act);
    }
  }
  return out;
}

/**
 * One seq-ordered stream from all four channels.
 *
 * Legacy items without seq sort stably to the end (Number.MAX_SAFE_INTEGER). seq is used
 * ONLY for sorting and is never rendered — it is an engine-layer ordinal (see the emission
 * -order contract), not something the story knows about itself.
 */
export function buildFeed(step: StepEvent): FeedItem[] {
  return [
    ...step.world_events.map((ev) => ({
      kind: "event" as const,
      seq: ev.seq ?? Number.MAX_SAFE_INTEGER,
      data: ev,
    })),
    ...foldExecutions(step.actions).map(({ lead, parts, led }) => ({
      kind: "action" as const,
      seq: lead.seq ?? Number.MAX_SAFE_INTEGER,
      data: lead,
      parts,
      led,
    })),
    ...step.messages.map((m) => ({
      kind: "message" as const,
      seq: m.seq ?? Number.MAX_SAFE_INTEGER,
      data: m,
    })),
    ...(step.broadcasts ?? []).map((b, i) => ({
      kind: "broadcast" as const,
      seq: b.seq ?? Number.MAX_SAFE_INTEGER,
      data: b,
      key: `bc-${step.step}-${i}`,
    })),
  ].sort((a, b) => a.seq - b.seq);
}

/** A stable React key for a feed row. `index` only ever backs an action, which has no id. */
export function itemKey(item: FeedItem, index: number): string {
  if (item.kind === "event") return `ev-${item.data.id}`;
  if (item.kind === "message") return `msg-${item.data.message_id}`;
  if (item.kind === "broadcast") return item.key;
  return `act-${index}`;
}

// Every agent a focus can match this beat against (ids, a join key, never rendered). A deed
// counts its whole cast plus who it was AIMED at and who CUT it.
function itemCast(item: FeedItem): string[] {
  if (item.kind === "action") {
    const ids: string[] = [];
    for (const p of item.parts) {
      ids.push(p.agent_id, ...actedOnAgents(p.target));
      if (p.interrupted_by) ids.push(p.interrupted_by);
    }
    return ids;
  }
  if (item.kind === "message") return [item.data.sender_id, ...item.data.receiver_ids];
  // An event's only id-level cast is a closed director injection's receipt (who it was
  // delivered to); affected_names is narrative-layer, not ids.
  if (item.kind === "event") return item.data.receipt?.delivered_to.map((d) => d.agent_id) ?? [];
  return []; // broadcast — senderless by construction; see `involves`.
}

// Is this beat about anyone in the focus? Two "yes" answers aren't cast matches:
//   - A BROADCAST is the world's own voice with no agent ids; a death announced to the realm
//     is never filtered out by a character focus. They are rare, so this costs no scrolling.
//   - A world EVENT that names nobody lands on everybody.
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
export function bornInStep(step: StepEvent): Set<string> {
  const table = step.entities ?? {};
  return new Set(Object.keys(table).filter((eid) => table[eid]?.created_step === step.step));
}
