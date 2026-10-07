/**
 * Per-action-type presentation in the feed: identity (icon · label, shared with the map through
 * lib/actionIdentity) plus the feed's own chip colour.
 *
 * Not owned by the action row: the collapsed line draws from the same table so a folded beat
 * stays recognisable.
 */

import { ACTION_IDENTITY } from "../../lib/actionIdentity";
import { ActionType, type ActionTypeValue, Phase } from "../../lib/contract";
import type { ActionSummary } from "../../types";
import type { FeedItem } from "./feed";

interface ActionMeta {
  icon: string;
  label: string;
  chip: string;
}

// Chip colour per action type — TOTAL, like the identity table it pairs with.
const CHIP: Record<ActionTypeValue, string> = {
  [ActionType.talk]:         "text-sky-300 border-sky-900/60 bg-sky-950/40",
  [ActionType.move]:         "text-slate-300 border-slate-700/60 bg-slate-800/40",
  [ActionType.physical]:     "text-rose-300 border-rose-900/60 bg-rose-950/40",
  [ActionType.covert]:       "text-violet-300 border-violet-900/60 bg-violet-950/40",
  [ActionType.work]:         "text-amber-300 border-amber-900/60 bg-amber-950/40",
  [ActionType.rest]:         "text-teal-300 border-teal-900/60 bg-teal-950/40",
  [ActionType.send_message]: "text-emerald-300 border-emerald-900/60 bg-emerald-950/40",
  [ActionType.errand]:       "text-lime-300 border-lime-900/60 bg-lime-950/40",
};
const DEFAULT_META: ActionMeta = { icon: "•", label: "行动", chip: "text-slate-300 border-slate-700/60 bg-slate-800/40" };
const INTERRUPT_META: ActionMeta = { icon: "⛔", label: "中断", chip: "text-slate-300 border-slate-600/60 bg-slate-800/50" };

/** How a deed presents itself. An interrupt overrides whatever kind of act it cut short. */
export function metaOf(act: ActionSummary): ActionMeta {
  if (act.phase === Phase.interrupt) return INTERRUPT_META;
  const type = act.action_type as ActionTypeValue;
  // The fallback is for the wire's non-types (step-0 seeding records carry an empty action_type).
  const identity = ACTION_IDENTITY[type];
  return identity ? { ...identity, chip: CHIP[type] } : DEFAULT_META;
}

// The one line a collapsed beat keeps: enough to know WHAT happened and WHO it was, never
// enough to read it. Reading it is what expanding is for.
export function collapsedOf(item: FeedItem): { icon: string; text: string } {
  if (item.kind === "event")
    return { icon: item.data.authored_by === "director" ? "🎬" : "⚡", text: item.data.narrative };
  if (item.kind === "message")
    return { icon: "✉", text: `${item.data.sender_name || "不知来源"} · ${item.data.perceived_summary}` };
  if (item.kind === "broadcast") return { icon: "📣", text: item.data.content };
  const act = item.data;
  return {
    icon: metaOf(act).icon,
    text: `${item.parts.map((p) => p.agent_name).join(" · ")} · ${act.action_description}`,
  };
}
