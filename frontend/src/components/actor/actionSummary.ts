/**
 * The public action summary for confirmation and history, with every id named from the
 * turn's own lists.
 *
 * Interface text, not narrative: nothing here reaches the world. Names come only from the
 * briefing, which holds what the character knows; only self identity comes from the public
 * cast. An unknown person reads as 「某人」.
 */

import { ACTION_IDENTITY } from "../../lib/actionIdentity";
import type { ActionTypeValue } from "../../lib/contract";
import type { ActorBriefing, ActorActionSummary, ActorRecordedIntent, ActorTarget, CharacterIdentity } from "../../types";
import { OPERATION_LABEL } from "./shortcuts";

export interface ActionSummaryText { headline: string; details: string[] }

export function targetName(target: ActorTarget, b: ActorBriefing, self?: Pick<CharacterIdentity, "agent_id" | "name">): string {
  if (target.kind === "agent") return (self?.agent_id === target.id ? self.name || "自己" : b.people.find((p) => p.agent_id === target.id)?.name) ?? "某人";
  if (target.kind === "npc") return b.npcs.find((p) => p.npc_id === target.id)?.name ?? "某人";
  if (target.kind === "location") return (b.location?.location_id === target.id ? b.location.name : b.places.find((p) => p.location_id === target.id)?.name) ?? "某处";
  return b.entities.find((p) => p.entity_id === target.id)?.name ?? "某物";
}

export function describeActionSummary(actionSummary: Pick<ActorActionSummary, "operation" | "description" | "target" | "parameters">, b: ActorBriefing, self?: Pick<CharacterIdentity, "agent_id" | "name">): ActionSummaryText {
  const identity = ACTION_IDENTITY[actionSummary.operation as ActionTypeValue];
  const label = OPERATION_LABEL[actionSummary.operation] ?? (identity ? `${identity.icon} ${identity.label}` : "行动");
  const details: string[] = [];
  if (actionSummary.target) details.push(`对象：${targetName(actionSummary.target, b, self)}`);
  const p = actionSummary.parameters;
  if (typeof p.destination_id === "string" && p.destination_id) details.push(`去往：${targetName({kind: "location", id: p.destination_id}, b, self)}`);
  if (typeof p.item_id === "string" && p.item_id) details.push(`带去：${targetName({kind: "item", id: p.item_id}, b, self)}`);
  if (typeof p.recipient_id === "string" && p.recipient_id) details.push(`交给：${targetName({kind: "agent", id: p.recipient_id}, b, self)}`);
  if (p.announce) details.push("送给：在场所有人");
  if (typeof p.message === "string" && p.message) details.push(`内容：${p.message}`);
  if (Array.isArray(p.person_ids) && p.person_ids.length) {
    const label = actionSummary.operation === "talk" ? "旁听" : "送给";
    const ids = actionSummary.operation === "talk" ? p.person_ids.filter((id) => id !== actionSummary.target?.id) : p.person_ids;
    if (ids.length) details.push(`${label}：${ids.map((id) => targetName({ kind: "agent", id }, b, self)).join("、")}`);
  }
  if (Array.isArray(p.move_carry_ids) && p.move_carry_ids.length) details.push(`强行带走：${p.move_carry_ids.map((id) => targetName({ kind: "agent", id }, b, self)).join("、")}`);
  if (typeof p.item_name === "string" && p.item_name) details.push(`物品：${p.item_name}`);
  return { headline: `${label} · ${actionSummary.description}`, details };
}

export function describeIntent(intent: ActorRecordedIntent, briefing: ActorBriefing, self?: Pick<CharacterIdentity, "agent_id" | "name">): ActionSummaryText | null {
  if (!intent.operation) return null;
  const parameters = Object.fromEntries(Object.entries(intent.parameters).filter((entry): entry is [string, string | boolean] => entry[1] !== null));
  const description = intent.description || "";
  const result = describeActionSummary({ operation: intent.operation, target: intent.target, parameters, description }, briefing, self);
  return { ...result, headline: description ? result.headline : result.headline.replace(/ · $/, "") };
}

// Why an answer or an amendment can't be taken, by the backend's reason slug (ActorAnswerRejected).
const REASON_TEXT: Record<string, string> = {
  already_here: "你已经在这里了。",
  broadcast_requires_current_location: "只能向你当前所在地的人广播。",
  npc_busy: "这个人正在办事，暂时无法差遣。",
  item_not_takeable: "这件东西无法拿起。",
  already_held: "这件东西已经在你手中了。",
  action_summary_not_pending: "这份行动概要已失效，请重新提交。",
  stale_intent: "回合或指令已经变化，请重新提交。",
  operation_constraint_conflict: "补充要求与选定操作或对象冲突，请修改后再试。",
  operation_not_applicable: "这个对象不支持所选操作。",
  message_requires_recipient_or_announce: "传话需要指定收件人，或选择向在场所有人传话。",
  errand_address_conflict: "请在指定收件人与向在场所有人传话之间选择一种。",
  message_required: "请先填写要说的原话。",
  description_required: "请先描述具体要做什么。",
  turn_not_pending: "当前回合已结束。",
  unreadable: "没读懂这句话，换个说法试试。",
  unbindable: "这件事此刻做不成，换个说法试试。",
  missing_action_description: "没说清要做什么。",
  talk_without_target: "要和谁说话？",
  talk_target_not_present: "你想说话的人不在眼前。",
  send_without_channel: "要把话传给谁？",
  send_recipients_unresolvable: "找不到你要传话的人。",
  move_without_destination: "要去哪里？",
  physical_without_target: "要对谁、或对什么动手？",
  physical_recipient_without_entity: "要递过去的是什么？",
  physical_recipient_unresolvable: "要把东西交给的人不在眼前。",
  errand_without_bearer: "要差遣谁去？",
  errand_without_destination: "要差人去哪里？",
  errand_item_not_in_hand: "要让人带去的东西不在你手上。",
  relation_not_listed: "你和这个人还没有交集。",
  goal_not_listed: "这个目标已结束或不在当前列表中，请刷新后再修改。",
  goal_duplicate: "同一个目标不能重复保存，请刷新后再修改。",
  nothing_to_break_off: "你手上没在做什么，不用停。",
  turn_too_long: "这一轮说得太长了，分几轮说吧。",
  unknown_emotion: "不认识这种情绪。",
};

export function reasonText(reason: string | null): string {
  if (!reason) return "";
  if (REASON_TEXT[reason]) return REASON_TEXT[reason];
  // An id off the turn's lists (`<slot>_not_perceived`): someone or something out of reach.
  if (reason.endsWith("_not_perceived")) return "提到的人、地方或东西，此刻不在你能触及的范围里。";
  return "这样做不成，换个说法试试。";
}
