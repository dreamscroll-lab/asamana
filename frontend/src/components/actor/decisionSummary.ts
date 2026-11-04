/**
 * The public decision summary for confirmation and history, with every id named from the
 * turn's own lists.
 *
 * Interface text, not narrative: nothing here reaches the world. Names come only from the
 * briefing, which holds what the character knows; only self identity comes from the public
 * cast. An unknown person reads as 「某人」.
 */

import { URGENCY_LABEL } from "../../lib/urgency";
import type { ActorBriefing, ActorDecisionSummary, ActorRecordedIntent, ActorTarget, CharacterIdentity } from "../../types";
import { OPERATION_LABEL } from "./shortcuts";

export interface DecisionSummaryText { headline: string; details: string[] }

export function targetName(target: ActorTarget, b: ActorBriefing, self?: Pick<CharacterIdentity, "agent_id" | "name">): string {
  if (target.kind === "agent") return (self?.agent_id === target.id ? self.name || "自己" : b.people.find((p) => p.agent_id === target.id)?.name) ?? "某人";
  if (target.kind === "npc") return b.npcs.find((p) => p.npc_id === target.id)?.name ?? "某人";
  if (target.kind === "location") return (b.location?.location_id === target.id ? b.location.name : b.places.find((p) => p.location_id === target.id)?.name) ?? "某地";
  return b.entities.find((p) => p.entity_id === target.id)?.name ?? "某物";
}

export function describeDecisionSummary(decisionSummary: Pick<ActorDecisionSummary, "operation" | "description" | "target" | "parameters">, b: ActorBriefing, self?: Pick<CharacterIdentity, "agent_id" | "name">): DecisionSummaryText {
  const label = OPERATION_LABEL[decisionSummary.operation] ?? "行动";
  const details: string[] = [];
  if (decisionSummary.target) details.push(`对象：${targetName(decisionSummary.target, b, self)}`);
  const p = decisionSummary.parameters;
  if (typeof p.destination_id === "string" && p.destination_id) details.push(`目的地：${targetName({kind: "location", id: p.destination_id}, b, self)}`);
  if (typeof p.item_id === "string" && p.item_id) details.push(`携带物品：${targetName({kind: "item", id: p.item_id}, b, self)}`);
  if (typeof p.recipient_id === "string" && p.recipient_id) details.push(`接收：${targetName({kind: "agent", id: p.recipient_id}, b, self)}`);
  if (typeof p.anonymous === "boolean") details.push(`签名：${p.anonymous ? "匿名" : "实名"}`);
  if (p.announce) details.push("接收：在场所有人");
  if (typeof p.message === "string" && p.message) details.push(`内容：${p.message}`);
  if (typeof p.talk_topic === "string" && p.talk_topic) details.push(`交谈主题：${p.talk_topic}`);
  if (Array.isArray(p.person_ids) && p.person_ids.length) {
    const label = decisionSummary.operation === "talk" ? "旁听" : "接收";
    const ids = decisionSummary.operation === "talk" ? p.person_ids.filter((id) => id !== decisionSummary.target?.id) : p.person_ids;
    if (ids.length) details.push(`${label}：${ids.map((id) => targetName({ kind: "agent", id }, b, self)).join("、")}`);
  }
  if (Array.isArray(p.move_carry_ids) && p.move_carry_ids.length) details.push(`强行带走：${p.move_carry_ids.map((id) => targetName({ kind: "agent", id }, b, self)).join("、")}`);
  if (typeof p.item_name === "string" && p.item_name) details.push(`物品：${p.item_name}`);
  if (typeof p.urgency === "string" && p.urgency in URGENCY_LABEL) details.push(`紧急程度：${URGENCY_LABEL[p.urgency]}`);
  if (typeof p.expected_outcome === "string" && p.expected_outcome) details.push(`预期结果：${p.expected_outcome}`);
  return { headline: decisionSummary.description ? `${label} · ${decisionSummary.description}` : label, details };
}

export function describeIntent(intent: ActorRecordedIntent, briefing: ActorBriefing, self?: Pick<CharacterIdentity, "agent_id" | "name">): DecisionSummaryText | null {
  if (!intent.operation) return null;
  const parameters = Object.fromEntries(Object.entries(intent.parameters).filter((entry): entry is [string, string | boolean] => entry[1] !== null));
  const description = intent.description || "";
  const result = describeDecisionSummary({ operation: intent.operation, target: intent.target, parameters, description }, briefing, self);
  return { ...result, headline: description ? result.headline : result.headline.replace(/ · $/, "") };
}

// Why an answer or an amendment can't be taken, by the backend's reason slug (ActorAnswerRejected).
const REASON_TEXT: Record<string, string> = {
  already_here: "你已经在这里了。",
  broadcast_requires_current_location: "只能向你当前所在地的人广播。",
  npc_busy: "对方正在执行任务，暂时无法接受新任务。",
  item_not_takeable: "该物品无法拿起。",
  already_held: "你已持有该物品。",
  decision_summary_not_pending: "该行动方案已失效，请重新提交。",
  stale_intent: "回合或指令已经变化，请重新提交。",
  operation_constraint_conflict: "补充要求与选定操作或对象冲突，请修改后再试。",
  operation_not_applicable: "这个对象不支持所选操作。",
  message_requires_recipient_or_announce: "发送消息需要指定接收角色，或选择发送给在场所有角色。",
  errand_address_conflict: "请在指定接收角色与发送给在场所有角色之间选择一种。",
  message_required: "请填写消息内容。",
  description_required: "请先描述具体要做什么。",
  turn_not_pending: "当前回合已结束。",
  unreadable: "无法理解该指令，请调整描述后重试。",
  unbindable: "当前无法执行该指令，请调整描述后重试。",
  missing_action_description: "请填写具体的行动描述。",
  talk_without_target: "请选择交谈对象。",
  talk_target_not_present: "交谈对象不在当前地点。",
  send_without_channel: "请选择消息接收角色。",
  send_recipients_unresolvable: "无法确定消息接收角色。",
  move_without_destination: "请选择目的地。",
  physical_without_target: "请选择动手的目标角色或物件。",
  physical_recipient_without_entity: "请选择要转交的物品。",
  physical_recipient_unresolvable: "物品接收角色不在当前地点。",
  physical_recipient_not_held: "你未持有要转交的物品。",
  errand_without_bearer: "请选择执行任务的人。",
  errand_without_destination: "请选择任务目的地。",
  errand_item_not_in_hand: "你未持有要运送的物品。",
  relation_not_listed: "当前关系列表中没有该角色。",
  goal_not_listed: "这个目标已结束或不在当前列表中，请刷新后再修改。",
  goal_duplicate: "同一个目标不能重复保存，请刷新后再修改。",
  nothing_to_break_off: "当前没有可中断的行动。",
  turn_too_long: "本轮内容过长，请缩短后重试。",
  unknown_emotion: "请选择有效的情绪类型。",
};

export function reasonText(reason: string | null): string {
  if (!reason) return "";
  if (REASON_TEXT[reason]) return REASON_TEXT[reason];
  // An id off the turn's lists (`<slot>_not_perceived`): someone or something out of reach.
  if (reason.endsWith("_not_perceived")) return "目标角色、地点或物件不在当前可感知范围内。";
  return "当前无法执行该操作，请调整后重试。";
}
