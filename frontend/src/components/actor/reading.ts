/**
 * The engine's reading of the actor's words, as the actor reads it back on a card before
 * confirming, with every id named from the turn's own lists.
 *
 * Interface text, not narrative: nothing here reaches the world. Names come only from the
 * briefing, which holds what the character knows; an id it doesn't list reads as 「某人」. Steps
 * read as world time (`secondsPerStep`, 0 when unknown), never as a count of steps.
 */

import { ACTION_IDENTITY } from "../../lib/actionIdentity";
import { ActionType, type ActionTypeValue } from "../../lib/contract";
import { durationLabel } from "../../lib/duration";
import type { ActorAction, ActorBriefing, ActorDecision } from "../../types";

export interface ReadingText {
  headline: string;
  details: string[];
}

function namer(b: ActorBriefing) {
  const people = new Map(b.people.map((p) => [p.agent_id, p.name]));
  const npcs = new Map(b.npcs.map((n) => [n.npc_id, n.name]));
  const places = new Map(b.places.map((p) => [p.location_id, p.name]));
  const things = new Map(b.entities.map((e) => [e.entity_id, e.name]));
  return {
    person: (id: string | null) => (id ? people.get(id) || "某人" : ""),
    npc: (id: string | null) => (id ? npcs.get(id) || "某人" : ""),
    place: (id: string | null) => (id ? places.get(id) || "某处" : ""),
    thing: (id: string | null) => (id ? things.get(id) || "某物" : ""),
  };
}

function actionDetails(a: ActorAction, b: ActorBriefing, secondsPerStep: number): string[] {
  const n = namer(b);
  const out: string[] = [];
  const add = (label: string, value: string) => value && out.push(`${label}：${value}`);
  switch (a.action_type as ActionTypeValue) {
    case ActionType.talk:
      add("对象", n.person(a.person_ids[0] ?? null));
      add("旁听", a.person_ids.slice(1).map((id) => n.person(id)).join("、"));
      break;
    case ActionType.send_message:
      add("送给", a.message_announce ? "在场所有人" : a.person_ids.map((id) => n.person(id)).join("、"));
      add("内容", a.message_content);
      break;
    case ActionType.move:
      add("去往", n.place(a.destination_id));
      add("带上", a.move_carry_ids.map((id) => n.person(id)).join("、"));
      break;
    case ActionType.physical:
      add(
        "对",
        n.person(a.physical_person_id) || n.npc(a.physical_npc_id) || n.thing(a.physical_entity_id) || a.item_name,
      );
      add("递给", n.person(a.physical_recipient_id));
      break;
    case ActionType.errand:
      add("差遣", n.npc(a.errand_npc_id));
      add("去往", n.place(a.errand_destination_id));
      add("带去", n.thing(a.errand_item_id));
      add("交给", a.errand_announce ? "当地所有人" : n.person(a.errand_recipient_id));
      add("捎话", a.errand_message);
      break;
    default:
      break;
  }
  if (a.estimated_steps > 1 && secondsPerStep) out.push(`预计${durationLabel(a.estimated_steps * secondsPerStep)}`);
  return out;
}

export function describeDecision(d: ActorDecision, b: ActorBriefing, secondsPerStep: number): ReadingText {
  const a = d.action!;
  const identity = ACTION_IDENTITY[a.action_type as ActionTypeValue];
  return {
    headline: `${identity ? `${identity.icon} ${identity.label}` : "行动"} · ${a.action_description}`,
    details: actionDetails(a, b, secondsPerStep),
  };
}

// Why an answer or an amendment can't be taken, by the backend's reason slug (ActorAnswerRejected).
const REASON_TEXT: Record<string, string> = {
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
