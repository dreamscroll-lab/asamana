import { describe, expect, it } from "vitest";

import { breakableAct, buildThread, conversationsOf, liveConversation, mergeEntries } from "./thread";
import type {
  ActorPublicAnswer, ActorBriefing, ActorActionSummary, ActorEntry, ActorTurn,
} from "../../types";

const briefing: ActorBriefing = { location: null, people: [], npcs: [], places: [], entities: [] };

const act = (description: string): ActorActionSummary => ({
  action_summary_id: `action_summary-${description}`, description, operation: "work", target: null, parameters: {}, estimated_duration_seconds: 7200,
});
const IDLE: ActorPublicAnswer = { idle: true };

const turn = (turn_id: string, kind: ActorTurn["kind"] = "decide"): ActorTurn => ({
  agent_id: "me", step: 1, kind, briefing: kind === "decide" ? briefing : null, doing: kind === "interrupt" ? "抄奏章" : "",
  invitation: kind === "invite"
    ? { inviter: { agent_id: "li", name: "李世民", gender: "男", present: true }, action_type: "talk" }
    : null,
  decided: "", conversation: null, reason: "", turn_id,
});

let n = 0;
const id = () => `e${++n}`;
const asked = (t: ActorTurn): ActorEntry => ({ kind: "asked", entry_id: id(), time_label: "", step: t.step, turn: t });
const said = (turn_id: string, words: string, action_summary: ActorActionSummary | null, reason: string | null = null): ActorEntry => ({
  kind: "said", entry_id: id(), time_label: "", step: 1, turn_id, words, action_summary, reason, intent: null,
});
const answered = (turn_id: string, answer: ActorPublicAnswer): ActorEntry => ({
  kind: "answered", entry_id: id(), time_label: "", step: 1, turn_id, answer,
});
const began = (step: number, total_steps: number): ActorEntry => ({
  kind: "began", entry_id: id(), time_label: "", step, description: "抄奏章", total_steps,
});
const ended = (step: number): ActorEntry => ({
  kind: "ended", entry_id: id(), time_label: "", step, fact: "我抄完了奏章", succeeded: true, happened: true,
});
const perceived = (): ActorEntry => ({
  kind: "perceived", entry_id: id(), time_label: "", step: 2, messages: [], broadcasts: ["宫中走水"], acted_upon: [],
});

describe("buildThread · 卡片", () => {
  it("回合开着时只有最后一次理解等确认,更早的未采用", () => {
    const t = turn("t1");
    const items = buildThread([asked(t), said("t1", "抄", act("抄奏章")), said("t1", "抄奏章", act("亲自抄"))], t);

    expect(items.filter((i) => i.kind === "said").map((i) => i.kind === "said" && i.state)).toEqual(["declined", "pending"]);
    expect(items[0]).toMatchObject({ kind: "asked", open: true });
  });

  it("用理解作答的那张打勾,其余与选了不动的都是未采用", () => {
    const confirmed = buildThread([asked(turn("t1")), said("t1", "抄", act("抄")), answered("t1", { action_summary_id: act("抄").action_summary_id })], null);
    expect(confirmed.find((i) => i.kind === "said")).toMatchObject({ state: "confirmed" });
    // Its answer is the tick on the card, not a line of its own.
    expect(confirmed.map((i) => i.kind)).toEqual(["asked", "said"]);

    const idle = buildThread([asked(turn("t2")), said("t2", "抄", act("抄")), answered("t2", IDLE)], null);
    expect(idle.find((i) => i.kind === "said")).toMatchObject({ state: "declined" });
    expect(idle.at(-1)).toMatchObject({ kind: "chose", choice: "idle" });
  });

  it("retains a failed structured operation and description in the actor thread", () => {
    const t = turn("t1");
    const entry = said("t1", "避开要害", null, "operation_constraint_conflict");
    if (entry.kind !== "said") throw new Error("Unexpected test entry");
    entry.intent = {
      turn_id: t.turn_id, text: null, operation: "physical", target: { kind: "agent", id: "li" },
      parameters: { destination_id: null, item_id: null, recipient_id: null, message: null, announce: null }, description: "避开要害",
    };
    expect(buildThread([asked(t), entry], t).at(-1)).toMatchObject({ kind: "said", intent: entry.intent });
  });

  it("declines an earlier action summary when the latest intent could not be read", () => {
    const t = turn("t1");
    const items = buildThread([asked(t), said("t1", "抄", act("抄")), said("t1", "唉", null, "unreadable")], t);
    expect(items.filter((item) => item.kind === "said").map((item) => item.kind === "said" && item.state)).toEqual(["declined", "declined"]);
  });

  it("confirms the exact action summary id rather than the last action summary", () => {
    const items = buildThread([asked(turn("t1")), said("t1", "抄", act("甲")), said("t1", "抄", act("乙")), answered("t1", { action_summary_id: act("甲").action_summary_id })], null);
    expect(items.filter((item) => item.kind === "said").map((item) => item.kind === "said" && item.state)).toEqual(["confirmed", "declined"]);
  });

  it("回合没答就关了,理解也算未采用;读不成的话带着原因", () => {
    const items = buildThread([asked(turn("t1")), said("t1", "抄", act("抄")), said("t1", "唉", null, "unreadable")], null);
    expect(items.filter((i) => i.kind === "said")).toMatchObject([{ state: "declined" }, { action_summary: null, reason: "unreadable" }]);
  });

  it("没有理解垫着的行动、打断的回答都各成一行", () => {
    const items = buildThread([
      asked(turn("t1")), answered("t1", { action_summary_id: act("抄奏章").action_summary_id }),
      asked(turn("t2", "interrupt")), answered("t2", { interrupt: false, thought: "先做完" }),
      asked(turn("t3", "interrupt")), answered("t3", { interrupt: true, thought: "我得先处理来信" }),
    ], null);
    expect(items.filter((i) => i.kind === "chose")).toMatchObject([
      { choice: "carry_on", description: "先做完" }, { choice: "break_off", description: "我得先处理来信" },
    ]);
  });

  it("应邀与不理会也各成一行", () => {
    const items = buildThread([
      asked(turn("t1", "invite")), answered("t1", { accept: true }),
      asked(turn("t2", "invite")), answered("t2", { accept: false }),
    ], null);
    expect(items.filter((i) => i.kind === "chose")).toMatchObject([{ choice: "accept" }, { choice: "refuse" }]);
  });
});

describe("buildThread · 行动", () => {
  it("进度只随后端记下的推进前进,开始即第一步,结束时满格", () => {
    const started = buildThread([began(3, 4)], null);
    expect(started[0]).toMatchObject({ kind: "act", elapsed: 1, total: 4, ended: false });
    const going = buildThread([began(3, 4), { kind: "progressed", entry_id: id(), time_label: "", step: 4, elapsed_steps: 2, total_steps: 4 }], null);
    expect(going).toHaveLength(1);
    expect(going[0]).toMatchObject({ kind: "act", elapsed: 2, total: 4, ended: false });

    const done = buildThread([began(3, 4), ended(6)], null);
    expect(done[0]).toMatchObject({ elapsed: 4, ended: true });
    expect(done[1]).toMatchObject({ kind: "ended", fact: "我抄完了奏章" });
  });

  it("只有跨步且没结束的事能中断;要求过停下就显示即将停下", () => {
    expect(breakableAct(buildThread([began(1, 1)], null))).toBeNull();
    expect(breakableAct(buildThread([began(1, 3), ended(2)], null))).toBeNull();
    expect(breakableAct(buildThread([began(1, 3)], null))).not.toBeNull();

    const items = buildThread([began(1, 3), { kind: "break_off", entry_id: id(), time_label: "", step: 1 }], null);
    expect(breakableAct(items)).toMatchObject({ breakingOff: true });
  });
});

describe("buildThread · 感知", () => {
  it("打断问题开着时,上次作答后的感知自动展开", () => {
    const t = turn("t9", "interrupt");
    const entries = [perceived(), asked(turn("t1")), answered("t1", IDLE), perceived(), asked(t)];
    const items = buildThread(entries, t).filter((i) => i.kind === "perceived");
    expect(items.map((i) => i.kind === "perceived" && i.asking)).toEqual([false, true]);

    const quiet = buildThread(entries, null).filter((i) => i.kind === "perceived");
    expect(quiet.map((i) => i.kind === "perceived" && i.asking)).toEqual([false, false]);
  });
});

describe("buildThread · 连续的感知", () => {
  it("中间没有别的条目就合成一张卡片,被别的条目隔开则各自成卡", () => {
    const together = buildThread([perceived(), perceived()], null);
    expect(together).toHaveLength(1);
    expect(together[0]).toMatchObject({ kind: "perceived", broadcasts: ["宫中走水", "宫中走水"] });

    const apart = buildThread([perceived(), began(2, 1), perceived()], null);
    expect(apart.map((i) => i.kind)).toEqual(["perceived", "act", "perceived"]);
  });

  it("合并后只要有一条属于正在问的打断,整张卡片就展开", () => {
    const t = turn("t9", "interrupt");
    const items = buildThread([perceived(), asked(turn("t1")), answered("t1", IDLE), perceived(), perceived(), asked(t)], t);
    expect(items.filter((i) => i.kind === "perceived").map((i) => i.kind === "perceived" && i.asking)).toEqual([false, true]);
  });
});

describe("mergeEntries", () => {
  it("同一条只收一次,顺序不变", () => {
    const a = perceived();
    const b = perceived();
    const merged = mergeEntries([a], [a, b]);
    expect(merged.map((e) => e.entry_id)).toEqual([a.entry_id, b.entry_id]);
    expect(mergeEntries(merged, [b])).toBe(merged);
  });
});

describe("buildThread · 时间", () => {
  it("一步只显示一次时间,在这一步的第一行之上", () => {
    const at = (e: ActorEntry, step: number, time_label: string): ActorEntry => ({ ...e, step, time_label });
    const t = turn("t1");
    const items = buildThread([
      at(perceived(), 1, "辰时"), at(asked(t), 1, "辰时"), at(answered("t1", IDLE), 1, "辰时"),
      at(perceived(), 2, "巳时"),
    ], null);
    expect(items.map((i) => (i.kind === "time" ? i.label : i.kind))).toEqual(
      ["辰时", "perceived", "asked", "chose", "巳时", "perceived"],
    );
  });

  it("并进上一张卡片的感知不另起时间,下一步新起的一行才有", () => {
    const at = (e: ActorEntry, step: number, time_label: string): ActorEntry => ({ ...e, step, time_label });
    const items = buildThread([at(perceived(), 1, "辰时"), at(perceived(), 2, "巳时"), at(began(2, 1), 2, "巳时")], null);
    expect(items.map((i) => (i.kind === "time" ? i.label : i.kind))).toEqual(["辰时", "perceived", "巳时", "act"]);
  });
});

describe("buildThread · 交谈", () => {
  const said = (speaker_id: string, line: string, end = false): ActorEntry => ({
    kind: "dialogue_turn", entry_id: id(), time_label: "", step: 3, end,
    dialogue_turn: { speaker_id, speaker: speaker_id === "me" ? "李元吉" : "李建成", line },
  });
  const ended = (): ActorEntry => ({
    kind: "ended", entry_id: id(), time_label: "", step: 3, fact: "我劝了大哥", succeeded: true, happened: true,
  });

  it("连续的台词并成一场交谈,说了 end 的那句之后它就结束了", () => {
    const items = buildThread([began(3, 1), said("me", "大哥留步"), said("li", "何事?"), said("me", "", true)], null);
    const [conversation, ...rest] = conversationsOf(items);
    expect(rest).toHaveLength(0);
    expect(conversation.dialogue.map((t) => t.line)).toEqual(["大哥留步", "何事?"]);
    expect(conversation.over).toBe(true);
    expect(liveConversation(items)).toBeNull();
  });

  it("还在说的是正在进行的那场;之后来的任何一条都意味着它已结束", () => {
    const live = buildThread([began(3, 1), said("me", "大哥留步")], null);
    expect(liveConversation(live)?.dialogue).toHaveLength(1);
    const closed = buildThread([began(3, 1), said("me", "大哥留步"), ended()], null);
    expect(conversationsOf(closed)[0].over).toBe(true);
    expect(liveConversation(closed)).toBeNull();
  });

  it("前一场结束后的台词另起一场", () => {
    expect(conversationsOf(buildThread([said("me", "甲", true), said("li", "乙")], null))).toHaveLength(2);
  });
});
