import { describe, expect, it } from "vitest";
import { describeDecisionSummary, describeIntent, reasonText } from "./decisionSummary";
import type { ActorBriefing, ActorDecisionSummary } from "../../types";

const briefing: ActorBriefing = {
  location: { location_id: "hall", name: "大堂", travel_seconds: 0 },
  people: [{ agent_id: "b", name: "李建成", gender: "男", present: true }, { agent_id: "c", name: "李元吉", gender: "男", present: true }],
  npcs: [], places: [{ location_id: "gate", name: "玄武门", travel_seconds: 600 }], entities: [],
};
type Ref = { kind: string; id: string };
const target = (acts_on: Ref[] = [], claims: Ref[] = [], reaches: Ref[] = []) => ({ acts_on, claims, reaches });
const NONE = target();
const B = { kind: "agent", id: "b" };
const C = { kind: "agent", id: "c" };
const decisionSummary = (over: Partial<ActorDecisionSummary> = {}): ActorDecisionSummary => ({
  decision_summary_id: "r", description: "劝他别进宫", operation: "talk", target: target([B], [B]), parameters: {}, estimated_duration_seconds: 7200, ...over,
});

describe("describeDecisionSummary", () => {
  it("names the target through the briefing", () => {
    expect(describeDecisionSummary(decisionSummary(), briefing)).toEqual({
      headline: "💬 交谈 · 劝他别进宫", details: ["对象：李建成"],
    });
  });
  it("names the listeners of a talk apart from the one addressed", () => {
    expect(describeDecisionSummary(decisionSummary({ target: target([B], [B], [C]) }), briefing).details).toEqual(["对象：李建成", "旁听：李元吉"]);
  });
  it("never resolves other people from the public cast", () => {
    const self = { agent_id: "self", name: "李世民" };
    const unknown = describeDecisionSummary(decisionSummary({ target: target([{ kind: "agent", id: "hidden" }]) }), briefing, self);
    expect(unknown.details).toEqual(["对象：某人"]);
  });
  it("shows nothing for an act with no target", () => {
    expect(describeDecisionSummary(decisionSummary({ operation: "rest", target: NONE }), briefing).details).toEqual([]);
  });
  it("retains the selected operation and target alongside a supplemental description", () => {
    const text = describeIntent({
      turn_id: "t", text: null, operation: "physical", target: { kind: "agent", id: "b" },
      parameters: { destination_id: null, item_id: null, recipient_id: null, message: null, announce: null, anonymous: null },
      description: "避开要害",
    }, briefing);
    expect(text).toEqual({ headline: "💪 动手 · 避开要害", details: ["对象：李建成"] });
  });
  it("renders direct self operations without requiring supplemental words", () => {
    for (const operation of ["work", "rest", "covert"] as const) {
      const text = describeIntent({
        turn_id: "t", text: null, operation, target: { kind: "agent", id: "self" },
        parameters: { destination_id: null, item_id: null, recipient_id: null, message: null, announce: null, anonymous: null }, description: "",
      }, briefing, { agent_id: "self", name: "李世民" });
      expect(text?.headline).not.toContain(" · ");
      expect(text?.details).toEqual(["对象：李世民"]);
    }
  });
  it("reads a decision to do nothing this step as such, with no target", () => {
    expect(describeDecisionSummary(decisionSummary({ operation: "idle", description: "", target: NONE, estimated_duration_seconds: 0 }), briefing))
      .toEqual({ headline: "⏸ 本步不行动", details: [] });
  });
  it("shows how pressing a message is and what the step is for", () => {
    const text = describeDecisionSummary(decisionSummary({
      operation: "send_message", target: NONE, parameters: { message: "速来", urgency: "high", expected_outcome: "让他赶来" },
    }), briefing);
    expect(text.details).toEqual(["内容：速来", "紧急程度：紧急", "预期结果：让他赶来"]);
  });
  it("names a move's destination and never displays unknown identifiers", () => {
    expect(describeDecisionSummary(decisionSummary({ operation: "move", target: target([{ kind: "location", id: "gate" }]) }), briefing).details).toEqual(["目的地：玄武门"]);
    expect(describeDecisionSummary(decisionSummary({ operation: "move", target: target([{ kind: "location", id: "unknown" }]) }), briefing).details).toEqual(["目的地：某地"]);
  });
  it("tells a hand-over's receiver from the holder of a thing taken", () => {
    const held = { ...briefing, entities: [{ entity_id: "seal", name: "印信", entity_type: "item", holder_id: "c", is_takeable: true }] };
    const seal = { kind: "item", id: "seal" };
    expect(describeDecisionSummary(decisionSummary({ operation: "take", target: target([seal], [], [C]) }), held).details).toEqual(["对象：印信", "持有者：李元吉"]);
    const mine = { ...held, entities: [{ ...held.entities[0], holder_id: "self" }] };
    expect(describeDecisionSummary(decisionSummary({ operation: "physical", target: target([seal], [], [B]) }), mine).details).toEqual(["对象：印信", "交给：李建成"]);
  });
  it("renders unconstrained action types and their meaningful parameters", () => {
    const text = describeDecisionSummary(decisionSummary({ operation: "send_message", target: target([B, C]), parameters: { message: "留步", announce: false } }), briefing);
    expect(text.headline).toBe("📨 发送消息 · 劝他别进宫");
    expect(text.details).toContain("接收：李建成、李元吉");
    expect(text.details).toContain("内容：留步");
    expect(describeDecisionSummary(decisionSummary({ operation: "rest", target: NONE }), briefing).headline).toContain("💤 休息");
  });
  it("shows the people a move carries off", () => {
    const text = describeDecisionSummary(decisionSummary({ operation: "move", target: target([{ kind: "location", id: "gate" }], [C]) }), briefing);
    expect(text.details).toEqual(["目的地：玄武门", "强行带走：李元吉"]);
  });
  it("shows an errand's order without leaking ids", () => {
    const text = describeDecisionSummary(decisionSummary({ operation: "errand", target: NONE, parameters: { destination_id: "gate", item_id: "unknown", recipient_id: "b" } }), briefing);
    expect(text.details).toEqual(["目的地：玄武门", "携带物品：某物", "接收：李建成"]);
  });
  it("names a thing off every list by its own name only", () => {
    const text = describeDecisionSummary(decisionSummary({ operation: "physical", target: target([{ kind: "object", id: "门闩" }]), parameters: { item_name: "门闩" } }), briefing);
    expect(text.details).toEqual(["物品：门闩"]);
  });
});
describe("reasonText", () => {
  it("maps known reasons and falls back for unknown reasons", () => {
    expect(reasonText("talk_target_not_present")).toBe("交谈对象不在当前地点。");
    expect(reasonText("physical_recipient_not_held")).toBe("你未持有要转交的物品。");
    expect(reasonText("destination_id_not_perceived")).toContain("不在当前可感知范围");
    expect(reasonText("something_new")).toBe("当前无法执行该操作，请调整后重试。");
    expect(reasonText(null)).toBe("");
  });
});

describe("signature confirmation", () => {
  it.each([false, true])("shows the resolved signature before sending (%s)", (anonymous) => {
    const text = describeDecisionSummary(decisionSummary({
      operation: "send_message", target: NONE, parameters: { message: "请留步", anonymous },
    }), briefing);
    expect(text.details).toContain(`签名：${anonymous ? "匿名" : "实名"}`);
    expect(text.details).toContain("内容：请留步");
  });
});
