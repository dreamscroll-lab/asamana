import { describe, expect, it } from "vitest";
import { describeDecisionSummary, describeIntent, reasonText } from "./decisionSummary";
import type { ActorBriefing, ActorDecisionSummary } from "../../types";

const briefing: ActorBriefing = {
  location: { location_id: "hall", name: "大堂", travel_seconds: 0 },
  people: [{ agent_id: "b", name: "李建成", gender: "男", present: true }, { agent_id: "c", name: "李元吉", gender: "男", present: true }],
  npcs: [], places: [{ location_id: "gate", name: "玄武门", travel_seconds: 600 }], entities: [],
};
const decisionSummary = (over: Partial<ActorDecisionSummary> = {}): ActorDecisionSummary => ({
  decision_summary_id: "r", description: "劝他别进宫", operation: "talk", target: { kind: "agent", id: "b" }, parameters: {}, estimated_duration_seconds: 7200, ...over,
});

describe("describeDecisionSummary", () => {
  it("names the target through the briefing", () => {
    expect(describeDecisionSummary(decisionSummary(), briefing)).toEqual({
      headline: "💬 交谈 · 劝他别进宫", details: ["对象：李建成"],
    });
  });
  it("names the performed character without putting it among other people", () => {
    const self = { agent_id: "self", name: "李世民" };
    const text = describeDecisionSummary(decisionSummary({ operation: "rest", target: { kind: "agent", id: "self" } }), briefing, self);
    expect(text.details).toEqual(["对象：李世民"]);
    expect(text.headline).toContain("歇息");
  });
  it("uses the loading fallback only for self and never resolves other people from the public cast", () => {
    const cast = { self: { agent_id: "self", name: "李世民" }, hidden: { agent_id: "hidden", name: "未见之人" } };
    const unknown = describeDecisionSummary(decisionSummary({ target: { kind: "agent", id: "hidden" } }), briefing, cast.self);
    expect(unknown.details).toEqual(["对象：某人"]);
    expect(unknown.details.join(" ")).not.toContain(cast.hidden.name);
    const loading = describeDecisionSummary(decisionSummary({ target: { kind: "agent", id: "self" } }), briefing, { agent_id: "self", name: "自己" });
    expect(loading.details).toEqual(["对象：自己"]);
  });
  it("retains the selected operation and target alongside a supplemental description", () => {
    const text = describeIntent({
      turn_id: "t", text: null, operation: "physical", target: { kind: "agent", id: "b" },
      parameters: { destination_id: null, item_id: null, recipient_id: null, message: null, announce: null, anonymous: null },
      description: "避开要害",
    }, briefing);
    expect(text).toEqual({ headline: "动手 · 避开要害", details: ["对象：李建成"] });
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
    expect(describeDecisionSummary(decisionSummary({ operation: "idle", description: "", target: null, estimated_duration_seconds: 0 }), briefing))
      .toEqual({ headline: "⏸ 这次先不动", details: [] });
  });
  it("shows how pressing a message is and what the step is for", () => {
    const text = describeDecisionSummary(decisionSummary({
      operation: "send_message", target: null, parameters: { message: "速来", urgency: "high", expected_outcome: "让他赶来" },
    }), briefing);
    expect(text.details).toEqual(["内容：速来", "紧急程度：急", "想要的结果：让他赶来"]);
  });
  it("names the current location outside the reachable places", () => {
    expect(describeDecisionSummary(decisionSummary({ operation: "broadcast", target: { kind: "location", id: "hall" } }), briefing).details).toEqual(["对象：大堂"]);
  });
  it("never displays unknown identifiers", () => {
    expect(describeDecisionSummary(decisionSummary({ operation: "move", target: { kind: "location", id: "unknown" } }), briefing).details).toEqual(["对象：某处"]);
  });
  it("renders unconstrained action types and their meaningful parameters", () => {
    const text = describeDecisionSummary(decisionSummary({ operation: "send_message", parameters: { message: "留步", person_ids: ["b", "c"], announce: false } }), briefing);
    expect(text.headline).toBe("📨 传信 · 劝他别进宫");
    expect(text.details).toContain("送给：李建成、李元吉");
    expect(text.details).toContain("内容：留步");
    expect(describeDecisionSummary(decisionSummary({ operation: "rest", target: null }), briefing).headline).toContain("💤 歇息");
  });
  it("shows carried people, recipients and objects without leaking ids", () => {
    const text = describeDecisionSummary(decisionSummary({ operation: "move", parameters: { move_carry_ids: ["c"], destination_id: "gate", item_id: "unknown", recipient_id: "b", item_name: "账本" } }), briefing);
    expect(text.details).toEqual(["对象：李建成", "去往：玄武门", "带去：某物", "交给：李建成", "强行带走：李元吉", "物品：账本"]);
  });
});
describe("reasonText", () => {
  it("maps known reasons and falls back for unknown reasons", () => {
    expect(reasonText("talk_target_not_present")).toBe("你想说话的人不在眼前。");
    expect(reasonText("destination_id_not_perceived")).toContain("不在你能触及的范围");
    expect(reasonText("something_new")).toBe("这样做不成，换个说法试试。");
    expect(reasonText(null)).toBe("");
  });
});

describe("signature confirmation", () => {
  it.each([false, true])("shows the resolved signature before sending (%s)", (anonymous) => {
    const text = describeDecisionSummary(decisionSummary({
      operation: "send_message", target: null, parameters: { message: "请留步", anonymous },
    }), briefing);
    expect(text.details).toContain(`签名：${anonymous ? "匿名" : "实名"}`);
    expect(text.details).toContain("内容：请留步");
  });
});
