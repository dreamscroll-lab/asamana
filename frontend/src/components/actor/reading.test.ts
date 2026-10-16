import { describe, expect, it } from "vitest";

import { describeDecision, reasonText } from "./reading";
import type { ActorAction, ActorBriefing, ActorDecision } from "../../types";

const briefing = {
  people: [{ agent_id: "b", name: "李建成", gender: "男", present: true }],
  npcs: [],
  places: [{ location_id: "gate", name: "玄武门", travel_seconds: 600 }],
  entities: [],
} satisfies ActorBriefing;

const action = (over: Partial<ActorAction>): ActorAction => ({
  action_type: "talk", action_description: "劝他别进宫", inner_monologue: "", expected_outcome: "",
  estimated_steps: 1, person_ids: [], message_announce: false, message_content: "", urgency: "normal",
  destination_id: null, move_carry_ids: [], physical_person_id: null, physical_npc_id: null,
  physical_entity_id: null, item_name: "", physical_recipient_id: null, errand_npc_id: null,
  errand_destination_id: null, errand_item_id: null, errand_recipient_id: null, errand_announce: false,
  errand_message: "", ...over,
});

const decision = (over: Partial<ActorDecision>): ActorDecision => ({
  action: null, idle: false, ...over,
});

describe("describeDecision", () => {
  it("用角色所知的名字说出对象", () => {
    const text = describeDecision(
      decision({ action: action({ person_ids: ["b"] }) }), briefing, 3600,
    );
    expect(text.headline).toBe("💬 交谈 · 劝他别进宫");
    expect(text.details).toEqual(["对象：李建成"]);
  });

  it("名单外的人不露出 id", () => {
    const text = describeDecision(
      decision({ action: action({ action_type: "move", destination_id: "nowhere" }) }), briefing, 3600,
    );
    expect(text.details).toEqual(["去往：某处"]);
  });

  it("时长说成世界的时间,不说步数", () => {
    const busy = describeDecision(decision({ action: action({ estimated_steps: 3 }) }), briefing, 3600);
    expect(busy.details).toContain("预计约 3 小时");
  });
});

describe("reasonText", () => {
  it("已知原因有自己的说法,名单外的对象统一说明,其余兜底", () => {
    expect(reasonText("talk_target_not_present")).toBe("你想说话的人不在眼前。");
    expect(reasonText("destination_id_not_perceived")).toContain("不在你能触及的范围");
    expect(reasonText("something_new")).toBe("这样做不成，换个说法试试。");
    expect(reasonText(null)).toBe("");
  });
});
