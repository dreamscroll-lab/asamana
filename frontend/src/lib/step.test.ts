import { describe, expect, it } from "vitest";

import { stepCast } from "../components/narrative/cast";
import { identityColor, toHex } from "./color";
import type { Npcs } from "./npcs";
import { indexStep } from "./step";
import { beatsOf, figuresOf } from "../phaser/figures";
import type { ActionSummary, AgentView, CharacterIdentity, EntityView, NpcView, StepView } from "../types";

const thing = (is_public = true): EntityView => ({
  name: "书信", entity_type: "item", state: "", description: "", is_public, created_step: 0, private: { content: "" },
});

const agent = (agent_id: string, carrying: { open?: Record<string, EntityView>; hidden?: Record<string, EntityView> } = {}): AgentView => ({
  agent_id, name: `名-${agent_id}`, gender: "", age: null, is_main_character: false, is_active: true,
  public: { activity_status: "", activity_label: "", condition: "", displaced: false, carrying: carrying.open ?? {} },
  private: {
    emotion: "", emotion_label: "", emotion_intensity: null, emotion_valence: null, dominant_need: null,
    dominant_need_label: "", vitality: 1, activity_status: "", activity_label: "", transit: null,
    long_term_goals: [], short_term_goals: [],
    relations: [], carrying: carrying.hidden ?? {},
  },
});

const runner: NpcView = {
  npc_id: "n1",
  public: { condition: "", displaced: false, carrying: {} },
  private: { outcome: "", ongoing: false, carrying: { letter: thing(false) } },
};

const npcs: Npcs = {
  n1: { npc_id: "n1", name: "王二", color: "#8a93a6", gender: "男", age: 34, description: "" },
};

const talk: ActionSummary = {
  kind: "action", seq: 7, execution_id: "x", initiator_id: "a", participant_ids: ["a", "b"], action_type: "talk",
  phase: "settled", deed: "talk", target: { acts_on: [], claims: [], reaches: [] },
  affected_entity_ids: [], overheard_by: [], interrupted_by: "",
  detected: false, not_executed: false, public: null,
  shared: {
    audience: ["a", "b"], outcome: "", gist: "", succeeded: true, failure_reason: "", dialogue: [],
    talk_topic: "", duration_label: "", elapsed_steps: 0, total_steps: 0, route: null,
  },
  private: [],
};

const step: StepView = {
  world_id: "w",
  step: 1,
  world_time: { label: "", hour: null, minute: null },
  locations: {
    hall: {
      location_id: "hall",
      agents: [agent("a", { open: { sword: thing() } }), agent("b", { hidden: { note: thing(false) } })],
      npcs: [], entities: { chest: thing() }, events: [talk],
    },
  },
  transit: { agents: [agent("c")], npcs: [runner], events: [] },
  world: { events: [] },
  observer: { injections: [] },
};

describe("indexStep", () => {
  it("每个身体带上它所在的作用域", () => {
    const index = indexStep(step);
    expect(index.agents.a.location_id).toBe("hall");
    // Between two waypoints: no place.
    expect(index.agents.c.location_id).toBe("");
    expect(index.npcs[0].location_id).toBe("");
  });

  it("物件按它列在哪里还原出位置,私藏的也算", () => {
    const { entities } = indexStep(step);
    expect([entities.chest.presence, entities.chest.presence_ref]).toEqual(["at_location", "hall"]);
    expect([entities.sword.presence, entities.sword.presence_ref]).toEqual(["held", "a"]);
    expect([entities.note.presence, entities.note.presence_ref]).toEqual(["held", "b"]);
    expect([entities.letter.presence, entities.letter.presence_ref]).toEqual(["held", "n1"]);
  });
});

describe("beatsOf", () => {
  it("一件多人的事,地图上每个参与者各演一拍,共用一个 seq", () => {
    const beats = beatsOf(indexStep(step));
    expect(beats.map((b) => [b.agent_id, b.seq])).toEqual([["a", 7], ["b", 7]]);
    expect(beats[1].agent_name).toBe("名-b");
  });

  // One act of one type, done by `doer`, seen by "me" from inside or out.
  const seen = (action_type: string, audience: string[], doer = "a"): StepView => ({
    ...step,
    locations: { hall: { ...step.locations.hall, events: [{
      ...talk, action_type, deed: action_type, initiator_id: doer, participant_ids: [doer],
      public: { text: "旁人所见", strength: null },
      shared: { ...talk.shared!, audience, gist: "要旨" },
      private: [{ agent_id: doer, action_description: "我自己的说法", inner_monologue: "" }],
    }] } },
  });

  it("自己做的事用自己的说法", () => {
    const [beat] = beatsOf(indexStep(seen("work", ["me"], "me")), "me");
    expect([beat.shared !== null, beat.caption]).toEqual([true, "我自己的说法"]);
  });

  it("亲历别人做的事只知要旨,不知他的打算", () => {
    const [beat] = beatsOf(indexStep(seen("work", ["a", "me"])), "me");
    expect([beat.shared !== null, beat.caption]).toEqual([true, "要旨"]);
  });

  it("旁观时动作能说明的事不配字", () => {
    expect(beatsOf(indexStep(seen("talk", [])), "me")[0].caption).toBe("");
  });

  it("旁观时动作说不清的事用旁人所见兜底", () => {
    const [beat] = beatsOf(indexStep(seen("work", [])), "me");
    expect([beat.shared, beat.caption]).toEqual([null, "旁人所见"]);
  });

  it("旁观动手不知成败,配上旁人所见", () => {
    const [beat] = beatsOf(indexStep(seen("physical", [])), "me");
    expect([beat.shared, beat.caption]).toEqual([null, "旁人所见"]);
  });
});

describe("figuresOf", () => {
  it("uses the same NPC identity color for map figures and narrative", () => {
    const identity: CharacterIdentity = {
      agent_id: "a", name: "甲", gender: "女", age: 24, role: "旅人",
      background: "", appearance: "", color: "#123456",
    };
    const index = indexStep(step);
    const figures = Object.fromEntries(figuresOf(index, { a: identity }, npcs).map((f) => [f.agent_id, f]));
    expect(figures.a.color).toBe(identity.color);
    expect(figures.b.color).toBe("");
    expect(figures.n1.color).toBe(npcs.n1.color);
    expect(figures.n1.agent_name).toBe(npcs.n1.name);
    expect(stepCast(step, {}, {}, npcs).inkOf("n1"))
      .toBe(toHex(identityColor(figures.n1.color, "n1")));
    expect(Object.keys(figures).sort()).toEqual([
      ...Object.keys(index.agents), ...index.npcs.map((npc) => npc.npc_id),
    ].sort());
  });

  it("走到的那条路挂在他那一拍 MOVE 上,身体从那里读", () => {
    const route = {
      from_location_id: "gate", to_location_id: "hall", path: ["gate", "yard", "hall"],
      arrivals: [0, 1, 1], elapsed_steps: 1, total_steps: 1,
    };
    const arrived: StepView = {
      ...step,
      locations: { hall: { ...step.locations.hall, events: [{ ...talk, action_type: "move", deed: "move", participant_ids: ["a"], shared: { ...talk.shared!, route } }] } },
    };
    const figures = Object.fromEntries(figuresOf(indexStep(arrived), {}, npcs).map((f) => [f.agent_id, f]));
    expect(figures.a.arrival).toEqual(route);
    expect(figures.b.arrival).toBeNull();
  });

  it("被带着走的人和带他的人走同一条路", () => {
    const route = {
      from_location_id: "gate", to_location_id: "hall", path: ["gate", "yard", "hall"],
      arrivals: [0, 1, 1], elapsed_steps: 1, total_steps: 1,
    };
    const carried: StepView = {
      ...step,
      locations: { hall: { ...step.locations.hall, events: [{ ...talk, action_type: "move", deed: "move", participant_ids: ["a", "b"], shared: { ...talk.shared!, route } }] } },
    };
    const figures = Object.fromEntries(figuresOf(indexStep(carried), {}, npcs).map((f) => [f.agent_id, f]));
    expect(figures.a.arrival).toEqual(route);
    expect(figures.b.arrival).toEqual(route);
  });

  it("参与者不在路的终点就不沿这条路走", () => {
    const route = {
      from_location_id: "gate", to_location_id: "yard", path: ["gate", "yard"],
      arrivals: [0, 1], elapsed_steps: 1, total_steps: 1,
    };
    const astray: StepView = {
      ...step,
      locations: { hall: { ...step.locations.hall, events: [{ ...talk, action_type: "move", deed: "move", participant_ids: ["a", "b"], shared: { ...talk.shared!, route } }] } },
    };
    const figures = Object.fromEntries(figuresOf(indexStep(astray), {}, npcs).map((f) => [f.agent_id, f]));
    expect(figures.a.arrival).toBeNull();
    expect(figures.b.arrival).toBeNull();
  });
});
