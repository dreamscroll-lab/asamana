import { describe, expect, it } from "vitest";

import { indexStep } from "./step";
import { beatsOf, figuresOf } from "../phaser/figures";
import type { ActionSummary, AgentView, EntityView, NpcView, StepView } from "../types";

const thing = (is_public = true): EntityView => ({
  name: "书信", entity_type: "item", state: "", description: "", is_public, created_step: 0, private: { content: "" },
});

const agent = (agent_id: string, carrying: { open?: Record<string, EntityView>; hidden?: Record<string, EntityView> } = {}): AgentView => ({
  agent_id, name: `名-${agent_id}`, gender: "", age: null, color: "", is_main_character: false, is_active: true,
  public: { activity_status: "", activity_label: "", condition: "", displaced: false, carrying: carrying.open ?? {} },
  private: {
    emotion: "", emotion_label: "", emotion_intensity: null, emotion_valence: null, dominant_need: null,
    dominant_need_label: "", vitality: 1, activity_status: "", activity_label: "", transit: null,
    long_term_goals: [], short_term_goals: [],
    relations: [], thoughts: [], carrying: carrying.hidden ?? {},
  },
});

const runner: NpcView = {
  npc_id: "n1", name: "王二", color: "#8a93a6", gender: "男", age: 34, description: "",
  public: { condition: "", displaced: false, carrying: {} },
  private: { outcome: "", ongoing: false, carrying: { letter: thing(false) } },
};

const talk: ActionSummary = {
  kind: "action", seq: 7, execution_id: "x", initiator_id: "a", participant_ids: ["a", "b"], action_type: "talk",
  phase: "settled", deed: "talk", target: { acts_on: [], claims: [], reaches: [] },
  affected_entity_ids: [], overheard_by: [], interrupted_by: "", succeeded: true, failure_reason: "",
  detected: false, not_executed: false, public: null,
  private: {
    audience: ["a", "b"], outcome: "", gist: "", dialogue: [], action_description: "", duration_label: "",
    elapsed_steps: 0, total_steps: 0, route: null,
  },
};

const step: StepView = {
  world_id: "w",
  step: 1,
  world_time: { label: "", hour: null, minute: null },
  locations: {
    hall: {
      location_id: "hall", name: "大殿",
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
    expect([index.agents.a.location_id, index.agents.a.location_name]).toEqual(["hall", "大殿"]);
    // Between two waypoints: no place, and the narrative name says so.
    expect([index.agents.c.location_id, index.agents.c.location_name]).toEqual(["", "途中"]);
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
});

describe("figuresOf", () => {
  it("走到的那条路挂在他那一拍 MOVE 上,身体从那里读", () => {
    const route = {
      from_location_id: "gate", to_location_id: "hall", path: ["gate", "yard", "hall"],
      arrivals: [0, 1, 1], elapsed_steps: 1, total_steps: 1,
    };
    const arrived: StepView = {
      ...step,
      locations: { hall: { ...step.locations.hall, events: [{ ...talk, action_type: "move", deed: "move", participant_ids: ["a"], private: { ...talk.private, route } }] } },
    };
    const figures = Object.fromEntries(figuresOf(indexStep(arrived)).map((f) => [f.agent_id, f]));
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
      locations: { hall: { ...step.locations.hall, events: [{ ...talk, action_type: "move", deed: "move", participant_ids: ["a", "b"], private: { ...talk.private, route } }] } },
    };
    const figures = Object.fromEntries(figuresOf(indexStep(carried)).map((f) => [f.agent_id, f]));
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
      locations: { hall: { ...step.locations.hall, events: [{ ...talk, action_type: "move", deed: "move", participant_ids: ["a", "b"], private: { ...talk.private, route } }] } },
    };
    const figures = Object.fromEntries(figuresOf(indexStep(astray)).map((f) => [f.agent_id, f]));
    expect(figures.a.arrival).toBeNull();
    expect(figures.b.arrival).toBeNull();
  });
});
