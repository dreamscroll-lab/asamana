import { describe, expect, it } from "vitest";

import { setupGraph } from "./setupGraph";
import type { SetupView } from "../types";

const character = (agent_id: string, is_main_character: boolean) => ({
  agent_id, is_main_character, name: agent_id, role: "旅人", age: 30, gender: "女",
  background: "", appearance: "", color: "#123456", core_traits: [], core_values: [],
  self_image: "", life_goal: "", secret: "", initial_location: "palace",
  initial_emotion: { emotion: "neutral", intensity: 0.2, valence: 0 },
  long_term_goals: [], hard_constraints: [], needs: [],
});

describe("setupGraph", () => {
  it("draws each relation of the setup as both its directions", () => {
    const setup: SetupView = {
      entities: [], npcs: [],
      seconds_per_step: 3600,
      characters: [character("a", true), character("b", false)],
      historical_events: [],
      relations: [{
        source_id: "a", target_id: "b", trust: 0.6, affection: 0.3, labels: ["父子:父亲"],
        reverse_trust: 0.4, reverse_affection: 0.1, reverse_labels: ["父子:儿子"],
      }],
    };
    const graph = setupGraph(setup);
    expect(graph.nodes.map((n) => [n.id, n.is_main_character])).toEqual([["a", true], ["b", false]]);
    expect(graph.nodes.map((n) => [n.name, n.role])).toEqual([["a", "旅人"], ["b", "旅人"]]);
    expect(graph.edges.map((e) => [e.from_id, e.to_id, e.labels[0]])).toEqual([
      ["a", "b", "父子:父亲"], ["b", "a", "父子:儿子"],
    ]);
  });
});
