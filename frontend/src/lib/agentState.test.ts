import { describe, expect, it } from "vitest";

import type { AgentStateSummary } from "../types";
import { unfinishedGoals } from "./agentState";

const state = (extra: Partial<AgentStateSummary>) => extra as AgentStateSummary;

describe("unfinishedGoals", () => {
  it("splits unfinished goals by origin and drops finished ones", () => {
    const s = state({
      short_term_goals: ["ignored"],
      short_term_goal_entities: [
        { text: "a", status: "active", origin: "cognitive" },
        { text: "b", status: "interrupted" },
        { text: "c", status: "active", origin: "residue" },
        { text: "d", status: "completed", origin: "residue" },
      ],
    });
    expect(unfinishedGoals(s)).toEqual({ planned: ["a", "b"], owed: ["c"] });
  });

  it("reads the flat list as all-planned when no entity is unfinished", () => {
    expect(unfinishedGoals(state({ short_term_goals: ["x"] }))).toEqual({ planned: ["x"], owed: [] });
    expect(unfinishedGoals(undefined)).toEqual({ planned: [], owed: [] });
  });
});
