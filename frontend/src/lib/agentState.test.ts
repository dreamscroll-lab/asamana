import { describe, expect, it } from "vitest";

import type { AgentPrivateView } from "../types";
import { unfinishedGoals } from "./agentState";

const state = (extra: Partial<AgentPrivateView>) => extra as AgentPrivateView;

describe("unfinishedGoals", () => {
  it("splits unfinished goals by origin and drops finished ones", () => {
    const s = state({
      short_term_goals: [
        { text: "a", status: "active", origin: "cognitive" },
        { text: "b", status: "interrupted", origin: "cognitive" },
        { text: "c", status: "active", origin: "residue" },
        { text: "d", status: "completed", origin: "residue" },
      ],
    });
    expect(unfinishedGoals(s)).toEqual({ planned: ["a", "b"], owed: ["c"] });
  });

  it("reads nothing when there is no queue", () => {
    expect(unfinishedGoals(undefined)).toEqual({ planned: [], owed: [] });
  });
});
