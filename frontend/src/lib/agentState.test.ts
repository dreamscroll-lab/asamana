import { describe, expect, it } from "vitest";

import type { AgentPrivateView } from "../types";
import { unfinishedGoals } from "./agentState";

const state = (extra: Partial<AgentPrivateView>) => extra as AgentPrivateView;

describe("unfinishedGoals", () => {
  it("merges both origins in queue order and drops finished goals", () => {
    const s = state({
      short_term_goals: [
        { text: "a", status: "active", origin: "cognitive" },
        { text: "b", status: "interrupted", origin: "cognitive" },
        { text: "c", status: "active", origin: "residue" },
        { text: "d", status: "completed", origin: "residue" },
        { text: "e", status: "failed", origin: "cognitive" },
      ],
    });
    expect(unfinishedGoals(s)).toEqual(["a", "b", "c"]);
  });

  it("reads nothing when there is no queue", () => {
    expect(unfinishedGoals(undefined)).toEqual([]);
  });
});
