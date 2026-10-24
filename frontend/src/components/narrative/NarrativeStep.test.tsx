import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { StepView } from "../../types";
import NarrativeStep from "./NarrativeStep";

function render(outcome: string, ongoing = false, condition = ""): string {
  const step: StepView = {
    world_id: "world", step: 1,
    world_time: { label: "清晨", hour: 6, minute: 0 },
    locations: {
      hall: {
        location_id: "hall", agents: [], entities: {}, events: [],
        npcs: [{
          npc_id: "runner",
          public: { condition, displaced: false, carrying: {} },
          private: { outcome, ongoing, carrying: {} },
        }],
      },
    },
    transit: { agents: [], npcs: [], events: [] },
    world: { events: [] }, observer: { injections: [] },
  };
  return renderToStaticMarkup(<NarrativeStep step={step} cast={{}} places={{}} npcs={{}} />);
}

describe("step inactivity", () => {
  it.each([
    ["回来向某人回了话", false],
    ["正往回走", true],
  ])("shows an NPC's activity without claiming nobody acted: %s", (outcome, ongoing) => {
    const html = render(outcome, ongoing);
    expect(html).toContain(outcome);
    expect(html).not.toContain("本步无人行动。");
  });

  it("keeps the empty notice when an NPC has no activity", () => {
    expect(render("  ")).toContain("本步无人行动。");
  });

  it("does not count an existing condition as activity", () => {
    const html = render("", false, "双手被反绑");
    expect(html).toContain("双手被反绑");
    expect(html).toContain("本步无人行动。");
  });
});
