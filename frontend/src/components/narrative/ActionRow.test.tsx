import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { Placed, PlacedEntity } from "../../lib/step";
import type { ActionSummary } from "../../types";
import type { StepCast } from "./cast";
import ActionRow from "./ActionRow";

const cast: StepCast = {
  nameOf: (id) => ({ maker: "甲", taker: "乙" })[id] ?? "",
  inkOf: () => "#336699", isMain: () => false, conditionOf: () => "",
  placeOf: (id) => id === "elsewhere" ? "书房" : "大殿",
};

const entity = (over: Partial<PlacedEntity> = {}): PlacedEntity => ({
  name: "密诏副本", entity_type: "item", state: "intact", description: "",
  is_public: true, created_step: 13, private: { content: "" },
  presence: "held", presence_ref: "maker", ...over,
});

function render(created: string[], over: Partial<Placed<ActionSummary>> = {}, thing = entity()): string {
  const act: Placed<ActionSummary> = {
    kind: "action", anonymous: false, seq: 1, execution_id: "work-1", initiator_id: "maker", participant_ids: ["maker"],
    action_type: "work", phase: "settled", deed: "work", target: { acts_on: [], claims: [], reaches: [] },
    affected_entity_ids: ["copy"], created_entity_ids: created, overheard_by: [], interrupted_by: "",
    detected: false, not_executed: false, public: null, private: [], location_id: "hall",
    shared: {
      audience: ["maker"], outcome: "", gist: "", succeeded: true, failure_reason: "", dialogue: [],
      talk_topic: "", duration_label: "", elapsed_steps: 0, total_steps: 0, route: null,
    },
    ...over,
  };
  return renderToStaticMarkup(<ActionRow
    act={act} privateParts={act.private} cast={cast} turns={[]} showConversation={false}
    entities={{ copy: thing }} shared={act.shared} actionDescription=""
  />);
}

describe("action products", () => {
  it("shows the action's anonymous status without requiring a private part", () => {
    const action = { action_type: "send_message", deed: "send_message", private: [] };
    expect(render([], { ...action, anonymous: true })).toContain("· 匿名");
    expect(render([], { ...action, anonymous: false })).not.toContain("· 匿名");
  });

  it("does not credit a same-step injected object to its taker", () => {
    expect(render([], { action_type: "physical", deed: "seize" })).not.toContain("留下：");
  });

  it("does not credit a same-step injected object to work that merely changed it", () => {
    expect(render([])).not.toContain("留下：");
  });

  it("shows an explicitly created product even when the overall work failed", () => {
    const html = render(["copy"], { shared: {
      audience: ["maker"], outcome: "", gist: "", succeeded: false, failure_reason: "", dialogue: [],
      talk_topic: "", duration_label: "", elapsed_steps: 0, total_steps: 0, route: null,
    } });
    expect(html).toContain("留下：");
    expect(html).toContain("密诏副本");
    expect(html).toContain("随身");
  });

  it("names the current holder if someone took the product after its creation", () => {
    const html = render(["copy"], {}, entity({ presence_ref: "taker" }));
    expect(html).toContain("现由乙持有");
    expect(html).not.toContain("随身");
  });

  it("does not call a product elsewhere local", () => {
    const html = render(["copy"], {}, entity({ presence: "at_location", presence_ref: "elsewhere" }));
    expect(html).toContain("在书房");
    expect(html).not.toContain("放在此地");
  });

  it("does not infer creation from the visible result for an outsider", () => {
    expect(render(["copy"], { shared: null })).not.toContain("留下：");
  });
});
