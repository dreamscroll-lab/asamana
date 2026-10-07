/** What a step brought into the world: read each entity's own created_step rather than diffing
 * against the previous step. */

import { describe, expect, it } from "vitest";

import { bornInStep } from "./feed";
import type { EntityView, StepEvent } from "../../types";

const thing = (createdStep: number): EntityView => ({
  name: "横刀",
  entity_type: "item",
  state: "intact",
  presence: "at_location",
  presence_ref: "loc-1",
  description: "",
  is_public: true,
  content: "",
  created_step: createdStep,
});

const at = (step: number, entities: Record<string, EntityView>): StepEvent =>
  ({ step, entities } as unknown as StepEvent);

describe("bornInStep", () => {
  it("只认出生步等于本步的物件", () => {
    const born = bornInStep(at(41, { seed: thing(0), made: thing(41) }));
    expect([...born]).toEqual(["made"]);
  });

  it("叙事流最早的一步不会把世界原有之物算成新造的", () => {
    // This is the edge of the 60-step sliding window: there's no step 40 to compare against.
    // Diffing against the previous step would mark every entity in this step's table as made
    // this beat, so an action touching an age-old knife would be shown as having made it.
    expect([...bornInStep(at(41, { seed: thing(0), old: thing(7) }))]).toEqual([]);
  });

  it("造物的下一步它就是寻常物件", () => {
    expect([...bornInStep(at(42, { made: thing(41) }))]).toEqual([]);
  });

  it("没有物件表的一步答空集,不报错", () => {
    expect([...bornInStep({ step: 3 } as unknown as StepEvent)]).toEqual([]);
  });
});
