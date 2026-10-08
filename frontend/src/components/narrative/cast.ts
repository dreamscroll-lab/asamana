/**
 * How the feed refers to the people in a step: name, identity color, place. Shared by every
 * row so one person never resolves two ways.
 *
 * Nothing reaches past `step.agent_states`: a person this step never mentioned gets "", never
 * an id.
 */

import type { ActionSummary, StepEvent } from "../../types";

export interface StepCast {
  /** A person's identity color; `undefined` when this step has none for them (callers fall back). */
  inkOf(id: string | undefined): string | undefined;
  /** The name this step reported for an agent id; "" when it never mentioned them. */
  nameOf(id: string): string;
  /** Where a deed happened: the actor's reported location name. */
  placeOf(act: ActionSummary): string;
  /**
   * An imposed condition this person is standing under ("双手被反绑"), "" when free.
   * Here because a row gets only a `cast`, never `step.agent_states`. Not a row of its own:
   * it rides the card of whoever acts under it.
   */
  conditionOf(id: string | undefined): string;
}

export function stepCast(step: StepEvent): StepCast {
  return {
    inkOf: (id) => (id ? step.agent_states[id]?.color || undefined : undefined),
    nameOf: (id) => step.agent_states[id]?.agent_name ?? "",
    // Never the record's location_id: an id printed in the feed is a membrane leak.
    placeOf: (act) => step.agent_states[act.agent_id]?.location || "某地",
    conditionOf: (id) => (id ? step.agent_states[id]?.condition ?? "" : ""),
  };
}
