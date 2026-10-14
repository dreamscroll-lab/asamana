/**
 * How the feed refers to the people and places in a step: name, identity color, place. Shared
 * by every row so one person never resolves two ways.
 *
 * Nothing reaches past the step's own bodies and places: a person this step never mentioned
 * gets "", never an id.
 */

import { type Placed, indexStep } from "../../lib/step";
import type { ActionSummary, StepView } from "../../types";

export interface StepCast {
  /** A person's identity color; `undefined` when this step has none for them (callers fall back). */
  inkOf(id: string | undefined): string | undefined;
  /** The name this step reported for an agent id; "" when it never mentioned them. */
  nameOf(id: string): string;
  isMain(id: string): boolean;
  /** Where a deed happened: the place it is recorded at, 「途中」 between two waypoints. */
  placeOf(act: Placed<ActionSummary>): string;
  /**
   * An imposed condition this person is standing under ("双手被反绑"), "" when free.
   * Here because a row gets only a `cast`, never the step. Not a row of its own: it rides the
   * card of whoever acts under it.
   */
  conditionOf(id: string | undefined): string;
}

export function stepCast(step: StepView): StepCast {
  const index = indexStep(step);
  return {
    inkOf: (id) => (id ? index.agents[id]?.color || undefined : undefined),
    nameOf: (id) => index.agents[id]?.name ?? "",
    isMain: (id) => index.agents[id]?.is_main_character ?? false,
    placeOf: (act) => act.location_name,
    conditionOf: (id) => (id ? index.agents[id]?.public.condition ?? "" : ""),
  };
}
