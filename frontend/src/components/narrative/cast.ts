/** Identity (people and places) is public; presence and private state stay scoped to the step. */

import { identityOf } from "../../lib/worldTable";
import { indexStep } from "../../lib/step";
import type { Npcs } from "../../lib/npcs";
import type { Cast } from "../../lib/cast";
import { type Places, placeName } from "../../lib/places";
import { identityColor, toHex } from "../../lib/color";
import type { StepView } from "../../types";

export interface StepCast {
  /** A person's identity color, using the shared fallback when absent. */
  inkOf(id: string | undefined): string | undefined;
  /** The public name of a character; "" when absent from the cast. */
  nameOf(id: string): string;
  isMain(id: string): boolean;
  /** A scope's narrative name (see `placeName`): where a deed happened, a body stands, a word was heard. */
  placeOf(locationId: string | null): string;
  /**
   * An imposed condition this person is standing under ("双手被反绑"), "" when free.
   * Here because a row gets only a `cast`, never the step. Not a row of its own: it rides the
   * card of whoever acts under it.
   */
  conditionOf(id: string | undefined): string;
}

/** Everyone in a deed, the one who started it first: the sentence belongs to them. */
export function initiatorFirst(initiator_id: string, participant_ids: string[]): string[] {
  return [initiator_id, ...participant_ids.filter((id) => id !== initiator_id)].filter(Boolean);
}

export function stepCast(step: StepView, cast: Cast, places: Places, npcs: Npcs): StepCast {
  const index = indexStep(step);
  return {
    inkOf: (id) => (id ? toHex(identityColor(identityOf(id, cast, npcs)?.color, id)) : undefined),
    nameOf: (id) => identityOf(id, cast, npcs)?.name ?? "",
    isMain: (id) => index.agents[id]?.is_main_character ?? false,
    placeOf: (locationId) => placeName(places, locationId),
    conditionOf: (id) => (id ? index.agents[id]?.public.condition ?? "" : ""),
  };
}
