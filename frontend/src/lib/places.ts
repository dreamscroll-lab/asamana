import { api } from "../api/client";
import type { PlaceIdentity } from "../types";
import type { TipLine } from "./tip";
import { type Table, worldTable } from "./worldTable";

export type Places = Table<PlaceIdentity>;

export interface PlacesState {
  status: "loading" | "ready" | "error";
  places: Places;
  error: Error | null;
}

const table = worldTable((worldId: string) => api.places(worldId), (place) => place.location_id);

export const loadPlaces = table.load;

export function usePlaces(worldId: string): PlacesState {
  const { status, table: places, error } = table.useTable(worldId);
  return { status, places, error };
}

/**
 * A scope's narrative name: the place's own, 「途中」 between two waypoints (""), "" for the
 * world scope (null). A place missing from the table reads 「某地」, never its id.
 */
export function placeName(places: Places, locationId: string | null): string {
  if (locationId === null) return "";
  if (locationId === "") return "途中";
  return places[locationId]?.name || "某地";
}

/** A place's hover card: its name, then what it is. A place without a description is its name alone. */
export function placeTipLines(name: string, description: string): TipLine[] {
  return [{ text: name, tone: "head" }, ...(description ? [{ text: description, tone: "background" as const }] : [])];
}
