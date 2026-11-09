import { api } from "../api/client";
import type { PlaceIdentity } from "../types";
import type { TipLine } from "./tip";
import { identityOf, type Table, worldTable } from "./worldTable";

export type Places = Table<PlaceIdentity>;

export interface PlacesState {
  status: "loading" | "ready" | "error";
  places: Places;
  error: Error | null;
}

const table = worldTable((worldId: string) => api.places(worldId), (place) => place.location_id);

export const loadPlaces = table.load;
export const reloadPlaces = table.reload;

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
  return identityOf(locationId, places)?.name || "某地";
}

/**
 * A place's hover card: its name, then what it is, then the sub-places in it. A place with
 * neither a description nor sub-places is its name alone.
 */
export function placeTipLines(name: string, description: string, subPlaces: readonly string[] = []): TipLine[] {
  const lines: TipLine[] = [{ text: name, tone: "head" }];
  if (description) lines.push({ text: description, tone: "background" });
  if (subPlaces.length) lines.push({ text: `${subPlaces.join("、")}都在此地`, tone: "background" });
  return lines;
}
