import { api } from "../api/client";
import type { CharacterIdentity } from "../types";
import { type Table, worldTable } from "./worldTable";

export type Cast = Table<CharacterIdentity>;

export interface CastState {
  status: "loading" | "ready" | "error";
  cast: Cast;
  error: Error | null;
}

const table = worldTable((worldId: string) => api.agents(worldId), (identity) => identity.agent_id);

export const loadCast = table.load;

export function useCast(worldId: string): CastState {
  const { status, table: cast, error } = table.useTable(worldId);
  return { status, cast, error };
}
