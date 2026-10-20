import { api } from "../api/client";
import type { NpcIdentity } from "../types";
import { type Table, worldTable } from "./worldTable";

export type Npcs = Table<NpcIdentity>;

export interface NpcsState {
  status: "loading" | "ready" | "error";
  npcs: Npcs;
  error: Error | null;
}

const table = worldTable((worldId: string) => api.npcs(worldId), (npc) => npc.npc_id);

export const loadNpcs = table.load;
export const reloadNpcs = table.reload;

export function useNpcs(worldId: string): NpcsState {
  const { status, table: npcs, error } = table.useTable(worldId);
  return { status, npcs, error };
}
