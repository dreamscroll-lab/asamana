// Shared world-list state for the persistent shell: the sidebar renders it, the
// home view refreshes it after a build, the observation view refreshes it after a
// delete. Provided once, by WorldsProvider above App.
import { createContext, useContext } from "react";
import type { WorldMeta } from "../types";

/**
 * Whether the list could be fetched at all, separate from what it holds.
 *
 * Don't merge this into the list: an unreachable backend would then render as an empty list,
 * "you have no worlds", when the truth is "I could not ask".
 */
export type WorldsStatus = "loading" | "ok" | "offline";

export interface WorldsContextValue {
  worlds: WorldMeta[];
  status: WorldsStatus;
  refresh: () => Promise<void>;
}

export const WorldsContext = createContext<WorldsContextValue>({
  worlds: [],
  status: "loading",
  refresh: async () => {},
});

export const useWorlds = () => useContext(WorldsContext);
