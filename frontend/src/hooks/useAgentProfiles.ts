import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { AgentProfile } from "../types";

/**
 * The profiles of `ids` in `worldId`, as this viewer may know them, each fetched once. A profile
 * still loading, or one that failed to load, is absent from the map.
 */
export function useAgentProfiles(worldId: string, ids: string[]): Record<string, AgentProfile> {
  const [profiles, setProfiles] = useState<Record<string, AgentProfile>>({});
  const key = [...new Set(ids.filter(Boolean))].sort().join(",");

  useEffect(() => {
    let alive = true;
    for (const id of key ? key.split(",") : []) {
      if (profiles[id]) continue;
      api
        .agentProfile(worldId, id)
        .then((p) => alive && setProfiles((prev) => ({ ...prev, [id]: p })))
        .catch(() => {});
    }
    return () => {
      alive = false;
    };
    // `profiles` is left out on purpose: a fetched profile needs no refetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [worldId, key]);

  return profiles;
}
