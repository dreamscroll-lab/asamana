import { useCallback, useEffect, useRef, useState } from "react";
import { Outlet } from "react-router-dom";

import { api } from "./api/client";
import { WorldsContext, type WorldsStatus } from "./lib/worldsContext";
import type { WorldMeta } from "./types";

const POLL_MS = 6000;
/**
 * Backoff cap once the backend stops answering, doubling from POLL_MS. Polling never stops,
 * so the shell heals itself when the backend returns; 收起/重试 skips the wait.
 */
const POLL_MAX_MS = 60_000;

/**
 * Newest first — a display choice, so it lives here and not in the API (which returns the
 * worlds oldest first, the order they came into being). Worlds with no usable creation
 * time sink to the bottom; ties keep the API's order.
 */
export function newestFirst(worlds: WorldMeta[]): WorldMeta[] {
  // An unparseable time counts as missing: a NaN in the comparator would scramble the order.
  const at = (w: WorldMeta) => {
    const t = w.created_at ? Date.parse(w.created_at) : NaN;
    return Number.isNaN(t) ? Number.NEGATIVE_INFINITY : t;
  };
  return [...worlds].sort((a, b) => at(b) - at(a));
}

/**
 * The world list, for everything under it. Above the shell rather than inside it: the dev
 * tools also pick worlds from it, and they live outside the shell (see main.tsx).
 */
export default function WorldsProvider() {
  const [worlds, setWorlds] = useState<WorldMeta[]>([]);
  const [status, setStatus] = useState<WorldsStatus>("loading");
  // Read by the poll loop without re-arming it — the interval must not restart on every
  // successful tick just because the delay is state.
  const backoff = useRef(POLL_MS);

  const refresh = useCallback(async () => {
    try {
      setWorlds(newestFirst(await api.listWorlds()));
      setStatus("ok");
      backoff.current = POLL_MS;
    } catch {
      // Don't swallow this: nothing else reports a failure to list, so a swallowed error
      // looks like a fresh install (empty sidebar, no sample themes).
      setStatus("offline");
      backoff.current = Math.min(backoff.current * 2, POLL_MAX_MS);
    }
  }, []);

  useEffect(() => {
    let timer: number;
    let stopped = false;
    const tick = async () => {
      await refresh();
      if (!stopped) timer = window.setTimeout(tick, backoff.current);
    };
    void tick();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [refresh]);

  return (
    <WorldsContext.Provider value={{ worlds, status, refresh }}>
      <Outlet />
    </WorldsContext.Provider>
  );
}
