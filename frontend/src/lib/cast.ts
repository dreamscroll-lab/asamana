import { useCallback, useEffect, useSyncExternalStore } from "react";

import { api } from "../api/client";
import type { CharacterIdentity } from "../types";

export type Cast = Readonly<Record<string, CharacterIdentity>>;

export interface CastState {
  status: "loading" | "ready" | "error";
  cast: Cast;
  error: Error | null;
}

interface CastEntry {
  state: CastState;
  pending: Promise<Cast> | null;
  listeners: Set<() => void>;
}

const EMPTY_CAST: Cast = Object.freeze({});
const worlds = new Map<string, CastEntry>();

function entryFor(worldId: string): CastEntry {
  let entry = worlds.get(worldId);
  if (!entry) {
    entry = { state: { status: "loading", cast: EMPTY_CAST, error: null }, pending: null, listeners: new Set() };
    worlds.set(worldId, entry);
  }
  return entry;
}

function publish(entry: CastEntry, state: CastState): void {
  entry.state = state;
  for (const listener of entry.listeners) listener();
}

/** Successful public identities outlive views, playback resets and viewpoint changes. */
export function loadCast(worldId: string): Promise<Cast> {
  const entry = entryFor(worldId);
  if (entry.pending) return entry.pending;
  if (entry.state.status === "ready") return Promise.resolve(entry.state.cast);
  publish(entry, { status: "loading", cast: EMPTY_CAST, error: null });
  entry.pending = api.agents(worldId).then(
    (candidates) => {
      const cast: Cast = Object.freeze(Object.fromEntries(candidates.map((candidate) => [candidate.agent_id, candidate])));
      entry.pending = null;
      publish(entry, { status: "ready", cast, error: null });
      return cast;
    },
    (cause: unknown) => {
      const error = cause instanceof Error ? cause : new Error(String(cause));
      entry.pending = null;
      publish(entry, { status: "error", cast: EMPTY_CAST, error });
      throw error;
    },
  );
  return entry.pending;
}

export function useCast(worldId: string): CastState {
  const entry = entryFor(worldId);
  const subscribe = useCallback((listener: () => void) => {
    entry.listeners.add(listener);
    return () => { entry.listeners.delete(listener); };
  }, [entry]);
  const snapshot = useCallback(() => entry.state, [entry]);
  const state = useSyncExternalStore(subscribe, snapshot);
  useEffect(() => {
    if (entry.state.status === "loading") void loadCast(worldId).catch(() => {});
  }, [worldId, entry]);
  return state;
}
