import { useCallback, useEffect, useSyncExternalStore } from "react";

/**
 * A world's identity table: entries fixed at build and identical for every viewer, keyed by id,
 * loaded once per world and shared by every view. Successful tables outlive views, playback
 * resets and viewpoint changes; a failed load stays failed until `load` is called again. The one
 * change a table sees is a setup revision before the world is confirmed, after which `reload`
 * re-reads it while the old entries stay shown.
 */
export type Table<T> = Readonly<Record<string, T>>;

export interface TableState<T> {
  status: "loading" | "ready" | "error";
  table: Table<T>;
  error: Error | null;
}

interface Entry<T> {
  state: TableState<T>;
  pending: Promise<Table<T>> | null;
  listeners: Set<() => void>;
}

export interface WorldTable<T> {
  load(worldId: string): Promise<Table<T>>;
  reload(worldId: string): Promise<Table<T>>;
  useTable(worldId: string): TableState<T>;
}

export function worldTable<T>(fetch: (worldId: string) => Promise<T[]>, keyOf: (row: T) => string): WorldTable<T> {
  const empty: Table<T> = Object.freeze({});
  const worlds = new Map<string, Entry<T>>();

  const entryFor = (worldId: string): Entry<T> => {
    let entry = worlds.get(worldId);
    if (!entry) {
      entry = { state: { status: "loading", table: empty, error: null }, pending: null, listeners: new Set() };
      worlds.set(worldId, entry);
    }
    return entry;
  };

  const publish = (entry: Entry<T>, state: TableState<T>): void => {
    entry.state = state;
    for (const listener of entry.listeners) listener();
  };

  const load = (worldId: string): Promise<Table<T>> => {
    const entry = entryFor(worldId);
    if (entry.pending) return entry.pending;
    if (entry.state.status === "ready") return Promise.resolve(entry.state.table);
    publish(entry, { status: "loading", table: empty, error: null });
    return read(worldId, entry);
  };

  const reload = (worldId: string): Promise<Table<T>> => read(worldId, entryFor(worldId));

  const read = (worldId: string, entry: Entry<T>): Promise<Table<T>> => {
    entry.pending = fetch(worldId).then(
      (rows) => {
        const table: Table<T> = Object.freeze(Object.fromEntries(rows.map((row) => [keyOf(row), row])));
        entry.pending = null;
        publish(entry, { status: "ready", table, error: null });
        return table;
      },
      (cause: unknown) => {
        const error = cause instanceof Error ? cause : new Error(String(cause));
        entry.pending = null;
        publish(entry, { status: "error", table: empty, error });
        throw error;
      },
    );
    return entry.pending;
  };

  const useTable = (worldId: string): TableState<T> => {
    const entry = entryFor(worldId);
    const subscribe = useCallback((listener: () => void) => {
      entry.listeners.add(listener);
      return () => { entry.listeners.delete(listener); };
    }, [entry]);
    const snapshot = useCallback(() => entry.state, [entry]);
    const state = useSyncExternalStore(subscribe, snapshot);
    useEffect(() => {
      if (entry.state.status === "loading") void load(worldId).catch(() => {});
    }, [worldId, entry]);
    return state;
  };

  return { load, reload, useTable };
}
