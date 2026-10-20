import { useCallback, useEffect, useSyncExternalStore } from "react";

/** Shared per-world identity records; explicit reloads replace the table for every subscriber. */
export type Table<T> = Readonly<Record<string, T>>;

type IdentityRow<T> = T extends Table<infer Row> ? Row : never;

interface IdentitySource {
  state(): TableState<unknown>;
  pending(): Promise<unknown> | null;
  recover(id: string): void;
}

const sources = new WeakMap<object, IdentitySource>();

/** Inspect every candidate before recovering a miss, since people live in separate identity tables. */
export function identityOf<Tables extends readonly Table<unknown>[]>(
  id: string, ...tables: Tables
): IdentityRow<Tables[number]> | undefined {
  if (!id) return undefined;
  for (const table of tables) {
    if (Object.hasOwn(table, id)) return table[id] as IdentityRow<Tables[number]>;
  }
  const candidates = [...new Set(tables.map((table) => sources.get(table)).filter(
    (source): source is IdentitySource => source !== undefined,
  ))];
  if (!candidates.length) return undefined;

  // Lookups run during React rendering; recovery and subscriber notifications must run afterward.
  queueMicrotask(() => {
    void (async () => {
      await Promise.allSettled(candidates.map((source) => source.pending()).filter(
        (pending): pending is Promise<unknown> => pending !== null,
      ));
      if (candidates.some((source) => source.state().status === "loading")) return;
      if (candidates.some((source) => Object.hasOwn(source.state().table, id))) return;
      for (const source of candidates) source.recover(id);
    })();
  });
  return undefined;
}

export interface TableState<T> {
  status: "loading" | "ready" | "error";
  table: Table<T>;
  error: Error | null;
}

interface Entry<T> {
  state: TableState<T>;
  pending: Promise<Table<T>> | null;
  listeners: Set<() => void>;
  source: IdentitySource;
  checkedMissing: Set<string>;
  recoveryQueued: boolean;
  loaded: boolean;
}

export interface WorldTable<T> {
  load(worldId: string): Promise<Table<T>>;
  reload(worldId: string): Promise<Table<T>>;
  useTable(worldId: string): TableState<T>;
}

export function worldTable<T>(fetch: (worldId: string) => Promise<T[]>, keyOf: (row: T) => string): WorldTable<T> {
  const worlds = new Map<string, Entry<T>>();

  const entryFor = (worldId: string): Entry<T> => {
    let entry = worlds.get(worldId);
    if (!entry) {
      const source: IdentitySource = {
        state: () => created.state,
        pending: () => created.pending,
        recover: (id) => recover(worldId, created, id),
      };
      const created: Entry<T> = {
        state: { status: "loading", table: Object.freeze({}), error: null },
        pending: null, listeners: new Set(), source, checkedMissing: new Set(), recoveryQueued: false, loaded: false,
      };
      entry = created;
      sources.set(entry.state.table, source);
      worlds.set(worldId, entry);
    }
    return entry;
  };

  const publish = (entry: Entry<T>, state: TableState<T>): void => {
    entry.state = state;
    sources.set(state.table, entry.source);
    for (const listener of entry.listeners) listener();
  };

  const load = (worldId: string): Promise<Table<T>> => {
    const entry = entryFor(worldId);
    if (entry.pending) return entry.pending;
    if (entry.state.status === "ready" && !entry.state.error) return Promise.resolve(entry.state.table);
    entry.checkedMissing.clear();
    if (!entry.loaded) publish(entry, { status: "loading", table: entry.state.table, error: null });
    return read(worldId, entry);
  };

  const reload = (worldId: string): Promise<Table<T>> => {
    const entry = entryFor(worldId);
    entry.checkedMissing.clear();
    return read(worldId, entry);
  };

  const recover = (worldId: string, entry: Entry<T>, id: string): void => {
    if (entry.state.status === "loading" || entry.checkedMissing.has(id)) return;
    entry.checkedMissing.add(id);
    if (entry.pending || entry.recoveryQueued) return;
    entry.recoveryQueued = true;
    queueMicrotask(() => {
      entry.recoveryQueued = false;
      if (!entry.pending && entry.state.status !== "loading") void read(worldId, entry).catch(() => {});
    });
  };

  const read = (worldId: string, entry: Entry<T>): Promise<Table<T>> => {
    const pending = fetch(worldId).then(
      (rows) => {
        const table: Table<T> = Object.freeze(Object.fromEntries(rows.map((row) => [keyOf(row), row])));
        if (entry.pending === pending) {
          entry.pending = null;
          entry.loaded = true;
          publish(entry, { status: "ready", table, error: null });
        }
        return table;
      },
      (cause: unknown) => {
        const error = cause instanceof Error ? cause : new Error(String(cause));
        if (entry.pending === pending) {
          entry.pending = null;
          publish(entry, { status: entry.loaded ? "ready" : "error", table: entry.state.table, error });
        }
        throw error;
      },
    );
    entry.pending = pending;
    return pending;
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
