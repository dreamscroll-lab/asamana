import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { identityOf, worldTable, type TableState, type WorldTable } from "./worldTable";

const subscriptions = vi.hoisted(() => [] as ((listener: () => void) => () => void)[]);

vi.mock("react", async (original) => ({
  ...await original<typeof import("react")>(),
  useSyncExternalStore: (subscribe: (listener: () => void) => () => void, getSnapshot: () => unknown) => {
    subscriptions.push(subscribe);
    return getSnapshot();
  },
}));

beforeEach(() => { subscriptions.length = 0; });

type Row = { id: string; name: string };
const row = (id: string): Row => ({ id, name: id });
const keyOf = (value: Row) => value.id;
const flush = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

function snapshot(cache: WorldTable<Row>, worldId: string): TableState<Row> {
  let state!: TableState<Row>;
  function Probe() {
    state = cache.useTable(worldId);
    return null;
  }
  renderToStaticMarkup(createElement(Probe));
  return state;
}

describe("identity lookup recovery", () => {
  it("notifies every existing subscriber when recovery succeeds or fails", async () => {
    const fetch = vi.fn<() => Promise<Row[]>>()
      .mockResolvedValueOnce([row("known")])
      .mockResolvedValueOnce([row("known"), row("recovered")])
      .mockRejectedValueOnce(new Error("Offline"));
    const cache = worldTable(fetch, keyOf);
    const initial = await cache.load("world");
    snapshot(cache, "world");
    snapshot(cache, "world");
    const first = vi.fn();
    const second = vi.fn();
    const unsubscribeFirst = subscriptions[0](first);
    const unsubscribeSecond = subscriptions[1](second);

    identityOf("recovered", initial);
    await flush();
    expect(first).toHaveBeenCalledTimes(1);
    expect(second).toHaveBeenCalledTimes(1);
    const recovered = snapshot(cache, "world").table;
    expect(identityOf("recovered", recovered)?.name).toBe("recovered");

    identityOf("absent", recovered);
    await flush();
    expect(first).toHaveBeenCalledTimes(2);
    expect(second).toHaveBeenCalledTimes(2);
    const failed = snapshot(cache, "world");
    expect(failed.table).toBe(recovered);
    expect(failed.status).toBe("ready");
    expect(failed.error?.message).toBe("Offline");

    unsubscribeFirst();
    unsubscribeSecond();
    fetch.mockResolvedValueOnce([row("known"), row("recovered"), row("absent")]);
    await cache.load("world");
    expect(first).toHaveBeenCalledTimes(2);
    expect(second).toHaveBeenCalledTimes(2);
  });

  it("keeps the table's identity when a recovery brings back the same rows", async () => {
    const fetch = vi.fn<() => Promise<Row[]>>().mockResolvedValue([row("known")]);
    const cache = worldTable(fetch, keyOf);
    const initial = await cache.load("world");

    identityOf("absent", initial);
    await flush();

    expect(fetch).toHaveBeenCalledTimes(2);
    expect(snapshot(cache, "world").table).toBe(initial);
  });

  it("defers and merges missing-ID recovery and replaces the whole table", async () => {
    const fetch = vi.fn<() => Promise<Row[]>>()
      .mockResolvedValueOnce([row("old")])
      .mockResolvedValueOnce([row("a"), row("b")]);
    const cache = worldTable(fetch, keyOf);
    const initial = await cache.load("world");
    expect(identityOf("old", initial)?.name).toBe("old");
    expect(identityOf("a", initial)).toBeUndefined();
    identityOf("a", initial);
    identityOf("b", initial);
    expect(fetch).toHaveBeenCalledTimes(1);
    await flush();
    const refreshed = await cache.load("world");
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(Object.keys(refreshed)).toEqual(["a", "b"]);
    expect(identityOf("a", refreshed)?.name).toBe("a");
  });

  it("checks every candidate before refreshing either people table", async () => {
    const agentsFetch = vi.fn(async () => [row("agent")]);
    const npcsFetch = vi.fn(async () => [row("npc")]);
    const agents = worldTable(agentsFetch, keyOf);
    const npcs = worldTable(npcsFetch, keyOf);
    const cast = await agents.load("world");
    const bodies = await npcs.load("world");
    expect(identityOf("npc", cast, bodies)?.name).toBe("npc");
    expect(identityOf("agent", cast, bodies)?.name).toBe("agent");
    await flush();
    expect(agentsFetch).toHaveBeenCalledTimes(1);
    expect(npcsFetch).toHaveBeenCalledTimes(1);
  });

  it("waits for an initial candidate load before declaring a mixed lookup missing", async () => {
    let finish!: (rows: Row[]) => void;
    const agentsFetch = vi.fn(async () => [row("agent")]);
    const npcsFetch = vi.fn(() => new Promise<Row[]>((resolve) => { finish = resolve; }));
    const agents = worldTable(agentsFetch, keyOf);
    const npcs = worldTable(npcsFetch, keyOf);
    const cast = await agents.load("world");
    const pending = npcs.load("world");
    const empty = snapshot(npcs, "world").table;
    expect(identityOf("npc", cast, empty)).toBeUndefined();
    await flush();
    expect(agentsFetch).toHaveBeenCalledTimes(1);
    finish([row("npc")]);
    await pending;
    await flush();
    expect(agentsFetch).toHaveBeenCalledTimes(1);
    expect(npcsFetch).toHaveBeenCalledTimes(1);
  });

  it("automatically checks an unknown ID once until an explicit reload", async () => {
    const fetch = vi.fn(async () => [row("known")]);
    const cache = worldTable(fetch, keyOf);
    let table = await cache.load("world");
    identityOf("unknown", table);
    await flush();
    table = await cache.load("world");
    for (let i = 0; i < 3; i++) identityOf("unknown", table);
    await flush();
    expect(fetch).toHaveBeenCalledTimes(2);
    table = await cache.reload("world");
    identityOf("unknown", table);
    await flush();
    expect(fetch).toHaveBeenCalledTimes(4);
  });

  it("keeps records after a recovery failure without retrying every lookup", async () => {
    const fetch = vi.fn<() => Promise<Row[]>>()
      .mockResolvedValueOnce([row("known")])
      .mockRejectedValueOnce(new Error("Offline"));
    const cache = worldTable(fetch, keyOf);
    const initial = await cache.load("world");
    identityOf("missing", initial);
    await flush();
    const state = snapshot(cache, "world");
    expect(state.status).toBe("ready");
    expect(state.error?.message).toBe("Offline");
    expect(identityOf("known", state.table)?.name).toBe("known");
    identityOf("missing", state.table);
    await flush();
    expect(fetch).toHaveBeenCalledTimes(2);
    fetch.mockResolvedValueOnce([row("known"), row("missing")]);
    const retried = await cache.load("world");
    expect(identityOf("missing", retried)?.name).toBe("missing");
    expect(snapshot(cache, "world").error).toBeNull();
  });

  it("isolates missing-ID recovery across worlds", async () => {
    const fetch = vi.fn(async (world: string) => [row(world)]);
    const cache = worldTable(fetch, keyOf);
    const left = await cache.load("left");
    const right = await cache.load("right");
    identityOf("missing", left);
    await flush();
    expect(fetch.mock.calls.map(([world]) => world)).toEqual(["left", "right", "left"]);
    identityOf("missing", right);
    await flush();
    expect(fetch.mock.calls.map(([world]) => world)).toEqual(["left", "right", "left", "right"]);
  });

  it("treats empty sentinel IDs and unmanaged fixtures as local lookups", async () => {
    const fetch = vi.fn(async () => [row("known")]);
    const cache = worldTable(fetch, keyOf);
    const table = await cache.load("world");
    expect(identityOf("", table)).toBeUndefined();
    expect(identityOf("known", { known: row("known") })?.name).toBe("known");
    expect(identityOf("missing", {})).toBeUndefined();
    await flush();
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("cannot let an older recovery overwrite an explicit reload", async () => {
    let finish!: (rows: Row[]) => void;
    const fetch = vi.fn<() => Promise<Row[]>>()
      .mockResolvedValueOnce([row("old")])
      .mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }))
      .mockResolvedValueOnce([row("confirmed")]);
    const cache = worldTable(fetch, keyOf);
    identityOf("missing", await cache.load("world"));
    await flush();
    const confirmed = await cache.reload("world");
    finish([row("stale")]);
    await flush();
    expect(await cache.load("world")).toBe(confirmed);
  });
});
