import { describe, expect, it, vi } from "vitest";

import type { WalkHost } from "./walking";
import type { WorldMap as MapType } from "./worldMap";

vi.mock("phaser", () => ({ default: {} }));
vi.mock("./effects", () => ({ footDust: vi.fn() }));
Object.assign(globalThis, { window: { devicePixelRatio: 2 } });
const { Walker } = await import("./walking");
const { WorldMap } = await import("./worldMap");

function setup() {
  const map: MapType = new WorldMap(null as never, null as never);
  map.cols = 9;
  map.rows = 9;
  map.blocked = Array.from({ length: 9 }, (_, y) => Array.from({ length: 9 }, (_, x) => x !== 4 && y !== 4));
  map.standable = map.blocked.map((row) => row.map((blocked) => !blocked));
  map.toll = map.blocked.map((row) => row.map(() => 0));
  map.rooms.set("room", {
    id: "room", name: "Room", gx0: 1, gy0: 1, gw: 7, gh: 7, gx: 4.5, gy: 4.5,
    atx: 4, aty: 4, sx: 0, sy: 0, stand: [], inRect: 0,
  });
  const walker = new Walker({ map } as WalkHost, new Set());
  walker.mount();
  return { walker, map };
}

describe("local presence routes", () => {
  it("reaches the visible room edge using the walkable network", () => {
    const { walker, map } = setup();
    const at = map.iso(4.5, 4.5);
    const route = walker.presenceRoute("room", at)!;
    expect(route[0]).toEqual(at);
    const end = map.unproject(route.at(-1)!.x, route.at(-1)!.y);
    expect(Math.min(end.gx - 1, 8 - end.gx, end.gy - 1, 8 - end.gy)).toBeCloseTo(0.5);
    for (let i = 1; i < route.length; i++) {
      for (let t = 0; t <= 1; t += 0.1) {
        const a = route[i - 1];
        const b = route[i];
        const g = map.unproject(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
        expect(map.blocked[Math.floor(g.gy)][Math.floor(g.gx)]).toBe(false);
      }
    }
  });

  it("returns no route for an unknown room or a room without reachable standing ground", () => {
    const { walker, map } = setup();
    expect(walker.presenceRoute("unknown", { x: 0, y: 0 })).toBeNull();
    map.standable = map.standable.map((row) => row.map(() => false));
    expect(walker.presenceRoute("room", map.iso(4.5, 4.5))).toBeNull();
  });

  it("does not detour outside the visible room to reach an obstructed edge", () => {
    const { walker, map } = setup();
    map.blocked = Array.from({ length: 9 }, (_, y) => Array.from({ length: 9 }, (_, x) => y === 3 && x >= 1 && x <= 7));
    map.standable = map.blocked.map((row) => row.map((blocked) => !blocked));
    walker.mount();
    const route = walker.presenceRoute("room", map.iso(4.5, 4.5))!;
    expect(route).not.toBeNull();
    for (const p of route) {
      const g = map.unproject(p.x, p.y);
      expect(g.gx).toBeGreaterThanOrEqual(1);
      expect(g.gx).toBeLessThanOrEqual(8);
      expect(g.gy).toBeGreaterThanOrEqual(1);
      expect(g.gy).toBeLessThanOrEqual(8);
    }
  });
});
