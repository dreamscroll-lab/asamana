/**
 * Standing slots must sit on the road network. A route's two end segments are unchecked
 * straight lines to where the person stands (see Walker.groundRoute in walking.ts), so a slot
 * the network can't reach (a walled courtyard, an unmarked lawn) sends him through the wall.
 *
 * These tests pin that reachable ground sorts first and routes walked from it never touch a
 * solid tile.
 */

import { describe, expect, it } from "vitest";

import { astar, computeMainComponent, foldStairs, snap } from "./pathfinding";
import type { Room } from "./worldMap";

// worldMap reads window.devicePixelRatio (text resolution) via types.ts, and these tests run in
// node. The stub has to exist before the import, hence the dynamic import.
Object.assign(globalThis, { window: { devicePixelRatio: 2 } });
const { WorldMap } = await import("./worldMap");

// A minimal map that keeps the kinds of ground a real map has:
//   `#` wall: neither walkable nor standable
//   `R` street: walkable and standable (the author-declared road network)
//   `.` open ground: standable but off the network (courtyards, lawns); this is the ground
//       that sends people through walls
const ART = [
  "..............",
  "..#########...",
  "..#.......#...",
  "..#.......#...",
  "..#########...",
  "..............",
  "RRRRRRRRRRRRRR",
  "....RR....RR..",
  "....RR....RR..",
  "..............",
];
const COLS = ART[0].length;
const ROWS = ART.length;

function makeMap(): InstanceType<typeof WorldMap> {
  const map = new WorldMap(null as never, null as never);
  map.cols = COLS;
  map.rows = ROWS;
  map.blocked = ART.map((row) => [...row].map((c) => c !== "R"));
  map.standable = ART.map((row) => [...row].map((c) => c !== "#"));
  return map;
}

function makeRoom(id: string, gx0: number, gy0: number, gw: number, gh: number): Room {
  return {
    id, name: id, gx0, gy0, gw, gh,
    gx: gx0 + gw / 2, gy: gy0 + gh / 2,
    sx: 0, sy: 0, atx: 0, aty: 0, stand: [], inRect: 0,
  };
}

const map = makeMap();
const roadNet = computeMainComponent(map.blocked, COLS, ROWS);
const onNet = (c: { x: number; y: number }): boolean => roadNet.has(c.y * COLS + c.x);

// A sealed courtyard: no street inside the rect; the nearest street is to the south.
const courtyard = makeRoom("courtyard", 2, 1, 7, 4);
// A plaza: the rect has both street (y6-8) and open ground off the network (y9).
const plaza = makeRoom("plaza", 4, 6, 2, 4);
map.findStanding(courtyard, roadNet);
map.findStanding(plaza, roadNet);

describe("findStanding", () => {
  it("把路网上的地面排在前面,哪怕矩形内另有更近的空地", () => {
    // (5,2) in the courtyard is standable, closer to the center and inside the rect, but no road
    // reaches it, so it must not be used first.
    expect(map.standable[2][5]).toBe(true);
    expect(onNet({ x: 5, y: 2 })).toBe(false);
    expect(courtyard.stand.slice(0, 8).every(onNet)).toBe(true);
    expect(courtyard.stand.findIndex((c) => c.x === 5 && c.y === 2)).toBeGreaterThan(8);
  });

  it("矩形内的可达地面又排在矩形外的可达地面之前", () => {
    const inRect = (c: { x: number; y: number }): boolean =>
      c.x + 0.5 >= plaza.gx0 && c.x + 0.5 <= plaza.gx0 + plaza.gw &&
      c.y + 0.5 >= plaza.gy0 && c.y + 0.5 <= plaza.gy0 + plaza.gh;
    expect(plaza.stand.slice(0, 6).every((c) => inRect(c) && onNet(c))).toBe(true);
  });

  it("inRect 报的是矩形内的**前缀长度**,不是总数", () => {
    // layoutFixtures uses it as an index bound, so it must be a prefix. The plaza rect has 8
    // standable tiles: 6 on the network sort first, and the 2 unreachable ones (y9) come after.
    expect(plaza.inRect).toBe(6);
    expect(plaza.stand.filter((c) =>
      c.x + 0.5 >= plaza.gx0 && c.x + 0.5 <= plaza.gx0 + plaza.gw &&
      c.y + 0.5 >= plaza.gy0 && c.y + 0.5 <= plaza.gy0 + plaza.gh).length).toBe(8);
    expect(plaza.stand.slice(0, plaza.inRect).every(onNet)).toBe(true);
  });

  it("一格可站的地都没有的房间,给空名单而不是瞎给一格", () => {
    const sealed = makeRoom("sealed", 100, 100, 2, 2);
    map.findStanding(sealed, roadNet);
    expect(sealed.stand).toEqual([]);
    expect(sealed.inRect).toBe(0);
  });
});

describe("走出来的路线", () => {
  // All of groundRoute: snap both ends to the network, run A*, fold into axis-aligned runs with
  // foldStairs, then pin both ends back to where the person actually stands.
  const route = (from: { x: number; y: number }, to: { x: number; y: number }) => {
    const start = snap(from.x, from.y, map.blocked, COLS, ROWS, roadNet);
    const goal = snap(to.x, to.y, map.blocked, COLS, ROWS, roadNet);
    const path = astar(start, goal, map.blocked, COLS, ROWS);
    expect(path).not.toBeNull();
    const passable = (x: number, y: number): boolean => !map.blocked[y]?.[x];
    const pts = foldStairs(path!, passable).map(([tx, ty]) => ({ x: tx + 0.5, y: ty + 0.5 }));
    pts[0] = { x: from.x + 0.5, y: from.y + 0.5 };
    pts[pts.length - 1] = { x: to.x + 0.5, y: to.y + 0.5 };
    return pts;
  };

  // Tiles swept by a segment. The isometric projection is linear, so a straight line on screen
  // is also straight in tile space.
  const cells = (a: { x: number; y: number }, b: { x: number; y: number }) => {
    const out: [number, number][] = [];
    for (let i = 0; i <= 200; i++) {
      const t = i / 200;
      const c: [number, number] = [
        Math.floor(a.x + (b.x - a.x) * t),
        Math.floor(a.y + (b.y - a.y) * t),
      ];
      if (!out.length || out[out.length - 1][0] !== c[0] || out[out.length - 1][1] !== c[1]) out.push(c);
    }
    return out;
  };

  it("从歇脚位到歇脚位,没有一段碰到实心的格", () => {
    const solid = (x: number, y: number): boolean => map.blocked[y][x] && !map.standable[y][x];
    for (const from of courtyard.stand.slice(0, 8)) {
      for (const to of plaza.stand.slice(0, 8)) {
        if (from.x === to.x && from.y === to.y) continue;
        const pts = route(from, to);
        for (let i = 0; i < pts.length - 1; i++) {
          const hit = cells(pts[i], pts[i + 1]).filter(([x, y]) => solid(x, y));
          expect(hit, `${JSON.stringify(from)}→${JSON.stringify(to)} 第${i}段穿过 ${JSON.stringify(hit)}`).toEqual([]);
        }
      }
    }
  });

  it("两端的钉合段长度为零 —— 人就站在搜索的起点上", () => {
    for (const slot of [...courtyard.stand.slice(0, 8), ...plaza.stand.slice(0, 8)]) {
      expect(snap(slot.x, slot.y, map.blocked, COLS, ROWS, roadNet)).toEqual([slot.x, slot.y]);
    }
  });
});
