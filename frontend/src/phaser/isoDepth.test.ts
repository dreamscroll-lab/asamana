/**
 * Front-to-back order and fading of people. Occlusion never touches the map, so these two
 * are all that can go wrong. Left for the eye (scene bench): how the fade looks, and whether it jumps.
 */

import { describe, expect, it } from "vitest";

import {
  HIDDEN_ALPHA,
  type IsoGeometry,
  SORT_BASE,
  SORT_SPAN,
  sortDepth,
} from "./isoDepth";

// The shape of the two shipped maps: 48×56 tiles, 128×64 isometric tiles.
const TILE_W = 128;
const TILE_H = 64;
const COLS = 48;
const ROWS = 56;
const G: IsoGeometry = {
  tileW: TILE_W,
  tileH: TILE_H,
  originX: ROWS * (TILE_W / 2),
  originY: 320, // how far the tallest map overhangs the grid
  worldH: (COLS + ROWS) * (TILE_H / 2) + 320 + TILE_H,
};

/** Screen y at the feet of someone standing on a tile. */
const at = (tx: number, ty: number) => (tx + ty + 1) * (TILE_H / 2) + G.originY;

describe("谁画在谁前面", () => {
  it("站得更靠前(屏幕更靠下)的画在后面那个之上", () => {
    expect(sortDepth(at(10, 11), G)).toBeGreaterThan(sortDepth(at(10, 10), G));
    expect(sortDepth(at(11, 10), G)).toBeGreaterThan(sortDepth(at(10, 10), G));
  });

  it("整张图从头到尾都在分先后,不会走到一半就并列", () => {
    // A formula like `20 + min(y * 0.002, 3.9)` caps at y=1950px, but both shipped maps are over
    // 3700px tall, so everyone on more than half the map would get the same depth.
    const north = sortDepth(at(2, 2), G);
    const middle = sortDepth(at(COLS / 2, ROWS / 2), G);
    const south = sortDepth(at(COLS - 2, ROWS - 2), G);
    expect(middle).toBeGreaterThan(north);
    expect(south).toBeGreaterThan(middle);
    expect(south - north).toBeGreaterThan(SORT_SPAN * 0.8); // uses the full band
  });

  it("相邻两格之间也分得开,不靠运气", () => {
    // One tile must map to a resolvable depth difference, or who is in front of two neighbours is random.
    expect(sortDepth(at(10, 11), G) - sortDepth(at(10, 10), G)).toBeGreaterThan(1e-4);
  });

  it("再离谱的 y 也留在自己的带子里,不撞上面的气泡和下面的地面装饰", () => {
    for (const y of [-5000, 0, G.worldH / 2, G.worldH, G.worldH * 3]) {
      const d = sortDepth(y, G);
      expect(d).toBeGreaterThanOrEqual(SORT_BASE);
      expect(d).toBeLessThanOrEqual(SORT_BASE + SORT_SPAN);
    }
  });
});

describe("被美术盖住时的虚实", () => {
  it("淡下去,但不淡到看不见 —— 大半段路上这是唯一能看见他的样子", () => {
    expect(HIDDEN_ALPHA).toBeGreaterThan(0.3);
    expect(HIDDEN_ALPHA).toBeLessThan(0.7);
  });
});
