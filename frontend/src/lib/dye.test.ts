import { describe, expect, it } from "vitest";

import { dyePixels, hsvOf } from "./dye";

const px = (...rgba: number[]) => new Uint8ClampedArray(rgba);

describe("dyePixels", () => {
  const red = hsvOf(0xcc0000);

  it("re-lights a garment pixel as the identity color", () => {
    // A cool mid-tone at the art's lit value lands on the identity color itself.
    const data = px(0, 0, Math.round(0.7 * 255), 255);
    dyePixels(data, red);
    // ±1: 178/255 can only approximate the art's lit value.
    expect(Math.abs(data[0] - 0xcc)).toBeLessThanOrEqual(1);
    expect([data[1], data[2], data[3]]).toEqual([0, 0, 255]);
  });

  it("keeps skin, outlines and transparent pixels", () => {
    const skin = px(0xff, 0xd7, 0xb1, 255);
    const outline = px(10, 10, 20, 255);
    const clear = px(0, 0, 200, 0);
    for (const data of [skin, outline, clear]) {
      const before = [...data];
      dyePixels(data, red);
      expect([...data]).toEqual(before);
    }
  });

  it("keeps the art's folds: a darker garment pixel stays darker", () => {
    const lit = px(40, 80, 180, 255);
    const fold = px(20, 40, 90, 255);
    dyePixels(lit, red);
    dyePixels(fold, red);
    expect(fold[0]).toBeLessThan(lit[0]);
    expect([fold[1], fold[2], lit[1], lit[2]]).toEqual([0, 0, 0, 0]);
  });
});

describe("hsvOf", () => {
  it("unpacks all three channels", () => {
    expect(hsvOf(0x00ff00)).toEqual([1 / 3, 1, 1]);
    expect(hsvOf(0x000000)).toEqual([0, 0, 0]);
  });
});
