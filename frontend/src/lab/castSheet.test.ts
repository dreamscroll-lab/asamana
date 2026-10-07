/** Deriving the frame sheet: what each cell should show, and how the missing ones are reported. */

import { describe, expect, it } from "vitest";

import type { FrameRect } from "../lib/atlasFrames";
import { bodyCells, frameStyle, poseRow, unusedFrames } from "./castSheet";
import { characterSetFrom } from "../phaser/skins";

const rect: FrameRect = { x: 0, y: 0, width: 192, height: 256 };
const rects = Object.fromEntries(
  [
    "idleNE", "idleSE", "idleSW", "idleNW",
    "duckSE",
    "walk0", "walk1",
    "spare",
  ].map((name) => [name, rect]),
);

function cast(overrides: Record<string, unknown> = {}) {
  return characterSetFrom({
    scale: 0.28,
    bodies: { male: { child: "b", young: "b", middle: "b", elder: "b" }, female: {} },
    atlases: { b: { image: "b.png", atlas: "b.xml" } },
    frames: {
      idle: { NE: "idleNE", SE: "idleSE", SW: "idleSW", NW: "idleNW" },
      duck: "duckSE",
      walk: ["walk0", "walk1"],
      talk: { NE: "idleNE", SE: "idleSE", SW: "idleSW" },
      hold: "neverDrawn",
      ...overrides,
    },
  });
}

describe("一行 pose", () => {
  it("清单里没有这个 pose:四格都是「没映射」—— 渲染器到这儿会静默什么都不画", () => {
    const row = poseRow(cast(), "b", "attack", rects);
    expect(Object.values(row).map((c) => c.kind)).toEqual([
      "unmapped", "unmapped", "unmapped", "unmapped",
    ]);
  });

  it("只给一个帧名:四格同名,朝西两格镜像", () => {
    const row = poseRow(cast(), "b", "duck", rects);
    expect(row.NE).toEqual({ kind: "frames", frames: [{ name: "duckSE", rect }], flip: false });
    expect(row.SE).toMatchObject({ flip: false });
    expect(row.SW).toMatchObject({ flip: true });
    expect(row.NW).toMatchObject({ flip: true });
  });

  it("逐向给:四格都不镜像 —— 翻转一个画好的朝向正是这种写法要防的事", () => {
    const row = poseRow(cast(), "b", "idle", rects);
    expect(Object.values(row).every((c) => c.kind === "frames" && !c.flip)).toBe(true);
  });

  it("逐向表少一个方向:少的那格没映射,其余照常", () => {
    const row = poseRow(cast(), "b", "talk", rects);
    expect(row.NW.kind).toBe("unmapped");
    expect(row.NE.kind).toBe("frames");
  });

  it("帧名图集里没有:说出是哪个名字,别画个空盒子", () => {
    expect(poseRow(cast(), "b", "hold", rects).SE).toEqual({
      kind: "missing",
      names: ["neverDrawn"],
    });
  });

  it("多帧按清单里的先后给全", () => {
    const cell = poseRow(cast(), "b", "walk", rects).NE;
    expect(cell.kind === "frames" && cell.frames.map((f) => f.name)).toEqual(["walk0", "walk1"]);
  });
});

describe("身形八格", () => {
  it("清单没给的那一格报 null —— 这正是 bodyFor 答不了的问题", () => {
    const cells = bodyCells(cast());
    expect(cells).toHaveLength(8);
    expect(cells.filter((c) => c.gender === "male").every((c) => c.atlas === "b")).toBe(true);
    expect(cells.filter((c) => c.gender === "female").every((c) => c.atlas === null)).toBe(true);
  });
});

describe("图集里没人要的帧", () => {
  it("画了却没写进清单的,数得出来", () => {
    expect(unusedFrames(cast(), "b", rects)).toEqual(["spare"]);
  });
});

describe("一帧怎么摆", () => {
  // Geometry of the shipped assets: 192×256 frames, a 1536×2560 atlas, 0.28× in game.
  const sheet = { width: 1536, height: 2560 };
  const walk = [0, 192, 384, 576].map((x) => ({ x, y: 512, width: 192, height: 256 }));

  it("缩的是图集不是框:一个 pose 的每一帧都共用同一张重采样过的图", () => {
    const sizes = new Set(walk.map((r) => frameStyle(r, sheet, 0.28, false, "b.png").clip.backgroundSize));
    expect(sizes.size).toBe(1);
    expect([...sizes][0]).toBe("432px 720px");
  });

  it("逐帧的偏移落在整像素上 —— 小数相位每帧不同,边缘就会逐帧爬", () => {
    expect(walk.map((r) => frameStyle(r, sheet, 0.28, false, "b.png").clip.backgroundPosition)).toEqual([
      "-0px -144px",
      "-54px -144px",
      "-108px -144px",
      "-162px -144px",
    ]);
  });

  it("框就是它最后占的大小,里外一致", () => {
    const { box, clip } = frameStyle(walk[0], sheet, 0.28, false, "b.png");
    expect(box).toMatchObject({ width: 54, height: 72 });
    expect(clip).toMatchObject({ width: 54, height: 72 });
    expect(clip.imageRendering).toBe("auto");
  });

  it("放大时不平滑,原寸不动图", () => {
    const one = frameStyle(walk[0], sheet, 1, false, "b.png");
    expect(one.clip.backgroundSize).toBe("1536px 2560px");
    expect(one.clip.imageRendering).toBe("pixelated");
  });

  it("镜像只是把这一格翻过来,不参与缩放", () => {
    expect(frameStyle(walk[0], sheet, 0.28, true, "b.png").clip.transform).toBe("scaleX(-1)");
    expect(frameStyle(walk[0], sheet, 0.28, false, "b.png").clip.transform).toBeUndefined();
  });
});
