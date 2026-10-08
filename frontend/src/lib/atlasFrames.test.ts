/** Reading frame rects from a Starling atlas description. */

import { describe, expect, it } from "vitest";

import { parseAtlasFrames } from "./atlasFrames";

const SHIPPED = `<TextureAtlas imagePath="tangCommonerYoungMale.png">
\t<SubTexture name="idleNE" x="0" y="0" width="192" height="256"/>
\t<SubTexture name="idleSE" x="192" y="0" width="192" height="256"/>
</TextureAtlas>`;

describe("读帧矩形", () => {
  it("现货那份的写法:制表符缩进、自闭合、外层标签不算一帧", () => {
    expect(parseAtlasFrames(SHIPPED)).toEqual({
      idleNE: { x: 0, y: 0, width: 192, height: 256 },
      idleSE: { x: 192, y: 0, width: 192, height: 256 },
    });
  });

  it("属性顺序无关", () => {
    const frames = parseAtlasFrames('<SubTexture y="1" name="a" height="4" width="3" x="2"/>');
    expect(frames.a).toEqual({ x: 2, y: 1, width: 3, height: 4 });
  });

  it("多余属性不碍事", () => {
    const frames = parseAtlasFrames(
      '<SubTexture name="a" x="0" y="0" width="1" height="1" frameX="-3" frameWidth="9"/>',
    );
    expect(frames.a).toEqual({ x: 0, y: 0, width: 1, height: 1 });
  });

  it("没名字的、坐标不是数的都丢掉 —— 宁可说缺这一帧,也不要画一个空盒子", () => {
    expect(parseAtlasFrames('<SubTexture x="0" y="0" width="1" height="1"/>')).toEqual({});
    expect(parseAtlasFrames('<SubTexture name="a" x="abc" y="0" width="1" height="1"/>')).toEqual({});
    expect(parseAtlasFrames('<SubTexture name="a" x="0" y="0"/>')).toEqual({});
  });

  it("重名以最后一条为准", () => {
    const frames = parseAtlasFrames(
      '<SubTexture name="a" x="0" y="0" width="1" height="1"/>' +
        '<SubTexture name="a" x="9" y="0" width="1" height="1"/>',
    );
    expect(frames.a.x).toBe(9);
  });

  it("读不懂就是没有帧,不抛 —— 面板要能说「读不到」而不是把工作台带崩", () => {
    expect(parseAtlasFrames("")).toEqual({});
    expect(parseAtlasFrames("<html>nope</html>")).toEqual({});
  });
});
