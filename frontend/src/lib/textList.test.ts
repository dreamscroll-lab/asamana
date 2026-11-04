import { describe, expect, it } from "vitest";

import { joinList, splitList } from "./textList";

describe("splitList", () => {
  it("splits on every separator a Chinese or English line may use", () => {
    expect(splitList("果决、多疑，隐忍,好胜")).toEqual(["果决", "多疑", "隐忍", "好胜"]);
  });

  it("drops blank items", () => {
    expect(splitList(" 、果决、 ，")).toEqual(["果决"]);
  });

  it("round-trips through joinList", () => {
    expect(splitList(joinList(["父子:父亲", "君臣:君"]))).toEqual(["父子:父亲", "君臣:君"]);
  });
});
