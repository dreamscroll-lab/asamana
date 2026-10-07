/** An entity's hover card: head, appearance and content lines, each with its own tone. */

import { describe, expect, it } from "vitest";

import { entityTipLines } from "./entity";

const letter = { name: "密信", state: "intact", description: "一封火漆封口的信", content: "三日后子时，玄武门" };

describe("物的悬停卡", () => {
  it("描述与内容都印,内容加引号、身份是 content", () => {
    expect(entityTipLines(letter)).toEqual([
      { text: "密信", tone: "head" },
      { text: "一封火漆封口的信", tone: "background" },
      { text: "「三日后子时，玄武门」", tone: "content" },
    ]);
  });

  it("状态非默认才进抬头", () => {
    expect(entityTipLines({ ...letter, state: "已拆封" })[0].text).toBe("密信 · 已拆封");
  });

  it("没有内容就没有那一行,不印空引号", () => {
    expect(entityTipLines({ ...letter, content: "" }).map((l) => l.tone)).toEqual(["head", "background"]);
    expect(entityTipLines({ name: "刀", state: "intact", description: "", content: "" }).map((l) => l.text)).toEqual(["刀"]);
  });
});
