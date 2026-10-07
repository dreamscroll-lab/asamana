/**
 * What the frontend still decides about an NPC's look: the order of the two sentences in a line,
 * and whether the line is worth printing. Wording is the backend's, so a Chinese assertion beyond
 * the fixture strings means Chinese has leaked back to this side.
 */

import { describe, expect, it } from "vitest";

import type { NpcStateSummary } from "../types";
import { npcStandingLine, npcTipLines } from "./npc";

function npc(over: Partial<NpcStateSummary> = {}): NpcStateSummary {
  return {
    npc_id: "npc_x", name: "东宫典膳", location: "东宫", location_id: "loc-east-palace", color: "#8a93a6",
    gender: "男", age: 45, description: "掌酒食采办", condition: "",
    outcome: "", ongoing: false,
    ...over,
  };
}

describe("状态行", () => {
  it("什么也没发生就不出声 —— 一具闲着的身体每步一行会埋掉整条流", () => {
    expect(npcStandingLine(npc())).toBe("");
  });

  it("点名那一步,光是在场就值一行", () => {
    expect(npcStandingLine(npc(), true)).toBe("东宫典膳（东宫）");
  });

  it("地点只出现一次 —— 它由这一侧加,后端那句里没有", () => {
    expect(npcStandingLine(npc({ outcome: "把话带到了" }))).toBe("东宫典膳（东宫）把话带到了");
  });

  it("被制住的排在前 —— 它解释了他为什么没在动", () => {
    const line = npcStandingLine(npc({ condition: "被按在地上", outcome: "正往宫门去" }));
    expect(line.indexOf("被按在地上")).toBeLessThan(line.indexOf("正往宫门去"));
  });

  it("两样各自都足以让这一行出现,不必凑齐", () => {
    for (const one of [{ condition: "被按在地上" }, { outcome: "正往回走" }]) {
      expect(npcStandingLine(npc(one))).not.toBe("");
    }
  });
});

describe("悬停卡", () => {
  const text = (n = npc()) => npcTipLines(n).map((l) => l.text);
  const tone = (n = npc()) => npcTipLines(n).map((l) => l.tone);

  it("档次是第一件说的事 —— 卡上别的每一项角色也都有", () => {
    expect(text()[0]).toBe("NPC · 东宫典膳（男·45岁）");
  });

  it("性别年龄缺了就省掉括号,不渲染空壳", () => {
    expect(text(npc({ gender: "", age: null }))[0]).toBe("NPC · 东宫典膳");
  });

  it("空项一律丢掉", () => {
    expect(text(npc({ description: "", outcome: "", condition: "" })))
      .toEqual(["NPC · 东宫典膳（男·45岁）"]);
  });

  it("三种事实各带各的身份 —— 一贯如此的、此刻在发生的、身上被加的,不许长成一个样", () => {
    const n = npc({ outcome: "正往宫门去当众传句话", condition: "被按在地上" });
    expect(tone(n)).toEqual(["head", "background", "now", "held"]);
  });
});
