import { describe, expect, it } from "vitest";

import { actionDescriptionOf, isWholeWorld, partsOf, sharedOf, viewerOf } from "./viewer";
import type { ActionSharedView, ActionSummary, WorldMeta } from "../types";

const meta = (actor: string | null, exited = false) =>
  ({ actor_character_id: actor, actor_exited: exited }) as WorldMeta;

const withAudience = (audience: string[]) => ({ shared: { audience } }) as unknown as ActionSummary;

// "a" did it, "b" was in it.
const act = {
  initiator_id: "a",
  shared: { audience: ["a", "b"] } as ActionSharedView,
  private: [
    { agent_id: "a", action_description: "甲的打算", inner_monologue: "甲的念头" },
    { agent_id: "b", action_description: "乙被叫住", inner_monologue: "" },
  ],
} as unknown as ActionSummary;

describe("viewerOf", () => {
  it("演员在场时看的是所演的角色", () => {
    expect(viewerOf(meta("a1"))).toBe("a1");
  });

  it("没人选角或演员已退出,就是导演", () => {
    expect(viewerOf(meta(null))).toBeNull();
    expect(viewerOf(meta("a1", true))).toBeNull();
    expect(viewerOf(null)).toBeNull();
  });
});

describe("sharedOf", () => {
  it("在 audience 里才知其中", () => {
    expect(sharedOf(withAudience(["a1", "b"]), "a1")).not.toBeNull();
    expect(sharedOf(withAudience([]), "a1")).toBeNull();
  });

  it("导演对每件事都知其中", () => {
    expect(sharedOf(withAudience([]), null)).not.toBeNull();
  });

  it("已滤去的共享层就是不知", () => {
    expect(sharedOf({ shared: null } as unknown as ActionSummary, null)).toBeNull();
  });
});

describe("partsOf / actionDescriptionOf", () => {
  it("每人只读得到自己那份", () => {
    expect(partsOf(act, "b").map((p) => p.agent_id)).toEqual(["b"]);
    expect(actionDescriptionOf(act, "b")).toBe("");
  });

  it("行动者知道自己的打算,导演都知道", () => {
    expect(actionDescriptionOf(act, "a")).toBe("甲的打算");
    expect(actionDescriptionOf(act, null)).toBe("甲的打算");
    expect(partsOf(act, null)).toHaveLength(2);
  });
});

describe("isWholeWorld", () => {
  it("只有导演看到的是整个世界", () => {
    expect(isWholeWorld(null)).toBe(true);
    expect(isWholeWorld("a1")).toBe(false);
  });
});
