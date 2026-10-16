import { describe, expect, it } from "vitest";

import { isWholeWorld, knowsInside, viewerOf } from "./viewer";
import type { ActionSummary, WorldMeta } from "../types";

const meta = (actor: string | null, exited = false) =>
  ({ actor_character_id: actor, actor_exited: exited }) as WorldMeta;

const withAudience = (audience: string[]) => ({ private: { audience } }) as unknown as ActionSummary;

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

describe("knowsInside", () => {
  it("在 audience 里才算亲历", () => {
    expect(knowsInside(withAudience(["a1", "b"]), "a1")).toBe(true);
    expect(knowsInside(withAudience([]), "a1")).toBe(false);
  });

  it("导演对每件事都算亲历", () => {
    expect(knowsInside(withAudience([]), null)).toBe(true);
  });
});

describe("isWholeWorld", () => {
  it("只有导演看到的是整个世界", () => {
    expect(isWholeWorld(null)).toBe(true);
    expect(isWholeWorld("a1")).toBe(false);
  });
});
