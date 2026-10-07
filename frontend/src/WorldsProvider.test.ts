import { describe, expect, it } from "vitest";

import { newestFirst } from "./WorldsProvider";
import type { WorldMeta } from "./types";

const world = (world_id: string, created_at: string | null) => ({ world_id, created_at }) as WorldMeta;

describe("newestFirst", () => {
  it("puts the newest world first and worlds with no usable time last", () => {
    const order = newestFirst([
      world("old", "2026-08-19T06:28:07.455957"),
      world("none", null),
      world("new", "2026-09-23T03:43:36.938163"),
      world("garbled", "not a date"),
      world("mid", "2026-09-19T15:30:37.870382"),
    ]).map((w) => w.world_id);
    expect(order.slice(0, 3)).toEqual(["new", "mid", "old"]);
    expect(order.slice(3).sort()).toEqual(["garbled", "none"]);
  });

  it("does not mutate the list it is given", () => {
    const input = [world("a", "2026-01-01T00:00:00"), world("b", "2026-02-01T00:00:00")];
    newestFirst(input);
    expect(input.map((w) => w.world_id)).toEqual(["a", "b"]);
  });
});
