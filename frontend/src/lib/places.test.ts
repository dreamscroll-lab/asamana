import { describe, expect, it, vi } from "vitest";

import { placeTipLines } from "./places";

vi.mock("../api/client", () => ({ api: { places: vi.fn() } }));

describe("place hover card", () => {
  it("leads with the name and adds the description under it", () => {
    expect(placeTipLines("太极宫", "宫城正殿")).toEqual([
      { text: "太极宫", tone: "head" }, { text: "宫城正殿", tone: "background" },
    ]);
  });

  it("is the name alone for a place without a description", () => {
    expect(placeTipLines("内院", "")).toEqual([{ text: "内院", tone: "head" }]);
  });
});
