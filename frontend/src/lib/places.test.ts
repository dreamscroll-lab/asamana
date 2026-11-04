import { describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { loadPlaces, reloadPlaces, placeTipLines } from "./places";

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


describe("world place identities", () => {
  it("replaces cached draft places with the confirmed identities", async () => {
    vi.mocked(api.places).mockResolvedValueOnce([
      { location_id: "hall", name: "草稿大殿", description: "待定" },
    ]);
    const draft = await loadPlaces("confirmed-places");
    vi.mocked(api.places).mockResolvedValueOnce([
      { location_id: "hall", name: "正殿", description: "议事之所" },
    ]);
    const confirmed = await reloadPlaces("confirmed-places");
    expect(confirmed).not.toBe(draft);
    expect(confirmed.hall.name).toBe("正殿");
    expect(await loadPlaces("confirmed-places")).toBe(confirmed);
  });
});
