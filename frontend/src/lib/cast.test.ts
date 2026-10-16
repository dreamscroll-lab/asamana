import { describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { loadCast } from "./cast";
import type { CharacterIdentity } from "../types";

vi.mock("../api/client", () => ({ api: { agents: vi.fn() } }));

const candidate = (name: string): CharacterIdentity => ({
  agent_id: "same-id", name, gender: "女", age: 30, role: "旅人", background: "远方来客",
  appearance: "长发", color: "#123456",
});

describe("world cast", () => {
  it("shares in-flight requests and reuses identities across repeated loads", async () => {
    let finish!: (value: CharacterIdentity[]) => void;
    vi.mocked(api.agents).mockReturnValueOnce(new Promise((resolve) => { finish = resolve; }));
    const first = loadCast("shared");
    expect(loadCast("shared")).toBe(first);
    finish([candidate("甲")]);
    const cast = await first;
    expect(await loadCast("shared")).toBe(cast);
    expect(api.agents).toHaveBeenCalledTimes(1);
    expect(cast["same-id"]).toEqual(candidate("甲"));
    expect(Object.isFrozen(cast)).toBe(true);
  });

  it("isolates the same character id in different worlds", async () => {
    vi.mocked(api.agents).mockResolvedValueOnce([candidate("乙")]).mockResolvedValueOnce([candidate("丙")]);
    const [left, right] = await Promise.all([loadCast("left"), loadCast("right")]);
    expect(left["same-id"].name).toBe("乙");
    expect(right["same-id"].name).toBe("丙");
  });

  it("allows a failed request to be retried without caching a false empty cast", async () => {
    vi.mocked(api.agents).mockRejectedValueOnce(new Error("Offline")).mockResolvedValueOnce([candidate("丁")]);
    await expect(loadCast("retry")).rejects.toThrow("Offline");
    expect((await loadCast("retry"))["same-id"].name).toBe("丁");
  });

  it("caches a successfully empty world", async () => {
    vi.mocked(api.agents).mockResolvedValueOnce([]);
    const empty = await loadCast("empty");
    expect(await loadCast("empty")).toBe(empty);
    expect(empty).toEqual({});
  });
});
