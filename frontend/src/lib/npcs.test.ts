import { describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import type { NpcIdentity } from "../types";
import { loadNpcs, reloadNpcs } from "./npcs";

vi.mock("../api/client", () => ({ api: { npcs: vi.fn() } }));

const identity = (name: string): NpcIdentity => ({
  npc_id: "npc-1", name, gender: "男", age: 40, description: "门房", color: "#8a93a6",
});

describe("world NPC identities", () => {
  it("shares requests and keeps the table when no current step contains the NPC", async () => {
    let finish!: (rows: NpcIdentity[]) => void;
    vi.mocked(api.npcs).mockReturnValueOnce(new Promise((resolve) => { finish = resolve; }));
    const pending = loadNpcs("shared-npcs");
    expect(loadNpcs("shared-npcs")).toBe(pending);
    finish([identity("甲")]);
    const table = await pending;
    expect(table["npc-1"]).toEqual(identity("甲"));
    expect(await loadNpcs("shared-npcs")).toBe(table);
    expect(api.npcs).toHaveBeenCalledTimes(1);
  });

  it("isolates identities by world and retries a failed load", async () => {
    vi.mocked(api.npcs).mockRejectedValueOnce(new Error("Offline"));
    await expect(loadNpcs("left-npcs")).rejects.toThrow("Offline");
    vi.mocked(api.npcs).mockResolvedValueOnce([identity("乙")]).mockResolvedValueOnce([identity("丙")]);
    const [left, right] = await Promise.all([loadNpcs("left-npcs"), loadNpcs("right-npcs")]);
    expect(left["npc-1"].name).toBe("乙");
    expect(right["npc-1"].name).toBe("丙");
  });

  it("replaces any draft identities with the confirmed roster", async () => {
    vi.mocked(api.npcs).mockResolvedValueOnce([identity("草稿门房")]);
    const draft = await loadNpcs("confirm-npcs");
    vi.mocked(api.npcs).mockResolvedValueOnce([identity("确认后的管事")]);
    const confirmed = await reloadNpcs("confirm-npcs");
    expect(confirmed["npc-1"].name).toBe("确认后的管事");
    expect(confirmed).not.toBe(draft);
    expect(await loadNpcs("confirm-npcs")).toBe(confirmed);
  });


  it.each([false, true])("ignores a draft response finishing after confirmation (failure=%s)", async (fail) => {
    let finish!: (rows: NpcIdentity[]) => void;
    let reject!: (error: Error) => void;
    vi.mocked(api.npcs).mockReturnValueOnce(new Promise((resolve, failRequest) => {
      finish = resolve;
      reject = failRequest;
    }));
    const world = `overlapping-confirm-${fail}`;
    const draft = loadNpcs(world);
    const settledDraft = draft.catch(() => undefined);
    vi.mocked(api.npcs).mockResolvedValueOnce([identity("确认后的管事")]);
    const confirmed = await reloadNpcs(world);
    if (fail) reject(new Error("Late draft failure"));
    else finish([identity("旧草稿门房")]);
    await settledDraft;
    expect(await loadNpcs(world)).toBe(confirmed);
    expect(confirmed["npc-1"].name).toBe("确认后的管事");
  });

});
