import { describe, expect, it } from "vitest";
import { operationTarget } from "./operationTarget";
const selectable = () => true;
const npc = (id: string) => id === "servant";

describe("right-click operation target", () => {
  it("prioritizes held item rows over the overlapping pocket and holder", () => {
    expect(operationTarget([{ agentId: "owner" }, { pocketPanel: true }, { entityId: "letter", entityKind: "item" }], "room", selectable, npc)).toEqual({ kind: "item", id: "letter" });
  });
  it("does not select the owner through an empty pocket area", () => {
    expect(operationTarget([{ pocketPanel: true }, { agentId: "owner" }], "room", selectable, npc)).toBeNull();
  });
  it("distinguishes agents, NPCs and landmarks", () => {
    expect(operationTarget([{ agentId: "servant" }], "room", selectable, npc)).toEqual({ kind: "npc", id: "servant" });
    expect(operationTarget([{ agentId: "character" }], "room", selectable, npc)).toEqual({ kind: "agent", id: "character" });
    expect(operationTarget([{ entityId: "gate", entityKind: "landmark" }], "room", selectable, npc)).toEqual({ kind: "landmark", id: "gate" });
  });
  it("falls back to a location and does not target arriving figures", () => {
    expect(operationTarget([{ agentId: "arriving" }], "room", () => false, npc)).toEqual({ kind: "location", id: "room" });
    expect(operationTarget([], null, selectable, npc)).toBeNull();
  });
});
