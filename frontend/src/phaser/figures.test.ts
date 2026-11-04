import { describe, expect, it } from "vitest";

import { ActionType, Phase } from "../lib/contract";
import { drawsVeil } from "./figures";

const covert = (phase: string, detected: boolean) => ({ action_type: ActionType.covert, phase, detected });

describe("drawsVeil", () => {
  it("veils him where he enters concealment, spotted or not", () => {
    expect(drawsVeil(covert(Phase.begin, false))).toBe(true);
    expect(drawsVeil(covert(Phase.settled, true))).toBe(true);
  });

  it("closes a longer act with the veil only when he was spotted, which is judged only then", () => {
    expect(drawsVeil(covert(Phase.ongoing_complete, true))).toBe(true);
    expect(drawsVeil(covert(Phase.ongoing_complete, false))).toBe(false);
  });

  it("is for covert acts only", () => {
    expect(drawsVeil({ action_type: ActionType.talk, phase: Phase.begin, detected: false })).toBe(false);
  });
});
