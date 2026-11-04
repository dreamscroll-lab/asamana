import { describe, expect, it } from "vitest";
import type { ActorOperationCapability } from "../../types";
import { missingParameters } from "./shortcuts";
const capability = (operation: ActorOperationCapability["operation"], required_parameters: string[]): ActorOperationCapability => ({ operation, required_parameters, available: true, reason: null });
describe("shortcut form requirements", () => {
  it("lets complete actions submit without additional description", () => {
    expect(missingParameters(capability("move", []), {}, "")).toBe(false);
    expect(missingParameters(capability("broadcast", ["message"]), { message: "大家留步" }, "")).toBe(false);
  });
  it("requires broadcast content and an errand description", () => {
    expect(missingParameters(capability("broadcast", ["message"]), { message: "  " }, "")).toBe(true);
    expect(missingParameters(capability("errand", ["description"]), {}, "")).toBe(true);
    expect(missingParameters(capability("errand", ["description"]), {}, "去院子看看")).toBe(false);
  });
  it("uses backend requirements rather than the operation name", () => {
    expect(missingParameters(capability("physical", []), {}, "")).toBe(false);
    expect(missingParameters(capability("physical", ["description"]), {}, "")).toBe(true);
    expect(missingParameters(capability("broadcast", []), {}, "")).toBe(false);
    expect(missingParameters(capability("broadcast", ["message"]), {}, "")).toBe(true);
    expect(missingParameters(capability("errand", []), {}, "")).toBe(false);
  });
  it("accepts false as a supplied boolean value", () => {
    expect(missingParameters(capability("errand", ["announce"]), { announce: false }, "")).toBe(false);
    expect(missingParameters(capability("errand", ["announce"]), { announce: true }, "")).toBe(false);
  });
  it("requires a concrete description for the custom physical operation", () => {
    expect(missingParameters(capability("physical", ["description"]), {}, " ")).toBe(true);
    expect(missingParameters(capability("physical", ["description"]), {}, "轻轻敲三下门")).toBe(false);
  });
});
