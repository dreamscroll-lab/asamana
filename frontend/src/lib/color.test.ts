import { describe, expect, it } from "vitest";

import { identityColor, parseHex, shade, toHex } from "./color";

describe("color", () => {
  it("parses #RRGGBB with or without the hash, and rejects anything else", () => {
    expect(parseHex("#0d9488")).toBe(0x0d9488);
    expect(parseHex(" 0d9488 ")).toBe(0x0d9488);
    expect(parseHex("#0d948")).toBeNull();
    expect(parseHex(undefined)).toBeNull();
  });

  it("shades per channel, clamped, and prints back as a hex string", () => {
    expect(shade(0x808080, 0.5)).toBe(0x404040);
    expect(shade(0xf0f0f0, 2)).toBe(0xffffff);
    expect(toHex(0x0a0b0c)).toBe("#0a0b0c");
  });
});

describe("identityColor", () => {
  it("is the world's colour when it gave one", () => {
    expect(identityColor("#112233", "agent-a")).toBe(0x112233);
  });

  it("falls back by id alone, so a colourless agent wears one colour everywhere", () => {
    expect(identityColor("", "agent-a")).toBe(identityColor(undefined, "agent-a"));
    expect(identityColor("not-a-hex", "agent-a")).toBe(identityColor(null, "agent-a"));
  });
});
