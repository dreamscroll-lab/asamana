import { parseHex } from "./color";

/**
 * Palette for JS/SVG/canvas, mirroring the brand + neutral ramps in tailwind.config.cjs (the source
 * of truth for Tailwind classes). Use these wherever JS needs hex (Phaser via `THEME.toInt`, SVG
 * attrs, canvas); don't duplicate raw hex literals across files.
 */

// --- Brand purple ramp (mirrors tailwind.config.cjs `brand`) ---
const _brand = {
  300: "#bda6e3",
  400: "#a689d8",
  500: "#8e72cc", // soft accent
  600: "#6b4eaa", // main accent
  700: "#573f8a",
} as const;

// --- Neutral (violet-grey ground) ramp (mirrors tailwind.config.cjs `neutral`) ---
const _neutral = {
  200: "#ddd6ea",
  300: "#c4bcd6",
  400: "#a094ba", // muted text
  700: "#463a68", // subtle borders
  800: "#322850", // borders / hover fills
  900: "#201936", // panels
  950: "#150f24", // darkest ground / bg
} as const;

export const THEME = {
  brand300:   _brand[300],
  brand400:   _brand[400],
  accentSoft: _brand[500],
  accent:     _brand[600],
  accentDeep: _brand[700],

  neutral200: _neutral[200],
  neutral300: _neutral[300],
  neutral400: _neutral[400],
  neutral700: _neutral[700],
  neutral800: _neutral[800],
  panel: _neutral[900],
  bg:    _neutral[950],

  // semantic
  good:    "#0d9488", // teal  — positive affection / success
  bad:     "#e11d48", // rose  — negative affection / failure
  neutral: "#475569", // slate — neutral edge / muted

  /** Convert "#RRGGBB" → Phaser/canvas integer 0xRRGGBB. Only this file's literals come
   *  through here, so a malformed one is a typo and throws. */
  toInt(hex: string): number {
    const n = parseHex(hex);
    if (n === null) throw new Error(`THEME colour is not #RRGGBB: ${hex}`);
    return n;
  },
} as const;
