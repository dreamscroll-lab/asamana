// Color arithmetic shared by the map and the React views — one parser, one shade, one blend.

/** "#RRGGBB" (or "RRGGBB") → 0xRRGGBB, or null when absent or malformed. */
export function parseHex(hex: string | null | undefined): number | null {
  if (!hex) return null;
  const m = /^#?([0-9a-fA-F]{6})$/.exec(hex.trim());
  return m ? parseInt(m[1], 16) : null;
}

// For an agent whose state carries no color (the read model sends "" for one seen only through an
// action). Keyed off the immutable id, so the same person wears it on the map and in every DOM
// avatar, step after step. Don't key it off a position in this step's list.
const FALLBACK_IDENTITY_COLORS = [0x7c83ff, 0x3fb98f, 0xe0607a, 0x59b0e6, 0xd98cf0, 0xe0a94f, 0x6fae7a];

/** An agent's identity color: its own when the world gave one, else the id's fallback. */
export function identityColor(color: string | null | undefined, id: string): number {
  const own = parseHex(color);
  if (own !== null) return own;
  let hash = 0;
  for (let i = 0; i < id.length; i++) hash = (hash * 31 + id.charCodeAt(i)) >>> 0;
  return FALLBACK_IDENTITY_COLORS[hash % FALLBACK_IDENTITY_COLORS.length];
}

export function toHex(c: number): string {
  return `#${c.toString(16).padStart(6, "0")}`;
}

// Scales each channel (clamped) to fake iso lighting: top face lightest, side faces darker.
export function shade(c: number, f: number): number {
  const r = Math.min(255, Math.round(((c >> 16) & 255) * f));
  const g = Math.min(255, Math.round(((c >> 8) & 255) * f));
  const b = Math.min(255, Math.round((c & 255) * f));
  return (r << 16) | (g << 8) | b;
}

// Linear blend of two 0xRRGGBB colors (t=0 → a, t=1 → b).
export function mix(a: number, b: number, t: number): number {
  const ar = (a >> 16) & 255, ag = (a >> 8) & 255, ab = a & 255;
  const br = (b >> 16) & 255, bg = (b >> 8) & 255, bb = b & 255;
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const bl = Math.round(ab + (bb - ab) * t);
  return (r << 16) | (g << 8) | bl;
}
