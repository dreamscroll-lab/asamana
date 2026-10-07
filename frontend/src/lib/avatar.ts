import type { CSSProperties } from "react";

import { identityColor, shade, toHex } from "./color";

// A world's icon gradient, picked by a stable hash of its id. Class strings are literals so
// Tailwind's JIT emits them. Characters don't use these: their avatars are their identity colour.
const AVATAR_GRADIENTS = [
  "from-violet-600 via-indigo-600 to-indigo-400",
  "from-rose-600 via-red-600 to-amber-500",
  "from-amber-600 via-yellow-600 to-yellow-400",
  "from-emerald-600 via-teal-600 to-cyan-500",
];

export function avatarGradient(id: string): string {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h + id.charCodeAt(i)) % AVATAR_GRADIENTS.length;
  return AVATAR_GRADIENTS[h];
}

// Inline gradient from a character's identity colour (light→base), so the avatar matches the map
// token and the graph node, the fallback colour included.
export function avatarStyle(color: string | null | undefined, id: string): CSSProperties {
  const base = identityColor(color, id);
  return { backgroundImage: `linear-gradient(to top right, ${toHex(shade(base, 1.28))}, ${toHex(shade(base, 0.82))})` };
}

export function emotionColor(v: number | null): string {
  if (v == null) return "text-slate-400";
  if (v > 0.15) return "text-teal-400";
  if (v < -0.15) return "text-rose-400";
  return "text-amber-400";
}
