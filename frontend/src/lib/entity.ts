import type { EntityView } from "../types";
import type { TipLine } from "./tip";

/**
 * The backend's default state (world/models.py), which its prompts skip (core/prompts.py
 * render_entity). The map skips it too, so a pocket doesn't repeat "intact" for every item.
 */
export const DEFAULT_ENTITY_STATE = "intact";

/**
 * The map is the god's-eye view, so content is printed as-is: who can read it in the world is the
 * backend's concern, and the observer can read everything. Content is quoted as a whole and kept
 * apart from the description: one says what it looks like, the other what is written on it.
 */
export function entityTipLines(e: Pick<EntityView, "name" | "state" | "description" | "content">): TipLine[] {
  const state = (e.state ?? "").trim();
  const head = state && state !== DEFAULT_ENTITY_STATE ? `${e.name} · ${state}` : e.name;
  const content = (e.content ?? "").trim();
  return ([
    { text: head, tone: "head" },
    { text: (e.description ?? "").trim(), tone: "background" },
    { text: content ? `「${content}」` : "", tone: "content" },
  ] as TipLine[]).filter((line) => line.text.length > 0);
}
