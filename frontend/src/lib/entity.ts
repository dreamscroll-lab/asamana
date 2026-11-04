import type { EntityView } from "../types";
import type { TipLine } from "./tip";

/**
 * The backend's default state (core/interfaces/perception.py), which its prompts skip (core/prompts.py
 * render_entity). The map skips it too, so a pocket doesn't repeat "intact" for every item.
 */
export const DEFAULT_ENTITY_STATE = "intact";

/**
 * The map is the god's-eye view, so content (the holder's private layer) is printed as-is: who
 * can read it in the world is settled by where the backend puts it, and the observer reads
 * everything. Content is quoted as a whole and kept apart from the description: one says what it
 * looks like, the other what is written on it.
 */
export function entityTipLines(e: Pick<EntityView, "name" | "state" | "description" | "private">): TipLine[] {
  const state = (e.state ?? "").trim();
  const head = state && state !== DEFAULT_ENTITY_STATE ? `${e.name} · ${state}` : e.name;
  const content = e.private.content.trim();
  return ([
    { text: head, tone: "head" },
    { text: (e.description ?? "").trim(), tone: "background" },
    { text: content ? `「${content}」` : "", tone: "content" },
  ] as TipLine[]).filter((line) => line.text.length > 0);
}
