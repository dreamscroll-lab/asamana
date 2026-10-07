import type { NpcStateSummary } from "../types";
import type { TipLine } from "./tip";

/**
 * How a body without a mind reads to a watcher: one renderer for the map's hover card and the
 * feed's standing row, so the two can't drift.
 *
 * No Chinese about a runner lives on this side: `condition` and `outcome` arrive as whole sentences
 * and print verbatim; this module only decides order and whether a row is worth printing. Don't
 * ship a kind just to reword it here: that is a second copy of the backend's vocabulary.
 *
 * The one thing added is the place, which is safe because a beat always happens where the body
 * ends the step (see `NpcStateSummary.outcome`).
 *
 * Don't add a standing mark on the figure: a direction glyph would only tell which leg of a job he
 * is on, a mechanism detail, at the cost of a permanent mark on up to six figures competing with the
 * condition pip and the pouch handle. His walking, his slate colour and the feed row already say it.
 */

/**
 * One feed line: who, where, and what happened to him this step. Empty when nothing did, or idle
 * bodies would bury the feed.
 *
 * `roll` lifts that floor for the opening roll call, where being here is the news (otherwise these
 * would be the only figures the reader is never told about): an idle body prints as name and place.
 */
export function npcStandingLine(npc: NpcStateSummary, roll = false): string {
  const outcome = npc.outcome.trim();
  const held = npc.condition.trim();
  if (!outcome && !held && !roll) return "";
  const name = npc.name.trim() || "某人";
  const at = npc.location.trim();
  // Being held comes first: it explains why he isn't moving.
  const what = [held, outcome].filter(Boolean).join("，");
  return at ? `${name}（${at}）${what}` : `${name}${what}`;
}

/**
 * The hover card's lines, each tagged with the kind of fact it is (see `TipTone`).
 *
 * The kinds must not look alike: who he always is and what he is doing now, in one colour, read as
 * one paragraph.
 */
export function npcTipLines(npc: NpcStateSummary): TipLine[] {
  const bits = [npc.gender.trim(), npc.age === null ? "" : `${npc.age}岁`].filter(Boolean);
  const head = bits.length ? `NPC · ${npc.name}（${bits.join("·")}）` : `NPC · ${npc.name}`;
  return ([
    { text: head, tone: "head" },
    { text: npc.description.trim(), tone: "background" },
    { text: npc.outcome.trim(), tone: "now" },
    { text: npc.condition.trim(), tone: "held" },
  ] as TipLine[]).filter((line) => line.text.length > 0);
}
