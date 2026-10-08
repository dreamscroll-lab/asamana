/**
 * The kind of a hover-card line, which decides how it looks (see phaser/hoverTip.ts).
 *
 * The tone travels with the line and is set by the module that writes it (npc.ts / entity.ts):
 * what kind of fact a line is belongs to the line; the scene only turns a tone into pixels.
 *   head        who/what it is: the heading
 *   background  what it always is: the backdrop, unchanging
 *   now         what is happening right now
 *   held        what has been imposed on it
 *   content     what it carries: the words in a letter, the inscription on a stele
 */
export type TipTone = "head" | "background" | "now" | "held" | "content";

export interface TipLine {
  text: string;
  tone: TipTone;
}
