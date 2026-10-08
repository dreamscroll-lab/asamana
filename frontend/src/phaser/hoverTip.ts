/** The hover card over a body with no mind, or over a thing. What it shows, and why hover, is on `show`. */

import Phaser from "phaser";

import type { EntityView, NpcStateSummary } from "../types";
import { entityTipLines } from "../lib/entity";
import { npcTipLines } from "../lib/npc";
import type { TipLine, TipTone } from "../lib/tip";
import { type AgentToken, FONT, NOTE_WRAP_W, TEXT_RES, drawPlate } from "./types";

// Tier chip plate: achromatic, so only the gray text inside it carries color.
const TIP_TAG_PAD_X = 4;
const TIP_TAG_PAD_Y = 2;
const TIP_TAG_GAP = 5;
const TIP_TAG_FILL = 0x232a3a;
const TIP_TAG_EDGE = 0x3b4358;
// Above the whole sorted cast band (isoDepth.SORT_BASE..+SORT_SPAN): a thing lies on the
// ground, so a card hung off its own marker would be drawn under anybody standing south of it.
const FREE_TIP_DEPTH = 60;

// One look per kind of fact, so the card doesn't read as one gray block. `held` matches the
// condition mark by the head; `content` (writing on paper) is warm on cool.
const TIP_TONE: Record<TipTone, { fontSize: string; color: string }> = {
  head:       { fontSize: "11px", color: "#e8ecf5" },
  background: { fontSize: "9px",  color: "#7c8699" },
  now:        { fontSize: "10px", color: "#cfe3f5" },
  held:       { fontSize: "10px", color: "#e0a94f" },
  content:    { fontSize: "10px", color: "#efe0b8" },
};

export class HoverTip {
  private tip: Phaser.GameObjects.Container | null = null;

  constructor(private host: Phaser.Scene) {}

  /**
   * The hover card over an NPC: who he is and what he is up to. Hover, not the nameplate:
   * the plate stays ONE line, and this is asked for one figure at a time.
   *
   * Agents get no hover card — the character card already owns those facts.
   */
  show(tok: AgentToken, npc: NpcStateSummary): void {
    this.hide();
    const card = this.build(npcTipLines(npc), { text: "NPC", color: npc.color });
    if (!card) return;
    card.setPosition(0, tok.headY - 16);
    tok.container.add(card);
    this.tip = card;
  }

  /**
   * The same card over a thing on the ground: what it is, what it looks like, what it carries.
   * Free-standing at the marker rather than its child — see FREE_TIP_DEPTH.
   */
  showEntity(x: number, y: number, entity: EntityView): void {
    this.hide();
    const card = this.build(entityTipLines(entity));
    if (!card) return;
    this.tip = card.setPosition(x, y).setDepth(FREE_TIP_DEPTH);
  }

  hide(): void {
    this.tip?.destroy();
    this.tip = null;
  }

  /** The card, bottom-centered on its own origin. Null when there is nothing to say. */
  private build(
    lines: TipLine[], chip?: { text: string; color: string },
  ): Phaser.GameObjects.Container | null {
    if (!lines.length) return null;
    const texts = lines.map((line) =>
      this.host.add
        .text(0, 0, line.text, {
          ...TIP_TONE[line.tone], fontFamily: FONT,
          wordWrap: { width: NOTE_WRAP_W, useAdvancedWrap: true },
        })
        .setOrigin(0, 0)
        .setResolution(TEXT_RES),
    );
    // ONE width for every card: shrink-to-fit makes the card jump as the cursor moves down a row.
    const w = NOTE_WRAP_W + 16;
    const h = texts.reduce((sum, t) => sum + t.height + 3, 0) + 10;
    const left = -w / 2 + 8;
    const top = -h + 6;
    const parts: Phaser.GameObjects.GameObject[] = [];
    let indent = 0;
    if (chip) {
      // The tier as a chip, not a word in the sentence: it says which channel drew this figure,
      // not who he is. It wears the figure's own gray, matching garment and ground ring.
      const tag = this.host.add
        .text(0, 0, chip.text, { fontFamily: FONT, fontSize: "9px", color: chip.color })
        .setOrigin(0, 0)
        .setResolution(TEXT_RES);
      const tagW = Math.ceil(tag.width) + TIP_TAG_PAD_X * 2;
      const tagH = Math.ceil(tag.height) + TIP_TAG_PAD_Y * 2;
      indent = tagW + TIP_TAG_GAP;
      // Centered on the header line: the chip is taller than its 9px text, so aligning tops hangs it above the name.
      const tagY = top + (texts[0].height - tagH) / 2;
      const tagPlate = this.host.add.graphics();
      tagPlate.fillStyle(TIP_TAG_FILL, 1).fillRoundedRect(left, tagY, tagW, tagH, 3);
      tagPlate.lineStyle(1, TIP_TAG_EDGE, 1).strokeRoundedRect(left, tagY, tagW, tagH, 3);
      tag.setPosition(left + TIP_TAG_PAD_X, tagY + TIP_TAG_PAD_Y);
      parts.push(tagPlate, tag);
    }
    let y = top;
    texts.forEach((t, i) => {
      t.setPosition(left + (i === 0 ? indent : 0), y);
      y += t.height + 3;
    });
    const plate = this.host.add.graphics();
    drawPlate(plate, -w / 2, -h, w, h, 0x151a26, 0x3b4358);
    return this.host.add.container(0, 0, [plate, ...parts, ...texts]);
  }
}
