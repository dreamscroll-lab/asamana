/**
 * Everything the map SAYS, as opposed to shows: bubbles, plates, chips, typed dialogue, floated
 * notes. Needs little of the scene; `HudHost` below is the whole list.
 *
 * The bubble primitives (titledBubble / layoutBubble / drawPanel / plateBehind) are shared with
 * the broadcast layer, so the map has one vocabulary of surfaces rather than several overlays.
 */

import Phaser from "phaser";

import type { ActionSummary, DialogueTurn } from "../types";
import { type FxHost, faceInteraction, faceOnlooker } from "./effects";
import type { PoseName } from "./skins";
import {
  type AgentToken, FONT, NOTE_WRAP_W, PLATE_R, PLATE_RISE, TEXT_ACTION, TEXT_RES, WRAP_W, drawPlate,
} from "./types";
import { mix, shade, toHex } from "../lib/color";

// A titled bubble: a drawn panel with a nameplate in the speaker's color and a tail pointing at
// them. The nameplate hugs the FAR edge and the tail leaves from the NEAR edge (relative to the
// speaker), so the two never collide whichever way the bubble grows.
interface Bubble {
  container: Phaser.GameObjects.Container;
  panel: Phaser.GameObjects.Graphics; // shadow + box + nameplate + tail, redrawn on every layout
  title: Phaser.GameObjects.Text;
  body: Phaser.GameObjects.Text;
  up: boolean; // grows upward (default) or downward (for tokens near the top edge)
  ax: number; // anchor (token) position — layoutBubble clamps the bubble around it
  ay: number;
  floor: number; // a downward bubble starts below this y (clears the name roster)
  color: number; // the speaker's identity color — nameplate fill, and the box's border tint
  variant: keyof typeof BUBBLE_VARIANTS;
}

const TAB_BITE = 2; // the nameplate bites into the box's edge, so the two read as one fixture

// Gap between the top of the nameplate and the near edge of a bubble floated over it.
const BUBBLE_GAP = 8;

// Action chip sits just below the feet (the nameplate is above), reading as a footer.
const CHIP_Y = 16;

// Room reserved along the chip's bottom edge for the accent/progress bar, INSIDE the plate.
const BAR_LANE = 7;
// Characters an unfocused chip line shows before clipping; the feed has the full text.
const CHIP_GLANCE = 16;

// How long a floated note takes to rise and fade. A beat that makes one waits exactly this long,
// so the note is never clipped and never outstays the beat.
export const FLOAT_NOTE_MS = 3000;

// The "cut short" amber, shared by the result chip and the interrupt's floated note.
const CUT_FG = "#ecd2a2";

export const CUT_ACCENT = 0xb08a3a;

// A floated note has no wrapping, so a long headline would run off across the map.
export const NOTE_CHARS = 20;

// The near-black a floated note's plate is tinted up from, same as the map's other surfaces.
export const NOTE_GROUND = 0x0a0e1a;

// Per-character typing delay at 1×. Scaled with the dwell by the animation-speed control, or a
// talky step at 2× plays at well under 2×.
const DIALOGUE_TYPE_MS = 42;

// Color carries the channel: speech cool blue, action warm amber. `fill` is numeric because the
// box is drawn (drawPanel), the only way to get a border, shadow and tail.
// An action caption is sized with the chip under the token (TEXT_ACTION), whose report it opens,
// not with speech. Neither is italic: an italic deed reads as an aside, not the thing that happened.
export const BUBBLE_VARIANTS = {
  speech: { fill: 0x161c2b, fg: "#e9ecf3", size: "14px" },
  action: { fill: 0x2b2413, fg: "#eeddb0", size: TEXT_ACTION },
};

// Dark ink on a light plate, light ink on a dark one: identity colors span pale gold to deep slate.
export function inkOn(bg: number): string {
  const lum = 0.299 * ((bg >> 16) & 255) + 0.587 * ((bg >> 8) & 255) + 0.114 * (bg & 255);
  return lum > 140 ? "#12161f" : "#f2f5fa";
}

/**
 * The slice of the scene a readable surface needs. Deliberately no token map, room table or step:
 * deciding WHAT to draw stays with the scene.
 */
export interface HudHost extends FxHost {
  time: Phaser.Time.Clock;
  /** Track a timer for step cleanup, the way `pushEphemeral` tracks an object. */
  pushTimer(t: Phaser.Time.TimerEvent): void;
  /** An awaitable, cancellable tween — resolves even when a scrub supersedes the step. */
  tweenP(config: Phaser.Types.Tweens.TweenBuilderConfig): Promise<void>;
  /** Which render this beat belongs to; a beat that resumes after a scrub checks it. */
  readonly renderGen: number;
  /** Projected map bounds — a bubble is clamped inside the world, not the viewport. */
  readonly worldW: number;
  readonly worldH: number;
  /** Is this the subject the viewer singled out? Decides compact text vs the whole sentence. */
  isFocused(agentId: string): boolean;
  /** Put a figure into a pose for a beat. The HUD's ONLY reach into the body. */
  holdPose(tok: AgentToken, pose: PoseName, sustained?: boolean): void;
}

export class MapHud {
  constructor(private host: HudHost) {}

  // Lay a plate behind an already-positioned Text (drawn around its padding). Returned so the
  // caller can fade the two as one.
  plateBehind(t: Phaser.GameObjects.Text, fill: number, border: number, radius = PLATE_R): Phaser.GameObjects.Graphics {
    const g = this.host.add
      .graphics()
      .setScrollFactor(t.scrollFactorX, t.scrollFactorY)
      .setDepth(t.depth - 1);
    drawPlate(g, t.x - t.width * t.originX, t.y - t.height * t.originY, t.width, t.height, fill, border, radius);
    this.host.pushEphemeral(g);
    return g;
  }

  // The container's origin sits at the NEAR edge (the tail's root); the box grows away from the token.
  titledBubble(
    tok: AgentToken,
    title: string,
    titleColor: number,
    variant: keyof typeof BUBBLE_VARIANTS = "speech",
  ): Bubble {
    const v = BUBBLE_VARIANTS[variant];
    // A bubble ALWAYS floats above its speaker, except at the map's top edge where it would clip.
    // Don't flip sides to dodge neighbors: on an isometric grid that splits a conversation above
    // and below, and a slight overlap costs less than losing a constant grammar. layoutBubble clamps.
    const up = tok.container.y > 170;
    const panel = this.host.add.graphics();
    const body = this.host.add
      .text(0, 0, "", {
        fontFamily: FONT,
        fontSize: v.size,
        color: v.fg,
        padding: { x: 11, y: 8 },
        align: "left",
        lineSpacing: 4,
        wordWrap: { width: WRAP_W, useAdvancedWrap: true },
      })
      .setOrigin(0, 0)
      .setResolution(TEXT_RES);
    // No text stroke: the box lifts the words off the tilemap, and an outline silts up CJK glyphs.
    const titleText = this.host.add
      .text(0, 0, title, {
        fontFamily: FONT,
        fontSize: "12px",
        fontStyle: "bold",
        color: inkOn(titleColor),
        padding: { x: 7, y: 3 },
      })
      .setOrigin(0, 0)
      .setResolution(TEXT_RES);
    const container = this.host.add.container(tok.container.x, tok.container.y, [panel, body, titleText]).setDepth(45);
    this.host.pushEphemeral(container);
    const b: Bubble = {
      container,
      panel,
      title: titleText,
      body,
      up,
      ax: tok.container.x,
      // Upward: off the nameplate above the measured head, NOT the container origin (the feet),
      // which would put the box over the chest. Downward: from the feet.
      ay: up ? tok.container.y + tok.headY - PLATE_RISE : tok.container.y,
      floor: up ? 0 : tok.floor, // downward bubbles clear this token's name roster
      color: titleColor,
      variant,
    };
    this.layoutBubble(b);
    return b;
  }

  // Keep the whole box inside the map on all edges; re-run each time the typewriter grows the body.
  layoutBubble(b: Bubble): void {
    const MARGIN = 8;
    const bw = b.body.width, bh = b.body.height;
    const tw = b.title.width, th = b.title.height;
    const w = Math.max(bw, tw + 20); // the box is never narrower than the plate it wears
    const h = bh + th - TAB_BITE;
    // y=0 is the NEAR edge (facing the token).
    b.body.setPosition(-bw / 2, b.up ? -bh : 0);
    b.title.setPosition(-w / 2 + 10, b.up ? -bh - th + TAB_BITE : bh - TAB_BITE);
    const halfW = w / 2 + 2;
    b.container.x = Math.max(MARGIN + halfW, Math.min(this.host.worldW - MARGIN - halfW, b.ax));
    if (b.up) {
      const y = b.ay - BUBBLE_GAP;
      b.container.y = y - h < MARGIN ? h + MARGIN : y;
    } else {
      const y = Math.max(b.ay + 28, b.floor + 6); // start below the name roster
      b.container.y = y + h > this.host.worldH - MARGIN ? this.host.worldH - MARGIN - h : y;
    }
    this.drawPanel(b, w, bh, tw, th);
  }

  // Paint order matters: each fixture covers the seam the one before it left.
  //   1. the plate, its border tinted toward the speaker's color
  //   2. the nameplate in the speaker's identity color, biting into the plate's far edge
  //   3. the tail, pointing at the speaker even when the box was shoved sideways to stay on the map
  drawPanel(b: Bubble, w: number, bh: number, tw: number, th: number): void {
    const v = BUBBLE_VARIANTS[b.variant];
    const g = b.panel;
    g.clear();
    const x = -w / 2;
    const boxY = b.up ? -bh : 0;
    const border = mix(v.fill, b.color, 0.55);
    drawPlate(g, x, boxY, w, bh, v.fill, border);

    const tabY = b.up ? boxY - th + TAB_BITE : bh - TAB_BITE;
    g.fillStyle(b.color, 1);
    g.fillRoundedRect(x + 10, tabY, tw, th, b.up
      ? { tl: PLATE_R, tr: PLATE_R, bl: 0, br: 0 }
      : { tl: 0, tr: 0, bl: PLATE_R, br: PLATE_R });
    g.lineStyle(2, shade(b.color, 0.6), 1);
    g.strokeRoundedRect(x + 10, tabY, tw, th, b.up
      ? { tl: PLATE_R, tr: PLATE_R, bl: 0, br: 0 }
      : { tl: 0, tr: 0, bl: PLATE_R, br: PLATE_R });

    // Rooted 1px INSIDE the box so its fill covers the plate's outline across the tail's base.
    const dir = b.up ? 1 : -1; // +1 = the token is below us
    const tx = Phaser.Math.Clamp(b.ax - b.container.x, x + 14, x + w - 14);
    g.fillStyle(v.fill, 1);
    g.fillTriangle(tx - 7, -dir, tx + 7, -dir, tx, dir * 9);
    g.lineStyle(2, border, 1);
    g.lineBetween(tx - 7, -dir, tx, dir * 9);
    g.lineBetween(tx + 7, -dir, tx, dir * 9);
  }

  // A non-dialogue action's caption: icon + who in the title, what in the body.
  async playActionCaption(
    tok: AgentToken, act: ActionSummary, icon: string, gen: number, say?: string,
  ): Promise<void> {
    // Only a genuine in-world failure gets the mark (a non-event never reaches here).
    const failed = act.succeeded === false;
    const b = this.titledBubble(tok, `${icon} ${act.agent_name}${failed ? "  ✗" : ""}`, failed ? 0xc05a5a : tok.color, "action");
    const desc = say || act.action_description || act.gist || "";
    b.body.setText([...desc].length > 48 ? [...desc].slice(0, 48).join("") + "…" : desc);
    this.layoutBubble(b);
    if (this.host.renderGen !== gen) return;
    // Linger long enough to read, then fade rather than sit lit for a long step.
    await this.host.delayP(1300 + Math.min([...desc].length, 30) * 42);
    if (this.host.renderGen !== gen) return;
    await this.host.tweenP({ targets: b.container, alpha: 0, duration: 420, ease: "Quad.easeIn" });
  }

  // The result pill under the token: what this agent did this step and how it went. Lands after
  // the intent beat, is step-scoped (cross-step history is the feed's), and a later action this
  // step overwrites it. Empty text → no chip.
  setActionChip(
    tok: AgentToken, icon: string, text: string,
    tone: "ok" | "fail" | "cut", progress?: number, intent?: string,
  ): void {
    const trimmed = (text ?? "").trim();
    tok.chipData = trimmed ? { icon, text: trimmed, tone, progress, intent } : undefined;
    this.renderChip(tok);
  }

  // The focused subject shows full wrapped text; everyone else a compact glance. Re-run on every
  // focus change (applyFocus).
  //
  // Intent (muted, above) and result (tone color, below) are two lines, not one string: they
  // style and clip separately, and the intent is first-person while the result is third-person.
  renderChip(tok: AgentToken): void {
    tok.actionChip?.destroy();
    tok.actionChip = undefined;
    const data = tok.chipData;
    if (!data) return;
    const focused = this.host.isFocused(tok.container.getData("agentId") as string);
    // Clip per line, not over the pair, so a long intent never eats the result.
    const clip = (s: string): string => {
      const chars = [...s];
      return focused || chars.length <= CHIP_GLANCE ? s : chars.slice(0, CHIP_GLANCE).join("") + "…";
    };
    const style = {
      ok: { fg: "#e8dfc4", dim: "#9a927c", fill: 0x171c11, accent: tok.color },
      fail: { fg: "#f3bcbc", dim: "#a88585", fill: 0x2a1414, accent: 0xc05a5a },
      cut: { fg: CUT_FG, dim: "#a2916f", fill: 0x241c0e, accent: CUT_ACCENT },
    }[data.tone];
    const glyph = data.tone === "fail" ? "✗ " : "";
    const line = (text: string, color: string, top: number, bottom: number) =>
      this.host.add
        .text(0, 0, text, {
          fontFamily: FONT,
          fontSize: TEXT_ACTION,
          color,
          padding: { left: 8, right: 8, top, bottom },
          align: "left",
          ...(focused ? { wordWrap: { width: WRAP_W, useAdvancedWrap: true } } : {}),
        })
        // Left origin so the two lines share a left edge (`align` only acts within one Text).
        .setOrigin(0, 0)
        .setResolution(TEXT_RES);
    // The icon labels the whole chip, so it leads the FIRST line. ✗ marks only the result line.
    const intentText = (data.intent ?? "").trim();
    const above = intentText
      ? line(`${data.icon} ${clip(intentText)}`, style.dim, 4, 0)
      : undefined;
    const label = line(
      above ? `${glyph}${clip(data.text)}` : `${data.icon} ${glyph}${clip(data.text)}`,
      style.fg, above ? 0 : 4, 4,
    );
    const w = Math.max(20, label.width, above?.width ?? 0);
    const h = (above?.height ?? 0) + label.height + BAR_LANE;
    above?.setPosition(-w / 2, CHIP_Y);
    label.setPosition(-w / 2, CHIP_Y + (above?.height ?? 0));
    const g = this.host.add.graphics();
    // Default radius: the plate is the shared shape; the radius argument is for the event banner.
    drawPlate(g, -w / 2, CHIP_Y, w, h, style.fill, mix(style.fill, style.accent, 0.5));
    // The accent line carries success/failure color, readable at any zoom; with PROGRESS it becomes
    // a bar. Inset into the plate: below it, it reads as a stray rule belonging to nothing.
    const barW = w - 12, barX = -barW / 2, barY = CHIP_Y + h - 6;
    if (data.progress === undefined) {
      g.fillStyle(style.accent, 1);
      g.fillRect(barX, barY, barW, 2);
    } else {
      const p = Phaser.Math.Clamp(data.progress, 0, 1);
      g.fillStyle(0x2a3242, 1);
      g.fillRect(barX, barY, barW, 3);
      g.fillStyle(style.accent, 1);
      g.fillRect(barX, barY, Math.max(2, barW * p), 3);
    }
    const chip = this.host.add.container(0, 0, above ? [g, above, label] : [g, label]);
    tok.container.add(chip); // child → auto-follows moves, destroyed with the token
    tok.actionChip = chip;
  }

  /**
   * A floated note for a MOMENT (a thing wrecked, a state turned, an act cut short): rises and fades.
   *
   * Uses the shared plate, not `Text.backgroundColor` (see drawPlate). `accent` does all the
   * tinting, so a new kind of note cannot bring a palette of its own.
   */
  floatNote(x: number, y: number, text: string, accent: number): void {
    const note = this.host.add
      .text(x, y, text, {
        fontFamily: FONT,
        fontSize: "10px",
        color: toHex(mix(accent, 0xffffff, 0.55)),
        padding: { x: 8, y: 4 },
        align: "left",
        wordWrap: { width: NOTE_WRAP_W, useAdvancedWrap: true },
      })
      // Bottom-anchored so extra lines grow upward instead of over what the note annotates.
      .setOrigin(0.5, 1)
      .setDepth(41)
      .setResolution(TEXT_RES);
    this.host.pushEphemeral(note);
    const plate = this.plateBehind(note, mix(NOTE_GROUND, accent, 0.14), mix(NOTE_GROUND, accent, 0.55));
    this.host.tweens.add({
      targets: [note, plate],
      y: "-=22",
      alpha: 0,
      duration: FLOAT_NOTE_MS,
      ease: "Quad.easeOut",
    });
  }

  // Each turn types out on its speaker's token, dwells, then fades as the next begins. Awaited:
  // the step does not finish until the exchange does.
  //
  // `party` is everyone in the exchange, the counterpart right after the record's actor. Each
  // speaker and whoever he answers (the last speaker, else the first other) face each other; the
  // rest of the party look at the speaker.
  async playDialogue(
    turns: DialogueTurn[],
    fallback: AgentToken,
    party: string[],
    gen: number,
  ): Promise<void> {
    let prev: Bubble | null = null;
    let lastSpeaker = "";
    // EVERY turn: don't cap here, a silent cut is invisible to the viewer. Conversation length
    // is the decision layer's limit to set.
    for (const turn of turns) {
      if (this.host.renderGen !== gen) return;
      const { speaker, line } = turn;
      if (!line) continue;
      const spk = this.host.token(turn.speaker_id) ?? fallback;
      const others = [...new Set(party)].filter((id) => id !== turn.speaker_id);
      const addressee = others.includes(lastSpeaker) ? lastSpeaker : others[0];
      lastSpeaker = turn.speaker_id;
      if (addressee) faceInteraction(this.host, spk, [addressee]);
      for (const id of others) {
        const ot = this.host.token(id);
        if (id !== addressee && ot) faceOnlooker(this.host, ot, [turn.speaker_id]);
      }
      // Pose per line, not once per exchange, so the bodies take turns with the talk.
      this.host.holdPose(spk, "talk", true);
      if (prev) this.host.tweens.add({ targets: prev.container, alpha: 0, duration: 240 });
      const b = this.speechBubble(spk, speaker);
      prev = b;
      const chars = [...line];
      await this.typeP(b, chars);
      if (this.host.renderGen !== gen) return;
      await this.host.delayP(700 + Math.min(chars.length, 24) * 40); // read time
    }
    if (prev) this.host.tweens.add({ targets: prev.container, alpha: 0, duration: 300 });
  }

  speechBubble(tok: AgentToken, speaker: string): Bubble {
    const b = this.titledBubble(tok, speaker, tok.color);
    this.layoutBubble(b);
    return b;
  }

  // Resolves when the whole line is shown, or immediately if superseded.
  typeP(b: Bubble, chars: string[]): Promise<void> {
    return new Promise((resolve) => {
      const render = (n: number) => {
        b.body.setText(chars.slice(0, n).join(""));
        this.layoutBubble(b);
      };
      if (!chars.length) {
        render(0);
        resolve();
        return;
      }
      render(0);
      let idx = 0;
      let timer: Phaser.Time.TimerEvent;
      const cancel = () => {
        timer?.remove(false);
        render(chars.length);
        resolve();
      };
      this.host.addCanceler(cancel);
      timer = this.host.time.addEvent({
        delay: DIALOGUE_TYPE_MS,
        repeat: chars.length - 1,
        callback: () => {
          idx += 1;
          render(idx);
          if (idx >= chars.length) {
            this.host.removeCanceler(cancel);
            resolve();
          }
        },
      });
      this.host.pushTimer(timer);
    });
  }

}
