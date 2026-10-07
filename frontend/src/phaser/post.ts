/**
 * A step's post on the map: letters reaching their readers, the one post that opens itself, and the
 * ✉ badge everyone else is left with. Staged by reader, not by letter — see `playDeliveries`.
 */

import Phaser from "phaser";

import { mix, toHex } from "../lib/color";
import type { MessageSummary } from "../types";
import type { Announcer } from "./announce";
import { type FxHost, arrivingMessage } from "./effects";
import { BUBBLE_VARIANTS, type MapHud, NOTE_GROUND } from "./hud";
import type { PocketLayer } from "./pocket";
import type { PoseName } from "./skins";
import { type AgentToken, FONT, PLATE_GAP, PLATE_RISE, TEXT_RES, WRAP_W, drawPlate } from "./types";

// A letter nobody signed: the same slate arrivingMessage flies its ✉？ in, so in the air and
// opened it looks the same.
const UNSIGNED_INK = 0x8a94a8;

// How many of one reader's letters are quoted before the rest are counted; past this the
// news is HOW MANY, not which.
const LETTER_LINES = 3;

/** The slice of the scene the post needs, on top of the shared FX one. */
export interface PostHost extends FxHost {
  tweenP(config: Phaser.Types.Tweens.TweenBuilderConfig): Promise<void>;
  readonly renderGen: number;
  focusActive(): boolean;
  isFocused(agentId: string): boolean;
  holdPose(tok: AgentToken, pose: PoseName, sustained?: boolean): void;
}

export class PostLayer {
  constructor(
    private host: PostHost,
    private hud: MapHud,
    private pockets: PocketLayer,
    private announcer: Announcer,
  ) {}

  /**
   * Letters DELIVERED this step. A letter has two beats, one step apart:
   *
   *   step N   send — directedMessages: the ✉ crosses the city. SPATIAL; no words.
   *   step N+1 read — HERE: the letter opens on its reader with what it says. EPISTEMIC.
   *
   * Either end may be missing from the map, and neither absence may silence it:
   *
   *   sender on map, recipient on map  → an arc between them (at the SEND step)
   *   NO SENDER (narrator, or off-cast / dead) → it ARRIVES from beyond the edge, here
   *   NO RECIPIENT (off-cast, or dead) → it DEPARTS past the edge, out of sight
   *
   * Staged by READER, not by letter: one letter to many readers must not become N identical
   * private letters, and many letters to one reader is one event, not a stack of plates.
   * The bubble belongs to the reader, and each reader opens his post exactly once.
   */
  async playDeliveries(messages: MessageSummary[], gen: number): Promise<void> {
    // On the map and alive. NOT focus: focus decides only whether a post opens itself (below).
    // The dead can't read — a ✉ on a corpse asserts a cognition that did not happen.
    const canRead = (rid: string) => {
      const tok = this.host.token(rid);
      return !!tok && !tok.dead;
    };
    const flights: Promise<void>[] = [];
    // reader → every `direct` letter addressed to him; a letter to five lands in all five piles.
    // Crowding is handled below by how many piles open themselves, never by withholding mail.
    const post = new Map<string, MessageSummary[]>();
    // Announcements heard this step, in arrival order; they address rooms, not heads.
    const heard: {
      content: string; location_scope: string | null; speaker: string;
      from: { name: string; ink: number };
    }[] = [];
    for (const m of messages) {
      const readers = m.receiver_ids.filter(canRead);
      if (!readers.length) continue;
      // The flight, for letters that left no hand (a signed one flew a step ago), all at once.
      // FOCUS-GATED, unlike the badge: a state readout is for everyone, a timed beat belongs
      // to the subject you asked to watch (as in playTrack).
      if (!this.host.token(m.sender_id)) {
        for (const rid of readers) {
          if (this.host.focusActive() && !this.host.isFocused(rid)) continue;
          flights.push(arrivingMessage(this.host, this.host.token(rid)!));
        }
      }
      const said = String(m.perceived_summary ?? "").trim();
      if (!said) continue;
      // For every scope the send step draws space (directedMessages / localAnnounce) and the
      // delivery draws the content. An announcement's audience is a room or the world, so it
      // lands there with the speaker's name; don't badge every listener — that turns one
      // proclamation into N private letters (see the `scope` contract). Badges are `direct` only.
      if (m.scope !== "direct") {
        const from = { name: m.sender_name, ink: this.host.token(m.sender_id)?.color ?? UNSIGNED_INK };
        heard.push({
          content: said, location_scope: m.scope === "place" ? m.place_id : null,
          speaker: m.sender_id, from,
        });
        continue;
      }
      for (const rid of readers) (post.get(rid) ?? post.set(rid, []).get(rid)!).push(m);
    }

    if (flights.length) await Promise.all(flights);
    if (this.host.renderGen !== gen) return;
    // Badges FIRST: they cost no screen time, so "who got word" is legible for the whole step.
    for (const [rid, letters] of post) {
      const tok = this.host.token(rid)!;
      tok.postData = letters.map((m) => ({
        from: m.sender_name,
        ink: this.host.token(m.sender_id)?.color ?? UNSIGNED_INK,
        said: String(m.spoken ?? m.perceived_summary ?? "").trim(),
      }));
      this.renderPostBadge(tok);
    }
    for (const h of heard) {
      if (this.host.renderGen !== gen) return;
      // Pose the speaker BEFORE the words open, so they read as one act. Only when he is still
      // in the room the words landed in: delivery lags the utterance a step, and an errand-bearer
      // may already be walking home. (A world broadcast has null `location_scope`, so no pose.)
      const mouth = this.host.token(h.speaker);
      if (mouth && h.location_scope && mouth.locationId === h.location_scope) this.host.holdPose(mouth, "talk");
      await this.announcer.playBroadcast({ content: h.content, location_scope: h.location_scope }, h.from);
    }
    // EXACTLY ONE post opens itself; the rest wait to be clicked. Not zero: the map is watched
    // far more than clicked. Not all: that queues the whole step behind plates. Without focus
    // the biggest pile opens. Under a focus it is a HARD filter, not a sort: otherwise an
    // unfocused stranger's letter opens when the subject got none (same gate as playTrack).
    const openable = this.host.focusActive() ? [...post.keys()].filter((r) => this.host.isFocused(r)) : [...post.keys()];
    const opener = openable.sort((a, b) => post.get(b)!.length - post.get(a)!.length)[0];
    if (opener !== undefined && this.host.renderGen === gen) {
      await this.openPost(this.host.token(opener)!);
    }
  }

  /**
   * A reader's post, opened — the EPISTEMIC beat of the letters in it (see playDeliveries).
   *
   * The same bubble spoken lines use: titled with the READER, one coloured heading + body per
   * letter up to LETTER_LINES, then a count of the rest.
   */
  private async openPost(tok: AgentToken): Promise<void> {
    const letters = tok.postData ?? [];
    if (!letters.length) return;
    const shown = letters.slice(0, LETTER_LINES);
    // The badge hides while its post is open: it is the closed form of the same thing, and
    // hiding it is the whole of the double-open guard.
    const badge = tok.postBadge;
    badge?.setVisible(false);
    // …and an open pocket closes: one panel at a time, and the letters are what just happened.
    this.pockets.close();

    // Titled with the READER, as every bubble names its subject (see playActionCaption). Not
    // the sender: that reads as the sender talking from the reader's feet. Senders are named
    // inside, by each paragraph's coloured heading.
    const b = this.hud.titledBubble(tok, `✉ ${tok.label.text}`, tok.color);
    // Phaser has no rich text, so each coloured heading is its own Text.
    const parts: Phaser.GameObjects.Text[] = [];
    for (const l of shown) {
      parts.push(this.host.add.text(0, 0, l.from, {
        fontFamily: FONT, fontSize: "12px", fontStyle: "bold",
        color: toHex(mix(l.ink, 0xffffff, 0.35)),
        padding: { x: 11, y: 0 },
      }).setOrigin(0, 0).setResolution(TEXT_RES));
      parts.push(this.host.add.text(0, 0, l.said, {
        fontFamily: FONT, fontSize: "14px", color: BUBBLE_VARIANTS.speech.fg,
        padding: { x: 11, y: 0 }, align: "left", lineSpacing: 4,
        wordWrap: { width: WRAP_W, useAdvancedWrap: true },
      }).setOrigin(0, 0).setResolution(TEXT_RES));
    }
    if (letters.length > shown.length) {
      parts.push(this.host.add.text(0, 0, `…另 ${letters.length - shown.length} 封`, {
        fontFamily: FONT, fontSize: "12px", color: "#8894a8", padding: { x: 11, y: 0 },
      }).setOrigin(0, 0).setResolution(TEXT_RES));
    }
    // layoutBubble sizes the panel off `body` alone, so `body` is a blank sizing shim for the
    // measured stack. Place the rows only AFTER layout: it moves the body's origin.
    const offs: number[] = [];
    let y = 6;
    let w = 0;
    for (const p of parts) {
      offs.push(y);
      y += p.height + (p.style.fontSize === "12px" ? 1 : 6);
      w = Math.max(w, p.width);
    }
    b.body.setText(" ").setFixedSize(w, y);
    b.container.add(parts);
    this.hud.layoutBubble(b);
    parts.forEach((p, i) => p.setPosition(b.body.x, b.body.y + offs[i]));
    // Same read-time curve as the action caption, measured on what is shown.
    const chars = shown.reduce((n, l) => n + [...l.said].length, 0);
    await this.host.delayP(1300 + Math.min(chars, 60) * 42);
    await this.host.tweenP({ targets: b.container, alpha: 0, duration: 420, ease: "Quad.easeIn" });
    // Only if it still exists: clearEphemeral may have destroyed it when the step ended.
    if (badge?.scene) badge.setVisible(true);
  }

  /**
   * The ✉ badge: what a delivery leaves behind, and the only mark an unopened reader gets.
   * Step-scoped like the action chip (see setActionChip); a child of the token.
   *
   * NOT a red dot: red is reserved for the dead mark (DEAD_MARK_FILL), and a red dot implies
   * UNREAD, a state this map does not keep.
   */
  private renderPostBadge(tok: AgentToken): void {
    tok.postBadge?.destroy();
    tok.postBadge = undefined;
    const letters = tok.postData ?? [];
    if (!letters.length) return;
    // The READER's colour: the badge hangs on him; who wrote is inside.
    const ink = tok.color;
    const label = this.host.add
      .text(0, 0, letters.length > 1 ? `✉ ${letters.length}` : "✉", {
        fontFamily: FONT, fontSize: "12px", color: toHex(mix(ink, 0xffffff, 0.55)),
        padding: { x: 5, y: 2 },
      })
      .setOrigin(0.5, 0)
      .setResolution(TEXT_RES);
    const g = this.host.add.graphics();
    drawPlate(g, -label.width / 2, 0, label.width, label.height, mix(NOTE_GROUND, ink, 0.18), mix(NOTE_GROUND, ink, 0.6));
    const badge = this.host.add.container(0, tok.headY - PLATE_GAP - PLATE_RISE - 16, [g, label]);
    // Its own hit area: a click falling through to the token would select the agent instead.
    badge
      .setSize(label.width, label.height)
      .setInteractive({ useHandCursor: true })
      .on("pointerdown", (p: Phaser.Input.Pointer, _x: number, _y: number, e: Phaser.Types.Input.EventData) => {
        e.stopPropagation();
        p.event.stopPropagation();
        void this.openPost(tok);
      });
    tok.container.add(badge);
    tok.postBadge = badge;
  }
}
