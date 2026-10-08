/**
 * What a person IS between beats: the condition laid on him, and the things in his hands.
 * (`effects.ts` draws what the world just did; `hud.ts` what someone just did or said.)
 *
 *   - written every step, never animated or cleared between steps (clearEphemeral would
 *     close a pocket mid-read);
 *   - READ ON DEMAND, so it owns a click and the panel it opens;
 *   - must not shout: prose or item lists on every nameplate crowd the step's own beat.
 *
 * Two tiers, by how much the viewer asked:
 *
 *   nothing asked  →  the MARK beside his head: an icon survives zoom-out where 9px CJK
 *                     does not, and most people have no condition, so it usually costs nothing.
 *   he is clicked  →  the POCKET: the condition in full, then what he carries and its state.
 *
 * No middle tier of words on the nameplate; see `sync`. AT MOST ONE POCKET IS OPEN on the map
 * (`openId`). It doesn't hold the cast: `refresh` is handed the registry, as body.ts is.
 */

import Phaser from "phaser";

import { mix, toHex } from "../lib/color";
import { DEFAULT_ENTITY_STATE } from "../lib/entity";
import { type FxHost } from "./effects";
import {
  type AgentToken, type CarriedItem, FONT, TEXT_RES, drawPlate,
} from "./types";

// A standing condition's amber, shared with the feed chip and character card. Never gold
// (carried things) and never red (death).
const COND_INK = 0xc08a3e;
const PIP_FILL = 0xd9a04a;
const PIP_EDGE = 0x4a2f10;
const PIP_R = 3.5;
// The mark rides beside the HEAD, at a fraction of the figure's own head height so it fits
// any body size. Don't lower it to the chest: there it reads as something worn or held, and
// a beat's effects play across that space.
const PIP_X = 11;
const PIP_HEAD_F = 0.9;
// The mark's slow breath — see `pulse`.
const PIP_BREATH_MS = 900;
const PIP_ALPHA_FLOOR = 0.62;

// Gold, as the item gems in the world wear.
const ITEM_INK = "#e0c079";

// The handle: a pointing hand, not a chevron (a ▸ reads as a widget pasted over the world).
// ☟ when open, because the panel hangs BELOW the name.
const HAND_SHUT = "☞";
const HAND_OPEN = "☟";

// The panel sits right of the nameplate, clear of the figure, pose and chip, and DROPPED
// below the name line: level with its handle, it would swallow the click meant to close it.
const PANEL_X = 15;
const PANEL_Y = 11;
const PANEL_W = 138;
const PANEL_PAD = 6;
const ROW_GAP = 3;
const RULE_GAP = 9; // room the hairline between the two blocks sits in
const TEXT_W = PANEL_W - PANEL_PAD * 2;

// Under each thing: its description, dimmed, then its content in the hover card's paper tone
// (a held thing has no hover card; this is it).
const ITEM_DESC_INK = "#7c8699";
const ITEM_CONTENT_INK = "#efe0b8";

/** The slice of the scene this layer needs: focus and the carry ledger, both read-only. */
export interface PocketHost extends FxHost {
  isFocused(agentId: string): boolean;
  /** Is the viewer watching ANYONE? Decides whether "not focused" means "background". */
  focusActive(): boolean;
  readonly carriedBy: Map<string, CarriedItem[]>;
}

export class PocketLayer {
  /** Whose pocket is open, if anyone's — the single-open rule. */
  private openId: string | null = null;

  constructor(private host: PocketHost) {}

  /**
   * The token's two fixtures, built but NOT placed (the scene places them). The MARK must
   * hang on the BODY: the plate is LOD-gated away on zoom-out, and the condition must stay
   * visible. The HANDLE is plate furniture and shares the plate's fate.
   */
  mount(agentId: string, headY: number, nameHalfWidth: number): {
    pip: Phaser.GameObjects.Ellipse;
    pouch: Phaser.GameObjects.Text;
  } {
    const pip = this.host.add
      .ellipse(PIP_X, headY * PIP_HEAD_F, PIP_R * 2, PIP_R * 2, PIP_FILL)
      .setStrokeStyle(1.5, PIP_EDGE)
      .setVisible(false);
    // At the name's right shoulder, not a plate row of its own that would stay lit over every holder.
    const pouch = this.host.add
      .text(nameHalfWidth + 5, 0, "", { fontFamily: FONT, fontSize: "10px", color: "#d9a441" })
      .setOrigin(0, 0.5)
      .setResolution(TEXT_RES)
      .setStroke("#0a0e1a", 3)
      .setVisible(false)
      .setInteractive({ useHandCursor: true });
    pouch.setData("pocketOf", agentId);
    return { pip, pouch };
  }

  /**
   * Make the token's mark and handle agree with the world. One call, because both depend on
   * the same inputs (condition, items, focus); split, the handle's glyph goes stale.
   *
   * `text` omitted → re-apply from the cache (focus can change with no step in flight).
   *
   * Don't put the condition's sentence on the watched man's nameplate: unbounded prose
   * stacks over heads and repeats once the pocket opens. The MARK is always on (the premise
   * must not need asking for); the WORDS live in the pocket (and in the narrative feed).
   *
   * The HANDLE is hidden on background figures, and shows whenever the pocket has anything.
   * Don't gate it on items alone: a bound, empty-handed man would have a mark nobody can open.
   */
  sync(agentId: string, tok: AgentToken, text?: string): void {
    if (text !== undefined) tok.condText = text;
    const background = this.host.focusActive() && !this.host.isFocused(agentId);
    const items = this.host.carriedBy.get(agentId) ?? [];

    tok.condPip.setVisible(!!tok.condText);

    if (background || (!items.length && !tok.condText)) {
      tok.pouch.setVisible(false);
      return;
    }
    // Bare hand when the pocket holds only a condition — no "✦0".
    const hand = tok.pocket ? HAND_OPEN : HAND_SHUT;
    tok.pouch.setText(items.length ? `✦${items.length} ${hand}` : hand).setVisible(true);
  }

  /**
   * A slow breath on the mark, so a condition reads as ONGOING, not a decal. Gentler than any
   * beat, so it never competes with the step. Called per frame, per token.
   */
  pulse(tok: AgentToken, now: number): void {
    if (!tok.condPip.visible) return;
    const breath = Math.abs(Math.sin(now / PIP_BREATH_MS + tok.phase));
    tok.condPip.setAlpha(PIP_ALPHA_FLOOR + (1 - PIP_ALPHA_FLOOR) * breath);
  }

  /**
   * Everything this layer draws is stale. Order matters: rebuild the open panel FIRST, so
   * the handle's glyph reads a pocket that survived or was emptied away by `build`.
   */
  refresh(tokens: Iterable<[string, AgentToken]>): void {
    for (const [id, tok] of tokens) {
      if (tok.pocket) {
        this.build(id, tok);
        if (!tok.pocket) this.openId = null;
      }
      this.sync(id, tok);
    }
  }

  /** Open this man's pocket, closing whatever was open before. */
  toggle(agentId: string): void {
    const tok = this.host.token(agentId);
    if (!tok) return;
    const reopen = !tok.pocket;
    this.close();
    if (reopen) {
      this.build(agentId, tok);
      this.openId = tok.pocket ? agentId : null;
    }
    this.sync(agentId, tok); // the handle's open/shut glyph
  }

  /**
   * Focus changed: close a panel on a man no longer watched (he dims to background). Only
   * while a focus is ACTIVE; clearing focus leaves a deliberately opened pocket open.
   */
  onFocusChanged(focusActive: boolean): void {
    if (focusActive && this.openId && !this.host.isFocused(this.openId)) this.close();
  }

  /**
   * Close the open pocket — any, or only `agentId`'s. Also used when something else takes the
   * stage (post bubble, death visual).
   */
  close(agentId?: string): void {
    if (!this.openId || (agentId && agentId !== this.openId)) return;
    const owner = this.openId;
    this.openId = null;
    const tok = this.host.token(owner); // gone already (a cleared corpse) → nothing to undo
    if (!tok) return;
    tok.pocket?.destroy();
    tok.pocket = undefined;
    this.sync(owner, tok); // the panel is gone → the handle shows shut
  }

  /**
   * The panel: the condition in full, then what he holds and each thing's state. Rebuilt,
   * never patched — it is a view of two live ledgers. A child of the token and deliberately
   * NOT ephemeral, or it would close every step.
   *
   * Nothing to show → no panel, so an open pocket empties itself away; the caller reads
   * `tok.pocket` back to find out.
   */
  private build(agentId: string, tok: AgentToken): void {
    tok.pocket?.destroy();
    tok.pocket = undefined;
    const items = this.host.carriedBy.get(agentId) ?? [];
    if (!items.length && !tok.condText) return;

    const rows: Phaser.GameObjects.Text[] = [];
    const row = (text: string, color: string): Phaser.GameObjects.Text =>
      this.host.add
        .text(0, 0, text, {
          // Smaller than the 10px NAME it hangs under.
          fontFamily: FONT, fontSize: "9px", color,
          wordWrap: { width: TEXT_W, useAdvancedWrap: true },
          lineSpacing: 2,
        })
        .setOrigin(0, 0)
        .setResolution(TEXT_RES);
    // The condition wears the head mark's amber bead; carried things the gems' gold star —
    // differing in shape AND color, since they answer different questions.
    if (tok.condText) rows.push(row(`● ${tok.condText}`, toHex(COND_INK)));
    for (const it of items) {
      // State rides the thing's own row; a DEFAULT state says nothing, so the one that matters stands out.
      const worth = it.state && it.state !== DEFAULT_ENTITY_STATE;
      // A HOLLOW star (no third ink) for a thing nobody around him can perceive; the suffix
      // says it in words.
      const hidden = it.concealed === true;
      const label = worth ? `${it.name} · ${it.state}` : it.name;
      rows.push(row(hidden ? `✧ ${label}（不公开）` : `✦ ${label}`, ITEM_INK));
      const desc = (it.description ?? "").trim();
      const content = (it.content ?? "").trim();
      if (desc) rows.push(row(desc, ITEM_DESC_INK));
      if (content) rows.push(row(`「${content}」`, ITEM_CONTENT_INK));
    }

    // A rule between the two blocks: at this size the bead and star alone don't part them.
    const split = tok.condText ? 1 : 0;
    let y = PANEL_PAD;
    let ruleY = 0;
    rows.forEach((r, i) => {
      if (split && i === split) {
        ruleY = y + RULE_GAP / 2;
        y += RULE_GAP;
      }
      r.setPosition(PANEL_PAD, y);
      y += r.height + ROW_GAP;
    });
    const h = y - ROW_GAP + PANEL_PAD;

    const g = this.host.add.graphics();
    // The holder's own color, like the ✉ badge, so an open panel is never orphaned in a crowd.
    const border = mix(PANEL_GROUND, tok.color, 0.55);
    drawPlate(g, 0, 0, PANEL_W, h, mix(PANEL_GROUND, tok.color, 0.16), border);
    if (ruleY) {
      g.lineStyle(1, border, 0.6);
      g.lineBetween(PANEL_PAD, ruleY, PANEL_W - PANEL_PAD, ruleY);
    }

    const panel = this.host.add.container(PANEL_X, tok.headY + PANEL_Y, [g, ...rows]);
    // Its own hit area, as the post badge: a click falling through would re-select the man.
    panel.setSize(PANEL_W, h).setInteractive({ useHandCursor: false });
    panel.setData("pocketPanel", agentId);
    tok.container.add(panel);
    tok.pocket = panel;
  }
}

// The near-black every map surface is tinted from (hud.ts's NOTE_GROUND; not imported, to
// avoid tying this layer to the bubble module for one number).
const PANEL_GROUND = 0x0a0e1a;
