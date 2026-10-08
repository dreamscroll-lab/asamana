/**
 * The FIGURE: poses, headings, the walk cycle, garment dye, head height, falling down, and the
 * vitality bar. Not where a person stands (the scene's staging) nor what the world did (effects.ts).
 *
 * A class because it owns state: head heights measured once per BODY, shared by every token
 * wearing it.
 *
 * Deliberately does NOT know which tokens exist or how an agent id maps to a body: methods that
 * need the token map take it as an argument. Holding it would let this module start making
 * decisions about the cast and turn into a second scene.
 */

import Phaser from "phaser";

import { Phase, actedOnAgents } from "../lib/contract";
import { hsvOf } from "../lib/dye";
import type { ActionSummary, AgentStateSummary } from "../types";
import { deedBody } from "./deedPose";
import { type FxHost, deathDissipate } from "./effects";
import { RECOLOR_FX, RecolorFX } from "./recolorPipeline";
import { type CharacterSet, type Dir, DIRECTIONS, POSE_NAMES, type PoseName } from "./skins";
import {
  type AgentToken, DEAD_MARK_EDGE, DEAD_MARK_FILL, DEAD_MARK_PAD_X, DEAD_MARK_PAD_Y,
  DEAD_MARK_R, FONT, INITIAL_DIR, PLATE_GAP, SPRITE_FOOT_Y, TEXT_RES, drawPlate,
} from "./types";
import { reportedDead } from "../lib/agentState";

const DEAD_COLOR = 0x5a606b; // grayed-out token for a fallen agent

// The vitality bar shows only below this, so it never reads as an always-on game HUD.
const VIT_SHOW_BELOW = 0.85;

// Cadence for cycled poses other than walk and swing (talking, working). Slow: a fast flicker
// reads as agitation rather than occupation.
const POSE_LOOP_FPS = 4;

// Matches animKey's walk keys; playWalk uses it to carry the stride.
const WALK_KEY = /_walk_(NE|SE|SW|NW)$/;

/** The slice of the scene a body needs: `FxHost` plus the cast and two more Phaser managers. */
export interface BodyHost extends FxHost {
  readonly cast: CharacterSet;
  anims: Phaser.Animations.AnimationManager;
  textures: Phaser.Textures.TextureManager;
}

export class BodyRig {
  /** Measured head heights, per BODY (not per token) — see measureHeadY. */
  private headYOf = new Map<string, number>();

  constructor(private host: BodyHost) {}

  animKey(body: string, pose: string, dir: Dir): string {
    return `${body}_${pose}_${dir}`;
  }

  /**
   * Register an animation for EVERY pose the art gives more than one frame for, not just walk
   * and swing. Which poses cycle is the ART's business; hard-coding a list would draw shipped
   * loops (a talking mouth, working hands) as stills.
   */
  buildCharacterAnims(): void {
    for (const body of this.host.cast.bodyKeys()) {
      if (!this.host.textures.exists(body)) continue; // art missing → nothing to register
      for (const dir of DIRECTIONS) {
        for (const pose of POSE_NAMES) {
          const art = this.host.cast.poseArt(body, pose, dir);
          if (!art || art.frames.length < 2) continue; // a still needs no anim
          this.host.anims.create({
            key: this.animKey(body, pose, dir),
            frames: art.frames.map(([texture, frame]) => ({ key: texture, frame })),
            // Walk: manifest cadence. Swing: one fast pass, no loop. Everything else: unhurried.
            frameRate: pose === "walk" ? this.host.cast.walkFps : pose === "attack" ? 14 : POSE_LOOP_FPS,
            repeat: pose === "attack" ? 0 : -1,
          });
        }
      }
    }
  }

  /**
   * How far this body's head reaches above its feet, in container-local pixels (negative).
   *
   * Measured off the art's pixels: a manifest number drifts out of sync with the drawing (bubbles
   * through chests), and the frame height has slack above the head (fixtures float). Measures
   * the idle frame so the nameplate does not bob when the figure ducks.
   */
  measureHeadY(body: string): number {
    const cached = this.headYOf.get(body);
    if (cached !== undefined) return cached;
    const art = this.host.cast.poseArt(body, "idle", INITIAL_DIR);
    const frame = art && this.host.textures.getFrame(art.frames[0][0], art.frames[0][1]);
    // No art (or no canvas to read it with) → fall back to the frame box, which is too tall
    // rather than too short: a fixture floating a little high still clears the head.
    let rise = frame ? frame.height : 0;
    const source = frame?.source.image as HTMLImageElement | HTMLCanvasElement | undefined;
    const ctx = source && document.createElement("canvas").getContext("2d", { willReadFrequently: true });
    if (frame && source && ctx) {
      ctx.canvas.width = frame.width;
      ctx.canvas.height = frame.height;
      ctx.drawImage(source, frame.cutX, frame.cutY, frame.width, frame.height, 0, 0, frame.width, frame.height);
      const { data } = ctx.getImageData(0, 0, frame.width, frame.height);
      for (let y = 0; y < frame.height; y++) {
        let opaque = false;
        for (let x = 0; x < frame.width && !opaque; x++) opaque = data[(y * frame.width + x) * 4 + 3] > 8;
        if (opaque) { rise = frame.height - y; break; }
      }
    }
    const headY = SPRITE_FOOT_Y - rise * this.host.cast.scale;
    this.headYOf.set(body, headY);
    return headY;
  }

  // Show one of the body's poses at the token's heading: cycled poses play, single frames are
  // stills. Mirroring is the ART's call (art.flip), not the caller's. A pose the art lacks
  // leaves the figure unchanged.
  applyPose(tok: AgentToken, pose: PoseName): void {
    const art = this.host.cast.poseArt(tok.skinId, pose, tok.dir);
    if (!art) return;
    const key = this.animKey(tok.skinId, pose, tok.dir);
    if (this.host.anims.exists(key)) {
      // Don't restart a running cycle: a multi-step job re-applies its pose every tick, and
      // restarting would stutter once per step.
      tok.sprite.play(key, true);
    } else {
      tok.sprite.stop();
      const [texture, frame] = art.frames[0];
      tok.sprite.setTexture(texture, frame);
    }
    tok.sprite.setFlipX(art.flip);
  }

  /**
   * Start or re-aim the walk cycle for a token's current heading.
   *
   * Re-aiming CARRIES THE STRIDE; never restart from frame 0 on a turn. Routes on a 4-connected
   * grid staircase and flip heading every tile, so restarting would make the legs twitch
   * instead of walk.
   */
  playWalk(tok: AgentToken): void {
    const key = this.animKey(tok.skinId, "walk", tok.dir);
    if (!this.host.anims.exists(key)) return;
    const anims = tok.sprite.anims;
    if (anims.isPlaying && anims.getName() === key) return;
    // Carry progress only out of a walk; out of a swing or crouch there is no stride to carry.
    const carry = anims.isPlaying && WALK_KEY.test(anims.getName()) ? anims.getProgress() : 0;
    tok.sprite.play(key);
    if (carry > 0) anims.setProgress(carry);
    tok.sprite.setFlipX(this.host.cast.poseArt(tok.skinId, "walk", tok.dir)?.flip ?? false);
  }

  // ---- action poses -----------------------------------------------------------------
  // effects.ts says what the WORLD did; this says what the FIGURE did (mapping: deedPose.ts).
  //
  // While a pose is held, `posed` keeps update() from yanking a stationary figure back to idle
  // mid-swing. Restoration goes through delayP, which resolves even when a scrub supersedes
  // the step, so a token is never stranded mid-blow.

  /** Act out one action: the actor's deed, and the reaction of anyone it was aimed at. */
  playDeed(tok: AgentToken, act: ActionSummary, tokens: Map<string, AgentToken>): void {
    const body = deedBody(act.deed, act.succeeded !== false);
    if (!body) return; // nothing was adjudicated → nothing was done → draw nothing
    // The CLOSING beat of a long act: he is no longer at it, so a held pose is shown once and
    // let go. Re-entering it would leave him stuck in it (sustained poses have no timer) and
    // strand a covert's shroud.
    const done = act.phase === Phase.ongoing_complete;
    if (body.actor === "attack") this.swing(tok);
    else this.holdPose(tok, body.actor, body.sustained && !done);
    this.setShroud(tok, done ? 1 : body.shroud ?? 1);
    if (!body.target) return;
    // The TARGET's pose is always a beat: sustained-ness describes the ACTOR's deed, not the
    // target's reaction.
    for (const id of actedOnAgents(act.target)) {
      const vt = tokens.get(id);
      if (vt) this.holdPose(vt, body.target);
    }
  }

  swing(tok: AgentToken): void {
    const key = this.animKey(tok.skinId, "attack", tok.dir);
    if (tok.dead || !this.host.anims.exists(key)) return;
    tok.posed = true;
    tok.sprite.play(key);
    tok.sprite.setFlipX(this.host.cast.poseArt(tok.skinId, "attack", tok.dir)?.flip ?? false);
    void this.host.delayP(420).then(() => this.restorePose(tok));
  }

  /**
   * Put a figure into a pose: for a beat, or until something replaces it.
   *
   * `sustained` (from deedPose.ts) means a thing he is DOING, not did. A beat drops back to idle
   * on a timer; a sustained pose has no timer and ends only on the next deed, walking off
   * (update), or a step where he does nothing (releaseIdlePoses).
   */
  holdPose(tok: AgentToken, pose: PoseName, sustained = false): void {
    if (tok.dead) return;
    tok.posed = true;
    this.applyPose(tok, pose);
    if (sustained) return;
    void this.host.delayP(360).then(() => this.restorePose(tok));
  }

  /**
   * How solid a figure looks in its current pose: 1, or dimmed into shadow.
   *
   * The nameplate dims too, or a hidden man would be announced by his full-strength name tag.
   * The result chip does NOT: it is the step's report to the viewer, not part of the figure.
   * Entering is eased; leaving is instant, at a step boundary.
   */
  setShroud(tok: AgentToken, alpha: number, instant = false): void {
    if (tok.sprite.alpha === alpha && tok.plate.alpha === alpha) return;
    if (instant) {
      tok.sprite.setAlpha(alpha);
      tok.plate.setAlpha(alpha);
      return;
    }
    this.host.tweens.add({ targets: [tok.sprite, tok.plate], alpha, duration: 420, ease: "Sine.easeInOut" });
  }

  // Back to idle, unless the figure died mid-pose (the death visual owns the body then).
  // The SHROUD ends here too: it belongs to the pose, so the pose's lifecycle owns it.
  restorePose(tok: AgentToken): void {
    tok.posed = false;
    this.setShroud(tok, 1, true);
    if (tok.dead) return;
    this.applyPose(tok, "idle");
  }

  /**
   * End any sustained pose whose owner is not acting this step.
   *
   * Not cleared with the step's other leftovers: a multi-step job would then drop to idle
   * between ticks. Someone with a deed coming keeps it (playDeed re-sets it).
   */
  releaseIdlePoses(tokens: Map<string, AgentToken>, acting: Set<string>): void {
    for (const [id, tok] of tokens) {
      if (tok.posed && !acting.has(id)) this.restorePose(tok);
    }
  }

  // Attach the garment-recolor shader with the identity color. All three HSV channels travel
  // (hue alone would merge the palette's two near-neutral slates); the art supplies relative
  // value, where the folds live.
  dyeGarment(sprite: Phaser.GameObjects.Sprite, color: number): void {
    sprite.setPostPipeline(RECOLOR_FX);
    const fx = sprite.getPostPipeline(RECOLOR_FX);
    const pipe = (Array.isArray(fx) ? fx[0] : fx) as RecolorFX | undefined;
    if (pipe) pipe.dye = hsvOf(color);
  }

  // (Re)assign a token's body — used when the cast's gender/age arrives after tokens were
  // created. Color is untouched: it came from the step state and was right from frame 1.
  assignSkin(tok: AgentToken, skinId: string): void {
    if (skinId === tok.skinId) return;
    tok.skinId = skinId;
    tok.sprite.setScale(this.host.cast.scale);
    // A different body is a different height. layoutLabels re-reads headY each step; the plate
    // moves here too so the swap does not wait a step.
    tok.headY = this.measureHeadY(skinId);
    tok.plate.setPosition(0, tok.headY - PLATE_GAP);
    this.applyPose(tok, tok.dead ? "hurt" : "idle");
  }

  setDeadMark(tok: AgentToken, dead: boolean): void {
    if (dead && !tok.deadMark) {
      const label = this.host.add
        // A size down so the four-letter badge stays about as wide as the body.
        .text(0, 0, "OVER", { fontFamily: FONT, fontSize: "10px", color: "#ffe3e3" })
        .setOrigin(0.5)
        .setResolution(TEXT_RES);
      // Sized from the measured label: the font is the theme's.
      const w = Math.ceil(label.width) + DEAD_MARK_PAD_X * 2;
      const h = Math.ceil(label.height) + DEAD_MARK_PAD_Y * 2;
      const plate = this.host.add.graphics();
      drawPlate(plate, -w / 2, -h / 2, w, h, DEAD_MARK_FILL, DEAD_MARK_EDGE, DEAD_MARK_R);
      tok.deadMark = this.host.add.container(0, -1, [plate, label]);
      tok.container.add(tok.deadMark);
    }
    tok.deadMark?.setVisible(dead);
  }

  /**
   * Everything the STEP REPORT says about a body (alive or fallen, how hurt), applied as one.
   * The single entry point for all timings: instant, deferred in playTrack, and the safety net.
   *
   * Must stay single: a deferred path landing only death sends the rest through the instant
   * path, ahead of the blow that caused it (the vitality bar drains a beat early).
   */
  applyBodyState(tok: AgentToken, state: AgentStateSummary, wasPlaced: boolean): void {
    if (reportedDead(state)) {
      this.applyDeathVisual(tok, state.emotion_valence, !tok.dead && wasPlaced);
      return;
    }
    // Revive on a scrub back before the death. Keyed on tok.dead, not the sprite's angle:
    // death uses the `down` frame and never rotates.
    if (tok.dead) { tok.sprite.setAngle(0).clearTint(); this.applyPose(tok, "idle"); }
    tok.baseAlpha = 1;
    tok.dead = false;
    this.setDeadMark(tok, false);
    this.updateVitalityBar(tok, state.vitality ?? 1, false);
  }

  /** Would applying the report change what the body looks like: falling, or the bar moving. */
  bodyChanged(tok: AgentToken, state: AgentStateSummary): boolean {
    if (reportedDead(state) !== tok.dead) return true;
    return Math.abs((state.vitality ?? 1) - tok.vit) > 0.001;
  }

  updateVitalityBar(tok: AgentToken, vitality: number, dead: boolean): void {
    const v = Math.max(0, Math.min(1, vitality));
    tok.vit = v; // what the bar is now DRAWN at — bodyChanged reads this, not the report
    const show = !dead && v > 0 && v < VIT_SHOW_BELOW;
    tok.hurt = show; // update() gates actual visibility by LOD zoom
    tok.vitTrack.setVisible(show);
    tok.vitFill.setVisible(show);
    if (!show) return;
    tok.vitFill.scaleX = Math.max(0.04, v);
    tok.vitFill.setFillStyle(v >= 0.6 ? 0x6ab04c : v >= 0.3 ? 0xe0a94f : 0xe0607a);
  }

  // A corpse is swept at the start of the step after its death (the scene drops it from its
  // token map first). A body is drawn only on the step it falls: later steps still report him
  // dead, so the scene's gone() tests for a token, not the report, or the corpse would be
  // recreated each step, blinking and wandering.
  removeToken(tok: AgentToken): void {
    const c = tok.container;
    this.host.killTweens(c);
    this.host.tweens.add({ targets: c, alpha: 0, scale: (c.scale || 1) * 0.6, duration: 300, ease: "Quad.easeIn", onComplete: () => c.destroy() });
  }

  // isNew fires the one-time dissipation, only on the step he first falls.
  applyDeathVisual(tok: AgentToken, _emotionValence: number | null | undefined, isNew: boolean): void {
    if (isNew) deathDissipate(this.host, tok);
    tok.sprite.stop();
    tok.posed = false; // the dead hold no pose; the death visual owns the body from here
    this.setShroud(tok, 1, true); // …nor any shroud: a man killed while hiding stops hiding
    // The sheet's `down` frame, not the standing figure rotated (reads as a tipped signboard).
    this.applyPose(tok, "down");
    tok.sprite.setTint(DEAD_COLOR);
    tok.baseAlpha = 0.5;
    tok.dead = true;
    this.setDeadMark(tok, true);
    this.updateVitalityBar(tok, 0, true);
  }

}
