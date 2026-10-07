// Visual FX for the world map: impacts, shockwaves, message arcs. Leaf effects that spawn
// ephemeral objects and tween them out, through an `FxHost` (the scene).
//
// No connector LINES between actor and target, by design: see "The language of FORCE" below.

import Phaser from "phaser";

import type { ActionSummary } from "../types";
import { toHex } from "../lib/color";
import { Deed, actedOnAgents } from "../lib/contract";
import type { Dir } from "./skins";
import { type AgentToken, FONT, TEXT_RES } from "./types";

// The slice of the scene the effects need; TiledWorldScene implements it.
export interface FxHost {
  add: Phaser.GameObjects.GameObjectFactory;
  tweens: Phaser.Tweens.TweenManager;
  delayP(ms: number): Promise<void>;
  pushEphemeral(obj: Phaser.GameObjects.GameObject): void; // track for step cleanup
  addCanceler(fn: () => void): void;
  removeCanceler(fn: () => void): void;
  /** Kill a target's tweens and settle whatever awaits them (see `awaitTweens`). */
  killTweens(targets: unknown): void;
  /**
   * Register `settle` as awaiting `targets`' tweens; returns it wrapped to run once. Phaser's
   * killTweensOf destroys a tween without firing onStop or onComplete, so anything that kills an
   * awaited target must go through `killTweens`, or the await never settles and the step hangs.
   */
  awaitTweens(targets: unknown, settle: () => void): () => void;
  token(id: string): AgentToken | undefined;
  // Redraw a standing figure on its current heading; which frame shows it is the art's answer.
  reface(tok: AgentToken): void;
  // Which drawn direction points at a screen offset. The scene answers: it owns the isometric
  // projection. See skins.ts `bearing`.
  bearing(dx: number, dy: number, previous: Dir): Dir;
  entityMarkers: Map<string, Phaser.GameObjects.Container>;
  // A point outside the world on the ray from its centre through (x,y): "somewhere beyond here".
  beyondTheMap(x: number, y: number): { x: number; y: number };
}

// Concentric rings radiating outward. `ground` draws flat floor ellipses; otherwise upright circles.
export function emanateRings(
  host: FxHost,
  x: number,
  y: number,
  color: number,
  opts: { count?: number; maxR?: number; duration?: number; ground?: boolean; width?: number; stagger?: number } = {},
): void {
  const { count = 1, maxR = 40, duration = 900, ground = true, width = 2.5, stagger = 150 } = opts;
  for (let i = 0; i < count; i++) {
    const ring = ground
      ? host.add.ellipse(x, y, 8, 4).setStrokeStyle(width, color)
      : host.add.circle(x, y, 4).setStrokeStyle(width, color);
    ring.setDepth(ground ? 14 : 31);
    host.pushEphemeral(ring);
    host.tweens.add({
      targets: ring,
      scale: maxR / 4,
      alpha: 0,
      delay: i * stagger,
      duration: duration,
      ease: "Cubic.easeOut",
    });
  }
}

// Shards flying outward and falling: splash / debris.
function burstParticles(
  host: FxHost,
  x: number,
  y: number,
  color: number,
  opts: { count?: number; speed?: number; size?: number; duration?: number } = {},
): void {
  const { count = 10, speed = 44, size = 3, duration = 620 } = opts;
  for (let i = 0; i < count; i++) {
    const a = (Math.PI * 2 * i) / count + (i % 2) * 0.5;
    const v = speed * (0.6 + (i % 3) * 0.28);
    const shard = host.add.rectangle(x, y, size, size, color).setDepth(32).setAngle(i * 47);
    host.pushEphemeral(shard);
    host.tweens.add({
      targets: shard,
      x: x + Math.cos(a) * v,
      y: y + Math.sin(a) * v + 34, // downward bias → arcs and falls
      angle: shard.angle + 220,
      alpha: 0,
      scale: 0.3,
      duration: duration,
      ease: "Quad.easeOut",
    });
  }
}

// ---------------------------------------------------------------------------------------
// The language of FORCE: "A did this to B" is carried by orientation, motion along the A→B
// axis, and where the impact lands and which way it throws, never by a connector line.
//
// A line is a diagram convention, and it is shortest and hidden under two bodies exactly when
// they share a room, which acting figures here always do. The one distant interaction, a
// letter, uses a thing that FLIES instead.
// ---------------------------------------------------------------------------------------

// Particles sprayed into a CONE: unlike a symmetric burst it says which way the force went. The
// main carrier of "A struck B".
function directionalBurst(
  host: FxHost,
  x: number,
  y: number,
  angle: number,
  color: number,
  opts: { count?: number; speed?: number; size?: number; duration?: number; spread?: number } = {},
): void {
  const { count = 12, speed = 80, size = 4, duration = 640, spread = Math.PI / 2.4 } = opts;
  for (let i = 0; i < count; i++) {
    const a = angle + (i / (count - 1 || 1) - 0.5) * spread;
    const v = speed * (0.55 + ((i * 7) % 5) * 0.14); // deterministic spread of speeds
    const shard = host.add.rectangle(x, y, size, size, color).setDepth(32).setAngle(i * 53);
    host.pushEphemeral(shard);
    host.tweens.add({
      targets: shard,
      x: x + Math.cos(a) * v,
      y: y + Math.sin(a) * v + 22, // gravity droop, so it arcs rather than beams
      angle: shard.angle + 200,
      alpha: 0,
      scale: 0.3,
      duration: duration,
      ease: "Quad.easeOut",
    });
  }
}

// A crescent on the victim, opening toward where the blow came from.
function impactArc(host: FxHost, x: number, y: number, facing: number, color: number): void {
  // Drawn about its own origin, then positioned, so scaling grows from the impact point.
  const g = host.add.graphics().setDepth(31).setPosition(x, y);
  host.pushEphemeral(g);
  g.lineStyle(3.5, color, 0.9);
  g.beginPath();
  g.arc(0, 0, 15, facing - 0.75, facing + 0.75, false);
  g.strokePath();
  host.tweens.add({
    targets: g,
    scaleX: 2.2, scaleY: 2.2, alpha: 0,
    duration: 420,
    ease: "Cubic.easeOut",
  });
}

// COVERT has no structured target, so the effect is about the ACTOR: a shadow closes in on him.
// `spotted` is DETECTION, not failure (the adjudicator answers them separately): feed it
// `detected`, never `succeeded`, or a clean but empty-handed getaway draws in alarm red.
export function stealthVeil(host: FxHost, tok: AgentToken, spotted: boolean): void {
  const { x, y } = tok.container;
  const color = spotted ? 0xc05a5a : 0x6b4f9e; // spotted → the shadow turns against him
  const veil = host.add.ellipse(x, y + 6, 54, 26).setStrokeStyle(2.5, color, 0.8).setDepth(14);
  host.pushEphemeral(veil);
  // INWARD, unlike every other ring: contraction is the grammar of hiding.
  host.tweens.add({ targets: veil, scale: 0.35, alpha: 0, duration: 620, ease: "Cubic.easeIn" });
  // The dimming is not here: being hidden outlasts a beat, so it belongs to the pose
  // (deedPose.ts `shroud`, BodyRig.setShroud). A yoyo here would undraw the result.
}

// Shove a figure along a screen vector and let it settle back: the axis performed, not drawn.
// Moves the SPRITE, never the container: the container is the token's authoritative resting
// slot, and bubbles/effects anchor to it.
export function lunge(host: FxHost, tok: AgentToken, angle: number, dist: number, ms = 90): void {
  const dx = Math.cos(angle) * dist;
  const dy = Math.sin(angle) * dist;
  const restX = tok.sprite.x;
  const restY = tok.sprite.y;
  host.tweens.add({
    targets: tok.sprite,
    x: restX + dx,
    y: restY + dy,
    duration: ms,
    yoyo: true,
    ease: "Quad.easeOut",
    onComplete: () => tok.sprite.setPosition(restX, restY),
  });
}

/** The screen-space angle from one token to another: the axis every impact is read along. */
function axisBetween(from: AgentToken, to: AgentToken): number {
  return Math.atan2(to.container.y - from.container.y, to.container.x - from.container.x);
}

// A dust puff at a token's feet on departure/arrival.
export function footDust(host: FxHost, x: number, y: number): void {
  for (let i = 0; i < 6; i++) {
    const a = Math.PI + (Math.random() - 0.5) * Math.PI;
    const v = 10 + Math.random() * 14;
    const p = host.add
      .circle(x + (Math.random() - 0.5) * 6, y, 2 + Math.random() * 1.5, 0xbfae90, 0.7)
      .setDepth(19);
    host.pushEphemeral(p);
    host.tweens.add({
      targets: p,
      x: p.x + Math.cos(a) * v,
      y: y - 4 - Math.random() * 6,
      alpha: 0,
      scale: 0.4,
      duration: 520,
      ease: "Quad.easeOut",
    });
  }
}

// A one-time dissipation when an agent falls, on top of the persistent grey + OVER badge.
export function deathDissipate(host: FxHost, tok: AgentToken): void {
  const { x, y } = tok.container;
  for (let i = 0; i < 9; i++) {
    const p = host.add
      .circle(x + (Math.random() - 0.5) * 16, y - 4, 2 + Math.random() * 2, 0xb8b2c2, 0.7)
      .setDepth(33);
    host.pushEphemeral(p);
    host.tweens.add({
      targets: p,
      y: y - 30 - Math.random() * 24,
      x: p.x + (Math.random() - 0.5) * 14,
      alpha: 0,
      scale: 0.3,
      duration: 1100,
      ease: "Sine.easeOut",
    });
  }
}

// Nearby items jolt from a shock. Markers are persistent, so restore their resting Y.
export function joltNearbyEntities(host: FxHost, x: number, y: number, radius = 95): void {
  for (const marker of host.entityMarkers.values()) {
    if (!marker.visible) continue;
    if (Phaser.Math.Distance.Between(x, y, marker.x, marker.y) > radius) continue;
    const restY = marker.y;
    host.tweens.add({
      targets: marker,
      y: restY - 12,
      duration: 85,
      yoyo: true,
      repeat: 3,
      ease: "Sine.easeInOut",
      onComplete: () => marker.setY(restY),
    });
  }
}

// A ✉ lofted along a quadratic Bézier arc, with a faint dashed trail and a ripple on arrival.
//
// Everything it spawns fades on its own; pushEphemeral is only a safety net for a scrub, and
// relying on clearEphemeral would CUT the effect mid-air. Resolves only once the letter has
// landed and been put away.
export function flyMessageArc(
  host: FxHost,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  // The sender's identity colour. Omitted → an unsigned letter (see arrivingMessage).
  seal?: { color: number; glyph?: string },
): Promise<void> {
  const inkInt = seal?.color ?? 0x8fbfe6;                                  // correspondence blue
  const ink = toHex(inkInt);
  const glyph = seal?.glyph ?? "✉";
  const dist = Math.hypot(x1 - x0, y1 - y0);
  const lift = Math.min(40 + dist * 0.28, 120); // longer throws arc higher, capped
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2 - lift; // control point lifted upward → the arc bows over
  const at = (t: number) => {
    const u = 1 - t;
    return { x: u * u * x0 + 2 * u * t * cx + t * t * x1, y: u * u * y0 + 2 * u * t * cy + t * t * y1 };
  };
  const g = host.add.graphics().setDepth(16);
  host.pushEphemeral(g);
  g.lineStyle(1.4, inkInt, 0.4);
  const N = Math.max(14, Math.round(dist / 12));
  for (let i = 0; i < N; i += 2) {
    const a = at(i / N);
    const b = at(Math.min(1, (i + 1) / N));
    g.lineBetween(a.x, a.y, b.x, b.y);
  }
  const env = host.add
    .text(x0, y0, glyph, { fontFamily: FONT, fontSize: "15px", color: ink })
    .setOrigin(0.5)
    .setDepth(43)
    .setResolution(TEXT_RES);
  host.pushEphemeral(env);
  const proxy = { t: 0 };
  return new Promise((resolve) => {
    const cancel = () => {
      host.tweens.killTweensOf(proxy);
      resolve();
    };
    host.addCanceler(cancel);
    host.tweens.add({
      targets: proxy,
      t: 1,
      duration: 1050,
      ease: "Sine.easeInOut",
      onUpdate: () => {
        const p = at(proxy.t);
        env.setPosition(p.x, p.y).setScale(1 + Math.sin(proxy.t * Math.PI) * 0.25); // gentle loft
      },
      onComplete: () => {
        host.removeCanceler(cancel);
        emanateRings(host, x1, y1, 0x8fbfe6, { count: 2, maxR: 30, ground: false, duration: 640, stagger: 100, width: 2.5 });
        // Delivered into the recipient's hands; the beat ends only once it is gone.
        host.tweens.add({
          targets: env,
          alpha: 0,
          scale: 0.5,
          duration: 320,
          ease: "Quad.easeIn",
          onComplete: () => resolve(),
        });
      },
    });
    // The trail dissolves behind the letter, reading as a path travelled, not a line left drawn.
    host.tweens.add({
      targets: g,
      alpha: 0,
      delay: 420,
      duration: 760,
      ease: "Quad.easeIn",
    });
  });
}

// SEND_MESSAGE to named recipients: a pulse at the sender, then a ✉ arc to EACH recipient.
// The directed arcs distinguish it from a local announcement and a broadcast.
export function directedMessages(host: FxHost, tok: AgentToken, targetIds: string[]): Promise<void> {
  emanateRings(host, tok.container.x, tok.container.y, tok.color, { count: 2, maxR: 30, ground: false, duration: 620, stagger: 110, width: 2.5 });
  const seal = { color: tok.color };
  const flights: Promise<void>[] = [];
  for (const rid of targetIds) {
    const rt = host.token(rid);
    if (rt === tok) continue; // a letter to oneself has nothing to show
    // A recipient with no token (off-cast or dead): don't skip, or the letter is sent with
    // nothing drawn. It arcs out past the edge of the world instead.
    flights.push(
      rt
        ? flyMessageArc(host, tok.container.x, tok.container.y, rt.container.x, rt.container.y, seal)
        : departingMessage(host, tok),
    );
  }
  // The caller must await this: it is last in its track, and fired-and-forgotten the next
  // step's clearEphemeral would delete the letter mid-air. (physicalEffect can be
  // fire-and-forget because the caption's await after it covers it.)
  return Promise.all(flights).then(() => undefined);
}

/**
 * A letter with NO SENDER (the EventSystem posts as narrator: "不知来源"), landing on its reader.
 * With no origin on the map, it arcs in from beyond the world, in neutral slate, marked ？.
 *
 * This idiom means only that: a signed letter is drawn once, when it leaves its sender's hand
 * (directedMessages), and never on the step it lands.
 */
export function arrivingMessage(host: FxHost, rt: AgentToken): Promise<void> {
  const origin = host.beyondTheMap(rt.container.x, rt.container.y);
  return flyMessageArc(host, origin.x, origin.y, rt.container.x, rt.container.y, {
    color: 0x8a94a8, glyph: "✉？",
  });
}

// The mirror image: a letter to someone not on the map arcs off past the edge of the world.
export function departingMessage(host: FxHost, st: AgentToken): Promise<void> {
  const to = host.beyondTheMap(st.container.x, st.container.y);
  return flyMessageArc(host, st.container.x, st.container.y, to.x, to.y, { color: st.color });
}

// SEND_MESSAGE with NO named recipient: an announcement called out where the sender stands.
// Rings alone, no ✉: the envelope is the directed letter's glyph, and this is a voice. They reach
// further than a personal effect's because calling out carries past the person in front of you.
export function localAnnounce(host: FxHost, tok: AgentToken): Promise<void> {
  const { x, y } = tok.container;
  emanateRings(host, x, y + 4, tok.color, { count: 3, maxR: 84, ground: true, duration: 900, stagger: 170, width: 2.5 });
  // Awaitable, as in directedMessages. delayP so a scrub cancels it cleanly.
  return host.delayP(1300);
}

// deed → what the WORLD did about it; the body's half is deedPose.ts. Keep them apart: strike
// and destroy share a swing but differ in debris, seize and destroy share an object but differ
// in the arm. One table would draw taking a letter as smashing it.
//
// Failure is a modifier: whatever the deed, the world shows "nothing landed" (missEffect),
// while the body still performs the deed (deedBody).
export function physicalEffect(host: FxHost, tok: AgentToken, act: ActionSummary): void {
  if (act.succeeded === false) {
    missEffect(host, tok, actedOnAgents(act.target));
    return;
  }
  switch (act.deed) {
    case Deed.strike:
      combatStrike(host, tok, actedOnAgents(act.target));
      break;
    case Deed.restrain:
      subdueEffect(host, tok, actedOnAgents(act.target));
      break;
    case Deed.destroy:
      smashEffect(host, tok, act);
      break;
    case Deed.seize:
      seizeEffect(host, tok, act);
      break;
    case Deed.operate:
      operateEffect(host, act);
      break;
    case Deed.exert:
      // Force spent on nothing in particular: no victim, no debris.
      emanateRings(host, tok.container.x, tok.container.y + 4, 0x9a8a6a, { count: 2, maxR: 46, duration: 560, width: 3 });
      burstParticles(host, tok.container.x, tok.container.y, 0xb0a080, { count: 12, speed: 62, size: 4 });
      break;
    // no default: an empty deed means nothing was done, so nothing is drawn.
  }
}

// Take a thing: no violence. A gold gleam gathers INWARD onto the actor, the opposite of a smash.
function seizeEffect(host: FxHost, tok: AgentToken, act: ActionSummary): void {
  for (const eid of act.affected_entity_ids) {
    const m = host.entityMarkers.get(eid);
    if (!m || !m.visible) continue;
    const gleam = host.add.circle(m.x, m.y, 12).setStrokeStyle(2.5, 0xd9a441).setDepth(31);
    host.pushEphemeral(gleam);
    host.tweens.add({ targets: gleam, x: tok.container.x, y: tok.container.y - 6, scale: 0.3, alpha: 0, duration: 480, ease: "Cubic.easeIn" });
  }
  burstParticles(host, tok.container.x, tok.container.y - 6, 0xe0c090, { count: 6, speed: 30, size: 2, duration: 420 });
}

// Work a thing where it stands: it pulses in place. No shards (nothing broke), no gleam (nothing
// changed hands).
function operateEffect(host: FxHost, act: ActionSummary): void {
  for (const eid of act.affected_entity_ids) {
    const m = host.entityMarkers.get(eid);
    if (!m || !m.visible) continue;
    const pulse = host.add.circle(m.x, m.y, 9).setStrokeStyle(2.5, 0x7fb4c9).setDepth(31);
    host.pushEphemeral(pulse);
    host.tweens.add({ targets: pulse, scale: 2.2, alpha: 0, duration: 560, ease: "Quad.easeOut" });
    host.tweens.add({ targets: m, y: m.y - 3, duration: 140, yoyo: true, ease: "Sine.easeInOut" });
  }
}

// Wreck an object: shards, a shockwave, a crack ring. No red, no victim flash.
function smashEffect(host: FxHost, tok: AgentToken, act: ActionSummary): void {
  // As in combatStrike: debris erupts from the THING along the blow, never from the man.
  let struck = false;
  for (const eid of act.affected_entity_ids) {
    const m = host.entityMarkers.get(eid);
    if (!m || !m.visible) continue;
    struck = true;
    const axis = Math.atan2(m.y - tok.container.y, m.x - tok.container.x);
    lunge(host, tok, axis, 9, 100); // he drives into it
    host.tweens.add({ targets: m, angle: 9, duration: 55, yoyo: true, repeat: 3, onComplete: () => m.setAngle(0) });
    impactArc(host, m.x, m.y, axis + Math.PI, 0xe0c090);
    directionalBurst(host, m.x, m.y, axis, 0x8a7a5a, { count: 14, speed: 84, size: 4, duration: 700 });
    directionalBurst(host, m.x, m.y, axis, 0xcabfa0, { count: 7, speed: 50, size: 3, duration: 640, spread: 1.9 });
    emanateRings(host, m.x, m.y + 4, 0xb0a080, { count: 2, maxR: 58, duration: 600, width: 3 });
    const ring = host.add.circle(m.x, m.y, 10).setStrokeStyle(2.5, 0xe0c090).setDepth(31);
    host.pushEphemeral(ring);
    host.tweens.add({ targets: ring, scale: 2.4, alpha: 0, duration: 520, ease: "Cubic.easeOut" });
  }
  if (!struck) {
    // The thing is off-screen or gone; the blow still happened, so it still shows.
    burstParticles(host, tok.container.x, tok.container.y, 0x8a7a5a, { count: 10, speed: 62, size: 3 });
  }
}

// ANY deed that failed: no red, shards, gleam or shockwave. A person aimed at gets a cold clank
// ring and pale sparks; a thing gets silence. It must not say WHICH deed failed: the body
// already performs that.
function missEffect(host: FxHost, tok: AgentToken, targetIds: string[]): void {
  const victims = targetIds.map((id) => host.token(id)).filter((v): v is AgentToken => !!v && v !== tok);
  for (const vt of victims) {
    const axis = axisBetween(tok, vt);
    // The actor OVERSHOOTS; the victim slips SIDEWAYS off the axis, not back along it: dodged,
    // not pushed.
    lunge(host, tok, axis, 13, 130);
    lunge(host, vt, axis + Math.PI / 2, 6, 120);
    const ring = host.add.circle(vt.container.x, vt.container.y - 2, 9).setStrokeStyle(2.5, 0xccd2dd).setDepth(30);
    host.pushEphemeral(ring);
    host.tweens.add({ targets: ring, scale: 1.9, alpha: 0, duration: 360, ease: "Quad.easeOut" });
    directionalBurst(host, vt.container.x, vt.container.y - 2, axis, 0xd7dce6, { count: 6, speed: 46, size: 2, duration: 420, spread: 0.7 });
  }
}

// Subdue a person: an INWARD grapple ring and a slow push. No blood, no shards.
function subdueEffect(host: FxHost, tok: AgentToken, targetIds: string[]): void {
  const victims = targetIds.map((id) => host.token(id)).filter((v): v is AgentToken => !!v && v !== tok);
  for (const vt of victims) {
    const axis = axisBetween(tok, vt);
    // Pressure, not a blow: same axis as a strike, but a long push instead of a sharp snap.
    lunge(host, tok, axis, 6, 260);
    lunge(host, vt, axis, 4, 300);
    const ring = host.add.circle(vt.container.x, vt.container.y, 26).setStrokeStyle(3, 0xcbb07a).setDepth(30);
    host.pushEphemeral(ring);
    host.tweens.add({ targets: ring, scale: 0.4, alpha: 0, duration: 560, ease: "Cubic.easeIn" });
  }
}

// A blow that LANDS: the whole impact is at the VICTIM, debris flying away from the blow. Don't
// put it on the actor: that reads as "the attacker exploded". The two bodies perform the axis.
function combatStrike(host: FxHost, tok: AgentToken, targetIds: string[]): Promise<void> {
  const victims = targetIds.map((id) => host.token(id)).filter((v): v is AgentToken => !!v && v !== tok);
  if (!victims.length) {
    // A blow with nobody on the end of it: a dusty swing, nothing struck.
    burstParticles(host, tok.container.x, tok.container.y, 0xb0a080, { count: 10, speed: 60, size: 3 });
    return host.delayP(420);
  }
  for (const vt of victims) {
    const axis = axisBetween(tok, vt);
    lunge(host, tok, axis, 10, 95);
    lunge(host, vt, axis, 7, 130);
    impactArc(host, vt.container.x, vt.container.y, axis + Math.PI, 0xff5555);
    directionalBurst(host, vt.container.x, vt.container.y, axis, 0xff6a5a, { count: 14, speed: 92, size: 4 });
    emanateRings(host, vt.container.x, vt.container.y + 4, 0xff7a4a, { count: 3, maxR: 70, duration: 640, stagger: 100, width: 3.5 });
    const flash = host.add.circle(vt.container.x, vt.container.y - 4, 13, 0xff5555, 0.55).setDepth(29);
    host.pushEphemeral(flash);
    host.tweens.add({ targets: flash, scale: 2.6, alpha: 0, duration: 420, ease: "Cubic.easeOut" });
    joltNearbyEntities(host, vt.container.x, vt.container.y);
  }
  return host.delayP(560);
}



// Uses the scene's `bearing`, NOT dirOf: dirOf is built for travel and inherits an ambiguous axis
// from the previous heading. See skins.ts `bearing`.
function turnToward(host: FxHost, tok: AgentToken, x: number, y: number): void {
  tok.dir = host.bearing(x - tok.container.x, y - tok.container.y, tok.dir);
  host.reface(tok);
}

// Turn the actor and its primary target to face each other, until either moves. The strongest
// carrier of "who is acting on whom".
//
// Returns whether anyone turned (the target may be off the map), so the caller can fall through.
export function faceInteraction(host: FxHost, tok: AgentToken, targetIds: string[]): boolean {
  for (const tid of targetIds) {
    const tt = host.token(tid);
    if (!tt || tt === tok) continue;
    turnToward(host, tok, tt.container.x, tt.container.y);
    turnToward(host, tt, tok.container.x, tok.container.y);
    return true; // face the primary target only
  }
  return false;
}

// Turn a bystander toward what he is listening to. One-way: mutual would draw an eavesdropper as
// a party to the exchange.
export function faceOnlooker(host: FxHost, tok: AgentToken, towardIds: string[]): boolean {
  for (const tid of towardIds) {
    const tt = host.token(tid);
    if (!tt || tt === tok) continue;
    turnToward(host, tok, tt.container.x, tt.container.y);
    return true;
  }
  return false;
}

// Turn the actor toward the THING he acts on (entities live in affected_entity_ids, so they never
// reach faceInteraction).
//
// Relies on the marker still being on the map: renderStep withholds a deed's changes until after
// its beat, so a seized or destroyed thing has not yet been removed.
export function faceEntity(host: FxHost, tok: AgentToken, entityIds: string[]): boolean {
  for (const eid of entityIds) {
    const m = host.entityMarkers.get(eid);
    if (!m || !m.visible) continue;
    turnToward(host, tok, m.x, m.y);
    return true; // face the primary one
  }
  return false;
}
