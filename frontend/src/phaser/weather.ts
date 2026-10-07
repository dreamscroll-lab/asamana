/**
 * Draws weather: turns a `WeatherPlan` into Phaser objects.
 *
 * This file only translates the plan; every decision (size, intensity, placement, duration)
 * lives in the pure, browser-free-tested `weatherPlan.ts`, leaving only aesthetics to check
 * by eye.
 *
 * 1. Weather is a state, not a beat. It stays out of `ephemeral`: `playWeather` returns a
 *    `WeatherSpell` the scene keeps and `retire()`s, which stops the emitters (airborne
 *    particles finish falling) and fades the veil and glows before destroying them. Never
 *    retire it through `ephemeral`'s `destroy()`, which cuts it off in one frame.
 *    `renderStep` starts weather before the actions: it is the condition the step happens in,
 *    and queued after them only its tail would be seen.
 * 2. Never use `emitter.setScale()` for zoom compensation: particle positions are relative to
 *    the emitter origin, so the effect is pushed off the map. Compensate particle size only
 *    (the plan already does).
 * 3. When weather is drawn, the broadcast's own presentation is skipped (see
 *    `TiledWorldScene.draw`); the rain already says it.
 */

import Phaser from "phaser";

import {
  type EmitZone,
  type GlowPlan,
  type ParticleLayer,
  type WeatherArea,
  type WeatherContext,
  planWeather,
} from "./weatherPlan";

export type { WeatherArea, WeatherContext } from "./weatherPlan";

/** Scene capabilities this module needs. TiledWorldScene provides them. */
export interface WeatherHost extends Phaser.Scene {
  /** The animation-speed multiplier, for the few clocks the scene's time scale doesn't reach. */
  readonly speed: number;
}

/**
 * A weather effect in progress. `playWeather` returns it and the scene holds it until it should stop.
 *
 * Don't add a hard `cancel()`: Phaser cleans up on scene shutdown, and a shortcut would bring
 * back weather that vanishes in a frame.
 */
export interface WeatherSpell {
  /** Stops emitting, lets airborne particles land, fades the veil and glows, then destroys itself. Safe to call twice. */
  retire(): void;
}

/** Fade duration for veil and glow. One number, symmetric in and out. */
const FADE_MS = 900;

const TEX = { dot: "wx-dot", streak: "wx-streak" } as const;

/** Weather depth: above the world (see isoDepth for the band people occupy) but below text
 *  (action chips start at 45). Rain should not cover names and result bars. */
const WEATHER_DEPTH = 44;

/**
 * Draws one weather effect. Returns its handle, or `null` if nothing was drawn. The caller uses
 * this to decide whether the broadcast keeps its normal presentation: an unknown phenomenon
 * draws nothing, and then the broadcast has to show as usual or the message is lost.
 */
export function playWeather(
  host: WeatherHost,
  phenomenon: string,
  severity: string,
  area: WeatherArea,
  ctx: WeatherContext,
): WeatherSpell | null {
  const plan = planWeather(phenomenon, severity, area, ctx);
  if (!plan) return null;
  ensureTextures(host);

  const emitters: Phaser.GameObjects.Particles.ParticleEmitter[] = [];
  const fading: Faded[] = [];
  // How long the last particle keeps flying after emission stops. Lifespan is wall-clock ms in the
  // emitter config, while the scene clock waits in playback ms, so the wait is converted or the
  // rain is cut mid-fall at speed.
  let lingerMs = 0;

  for (const layer of plan.layers) {
    emitters.push(paintLayer(host, layer));
    // Use the upper bound: lifespan can be a range, and we wait for the last particle, not the average one.
    lingerMs = Math.max(
      lingerMs,
      typeof layer.lifespanMs === "number" ? layer.lifespanMs : layer.lifespanMs.max,
    );
  }
  for (const glow of plan.glows ?? []) fading.push(paintGlow(host, glow));
  if (plan.shake) host.cameras.main.shake(plan.shake.durationMs / host.speed, plan.shake.intensity);
  if (plan.veil) fading.push(paintVeil(host, plan.veil.alpha, area));
  return spellOf(host, emitters, fading, lingerMs);
}

/** Objects whose alpha is faded in and out (veil, glows). Particles aren't included; they finish falling on their own. */
type Faded = Phaser.GameObjects.Graphics | Phaser.GameObjects.Image;

function spellOf(
  host: WeatherHost,
  emitters: Phaser.GameObjects.Particles.ParticleEmitter[],
  fading: Faded[],
  lingerMs: number,
): WeatherSpell {
  let retired = false;
  return {
    retire() {
      if (retired) return;
      retired = true;
      // stop(), not destroy(): destroy() would also remove particles still in the air.
      for (const emitter of emitters) emitter.stop();
      for (const obj of fading) {
        host.tweens.killTweensOf(obj);   // the glow's pulse, or an unfinished fade-in
        host.tweens.add({
          targets: obj, alpha: 0, duration: FADE_MS, ease: "Sine.easeIn",
        });
      }
      host.time.delayedCall(Math.max(lingerMs * host.speed, FADE_MS), () => {
        for (const emitter of emitters) emitter.destroy();
        for (const obj of fading) obj.destroy();
      });
    },
  };
}

// ---------------------------------------------------------------------------

/** Highest rate (particles/s) emitted one at a time. Above this, particles are emitted in batches. */
const SMOOTH_RATE = 240;

function paintLayer(
  host: WeatherHost, layer: ParticleLayer,
): Phaser.GameObjects.Particles.ParticleEmitter {
  // perSecond → Phaser's (frequency, quantity). One particle per interval is smoothest, but the
  // 4ms minimum interval caps that at 250/s while world-wide rain needs 1680/s, so above the
  // ceiling emit in batches; otherwise the extra particles silently never appear.
  const quantity = Math.max(1, Math.ceil(layer.perSecond / SMOOTH_RATE));
  const config: Phaser.Types.GameObjects.Particles.ParticleEmitterConfig = {
    emitZone: randomZone(layer.zone),
    lifespan: layer.lifespanMs,
    frequency: (1000 * quantity) / layer.perSecond,
    quantity,
    alpha: layer.alpha,
    blendMode: layer.additive ? Phaser.BlendModes.ADD : Phaser.BlendModes.NORMAL,
  };
  if (layer.velocity.x !== undefined) config.speedX = layer.velocity.x;
  if (layer.velocity.y !== undefined) config.speedY = layer.velocity.y;
  if (layer.velocity.accelX !== undefined) config.accelerationX = layer.velocity.accelX;
  if (layer.velocity.accelY !== undefined) config.accelerationY = layer.velocity.accelY;
  if (layer.size.uniform !== undefined) config.scale = layer.size.uniform;
  if (layer.size.x !== undefined) config.scaleX = layer.size.x;
  if (layer.size.y !== undefined) config.scaleY = layer.size.y;
  if (layer.rotate !== undefined) config.rotate = layer.rotate;
  if (layer.colorRamp) {
    config.color = layer.colorRamp;
    config.colorEase = "quad.out";
  } else if (layer.tint !== undefined) {
    config.tint = layer.tint;
  }

  const emitter = host.add.particles(0, 0, TEX[layer.texture], config);
  emitter.setDepth(WEATHER_DEPTH);
  // Particles need no fade-in: the sky takes a full lifespan to fill up, which acts as one.
  return emitter;
}

function paintGlow(host: WeatherHost, glow: GlowPlan): Phaser.GameObjects.Image {
  const image = host.add.image(glow.x, glow.y, TEX.dot)
    .setTint(glow.tint)
    .setAlpha(0)
    .setScale(glow.radiusPx / 11)
    .setBlendMode(Phaser.BlendModes.ADD)
    .setDepth(WEATHER_DEPTH - 1);
  // Fade in, then pulse: two tweens, since the infinite yoyo can't also be the fade-in. A
  // retire() midway kills this tween, so the pulse never starts.
  host.tweens.add({
    targets: image, alpha: glow.alphaRest, duration: FADE_MS, ease: "Sine.easeOut",
    onComplete: () => {
      host.tweens.add({
        targets: image, alpha: glow.alphaPeak, scale: image.scale * 1.12,
        duration: 620, yoyo: true, repeat: -1, ease: "Sine.easeInOut",
      });
    },
  });
  return image;
}

/** The veil is drawn in the shape of the footprint, so darkness in one place doesn't dim the rest of the map. */
function paintVeil(host: WeatherHost, alpha: number, area: WeatherArea): Phaser.GameObjects.Graphics {
  const veil = host.add.graphics()
    .fillStyle(0x0b0e18, 1)
    .fillPoints(area.footprint.map((p) => new Phaser.Geom.Point(p.x, p.y)), true)
    .setDepth(WEATHER_DEPTH)
    .setAlpha(0);
  host.tweens.add({ targets: veil, alpha, duration: FADE_MS, ease: "Sine.easeOut" });
  return veil;
}

// ---------------------------------------------------------------------------
// Emit zones
// ---------------------------------------------------------------------------

/**
 * Polygon emit zone.
 *
 * Phaser's `Geom.Polygon` has no `getRandomPoint`, despite the docs listing Polygon as a
 * RandomZone source; passing one silently emits no particles. This samples the bounding rect
 * and retries outside the polygon (about two tries for an isometric diamond).
 */
class PolygonZone {
  private readonly poly: Phaser.Geom.Polygon;
  private readonly box: Phaser.Geom.Rectangle;

  constructor(points: { x: number; y: number }[]) {
    this.poly = new Phaser.Geom.Polygon(points.map((p) => new Phaser.Geom.Point(p.x, p.y)));
    this.box = Phaser.Geom.Polygon.GetAABB(this.poly);
  }

  getRandomPoint(point: Phaser.Types.Math.Vector2Like): Phaser.Types.Math.Vector2Like {
    for (let i = 0; i < 12; i += 1) {
      const x = this.box.x + Math.random() * this.box.width;
      const y = this.box.y + Math.random() * this.box.height;
      if (Phaser.Geom.Polygon.Contains(this.poly, x, y)) {
        point.x = x;
        point.y = y;
        return point;
      }
    }
    // Fall back to the centre. The point must always be set, or the whole batch spawns at the world origin.
    point.x = this.box.centerX;
    point.y = this.box.centerY;
    return point;
  }
}

/** Anything with a random-point sampler. Deliberately loose: `Rectangle.getRandomPoint` uses the
 *  generic `Point` signature, which doesn't match `PolygonZone`'s `Vector2Like` one, though both
 *  work at runtime. */
type ZoneSource = { getRandomPoint(...args: never[]): unknown };

function randomZone(zone: EmitZone): Phaser.Types.GameObjects.Particles.EmitZoneData {
  const source: ZoneSource = zone.kind === "ward"
    ? new PolygonZone(zone.points)
    : new Phaser.Geom.Rectangle(zone.x, zone.y, zone.width, zone.height);
  // Guard: a source without this method throws nothing and just emits no particles, which gets
  // past both the compiler and runtime. Fail loudly here instead.
  if (typeof source.getRandomPoint !== "function") {
    throw new Error(`weather: emit zone source has no getRandomPoint (kind=${zone.kind})`);
  }
  // The only cast: Phaser's own Geom types don't satisfy its own RandomZoneSourceCallback.
  return { type: "random", source: source as never };
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

/**
 * Two textures, generated once on first use.
 *
 * Rain and snow convince mostly through soft edges. Stacked alpha circles build a radial
 * falloff (Graphics can't draw gradients): a soft dot and a vertical streak fading at both
 * ends, covering all seven phenomena. Don't make them softer: too soft (e.g. 0.2 with
 * exponent 1.7) and small particles vanish over a base map. The quake errs the other way:
 * it is mainly a camera shake, and a little too much amplitude overwhelms everything.
 */
function ensureTextures(scene: Phaser.Scene): void {
  if (scene.textures.exists(TEX.dot)) return;

  const R = 16;
  const dot = scene.make.graphics({ x: 0, y: 0 }, false);
  for (let r = R; r > 0; r -= 1) {
    dot.fillStyle(0xffffff, Math.pow(1 - r / R, 1.15) * 0.4);
    dot.fillCircle(R, R, r);
  }
  dot.generateTexture(TEX.dot, R * 2, R * 2);
  dot.destroy();

  const H = 28;
  const W = 3;   // a 2px streak is too thin to see once scaled; 3px is the narrowest that still shows
  const streak = scene.make.graphics({ x: 0, y: 0 }, false);
  for (let y = 0; y < H; y += 1) {
    // Fading at both ends suggests speed (hard ends look like falling matchsticks), but the middle must stay solid.
    streak.fillStyle(0xffffff, Math.pow(Math.sin((y / (H - 1)) * Math.PI), 0.6));
    streak.fillRect(0, y, W, 1);
  }
  streak.generateTexture(TEX.streak, W, H);
  streak.destroy();
}
