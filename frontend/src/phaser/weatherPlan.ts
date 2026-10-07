/**
 * Decides what weather to draw: what, how big, how hard, where and for how long. This file
 * contains no Phaser code.
 *
 * Weather bugs are almost always wrong decisions, not bad pixels, so decisions live here where
 * weatherPlan.test.ts can assert them, one assertion per known trap:
 *
 *   · Fire sized from the ward's width: a 500px blob in a wide ward.
 *   · Rain travel taken from the ward's height: isometric wards are ~100px tall, so a drop
 *     lives 220ms and reads as a flicker.
 *   · Zoom compensation via `emitter.setScale()`: that scales the transform, so particle
 *     positions spread out too.
 *   · Local darkness covering the whole screen, or a local quake shaking the whole camera.
 *   · Treating unknown and known phenomena in separate branches.
 *   · Severity changing only the particle count, so heavier rain or snow is just "a few more".
 *   · Rain travel taken from the whole map's height: a drop lives 3.5s, ~10k live particles.
 *   · Splashes 3.5× wider than the streaks and denser: white dots everywhere, which reads as hail.
 *   · Smoke scattered at random over the footprint: grey blotches with no source or direction.
 *
 * All sizes are in tiles, not pixels: a pixel size means nothing on a map with another tile size.
 * The draw layer converts once, using the map's own tile height.
 */

/** The engine's closed vocabulary (core/interfaces/phenomenon.py), minus `none`. */
export type Phenomenon = "rain" | "snow" | "wind" | "fire" | "smoke" | "quake" | "dark";

/**
 * Sited phenomena always happen somewhere: a burning house, collapsing ground. The others are
 * diffuse: rain, snow, wind and darkness cover a stretch of sky and need no site.
 *
 * The engine owns this split (`core/interfaces/phenomenon.py`): a sited broadcast with no
 * location is lowered to `none` at the `Broadcast` boundary. This mirror decides nothing; it keeps
 * the scene bench from building impossible scenes and explains why fire and smoke are point sources.
 */
export const SITED_PHENOMENA: readonly Phenomenon[] = ["fire", "smoke", "quake"];

/**
 * The area a weather effect covers. It is always a single footprint polygon.
 *
 * A local effect uses the ward's outline, a world-wide one the whole map's: same code path.
 * World-wide is not the camera's view, or the rain would move whenever someone dragged the map.
 *
 * `box` (the bounding rect) is only for measurements. Particles are emitted from the footprint:
 * an isometric diamond's bounding rect has about twice its area and would rain onto neighbours.
 */
export interface WeatherArea {
  /** Polygon vertices in world coordinates: one ward's, or the whole map's. */
  footprint: { x: number; y: number }[];
  box: { x: number; y: number; width: number; height: number };
  /** Whether this area is currently on camera. Only the quake uses it; see plan.shake. */
  onScreen: boolean;
}

/** World measurements supplied by the draw layer. */
export interface WeatherContext {
  /** Height of one tile in pixels. Every tile quantity converts through it. */
  tilePx: number;
  /** Camera zoom. Only used to compensate particle size; never used for positions. */
  zoom: number;
}

/**
 * Severity is a small vector, not a single knob.
 *
 * One density on the particle count would make heavy rain read as "the same rain with a few more
 * drops". Four dimensions:
 *
 * - `count`    how many: particles per unit time
 * - `vigor`    how hard: speed, acceleration, stretch
 * - `weight`   how big: particle size, fire radius
 * - `presence` how visible: opacity
 *
 * Each phenomenon's plan function uses only the dimensions that physically change for it (heavier
 * snow gets bigger flakes, not faster ones, or it would look like white rain).
 */
export interface Intensity {
  count: number;
  vigor: number;
  weight: number;
  presence: number;
}

// severity is the engine's existing "how big is this" scale; the renderer reuses it rather than
// inventing an intensity scale for authors to learn. (Cost: a "narratively major drizzle" is
// drawn as a downpour. For weather the two correlate strongly; accepted.)
const INTENSITY: Record<string, Intensity> = {
  low:    { count: 0.5, vigor: 0.78, weight: 0.82, presence: 0.72 },
  medium: { count: 1,   vigor: 1,    weight: 1,    presence: 1 },
  high:   { count: 1.9, vigor: 1.32, weight: 1.28, presence: 1.15 },
};

// ── Spatial constants, in tiles ─────────────────────────────────────────────────

/**
 * Travel distance (tiles) for falling weather: how far a drop moves between birth and death. It
 * does not depend on the size of the area.
 *
 * Don't use `max(box.height, floor)`: a ward's box gives a 200ms flicker, the whole map's a 3.5s
 * lifespan and ~10k live particles. Streak length is an aesthetic constant, not a map property.
 */
export const FALL_TILES = 10;

/**
 * Maximum number of particles alive at once for one plan.
 *
 * Live particles = emission rate × lifespan, a product that can explode. Per plan, not per layer,
 * since frame drops hit the whole screen; over the cap, every layer's rate scales by one factor,
 * so the weather thins instead of losing a layer.
 */
export const MAX_LIVE_PARTICLES = 2400;

/** Base fire radius in tiles; multiplied by weight. It does not depend on the size of the place:
 *  a fire is big because it burns hard, not because it happens to be in a big ward. */
export const FIRE_RADIUS_TILES = 1;

/** Typical ward area (tiles²). All base emission rates below are tuned at this scale. */
const WARD_REF_TILES2 = 8;

/**
 * Upper bound on how much emission scales with area.
 *
 * An absolute rate dilutes world-wide rain to invisibility; constant density needs over 10k
 * particles. So the rate scales with `sqrt(area ratio)`, capped by the particle budget (heavy
 * world-wide rain comes to about 2–3k particles).
 */
export const MAX_AREA_SCALE = 8;

/** Maximum number of sources in one area. Point-source weather (fire, smoke) covers a larger area
 *  with more sites, not a higher rate: a big ward on fire means several fires, not one bigger
 *  fire in the middle. */
export const MAX_SITES = 4;

/** Native pixel sizes of the two textures. Tile size → scale factor divides by these. */
const TEX_PX = { dot: 32, streakW: 3, streakH: 28 } as const;

/**
 * Minimum on-screen size of a particle in pixels; anything smaller is effectively not drawn.
 *
 * This is the only place zoom affects size, and only when zoomed far out. At whole-city view a
 * raindrop should be too thin to make out individually, but if it shrinks to nothing the rain
 * disappears.
 */
const MIN_SCREEN_PX = 1.5;

/**
 * Converts a size in tiles into a texture scale factor. This is a world quantity and ignores zoom.
 *
 * Don't multiply by `1/zoom`: this is a world the viewer looks into, not a screen filter, and
 * zoomed out such rain ends up longer than the roofs.
 */
function texScale(tiles: number, ctx: WeatherContext, texPx: number): number {
  return (tiles * ctx.tilePx) / texPx;
}

/**
 * Visibility boost at extreme zoom-out. Scales the whole particle uniformly, computed once from
 * its longest dimension.
 *
 * Don't apply the minimum per axis (`max(axis, MIN/zoom)`): only a drop's narrow axis would hit
 * it, so the drop would fatten as the camera pulls back. One factor for both axes.
 */
function boost(longestTiles: number, ctx: WeatherContext): number {
  const screenPx = longestTiles * ctx.tilePx * ctx.zoom;
  return screenPx >= MIN_SCREEN_PX ? 1 : MIN_SCREEN_PX / screenPx;
}

/** Raindrop width and length in tiles for the near layer; the far layer is scaled down.
 *  Rain is small and dense: heavy rain looks convincing because the drops are uncountable, not
 *  because each one is visible. */
const RAIN_DROP_TILES = { width: 0.02, length: 0.4 };
/** Snowflake diameter in tiles. Snow is the opposite of rain: big and sparse, so each flake's outline shows. */
const SNOW_FLAKE_TILES = 0.23;

/** Fall speed (px/s) for the far and near layers. Shared by rain and snow; see the comment on snow(). */
const FALL_SPEED = { far: 720, near: 980 };
/** Wind: size of gust streaks and debris in tiles. Wind is the only phenomenon with nothing of its
 *  own to draw; it only shows through what it pushes, so the gust streaks have to be solid enough
 *  to see. */
const WIND_STREAK_TILES = { width: 0.10, length: 2.1 };
const WIND_MOTE_TILES = 0.14;
/** Fire: size of flame tongues, embers and smoke puffs (tiles). */
const EMBER_TILES = 0.16;
const SMOKE_PUFF_TILES = 0.5;
/** Quake dust fall (tiles). */
const DUST_TILES = 0.07;

// ── Shape of a plan ─────────────────────────────────────────────────────────────

/** Where particles spawn, in world pixels. No zoom compensation is applied. */
export type EmitZone =
  | { kind: "ward"; points: { x: number; y: number }[] }
  | { kind: "rect"; x: number; y: number; width: number; height: number };

export interface ParticleLayer {
  texture: "dot" | "streak";
  zone: EmitZone;
  /** Particles per second. Easier to assert and read than the coupled frequency/quantity pair. */
  perSecond: number;
  lifespanMs: number | { min: number; max: number };
  velocity: {
    x?: number | { min: number; max: number };
    y?: number | { min: number; max: number };
    accelX?: number | { min: number; max: number };
    accelY?: number;
  };
  /** Particle size, with zoom compensation already applied. Compensation may only affect size;
   *  applying it to positions throws the whole effect out of view. */
  size: {
    x?: number | { min: number; max: number } | { start: number; end: number };
    y?: number | { min: number; max: number } | { start: number; end: number };
    uniform?: number | { min: number; max: number } | { start: number; end: number };
  };
  alpha: number | { min: number; max: number } | { start: number; end: number };
  tint?: number | number[];
  colorRamp?: number[];
  rotate?: number | { min: number; max: number };
  additive?: boolean;
}

/** A pulsing glow (only fire uses it). The radius is a world quantity and is not zoom-compensated. */
export interface GlowPlan {
  x: number;
  y: number;
  radiusPx: number;
  tint: number;
  alphaRest: number;
  alphaPeak: number;
}

export interface WeatherPlan {
  layers: ParticleLayer[];
  /** Pulsing glows, one per fire site (only fire uses them). */
  glows?: GlowPlan[];
  /** Camera shake, or null for none. A quake in a distant ward must not shake the whole view; the
   *  viewer isn't there. */
  shake: { durationMs: number; intensity: number } | null;
  /** Darkening veil, drawn in the shape of `area.footprint`. Darkness in one place must not dim
   *  the rest of the map. */
  veil: { alpha: number } | null;
}

const cap = (v: number): number => Math.min(v, 1);

/**
 * Plans one weather effect. Returns `null` for an unknown phenomenon so the draw layer lets the
 * broadcast use its normal presentation; otherwise the message would not be shown at all.
 */
export function planWeather(
  phenomenon: string,
  severity: string,
  area: WeatherArea,
  ctx: WeatherContext,
): WeatherPlan | null {
  const s = INTENSITY[severity] ?? INTENSITY.medium;
  const plan = ((): WeatherPlan | null => {
    switch (phenomenon) {
      case "rain":  return rain(area, s, ctx);
      case "snow":  return snow(area, s, ctx);
      case "wind":  return wind(area, s, ctx);
      case "fire":  return fire(area, s, ctx);
      case "smoke": return smoke(area, s, ctx);
      case "quake": return quake(area, s, ctx);
      case "dark":  return dark(area, s, ctx);
      default:      return null;
    }
  })();
  return plan ? withinBudget(plan) : null;
}

/** Mean lifespan (s). Emission rate times this is the number alive at once. */
function avgLifeSec(life: number | { min: number; max: number }): number {
  return (typeof life === "number" ? life : (life.min + life.max) / 2) / 1000;
}

/**
 * Fits the whole plan into the particle budget. This is done once here, not separately in each
 * phenomenon.
 *
 * Each phenomenon picks locally reasonable values (area sets the rate, travel sets the lifespan).
 * The overflow only appears when they are multiplied together, and only this function sees the
 * whole product.
 */
function withinBudget(plan: WeatherPlan): WeatherPlan {
  const live = plan.layers.reduce((n, l) => n + l.perSecond * avgLifeSec(l.lifespanMs), 0);
  if (live <= MAX_LIVE_PARTICLES) return plan;
  const k = MAX_LIVE_PARTICLES / live;
  return { ...plan, layers: plan.layers.map((l) => ({ ...l, perSecond: l.perSecond * k })) };
}

/** Spawn zone for falling weather: directly above the place.
 *
 *  The ward's diamond is lifted above it, so rain falls from the sky above the ward and through
 *  it instead of appearing in the middle of it. The lift is derived from the travel distance
 *  (see travelPx), not from the ward's height. */
function fallZone(area: WeatherArea, travelPx: number): EmitZone {
  const lift = travelPx * 0.62;
  return { kind: "ward", points: area.footprint.map((p) => ({ x: p.x, y: p.y - lift })) };
}

/** Birth zone for diffuse / rising weather: inside the whole footprint. */
function areaZone(area: WeatherArea): EmitZone {
  return { kind: "ward", points: area.footprint };
}

/** Fall distance. Does not depend on the size of the area; see FALL_TILES. */
function travelPx(tilePx: number): number {
  return FALL_TILES * tilePx;
}

// ── "How big is this place" ─────────────────────────────────────────────────────

/** Footprint area in tiles², by the shoelace formula. Uses the footprint itself, since the
 *  bounding rect doubles the area of an isometric diamond. */
function footprintTiles2(area: WeatherArea, ctx: WeatherContext): number {
  const pts = area.footprint;
  let twice = 0;
  for (let i = 0; i < pts.length; i += 1) {
    const j = (i + 1) % pts.length;
    twice += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
  }
  return Math.abs(twice) / 2 / (ctx.tilePx * ctx.tilePx);
}

/**
 * Emission multiplier for diffuse weather: 1 for a single ward, larger for larger areas, capped at
 * `MAX_AREA_SCALE`.
 *
 * This multiplier is the only difference between world-wide and local weather. Both still use the
 * same code path with a different polygon; the polygon's area also feeds into the rate.
 */
function areaScale(area: WeatherArea, ctx: WeatherContext): number {
  const ratio = footprintTiles2(area, ctx) / WARD_REF_TILES2;
  return Math.min(Math.max(Math.sqrt(ratio), 1), MAX_AREA_SCALE);
}

/** Point-in-polygon (ray casting). Filters sites for multi-source weather. */
function contains(poly: { x: number; y: number }[], x: number, y: number): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i, i += 1) {
    const a = poly[i];
    const b = poly[j];
    if ((a.y > y) !== (b.y > y) && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) {
      inside = !inside;
    }
  }
  return inside;
}

/**
 * Places n sites deterministically inside the footprint (golden-angle spiral, filtered to the
 * polygon).
 *
 * A burning area means several fires, not one fire at a higher rate. A spiral rather than random
 * placement so the same plan always produces the same sites.
 */
function sitesInside(area: WeatherArea, n: number): { x: number; y: number }[] {
  const { x, y, width, height } = area.box;
  const center = { x: x + width / 2, y: y + height * 0.66 };
  if (n <= 1) return [center];
  const out: { x: number; y: number }[] = [];
  for (let i = 0; i < n; i += 1) {
    const angle = i * 2.399963229728653;             // golden angle: even spread, no rings or spokes
    // A sqrt radius spreads points evenly by area. Point i's radius comes from i/n, not the
    // attempt index, or only the innermost ring would be used (six fires bunched at the centre).
    const base = Math.sqrt((i + 0.5) / n) * 0.46;
    for (let k = 0; k < 6; k += 1) {
      const r = base * (1 - k * 0.17);               // outside the footprint: pull toward the centre, don't discard
      const px = x + width * (0.5 + r * Math.cos(angle));
      const py = y + height * (0.5 + r * Math.sin(angle));
      if (contains(area.footprint, px, py)) {
        out.push({ x: px, y: py });
        break;
      }
    }
  }
  return out.length > 0 ? out : [center];
}

// ── Falling ───────────────────────────────────────────────────────────────────────

/** Rain: two parallax layers plus ground splashes. Uses all four dimensions; heavier rain falls faster, which is what separates it from snow. */
function rain(area: WeatherArea, s: Intensity, ctx: WeatherContext): WeatherPlan {
  const fall = travelPx(ctx.tilePx);
  const zone = fallZone(area, fall);
  const g = areaScale(area, ctx);
  const layers: ParticleLayer[] = [
    // Far layer: thin, slow, faint. Near layer: thick, fast, bright. Single-layer rain looks like a pasted-on pattern.
    { k: 0.62, a: 0.42, v: FALL_SPEED.far, tint: 0x93b6d0, rate: 130 },
    { k: 1,    a: 0.78, v: FALL_SPEED.near, tint: 0xe2f0fb, rate: 210 },
  ].map((L) => {
    const speed = L.v * s.vigor;
    const len = RAIN_DROP_TILES.length * L.k * s.vigor;
    const b = boost(len, ctx);          // one drop, one size: the same compensation on both axes
    return {
      texture: "streak" as const,
      zone,
      perSecond: L.rate * s.count * g,
      lifespanMs: (fall / speed) * 1000,
      // Slanted rain: the sideways component is proportional to fall speed, so streaks don't look "broken"
      velocity: { y: speed, x: -speed * 0.14 },
      size: {
        x: texScale(RAIN_DROP_TILES.width * L.k * s.weight * b, ctx, TEX_PX.streakW),
        // harder rain stretches longer
        y: texScale(len * b, ctx, TEX_PX.streakH),
      },
      alpha: cap(L.a * s.presence),
      tint: L.tint,
      rotate: -8,
    };
  });
  // Splashes show that the rain lands somewhere. They must stay secondary to the streaks: wider
  // and denser, they fill the screen with white dots that read as hail.
  const splash = RAIN_DROP_TILES.width * 1.6 * s.weight;
  const bp = boost(splash, ctx);
  layers.push({
    texture: "dot",
    zone: areaZone(area),
    perSecond: 14 * s.count * g,
    lifespanMs: 380,
    velocity: {},
    size: {
      x: { start: texScale(splash * 0.25 * bp, ctx, TEX_PX.dot), end: texScale(splash * bp, ctx, TEX_PX.dot) },
      y: { start: texScale(splash * 0.2 * bp, ctx, TEX_PX.dot), end: texScale(splash * 0.4 * bp, ctx, TEX_PX.dot) },
    },
    alpha: { start: cap(0.55 * s.presence), end: 0 },
    tint: 0xeaf6ff,
  });
  return { layers, shake: null, veil: null };
}

/** Snow: falls at the same speed as rain; only the shape differs. A soft round dot with spin and
 *  sideways sway, versus rain's stretched streak.
 *
 *  Don't slow it to a realistic terminal velocity (about a thirtieth of rain's): within one step
 *  it would look like a frozen frame. Readability wins, so rain and snow share one speed constant.
 *  Heavier snow means bigger, denser flakes, not thinner, longer ones. */
function snow(area: WeatherArea, s: Intensity, ctx: WeatherContext): WeatherPlan {
  const fall = travelPx(ctx.tilePx);
  const speed = FALL_SPEED.near * s.vigor;
  const b = boost(SNOW_FLAKE_TILES * s.weight, ctx);
  return {
    layers: [{
      texture: "dot",
      zone: fallZone(area, fall),
      perSecond: 158 * s.count * areaScale(area, ctx),
      lifespanMs: (fall / speed) * 1000,
      // Sway is small compared with fall speed, so flakes drift but don't fly across half the map.
      velocity: { y: speed, x: { min: -60, max: 60 }, accelX: { min: -30, max: 30 } },
      size: {
        uniform: {
          min: texScale(SNOW_FLAKE_TILES * 0.45 * s.weight * b, ctx, TEX_PX.dot),
          max: texScale(SNOW_FLAKE_TILES * s.weight * b, ctx, TEX_PX.dot),
        },
      },
      alpha: { min: cap(0.72 * s.presence), max: 1 },
      tint: [0xffffff, 0xeaf4ff],
      rotate: { min: 0, max: 360 },
    }],
    shake: null,
    veil: null,
  };
}

// ── Sweeping ───────────────────────────────────────────────────────────────────────

/** Wind: invisible itself, so it is drawn through what it pushes, as debris and gust streaks.
 *  vigor is its main dimension (stronger wind blows faster).
 *
 *  Gusts start everywhere in the area, not from one edge: a particle travels under 1000px in its
 *  1.5s, so over the whole map an inflow band would cover only one edge.
 *
 *  Density is kept high: the motion already tells it from rain, and the usual failure is wind
 *  that can't be seen at all. */
function wind(area: WeatherArea, s: Intensity, ctx: WeatherContext): WeatherPlan {
  const zone = areaZone(area);
  const g = areaScale(area, ctx);
  const bs = boost(WIND_STREAK_TILES.length * s.vigor, ctx);
  const bm = boost(WIND_MOTE_TILES, ctx);
  return {
    layers: [
      {
        texture: "streak",
        zone,
        perSecond: 34 * s.count * g,
        lifespanMs: { min: 900, max: 1500 },
        velocity: { x: { min: 420 * s.vigor, max: 720 * s.vigor }, y: { min: -30, max: 30 } },
        size: {
          x: { min: texScale(WIND_STREAK_TILES.width * 0.6 * bs, ctx, TEX_PX.streakW),
               max: texScale(WIND_STREAK_TILES.width * bs, ctx, TEX_PX.streakW) },
          y: { min: texScale(WIND_STREAK_TILES.length * 0.5 * s.vigor * bs, ctx, TEX_PX.streakH),
               max: texScale(WIND_STREAK_TILES.length * s.vigor * bs, ctx, TEX_PX.streakH) },
        },
        alpha: { start: cap(0.72 * s.presence), end: 0 },
        tint: 0xeef5f0,
        rotate: 88,
      },
      {
        texture: "dot",
        zone,
        perSecond: 46 * s.count * g,
        lifespanMs: { min: 700, max: 1200 },
        velocity: {
          x: { min: 360 * s.vigor, max: 640 * s.vigor },
          y: { min: -70, max: 70 },
          accelX: { min: -40, max: 40 },
        },
        size: {
          uniform: { min: texScale(WIND_MOTE_TILES * 0.5 * bm, ctx, TEX_PX.dot),
                     max: texScale(WIND_MOTE_TILES * bm, ctx, TEX_PX.dot) },
        },
        alpha: { start: cap(0.85 * s.presence), end: 0 },
        tint: [0xd6c8a8, 0xb3c2aa],
      },
    ],
    shake: null,
    veil: null,
  };
}

// ── Rising ─────────────────────────────────────────────────────────────────────────

/** Fire: glow, flame tongues, embers and a column of smoke.
 *
 *  The radius is `FIRE_RADIUS_TILES × weight` and does not depend on the size of the place.
 *  Sizing by place width (e.g. `min(ward width × 0.28, 120)`) hits the cap in wide wards and turns
 *  the glow into an additive blob hundreds of pixels across that washes out the flames and embers. */
function fire(area: WeatherArea, s: Intensity, ctx: WeatherContext): WeatherPlan {
  const spread = FIRE_RADIUS_TILES * ctx.tilePx * s.weight;
  // Fire is a point source, not diffuse: a burning area means several fires, not one bigger fire
  // in the middle. So a larger area gets more sites rather than a higher rate, and a world-wide
  // fire appears all over the map instead of in one cluster at the centre.
  const sites = sitesInside(area, Math.round(Math.min(areaScale(area, ctx), MAX_SITES)));
  const band = (at: { x: number; y: number }, k: number, dy: number, h: number): EmitZone =>
    ({ kind: "rect", x: at.x - spread * k, y: at.y + dy, width: spread * 2 * k, height: h });

  const layers: ParticleLayer[] = [];
  for (const at of sites) {
    for (const t of [
      { life: [420, 780], speed: [70, 150], scale: [0.55, 0.05], rate: 167, k: 0.35 },
      { life: [1100, 2000], speed: [40, 110], scale: [0.22, 0.02], rate: 59, k: 1 },
    ]) {
      layers.push({
        texture: "dot",
        zone: band(at, t.k, -10, 20),
        perSecond: t.rate * s.count,
        lifespanMs: { min: t.life[0], max: t.life[1] },
        velocity: {
          y: { min: -t.speed[1] * s.vigor, max: -t.speed[0] * s.vigor },
          x: { min: -34, max: 34 },
          accelY: -70 * s.vigor,                     // accelerates upward like hot air
        },
        size: {
          uniform: {
            start: texScale(EMBER_TILES * t.scale[0] * s.weight * boost(EMBER_TILES * t.scale[0] * s.weight, ctx), ctx, TEX_PX.dot),
            end: texScale(EMBER_TILES * t.scale[1] * boost(EMBER_TILES * t.scale[0] * s.weight, ctx), ctx, TEX_PX.dot),
          },
        },
        alpha: { start: 0.95, end: 0 },
        colorRamp: [0xfff3b0, 0xffa53d, 0xe8471c, 0x5a1b0c], // this ramp is what makes it look like fire rather than orange snow
        additive: true,                                      // fire is light, so it blends additively
      });
    }
    // Fire always gets smoke; without it the fire looks like a decoration floating in the air.
    layers.push({
      texture: "dot",
      zone: band(at, 1, -90, 30),
      perSecond: 11 * s.count,
      lifespanMs: { min: 2200, max: 3600 },
      velocity: { y: { min: -46, max: -18 }, x: { min: -20, max: 20 } },
      size: {
        uniform: {
          start: texScale(SMOKE_PUFF_TILES * 0.4 * s.weight, ctx, TEX_PX.dot),
          end: texScale(SMOKE_PUFF_TILES * 1.3 * s.weight, ctx, TEX_PX.dot),
        },
      },
      alpha: { start: cap(0.3 * s.presence), end: 0 },
      tint: [0x54514e, 0x2f2d2b],
      rotate: { min: -40, max: 40 },
    });
  }

  return {
    layers,
    // A halo at the base of each fire rather than an orange screen filter; a large, bright filter
    // would hide the flames, which are what actually reads as fire. The radius is a world
    // quantity with no zoom compensation.
    glows: sites.map((at) => ({
      x: at.x, y: at.y, radiusPx: spread,
      tint: 0xff7a2a,
      alphaRest: cap(0.34 * s.presence),
      alphaPeak: cap(0.5 * s.presence),
    })),
    shake: null,
    veil: null,
  };
}

/**
 * Smoke: columns, not a layer of fog. It reads as smoke because the puffs expand as they rise.
 *
 * Don't scatter it over the footprint like rain: that gives grey blotches with no source, like a
 * dirty screen. Smoke rises from a source (see SITED_PHENOMENA).
 *
 * A column needs all three: a narrow spawn band at the base, vertical speed much larger than
 * horizontal, and horizontal spread growing slowly through accelX (else a straight pipe).
 *
 * Like snow, smoke ignores vigor: thicker smoke doesn't rise faster.
 */
function smoke(area: WeatherArea, s: Intensity, ctx: WeatherContext): WeatherPlan {
  const sites = sitesInside(area, Math.round(Math.min(areaScale(area, ctx), MAX_SITES)));
  const root = SMOKE_PUFF_TILES * 0.7 * ctx.tilePx;   // width of the column's base
  return {
    layers: sites.map((at) => ({
      texture: "dot" as const,
      zone: { kind: "rect" as const, x: at.x - root / 2, y: at.y - 8, width: root, height: 16 },
      perSecond: 34 * s.count,
      lifespanMs: { min: 2600, max: 4400 },
      velocity: { y: { min: -76, max: -44 }, x: { min: -9, max: 9 }, accelX: { min: -8, max: 8 } },
      size: {
        uniform: {
          start: texScale(SMOKE_PUFF_TILES * 0.4 * s.weight, ctx, TEX_PX.dot),
          end: texScale(SMOKE_PUFF_TILES * 1.9 * s.weight, ctx, TEX_PX.dot),
        },
      },
      alpha: { start: cap(0.62 * s.presence), end: 0 },
      tint: [0x76767c, 0x4f4f55, 0x3a3a3f],
      rotate: { min: -60, max: 60 },
    })),
    shake: null,
    veil: null,
  };
}

// ── Non-particle effects ──────────────────────────────────────────────────────────

/** Quake: mainly a camera shake; the particles are just the aftermath.
 *
 *  A single sharp shake (the quake itself) plus dust falling for the whole step (the aftershock).
 *  The camera stands for the viewer's eyes, so it only shakes when the quaking ground is in view;
 *  shaking the whole view for a quake in a distant ward puts the viewer somewhere he isn't.
 *  Amplitude follows vigor, not count: a stronger quake shakes harder, not more often. */
function quake(area: WeatherArea, s: Intensity, ctx: WeatherContext): WeatherPlan {
  const inView = area.onScreen;
  const bd = boost(DUST_TILES * s.weight, ctx);
  return {
    layers: [{
      texture: "dot",
      zone: areaZone(area),
      perSecond: 50 * s.count * areaScale(area, ctx),
      lifespanMs: { min: 900, max: 1800 },
      velocity: { y: { min: 60, max: 150 }, x: { min: -18, max: 18 }, accelY: 90 },
      size: {
        uniform: { min: texScale(DUST_TILES * 0.5 * bd, ctx, TEX_PX.dot),
                   max: texScale(DUST_TILES * s.weight * bd, ctx, TEX_PX.dot) },
      },
      alpha: { start: cap(0.75 * s.presence), end: 0 },
      tint: [0xb59a7d, 0x8a7358],
    }],
    // 0.006×density (0.0108 at high) makes the whole frame jump at Phaser's scale and drowns out
    // everything else in the step. A short jolt reads as a quake; sustained heavy shaking looks
    // like a broken screen.
    shake: inView ? { durationMs: 300, intensity: 0.0016 * s.vigor } : null,
    veil: null,
  };
}

/** Darkness: the light going out, not something falling.
 *
 *  presence is its only dimension; count, vigor and weight mean nothing for darkness, which shows
 *  why a single density can't drive all seven phenomena. When local, only that ward darkens. */
function dark(_area: WeatherArea, s: Intensity, _ctx: WeatherContext): WeatherPlan {
  return {
    // No particle layers. Don't fake clouds with translucent dark blobs: soft dots can't draw clouds,
    // which are recognised by their outline, and blurry dark patches look like a dirty screen.
    // The veil alone conveys darkness.
    layers: [],
    shake: null,
    // The veil follows the footprint: the whole map when world-wide, the one ward when local.
    // It is the only visual for this phenomenon, so it is fairly opaque.
    veil: { alpha: Math.min(0.46 * s.presence, 0.7) },
  };
}
