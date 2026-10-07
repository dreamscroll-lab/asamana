/**
 * EmergingWorld — the world-building wait, drawn as the thing it is waiting for: a dark,
 * unnamed planet turns while lights come on one at a time and threads grow between
 * neighbours (inhabitants appear, connections emerge).
 *
 * 1. Not Earth: recognisable continents would assert a geography no world here has.
 * 2. Slow and sparse, to avoid the "global network" landing-page cliché.
 * 3. It settles rather than loops: elapsed time is all the frontend knows about a build.
 *
 * Canvas 2D with a hand-rolled orthographic projection — no 3D dependency for one screen.
 */
import { useEffect, useRef } from "react";

import { THEME } from "../lib/theme";

const R = 116; // sphere radius, CSS px
/** The atmosphere reaches this many radii; the canvas must fit it or the glow is squared
 *  off. Exported: anything composed against the sphere (the splash wordmark) divides the
 *  box by this to find the world's edge. */
export const HALO = 1.55;
const SIZE = Math.ceil(R * HALO * 2);
const SPIN_MS = 42_000; // one revolution
const TILT = 0.34; // ~19°, so we look at it from slightly above
const LIGHT = { x: -0.5, y: -0.5 }; // where the light falls from, in radii
const POP = 44; // inhabitants, arriving one at a time
const BIRTH_MS = 260; // gap between arrivals
const FLARE_MS = 1100; // how long the arrival ripple lives
const LINK_MS = 900; // how long a thread takes to draw itself in
const LINK_DELAY = 700; // a thread waits this long after its later end arrives
const LINK_MAX = 0.62; // longest chord (unit sphere) that may become a thread
const LINKS_PER = 2; // threads a new arrival may grow back to earlier ones
const SETTLED_MS = POP * BIRTH_MS + LINK_DELAY + LINK_MS; // fully-populated frame

interface V3 {
  x: number;
  y: number;
  z: number;
}

/** Fixed seed: this is brand art, so it must look the same every time it plays. */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

function norm(v: V3): V3 {
  const m = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / m, y: v.y / m, z: v.z / m };
}

function onSphere(rand: () => number): V3 {
  const phi = Math.acos(2 * rand() - 1);
  const theta = 2 * Math.PI * rand();
  return {
    x: Math.sin(phi) * Math.cos(theta),
    y: Math.cos(phi),
    z: Math.sin(phi) * Math.sin(theta),
  };
}

/** Nudge a point off `c` by roughly `amt` radians — settlements near a region. */
function near(c: V3, amt: number, rand: () => number): V3 {
  const d = onSphere(rand);
  return norm({ x: c.x + d.x * amt, y: c.y + d.y * amt, z: c.z + d.z * amt });
}

function at(lat: number, lon: number): V3 {
  return {
    x: Math.cos(lat) * Math.cos(lon),
    y: Math.sin(lat),
    z: Math.cos(lat) * Math.sin(lon),
  };
}

// --- The world, built once at module load (fixed seed → identical every mount) ---

const rand = rng(0x10c3b0);

/** Soft overlapping caps that read as land, with no silhouette to recognise. */
const LAND = Array.from({ length: 7 }, () => ({
  c: onSphere(rand),
  r: 0.30 + rand() * 0.26,
}));

/** Inhabitants cluster around the land — people live in places, not on a lattice. */
const PEOPLE: V3[] = Array.from({ length: POP }, (_, i) =>
  near(LAND[i % LAND.length].c, 0.16 + rand() * 0.36, rand),
);

/** Threads: each arrival reaches back to its nearest few predecessors. */
const LINKS: { a: number; b: number; born: number }[] = [];
for (let i = 1; i < PEOPLE.length; i++) {
  const p = PEOPLE[i];
  const cands = [];
  for (let j = 0; j < i; j++) {
    const q = PEOPLE[j];
    const d = Math.hypot(p.x - q.x, p.y - q.y, p.z - q.z);
    if (d < LINK_MAX) cands.push({ j, d });
  }
  cands.sort((m, n) => m.d - n.d);
  for (const { j } of cands.slice(0, LINKS_PER)) {
    LINKS.push({ a: i, b: j, born: i * BIRTH_MS + LINK_DELAY });
  }
}

/** The graticule, sampled in world space once; each frame projects and drops the far side. */
const D = Math.PI / 180;
const MERIDIANS: V3[][] = Array.from({ length: 6 }, (_, m) =>
  Array.from({ length: 61 }, (_, i) => at((-90 + i * 3) * D, m * 30 * D)),
);
const PARALLELS: V3[][] = [-60, -30, 0, 30, 60].map((lat) =>
  Array.from({ length: 121 }, (_, i) => at(lat * D, i * 3 * D)),
);

/** Spin about the world's axis, then tilt the axis toward the viewer. +z faces us. */
function project(v: V3, spin: number): { sx: number; sy: number; z: number } {
  const cs = Math.cos(spin);
  const sn = Math.sin(spin);
  const x = v.x * cs + v.z * sn;
  const z0 = -v.x * sn + v.z * cs;
  const ct = Math.cos(TILT);
  const st = Math.sin(TILT);
  return {
    sx: SIZE / 2 + x * R,
    sy: SIZE / 2 + (v.y * ct - z0 * st) * R,
    z: v.y * st + z0 * ct,
  };
}

function hex(a: number): string {
  return Math.round(Math.max(0, Math.min(1, a)) * 255)
    .toString(16)
    .padStart(2, "0");
}

/** Stroke a world-space curve, lifting the pen wherever it passes behind. */
function strokeCurve(
  ctx: CanvasRenderingContext2D,
  pts: V3[],
  spin: number,
): void {
  ctx.beginPath();
  let down = false;
  for (const v of pts) {
    const p = project(v, spin);
    if (p.z <= 0) {
      down = false;
      continue;
    }
    if (down) ctx.lineTo(p.sx, p.sy);
    else {
      ctx.moveTo(p.sx, p.sy);
      down = true;
    }
  }
  ctx.stroke();
}

function draw(ctx: CanvasRenderingContext2D, t: number): void {
  const cx = SIZE / 2;
  const cy = SIZE / 2;
  const spin = (t / SPIN_MS) * Math.PI * 2;

  ctx.clearRect(0, 0, SIZE, SIZE);

  // Atmosphere: the world's own halo, so the sphere sits in something.
  const air = ctx.createRadialGradient(cx, cy, R * 0.92, cx, cy, R * HALO);
  air.addColorStop(0, `${THEME.accent}55`);
  air.addColorStop(0.45, `${THEME.accent}1c`);
  air.addColorStop(1, `${THEME.accent}00`);
  ctx.fillStyle = air;
  ctx.beginPath();
  ctx.arc(cx, cy, R * HALO, 0, Math.PI * 2);
  ctx.fill();

  // Everything on the surface is clipped to the disc, so land can run off the limb.
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, Math.PI * 2);
  ctx.clip();

  ctx.fillStyle = THEME.bg;
  ctx.fill();

  for (const l of LAND) {
    const p = project(l.c, spin);
    if (p.z <= 0) continue; // far side of an opaque world
    const rad = l.r * R;
    // Squash along the line to the centre by z: a patch turning away from us
    // narrows, which is most of what makes the surface read as curved.
    ctx.save();
    ctx.translate(p.sx, p.sy);
    ctx.rotate(Math.atan2(p.sy - cy, p.sx - cx));
    ctx.scale(Math.max(0.1, p.z), 1);
    const g = ctx.createRadialGradient(0, 0, 0, 0, 0, rad);
    // Land is a lift, not a stain: the brand ramp's deep end at low alpha, so it reads
    // as ground catching light rather than a pasted purple patch.
    const a = Math.min(1, p.z * 2.2);
    g.addColorStop(0, `${THEME.accentDeep}${hex(0.45 * a)}`);
    g.addColorStop(1, `${THEME.accentDeep}00`);
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(0, 0, rad, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  ctx.lineWidth = 0.7;
  ctx.strokeStyle = `${THEME.brand400}1f`;
  for (const m of MERIDIANS) strokeCurve(ctx, m, spin);
  for (const p of PARALLELS) strokeCurve(ctx, p, spin);
  ctx.strokeStyle = `${THEME.brand300}30`; // equator, a shade brighter
  strokeCurve(ctx, PARALLELS[2], spin);

  // A bright ring concentric with the disc: an offset highlight leaves a visible seam
  // that reads as glass. The shadow below turns it into a crescent.
  const limb = ctx.createRadialGradient(cx, cy, R * 0.87, cx, cy, R);
  limb.addColorStop(0, `${THEME.brand300}00`);
  limb.addColorStop(1, `${THEME.brand300}4d`);
  ctx.fillStyle = limb;
  ctx.globalCompositeOperation = "lighter";
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.globalCompositeOperation = "source-over";

  // The far shoulder (and the ring) rolls off into shadow. Last inside the clip, so it
  // darkens everything above.
  const lx = cx + LIGHT.x * R;
  const ly = cy + LIGHT.y * R;
  const shade = ctx.createRadialGradient(lx, ly, 0, lx, ly, R * 2.1);
  shade.addColorStop(0, "#00000000");
  shade.addColorStop(0.42, "#0000002b");
  shade.addColorStop(1, "#000000cc");
  ctx.fillStyle = shade;
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.restore();

  // Silhouette: bright where the light hits it, nearly gone on the shadow side.
  const rim = ctx.createLinearGradient(cx - R, cy - R, cx + R, cy + R);
  rim.addColorStop(0, `${THEME.brand300}cc`);
  rim.addColorStop(0.5, `${THEME.brand400}55`);
  rim.addColorStop(1, `${THEME.accentDeep}44`);
  ctx.strokeStyle = rim;
  ctx.lineWidth = 1.1;
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, Math.PI * 2);
  ctx.stroke();

  // Threads first, so the lights sit on top of their own connections.
  for (const link of LINKS) {
    const age = t - link.born;
    if (age <= 0) continue;
    const a = project(PEOPLE[link.a], spin);
    const b = project(PEOPLE[link.b], spin);
    if (a.z <= 0.02 || b.z <= 0.02) continue;
    const grow = Math.min(1, age / LINK_MS);
    const depth = Math.min(a.z, b.z);
    ctx.strokeStyle = `${THEME.accentSoft}${hex(0.5 * grow * depth)}`;
    ctx.lineWidth = 0.9;
    const mx = (a.sx + b.sx) / 2;
    const my = (a.sy + b.sy) / 2;
    ctx.beginPath();
    ctx.moveTo(a.sx, a.sy);
    // Bow the thread away from the centre so it reads as arcing over the surface.
    ctx.quadraticCurveTo(cx + (mx - cx) * 1.1, cy + (my - cy) * 1.1, b.sx, b.sy);
    ctx.stroke();
  }

  for (let i = 0; i < PEOPLE.length; i++) {
    const age = t - i * BIRTH_MS;
    if (age <= 0) continue;
    const p = project(PEOPLE[i], spin);
    if (p.z <= 0) continue;

    // The arrival ripple: one ring, expanding out and fading, then gone for good.
    if (age < FLARE_MS) {
      const k = age / FLARE_MS;
      ctx.strokeStyle = `${THEME.brand300}${hex((1 - k) * 0.55 * p.z)}`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, 2 + k * 13, 0, Math.PI * 2);
      ctx.stroke();
    }

    const on = Math.min(1, age / 400);
    const rad = (1.5 + p.z * 1.1) * on;
    const halo = ctx.createRadialGradient(p.sx, p.sy, 0, p.sx, p.sy, rad * 4);
    halo.addColorStop(0, `${THEME.brand300}${hex(0.36 * on * p.z)}`);
    halo.addColorStop(1, `${THEME.brand300}00`);
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(p.sx, p.sy, rad * 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = `#f2ecff${hex(0.35 + 0.65 * p.z)}`;
    ctx.beginPath();
    ctx.arc(p.sx, p.sy, rad, 0, Math.PI * 2);
    ctx.fill();
  }
}

/** `size` only scales the backing store; everything above is in the sphere's own
 *  SIZE-px space, so strokes and lights stay in proportion. */
export default function EmergingWorld({ size = SIZE }: { size?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  // Elapsed time lives outside the effect so a resize (the splash sizes to the pane)
  // doesn't replay the arrivals from zero.
  const t0 = useRef(0);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    // Cap DPR at 2 — past that this costs fill rate for nothing visible.
    const dpr = Math.min(window.devicePixelRatio || 1, 2) * (size / SIZE);
    canvas.width = SIZE * dpr;
    canvas.height = SIZE * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr); // after sizing — setting width resets the transform

    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
      draw(ctx, SETTLED_MS); // one frame: the world already populated, and still
      return;
    }

    let raf = 0;
    const tick = (now: number) => {
      if (!t0.current) t0.current = now;
      draw(ctx, now - t0.current);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [size]);

  return <canvas ref={ref} aria-hidden style={{ width: size, height: size }} />;
}
