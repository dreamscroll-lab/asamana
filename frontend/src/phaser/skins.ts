/**
 * The cast's art, as the world's own map declares it. Pure data and deterministic
 * selection; no Phaser.
 *
 * Nothing about a body is compiled in: frames, sheets and scale arrive per world from
 * `GET /api/worlds/{id}/characters`. A body is dressed for the map's period, so the figures
 * ship with the template, are frozen into the world at build and served from that copy.
 * Art bundled with the renderer would restyle every past replay on an edit and make a map of
 * a new period cost a frontend change.
 *
 * Three independent axes:
 *
 *   BODY      = f(gender, age bracket) — chosen here, from the manifest's table.
 *   DIRECTION = f(travel)              — chosen here, from screen-space movement.
 *   COLOUR    = the agent's identity colour — applied by the GPU at draw time
 *               (recolorPipeline.ts), never baked into an asset.
 *
 * Taking colour from SoulLayer.color keeps the art theme-agnostic. Body and colour are pure
 * functions of build-time facts, so an agent looks the same in every step, replay and session.
 */

/** A texture key + a frame within it. */
export type Pose = [string, string];

/** The still poses a token can be put into. `attack` is an animation, not a still. */
export type PoseName =
  | "idle" | "hurt" | "shove" | "duck" | "hold" | "interact" | "talk" | "show" | "down";

/** Every pose the renderer asks a body for — the manifest must map all of them. */
export const POSE_NAMES = [
  "idle", "walk", "down", "attack", "hurt", "shove", "duck", "hold", "interact", "talk", "show",
] as const;

/**
 * Isometric travel directions, named by where they head on screen. On a 2:1
 * isometric grid the four ways to walk project to the four screen diagonals, so
 * these are the only headings a figure can have; "left" and "right" are not.
 */
export const DIRECTIONS = ["NE", "SE", "SW", "NW"] as const;
export type Dir = (typeof DIRECTIONS)[number];

/** Heading to screen-left. Art drawn facing east is mirrored for these. */
const isWest = (dir: Dir): boolean => dir[1] === "W";

/**
 * How art declares one pose. Which shape it uses is the ART's business, not the
 * renderer's: side-view art has no directions to give and is mirrored for
 * west-facing travel; isometric art draws all four and is never mirrored (mirroring
 * a drawn direction would flip the asymmetries the artist put in it).
 */
type FrameSpec = string | string[] | Record<Dir, string | string[]>;

interface Manifest {
  scale: number;
  walk_fps?: number;
  bodies: Record<string, Record<string, string>>;
  atlases: Record<string, { image: string; atlas: string; portrait?: string }>;
  frames: Record<string, FrameSpec>;
}

/**
 * A sheet to preload: one texture key, its image URL and its atlas descriptor URL. `portrait`
 * is the body's high-resolution close-up, standing as its SE idle frame stands; the map never
 * loads it.
 */
export interface AtlasLoad {
  key: string;
  image: string;
  atlas: string;
  portrait?: string;
}

/** The agent facts that pick a body. Both come from the step-0 profile. */
export interface Demographics {
  gender: string;
  age: number | null;
}

/**
 * Age brackets, as distinctions readable on a ~96px figure: head-to-body ratio (child),
 * build and dress (middle), white hair and a shortened stance (elder). A finer split would
 * be invisible at this size.
 *
 * The boundaries fall between the multiples of five an LLM reaches for when writing ages,
 * so two characters of the same apparent age (45 and 46) never get different bodies.
 */
const BRACKETS: { key: string; maxAge: number }[] = [
  { key: "child", maxAge: 13 },
  { key: "young", maxAge: 37 },
  { key: "middle", maxAge: 57 },
  { key: "elder", maxAge: Infinity },
];

/**
 * The two axes a body is chosen on, as the manifest's `bodies` table is keyed. Derived
 * from the table above so a bracket can't be added in one place and missed in the other.
 */
export const GENDERS = ["male", "female"] as const;
export const AGE_BRACKETS = BRACKETS.map((b) => b.key);

/** An age we didn't get: assume an adult rather than a youth — casts skew adult. */
const ASSUMED_AGE = 40;

/** The bracket an age falls in. */
function bracketFor(age: number | null | undefined): string {
  const value = age ?? ASSUMED_AGE;
  return (BRACKETS.find((b) => value <= b.maxAge) ?? BRACKETS[BRACKETS.length - 1]).key;
}

/**
 * The cast's art for one world: what to preload, which body an agent gets, and
 * what each pose looks like in this particular art.
 */
export class CharacterSet {
  readonly scale: number;
  readonly walkFps: number;
  readonly atlases: AtlasLoad[];
  private readonly bodies: Record<string, Record<string, string>>;
  private readonly frames: Record<string, FrameSpec>;
  /** Body key used when the manifest cannot supply one — never undefined downstream. */
  private readonly fallbackBody: string;

  constructor(manifest: Manifest) {
    this.scale = Number(manifest.scale) || 1;
    this.walkFps = Number(manifest.walk_fps) || 10;
    this.bodies = manifest.bodies ?? {};
    this.frames = manifest.frames ?? {};
    this.atlases = Object.entries(manifest.atlases ?? {}).map(([key, entry]) => ({
      key,
      image: entry.image,
      atlas: entry.atlas,
      portrait: entry.portrait,
    }));
    this.fallbackBody = this.atlases[0]?.key ?? "";
  }

  /** Every body key this set can hand out — what the scene registers anims for. */
  bodyKeys(): string[] {
    return this.atlases.map((a) => a.key);
  }

  /**
   * The body for an agent: a pure function of its gender and age, so it can't shift
   * with which agents are loaded. Unknown demographics (profile not yet landed) fall
   * through to an adult male body until it does.
   */
  bodyFor(who: Demographics | undefined): string {
    // The backend's gender is 「男」/「女」 or "" (world/builders/theme_analyzer._normalize_gender).
    const gender = who?.gender === "女" ? "female" : "male";
    const byAge = this.bodies[gender] ?? this.bodies.male ?? {};
    return byAge[bracketFor(who?.age)] ?? this.fallbackBody;
  }

  /**
   * The body the manifest maps this exact cell to, or null where it maps none.
   *
   * Unlike `bodyFor`, which falls back so a figure always has something to draw and so
   * hides the gap an art review is looking for.
   */
  bodyAt(gender: string, bracket: string): string | null {
    return this.bodies[gender]?.[bracket] ?? null;
  }

  /**
   * The art for one pose in one direction: which frames to draw, and whether to
   * mirror them.
   *
   * The art decides mirroring: a pose given per-direction is already drawn facing that
   * way and is never flipped; a pose given once faces east and is flipped to head west.
   * Null when the manifest lacks the pose, so the caller leaves the figure as it is.
   */
  poseArt(body: string, pose: string, dir: Dir): { frames: Pose[]; flip: boolean } | null {
    const spec = this.frames[pose];
    if (spec === undefined) return null;
    const directional = !Array.isArray(spec) && typeof spec === "object";
    const chosen = directional ? (spec as Record<Dir, string | string[]>)[dir] : spec;
    if (chosen === undefined) return null;
    const names = Array.isArray(chosen) ? chosen : [chosen];
    if (!names.length) return null;
    return {
      frames: names.map((frame): Pose => [body, frame]),
      flip: !directional && isWest(dir),
    };
  }
}

/**
 * The direction a figure heads, from its screen-space movement.
 *
 * Both axes carry part of an isometric heading: vertical travel tells walking away from
 * the camera from walking toward it. Each axis keeps its previous value when movement
 * along it is too small to read, so a figure crossing the screen doesn't flicker.
 */
export function dirOf(dx: number, dy: number, previous: Dir): Dir {
  const EPS = 0.15;
  const ns = Math.abs(dy) > EPS ? (dy > 0 ? "S" : "N") : previous[0];
  const ew = Math.abs(dx) > EPS ? (dx > 0 ? "E" : "W") : previous[1];
  return `${ns}${ew}` as Dir;
}

/**
 * The direction a figure TURNS TO FACE something, from the screen offset to it and the
 * map's tile shape.
 *
 * Don't reuse dirOf: its "keep the previous value when an axis is too small" rule, right for
 * travel noise, here means keep facing where you already were. Co-seated figures very often
 * have dx == 0 or dy == 0 exactly, so a man who last walked west would reach south-west for
 * a crate due south of him.
 *
 * So this decides from geometry alone, in grid space: screen y is compressed by the
 * projection (raw dx vs dy under-weights north/south), and the four drawable directions are
 * the four grid axes, so the answer is just "which grid axis dominates".
 *
 * Inverting the projection: sx = (gx - gy)·tw/2 and sy = (gx + gy)·th/2, so gx - gy = 2·sx/tw
 * and gx + gy = 2·sy/th. On this map +gx runs SE and +gy runs SW (see pathfinding.ts).
 *
 * `previous` breaks a genuine tie only: |gx| == |gy|, a grid diagonal where both headings
 * are equally correct.
 */
export function bearing(dx: number, dy: number, tw: number, th: number, previous: Dir): Dir {
  const u = (2 * dx) / (tw || 1); // gx - gy
  const v = (2 * dy) / (th || 1); // gx + gy
  const gx = (u + v) / 2;
  const gy = (v - u) / 2;
  const ax = Math.abs(gx);
  const ay = Math.abs(gy);
  if (Math.abs(ax - ay) < 1e-6) {
    const tied: Dir[] = [gx > 0 ? "SE" : "NW", gy > 0 ? "SW" : "NE"];
    return tied.includes(previous) ? previous : tied[0];
  }
  return ax > ay ? (gx > 0 ? "SE" : "NW") : gy > 0 ? "SW" : "NE";
}

/** Parse a manifest served by the backend. Throws if it carries no usable art. */
export function characterSetFrom(manifest: unknown): CharacterSet {
  const set = new CharacterSet(manifest as Manifest);
  if (!set.atlases.length) throw new Error("character manifest names no atlases");
  return set;
}
