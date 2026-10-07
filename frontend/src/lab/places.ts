/**
 * Which of this map's places the fixtures act in.
 *
 * Scenes need places in particular relations (one due NE of another, a pair the map's graph joins,
 * roomy ground, tight ground). Naming real places in the fixtures would make the bench work on one
 * map only: on any other, nobody could be staged. So the fixtures own the roles and the loaded map
 * casts them. Only ids are picked here; the renderer still stages, routes and animates from the
 * payload alone.
 *
 * Read off the map document (no need to wait on a Phaser boot), using the same fields the renderer
 * reads (`type="location"` objects and the backend's standable grid).
 */

import type { GroundGrids, MapSource } from "../phaser/mapSource";

/** A place, as the fixtures refer to one: the wire carries both its id and its name. */
export interface LabRoom {
  id: string;
  name: string;
}

/** The seven roles the fixtures cast: the relations a map has to be able to supply. */
export interface LabPlaces {
  /** The heading loop's hub: the one place both `ne` and `se` are cleanly reached from. */
  pivot: LabRoom;
  /** Due NE of `pivot` — walking there reads as NE, walking back as SW. */
  ne: LabRoom;
  /** Due SE of `pivot` — walking there reads as SE, walking back as NW. */
  se: LabRoom;
  /** Far enough out that `pivot → ne → far` is a genuine multi-leg trek. */
  far: LabRoom;
  /** The most open ground on the map: where the deeds are staged and the weather falls. */
  open: LabRoom;
  /** The tightest ground: six figures in here have to be crowded apart. */
  tight: LabRoom;
  /** Joined to `tight` in the map's own connection graph — so a move between them is ONE step. */
  next: LabRoom;
}

/** A walk shorter than this shows no gait worth judging, in grid cells. */
const MIN_SPAN = 3;

interface Candidate extends LabRoom {
  gx: number; // centre, in grid cells
  gy: number;
  gx0: number; // rectangle origin and extent, same units
  gy0: number;
  gw: number;
  gh: number;
  connections: string[];
  ground: number; // standable cells inside the rectangle
}

interface RawObject {
  type?: unknown;
  name?: unknown;
  x?: unknown;
  y?: unknown;
  width?: unknown;
  height?: unknown;
  properties?: { name?: unknown; value?: unknown }[];
}

interface RawLayer {
  type?: unknown;
  layers?: unknown;
  objects?: unknown;
}

/** Every object on the map, groups walked through — Tiled nests object layers in them. */
function objectsOf(layers: unknown): RawObject[] {
  const found: RawObject[] = [];
  const walk = (ls: unknown): void => {
    if (!Array.isArray(ls)) return;
    for (const layer of ls as RawLayer[]) {
      if (layer.type === "group") walk(layer.layers);
      else if (layer.type === "objectgroup" && Array.isArray(layer.objects)) {
        found.push(...(layer.objects as RawObject[]));
      }
    }
  };
  walk(layers);
  return found;
}

/**
 * How many cells inside a place's rectangle a body may stand on, by the same test `findStanding`
 * applies (cell centre inside the rectangle). With no grid served, falls back to the rectangle's
 * area, which is blind to what the art puts inside it.
 */
function standingRoom(ground: GroundGrids | null, c: Omit<Candidate, "ground">): number {
  if (!ground) return Math.max(1, Math.round(c.gw * c.gh));
  const inside = (x: number, y: number): boolean =>
    x + 0.5 >= c.gx0 && x + 0.5 <= c.gx0 + c.gw && y + 0.5 >= c.gy0 && y + 0.5 <= c.gy0 + c.gh;
  let n = 0;
  for (let y = Math.max(0, Math.floor(c.gy0)); y <= Math.min(ground.rows - 1, Math.ceil(c.gy0 + c.gh)); y++) {
    for (let x = Math.max(0, Math.floor(c.gx0)); x <= Math.min(ground.cols - 1, Math.ceil(c.gx0 + c.gw)); x++) {
      if (ground.standable[y]?.[x] && inside(x, y)) n++;
    }
  }
  return n;
}

function candidates(source: MapSource): Candidate[] {
  const doc = source.doc;
  // On an isometric map Tiled's object space unit on both axes is the tile height (as in
  // parseLocations and worlds/tiled.py); dividing by the width halves every distance.
  const unit = Number(doc.tileheight) || 1;
  const out: Candidate[] = [];
  for (const obj of objectsOf(doc.layers)) {
    if (obj.type !== "location") continue;
    const props = Object.fromEntries(
      (obj.properties ?? []).map((p) => [String(p.name ?? ""), p.value]),
    );
    const id = String(props.location_id ?? "").trim();
    if (!id) continue;
    const gx0 = Number(obj.x ?? 0) / unit;
    const gy0 = Number(obj.y ?? 0) / unit;
    const gw = Number(obj.width ?? 0) / unit;
    const gh = Number(obj.height ?? 0) / unit;
    const geometry = {
      id,
      name: String(obj.name ?? id),
      gx: gx0 + gw / 2,
      gy: gy0 + gh / 2,
      gx0,
      gy0,
      gw,
      gh,
      connections: String(props.connections ?? "")
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
    };
    out.push({ ...geometry, ground: standingRoom(source.ground, geometry) });
  }
  return out;
}

/**
 * The two grid axes, as screen headings.
 *
 * On a 2:1 isometric grid `sx = (gx - gy)·tw/2` and `sy = (gx + gy)·th/2`, so +gx alone reads as SE
 * and -gy alone as NE (skins.dirOf reads each screen delta's sign independently). `along` is the
 * distance that produces the heading, `off` the drift that muddies it: with comparable drift the
 * figure flickers between two headings, so purity, not distance, leads the cost.
 */
const AXES: Record<"ne" | "se", { along: (dgx: number, dgy: number) => number; off: (dgx: number, dgy: number) => number }> = {
  ne: { along: (_dgx, dgy) => -dgy, off: (dgx) => Math.abs(dgx) },
  se: { along: (dgx) => dgx, off: (_dgx, dgy) => Math.abs(dgy) },
};

/**
 * The best partner for one heading out of `from`, or null if the map has none.
 *
 * The distance term (a thousandth per cell) only breaks ties between equally clean pairs, preferring
 * the nearer: a longer trek bends more along the roads.
 */
function partner(
  from: Candidate,
  all: Candidate[],
  axis: keyof typeof AXES,
): { room: Candidate; cost: number } | null {
  const { along, off } = AXES[axis];
  let best: { room: Candidate; cost: number } | null = null;
  for (const to of all) {
    if (to.id === from.id) continue;
    const span = along(to.gx - from.gx, to.gy - from.gy);
    if (span < MIN_SPAN) continue;
    const cost = off(to.gx - from.gx, to.gy - from.gy) / span + span / 1000;
    if (!best || cost < best.cost) best = { room: to, cost };
  }
  return best;
}

/** Straight-line distance in grid cells — only ever compared, never displayed. */
const gap = (a: Candidate, b: Candidate): number => Math.hypot(a.gx - b.gx, a.gy - b.gy);

/**
 * Cast every role from the loaded map, or null if the map cannot fill them; otherwise scenes would
 * run against places lacking the relation they check and quietly certify nothing.
 */
export function resolvePlaces(source: MapSource): LabPlaces | null {
  const all = candidates(source);
  if (all.length < 3) return null;

  // The hub is the place with the cleanest pair of headings out of it, scored by the worse one:
  // the loop walks both.
  let pivot: Candidate | null = null;
  let ne: Candidate | null = null;
  let se: Candidate | null = null;
  let bestCost = Infinity;
  for (const from of all) {
    const a = partner(from, all, "ne");
    const b = partner(from, all, "se");
    if (!a || !b || a.room.id === b.room.id) continue;
    const cost = Math.max(a.cost, b.cost);
    if (cost < bestCost) {
      bestCost = cost;
      pivot = from;
      ne = a.room;
      se = b.room;
    }
  }
  if (!pivot || !ne || !se) return null;

  // Far from both the hub and `ne`, so both legs of the trek are worth watching.
  const taken = new Set([pivot.id, ne.id, se.id]);
  const rest = all.filter((c) => !taken.has(c.id));
  const far = rest.length
    ? rest.reduce((best, c) => (gap(pivot, c) + gap(ne, c) > gap(pivot, best) + gap(ne, best) ? c : best))
    : se;

  const open = all.reduce((best, c) => (c.ground > best.ground ? c : best));

  // A move takes one step when the map's graph joins the two places, so the pair comes from
  // `connections`, not distance. The near end is the tightest such place so it also serves the
  // crowding scene.
  const byId = new Map(all.map((c) => [c.id, c]));
  let tight: Candidate | null = null;
  let next: Candidate | null = null;
  for (const from of all) {
    if (from.id === open.id) continue;
    for (const id of from.connections) {
      const to = byId.get(id);
      if (!to || to.id === from.id) continue;
      if (!tight || from.ground < tight.ground) {
        tight = from;
        next = to;
      }
      break;
    }
  }
  // No declared connections: the scene still renders (the payload doesn't consult the graph), it
  // just stops being a faithful one-step move.
  if (!tight || !next) {
    const spare = all.filter((c) => c.id !== open.id).sort((a, b) => a.ground - b.ground);
    if (spare.length < 2) return null;
    [tight, next] = spare;
  }

  const room = ({ id, name }: Candidate): LabRoom => ({ id, name });
  return {
    pivot: room(pivot),
    ne: room(ne),
    se: room(se),
    far: room(far),
    open: room(open),
    tight: room(tight),
    next: room(next),
  };
}
