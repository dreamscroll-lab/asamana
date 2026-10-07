/**
 * Turns a world's raw .tmj into something Phaser can render, without the map knowing about
 * Phaser. Tile sizes, tilesets, layer names are read off the document, never renderer
 * constants — otherwise a new map costs a code change.
 *
 * The exception is the projection: every map is isometric (enforced by
 * `template_check.REQUIRED_ORIENTATION`), since the cast has only the four isometric headings.
 */

import { type CharacterSet, characterSetFrom } from "./skins";

export interface TilesetSource {
  name: string; // as named in the .tmj — addTilesetImage matches on this
  key: string; // Phaser texture key
  url: string; // served by the backend, next to the map itself
}

/**
 * Which cells a body may be at, from the backend (`/map/ground`). Don't derive it here: it
 * comes off tileset alpha, and its three consumers (staging, A*, the template validator)
 * must share one answer, so it lives in `worlds/ground.py`.
 *
 * Two grids because a street may pass behind a wall but nobody may rest there. Don't
 * intersect them: that shatters the route network; the renderer bridges them by cost.
 */
export interface GroundGrids {
  cols: number;
  rows: number;
  walkable: boolean[][]; // a route may run here
  standable: boolean[][]; // a body may be seen at rest here
}

export interface MapSource {
  doc: Record<string, unknown>; // the .tmj, with render-only fixups applied
  tilesets: TilesetSource[];
  ground: GroundGrids | null; // null → the backend could not derive it; see the scene
  // The cast is part of the map's artifact (a body is dressed for the map's period):
  // authored beside the .tmj, frozen into the world at build.
  characters: CharacterSet;
}

interface RawTileset {
  name?: unknown;
  image?: unknown;
  tileheight?: unknown;
  tileoffset?: unknown;
}

/**
 * Phaser and Tiled disagree on where a tile image sits once tiles exceed the grid size.
 * Tiled anchors a tile's BOTTOM-LEFT on the cell's bottom-left (a tall tile grows up);
 * Phaser (`Tile.updatePixelXY`) puts its TOP-LEFT at the cell origin, so it lands down by
 * the extra height and half a tile right. The `tileoffset` Phaser subtracts corrects it.
 *
 * Applied here, never saved into the .tmj: Tiled honours the same field, so the map would
 * then render wrongly in the author's editor.
 */
function anchorOffset(ts: RawTileset, mapTileW: number, mapTileH: number) {
  const tileH = Number(ts.tileheight) || mapTileH;
  return { x: mapTileW / 2, y: tileH - mapTileH };
}

/** Unpack the wire form: one "0110…" string per row. */
function groundFrom(payload: unknown): GroundGrids | null {
  const raw = payload as { cols?: unknown; rows?: unknown; walkable?: unknown; standable?: unknown };
  const cols = Number(raw?.cols);
  const rows = Number(raw?.rows);
  if (!Number.isFinite(cols) || !Number.isFinite(rows) || cols <= 0 || rows <= 0) return null;
  const grid = (lines: unknown): boolean[][] | null => {
    if (!Array.isArray(lines) || lines.length !== rows) return null;
    return lines.map((line) => Array.from(String(line)).map((c) => c === "1"));
  };
  const walkable = grid(raw.walkable);
  const standable = grid(raw.standable);
  return walkable && standable ? { cols, rows, walkable, standable } : null;
}

/**
 * Prepare a fetched .tmj for the scene. `assets` maps each image path as written in the
 * map to its served URL (GET …/map/assets). The client never composes that URL — where art
 * lives is a deployment fact. An image with no entry is skipped, not guessed at.
 */
function prepareMapSource(
  doc: Record<string, unknown>,
  assets: Record<string, string>,
  characters: unknown,
  ground: GroundGrids | null,
): MapSource {
  const mapTileW = Number(doc.tilewidth) || 32;
  const mapTileH = Number(doc.tileheight) || 32;
  const raw = Array.isArray(doc.tilesets) ? (doc.tilesets as RawTileset[]) : [];

  const tilesets: TilesetSource[] = [];
  raw.forEach((ts, i) => {
    const image = String(ts.image ?? "");
    const url = assets[image];
    // No sheet or no served URL → skip; a missing tileset must not blank the world.
    if (!image || !url) return;
    const name = String(ts.name ?? `tileset_${i}`);
    ts.tileoffset = anchorOffset(ts, mapTileW, mapTileH);
    tilesets.push({ name, key: `tiles_${i}_${name}`, url });
  });

  // The cast is NOT optional like a tileset: this throws rather than draw an empty city.
  return { doc, tilesets, ground, characters: characterSetFrom(characters) };
}

/**
 * Fetch a whole map artifact (document, art index, ground, cast). `base` is the only thing
 * that varies: `/api/worlds/{id}` (frozen copy) or `/api/templates/{name}` (live files).
 *
 * Degrade rules belong to the artifact, not the caller: no asset index → draw without art;
 * no ground → the scene uses the map's own declarations (adoptGround); no cast or no map → throw.
 */
export async function loadMapSource(base: string): Promise<MapSource> {
  const [map, assets, cast, ground] = await Promise.all([
    fetch(`${base}/map`),
    fetch(`${base}/map/assets`),
    fetchCast(base),
    fetch(`${base}/map/ground`),
  ]);
  if (!map.ok) throw new Error(`map unavailable (${map.status})`);
  return prepareMapSource(
    await map.json(),
    assets.ok ? await assets.json() : {},
    cast,
    ground.ok ? groundFrom(await ground.json()) : null,
  );
}

/** The cast manifest alone, for a figure drawn outside the map. Same base as ``loadMapSource``. */
export async function loadCast(base: string): Promise<CharacterSet> {
  return characterSetFrom(await fetchCast(base));
}

async function fetchCast(base: string): Promise<unknown> {
  const r = await fetch(`${base}/characters`);
  if (!r.ok) throw new Error(`cast art unavailable (${r.status})`);
  return r.json();
}
