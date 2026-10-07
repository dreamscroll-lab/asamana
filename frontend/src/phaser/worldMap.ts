/**
 * The map as this renderer understands it: one coordinate system and everything in it. The
 * isometric projection, the grids the backend derived from the art (walkable, standable, extra
 * cost), the location regions from the Tiled object layer, and the terrain and place-names drawn
 * from them all share one space (`iso()` needs the tile size, rooms are placed by `iso()`, the
 * grids are indexed in it), so they stay together. It calls nothing back: it needs only the map
 * document and a scene.
 *
 * The camera is deliberately not here: zoom limits and whether the user grabbed the view are
 * facts about a viewer, not the map.
 */

import Phaser from "phaser";

import type { IsoGeometry } from "./isoDepth";
import type { MapSource } from "./mapSource";
import { FONT, TEXT_RES } from "./types";

export interface Room {
  id: string;
  name: string;
  gx: number; // grid-space centre (tiles) — drives A* endpoints + iso projection
  gy: number;
  gx0: number; // grid-space rect (tiles) — for click hit-testing which location was clicked
  gy0: number;
  gw: number;
  gh: number;
  sx: number; // projected screen anchor — agents rest here, labels/bubbles hang off it
  sy: number;
  atx: number; // walkable anchor tile (for A* endpoints), snapped to the road net
  aty: number;
  // Ground anyone in this room may be placed on (see findStanding), best first.
  // `inRect` counts the leading entries that lie inside the room's own rectangle;
  // anything past it came from widening the search, which a room drawn entirely over
  // rooftops needs — better its people stand on the street outside than on the roofs.
  stand: { x: number; y: number }[];
  inRect: number;
}

// What crossing one cell of hidden ground costs A*, on top of the 1 every step costs. A
// faded walker (see hiddenGround) reads worse than a plain one, so at 3 a route detours up to
// three extra cells to avoid one hidden cell. Don't forbid hidden cells: that breaks the
// route network into dozens of pieces, and 41% of the metro map's streets run under a canopy.
const HIDDEN_STEP_TOLL = 3;

// How far past its own rectangle a room may reach for ground, in cells.
const ROOM_STAND_REACH = 10;

// How many standable cells to keep per room. A room never stages more figures than it
// has cells here in practice; past this the layout is hopeless anyway and they double up.
const ROOM_STAND_CELLS = 64;

export class WorldMap {
  /** The location regions this map declares, by id. */
  readonly rooms = new Map<string, Room>();

  // --- the grid, as the backend derived it from the art (see mapSource.GroundGrids) ---
  cols = 0;
  rows = 0;
  blocked: boolean[][] = [];
  standable: boolean[][] = [];
  toll: number[][] = [];

  // --- the projection ---
  tileW = 32; // the MAP's grid cell size (not a tileset's tile size,
  tileH = 32; // which may be far larger — a wall spanning many cells)
  originX = 0; // x shift so the leftmost projected column lands at x≥0
  originY = 0; // top headroom for tiles that stand up out of their cell
  worldW = 0; // projected map bounds — camera fit + bubble clamping
  worldH = 0;


  constructor(private scene: Phaser.Scene, private source: MapSource) {}

  /**
   * Read the document and draw the world it describes.
   *
   * Split from `drawLabels` because the scene snaps each room's A* anchor onto the walkable
   * network in between, and creation order matters: same-depth objects draw in creation order.
   */
  build(map: Phaser.Tilemaps.Tilemap): void {
    this.tileW = map.tileWidth;
    this.tileH = map.tileHeight;
    this.cols = map.width;
    this.rows = map.height;
    this.setupProjection();
    this.adoptGround(map);
    this.drawTerrain(map);
    this.parseLocations(map);
  }

  /** The place-names, drawn once the rooms have been staged. */
  drawLabels(): void {
    this.drawLocationLabels();
  }

  /**
   * Project a (fractional) GRID coordinate to screen space.
   *
   * Tiled's own isometric formula (which is also Phaser's, up to the origin
   * shift), so an agent standing on a tile lands on that tile and not half a
   * cell away.
   */
  iso(gx: number, gy: number): { x: number; y: number } {
    return {
      x: (gx - gy) * (this.tileW / 2) + this.originX,
      y: (gx + gy) * (this.tileH / 2) + this.originY,
    };
  }

  /** Inverse of `iso`: a screen point back to a fractional grid coordinate. */
  unproject(wx: number, wy: number): { gx: number; gy: number } {
    const dx = wx - this.originX;
    const dy = wy - this.originY;
    const u = dx / (this.tileW / 2);
    const v = dy / (this.tileH / 2);
    return { gx: (u + v) / 2, gy: (v - u) / 2 };
  }

  /**
   * Fix the world's screen extent and the shift that keeps it at x ≥ 0.
   *
   * Isometric projection sends the gy axis into negative x, so the whole diamond
   * is pushed right by half the grid's height — the same origin Tiled uses, which
   * is why a location object's coordinates and a rendered tile agree. `originY`
   * is headroom: tall tiles (a wall, a pagoda) are anchored by their base and
   * grow UP out of their cell, so the top row would otherwise be clipped.
   */
  setupProjection(): void {
    const headroom = this.tallestTileOverhang();
    this.originX = this.rows * (this.tileW / 2);
    this.originY = headroom;
    this.worldW = (this.cols + this.rows) * (this.tileW / 2);
    this.worldH = (this.cols + this.rows) * (this.tileH / 2) + headroom + this.tileH;
  }

  // How far the tallest tileset tile rises above its own cell. Read off the map,
  // so a setting with taller art simply gets more headroom.
  tallestTileOverhang(): number {
    let overhang = 0;
    for (const ts of (this.source.doc.tilesets as { tileheight?: unknown }[]) ?? []) {
      overhang = Math.max(overhang, (Number(ts.tileheight) || 0) - this.tileH);
    }
    return overhang;
  }

  /**
   * How many CELLS beyond the viewport a tile may sit and still paint into it — the
   * padding Phaser's culling needs in order to be correct for this map's art.
   *
   * Phaser culls by the tile's cell, not its picture (see CheckIsoBounds), and its default of
   * 1 cell assumes art about a cell big. Tiled anchors a tile at its bottom-left, so a 768×384
   * wall on a 128×64 grid paints six cells right and five up; at the default, whole buildings
   * vanish along the view's left and bottom edges. So, like the headroom above, the padding is
   * read off the largest tile in this map's tilesets.
   */
  cullPadding(): { x: number; y: number } {
    let w = this.tileW;
    let h = this.tileH;
    for (const ts of (this.source.doc.tilesets as { tilewidth?: unknown; tileheight?: unknown }[]) ?? []) {
      w = Math.max(w, Number(ts.tilewidth) || 0);
      h = Math.max(h, Number(ts.tileheight) || 0);
    }
    return { x: Math.ceil(w / this.tileW), y: Math.ceil(h / this.tileH) };
  }

  /**
   * Draw the terrain: blit the map's own tilesets, in the author's layer order.
   *
   * Don't bake it procedurally from a table of tile names: that draws only the one map whose
   * names are in the table, in flat colour. Handing the layers to Phaser renders it as drawn.
   */
  drawTerrain(map: Phaser.Tilemaps.Tilemap): void {
    const sets = this.source.tilesets
      .map((ts) => map.addTilesetImage(ts.name, ts.key))
      .filter((ts): ts is Phaser.Tilemaps.Tileset => ts !== null);
    if (!sets.length) return; // art failed to load — labels/agents still render

    // Depth stays under the cast's band (isoDepth.SORT_BASE), in the author's layer order;
    // nothing is re-sorted against the figures (hiding a figure is isoDepth's job).
    // Layers are addressed by index, not name: Tiled allows duplicate names and
    // `createLayer(name)` takes the first match, so the second would silently never render.
    // Culling is told how far this map's pictures reach beyond their cells (see cullPadding).
    const pad = this.cullPadding();
    map.layers.forEach((_layerData, i) => {
      const layer = map.createLayer(i, sets, this.originX, this.originY);
      layer?.setDepth(i * 0.01).setCullPadding(pad.x, pad.y);
    });
  }

  /** The projection, as the plain numbers isoDepth reasons about. */
  get geometry(): IsoGeometry {
    return {
      tileW: this.tileW, tileH: this.tileH,
      originX: this.originX, originY: this.originY, worldH: this.worldH,
    };
  }

  /**
   * Pixels per grid step in Tiled's OBJECT coordinate space.
   *
   * On an isometric map the unit on both axes is the tile height, not the width; dividing
   * by the width halves every location's distance from the origin. The backend applies the
   * same rule (see worlds/tiled.py).
   */
  objectUnitPx(): number {
    return this.tileH;
  }

  parseLocations(map: Phaser.Tilemaps.Tilemap): void {
    for (const layer of map.objects) {
      for (const obj of layer.objects) {
        if (obj.type !== "location") continue;
        const props = Object.fromEntries(
          (obj.properties ?? []).map((p: { name: string; value: unknown }) => [p.name, p.value]),
        );
        const id = String(props.location_id ?? "").trim();
        if (!id) continue;
        // Object coords are pixels in Tiled's own object space; divide to grid
        // tiles, then project to the screen anchor agents rest on.
        const unit = this.objectUnitPx();
        const gx0 = (obj.x ?? 0) / unit;
        const gy0 = (obj.y ?? 0) / unit;
        const gw = (obj.width ?? 0) / unit;
        const gh = (obj.height ?? 0) / unit;
        const gx = gx0 + gw / 2;
        const gy = gy0 + gh / 2;
        const s = this.iso(gx, gy);
        const name = String(obj.name ?? id);
        // atx/aty and stand/inRect are filled in once the walkable and standable grids exist.
        this.rooms.set(id, {
          id, name, gx, gy, gx0, gy0, gw, gh, sx: s.x, sy: s.y, atx: 0, aty: 0, stand: [], inRect: 0,
        });
      }
    }
  }

  drawLocationLabels(): void {
    for (const room of this.rooms.values()) {
      this.scene.add
        .text(room.sx, room.sy - 6, room.name, {
          fontFamily: FONT,
          fontSize: "22px",
          color: "#f0e4c8",
        })
        .setOrigin(0.5)
        .setDepth(5)
        .setResolution(TEXT_RES)
        .setStroke("#1a1420", 5);
    }
  }

  // --- walkable grid + A* so transit follows a wall-avoiding pixel route -----
  /**
   * Adopt the two grids the backend derived from this map's art (see
   * `worlds/ground.py` and mapSource.GroundGrids), or fall back to reading the
   * map's own tile marks if it could not.
   *
   * The fallback is deliberately crude (a tile marks only its own cell): without pixels a tall
   * tile's footprint is unknowable, and a guessed one routes people through walls. It keeps a
   * world renderable when `/map/ground` fails; big road art then marks less than it paints and
   * nothing reads as hidden.
   */
  adoptGround(map: Phaser.Tilemaps.Tilemap): void {
    const ground = this.source.ground;
    if (ground && ground.cols === this.cols && ground.rows === this.rows) {
      this.blocked = ground.walkable.map((row) => row.map((ok) => !ok));
      this.standable = ground.standable;
      this.toll = this.standable.map((row) => row.map((ok) => (ok ? 0 : HIDDEN_STEP_TOLL)));
      return;
    }
    const hasRoads = this.mapDeclaresRoads(map);
    this.blocked = Array.from({ length: this.rows }, () => Array<boolean>(this.cols).fill(hasRoads));
    this.standable = Array.from({ length: this.rows }, () => Array<boolean>(this.cols).fill(true));
    this.toll = Array.from({ length: this.rows }, () => Array<number>(this.cols).fill(0));
    for (const layer of map.layers) {
      layer.data.forEach((row) =>
        row.forEach((t) => {
          if (!t || t.index < 0 || t.x >= this.cols || t.y >= this.rows) return;
          const props = t.properties as { collides?: boolean; road?: boolean } | undefined;
          if (props?.collides) {
            this.blocked[t.y][t.x] = true;
            this.standable[t.y][t.x] = false;
          } else if (hasRoads && props?.road) {
            this.blocked[t.y][t.x] = false;
          }
        }),
      );
    }
  }

  /**
   * Is a figure standing at this screen point on ground the map's art HIDES?
   *
   * The fade's trigger. Reuses `standable`, the backend's pixel-accurate cover answer
   * (worlds/ground.py) that A* pays HIDDEN_STEP_TOLL to cross and `findStanding` won't rest
   * anyone on: one truth, no second opinion here. Not rare: 18–41% of the walkable network is
   * covered, and the open cells alone fall apart into dozens of pieces.
   *
   * Degrades to "nothing is hidden" without a ground grid: full opacity is the honest default.
   */
  hiddenGround(worldX: number, worldY: number): boolean {
    const { gx, gy } = this.unproject(worldX, worldY);
    const row = this.standable[Math.floor(gy)];
    return row ? row[Math.floor(gx)] === false : false;
  }

  /**
   * The ground this room can stage people and things on, best first.
   *
   * Searched over the rectangle and a band around it: a ward drawn as solid housing may have no
   * standable cell of its own (huaide_fang on the Chang'an map), and a figure in the lane outside
   * still reads as being there.
   *
   * Ranked by reachability first, then inside-the-rectangle, then distance to the centre. Don't
   * demote reachability: a walk's two end segments are unchecked straight lines to where the
   * figure stands (groundRoute), so resting him on unroutable ground (a walled courtyard, an
   * unmarked lawn) sends him through the wall. The cost: about one room in twenty-five gives up
   * its innermost spot for a reachable one further out.
   *
   * `roadNet` is the main walkable component, keyed y * cols + x — the same set `snap`
   * pulls route endpoints onto, so "reachable" means the same thing in both places.
   */
  findStanding(room: Room, roadNet: ReadonlySet<number>): void {
    const inRect = (x: number, y: number): boolean =>
      x + 0.5 >= room.gx0 && x + 0.5 <= room.gx0 + room.gw && y + 0.5 >= room.gy0 && y + 0.5 <= room.gy0 + room.gh;
    const onNet = (x: number, y: number): boolean => !this.blocked[y]?.[x] && roadNet.has(y * this.cols + x);
    const x0 = Math.max(0, Math.floor(room.gx0 - ROOM_STAND_REACH));
    const x1 = Math.min(this.cols - 1, Math.ceil(room.gx0 + room.gw + ROOM_STAND_REACH));
    const y0 = Math.max(0, Math.floor(room.gy0 - ROOM_STAND_REACH));
    const y1 = Math.min(this.rows - 1, Math.ceil(room.gy0 + room.gh + ROOM_STAND_REACH));
    const found: { x: number; y: number; net: boolean; rect: boolean; d: number }[] = [];
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        if (!this.standable[y]?.[x]) continue;
        const d = (x + 0.5 - room.gx) ** 2 + (y + 0.5 - room.gy) ** 2;
        found.push({ x, y, net: onNet(x, y), rect: inRect(x, y), d });
      }
    }
    found.sort((a, b) => Number(b.net) - Number(a.net) || Number(b.rect) - Number(a.rect) || a.d - b.d);
    room.stand = found.slice(0, ROOM_STAND_CELLS).map(({ x, y }) => ({ x, y }));
    // The LEADING run, not the total: with reachability ranked above the rectangle, a cell
    // outside it can outrank one inside, and layoutFixtures reads this as an index bound.
    const outside = room.stand.findIndex((c) => !inRect(c.x, c.y));
    room.inRect = outside < 0 ? room.stand.length : outside;
  }

  // Does this map mark its walkable tiles with `road`? One such tile is enough —
  // the question is which convention the author used, not how much of it there is.
  mapDeclaresRoads(map: Phaser.Tilemaps.Tilemap): boolean {
    return map.layers.some((layer) =>
      layer.data.some((row) =>
        row.some((t) => t && t.index >= 0 && (t.properties as { road?: boolean } | undefined)?.road),
      ),
    );
  }

  // World (screen-space) point → the location region that contains it, or null.
  roomAt(wx: number, wy: number): Room | null {
    const { gx, gy } = this.unproject(wx, wy);
    for (const room of this.rooms.values()) {
      if (gx >= room.gx0 && gx < room.gx0 + room.gw && gy >= room.gy0 && gy < room.gy0 + room.gh) {
        return room;
      }
    }
    return null;
  }

  locationName(id: string): string {
    return this.rooms.get(id)?.name ?? id;
  }
}
