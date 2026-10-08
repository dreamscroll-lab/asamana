/**
 * Where a figure sits in the picture: how far forward it is drawn, and how solid it is
 * when the map's art is in front of it. Pure (no Phaser), so both decisions are tested
 * without a screen.
 *
 * Deliberately does NOT touch the map: no tile moves, the terrain's layer order stays as
 * the author drew it. Don't lift standing art into per-tile sprites sorted against the
 * cast: a multi-cell canvas gets ONE depth (its anchor cell's), which is only right when
 * the canvas is all Thing. A canvas that is floor (a river under drawn bridges) climbs
 * over what is drawn on it; one that is both (a palace with its forecourt) cuts a figure
 * at the foot of the steps in half. Nothing in the data says which is which, so the fix
 * would need an author's mark on every large tile of every map (cf. `worlds/ground.py`).
 *
 * Instead only the FIGURE is decided: standing on ground the art hides, it fades
 * (`HIDDEN_ALPHA`) — reads as behind, yet stays findable. That matters: much of a city's
 * walkable network lies under canopies or behind walls, and most cross-city walks have
 * no fully open route, so a figure that vanished would vanish for stretches of every walk.
 *
 * Trade-off: the fade is per CELL and whole-figure, so a figure half behind a trunk fades
 * entirely rather than being cut. In exchange this cannot break a map, since it never reads one.
 */

/** Everything the projection needs, as plain numbers. Mirrors WorldMap's own. */
export interface IsoGeometry {
  tileW: number;
  tileH: number;
  originX: number;
  originY: number;
  /** Full projected height of the world, in pixels — the depth normalizer. */
  worldH: number;
}

/**
 * The band everything standing on the ground is drawn in: the cast, and the markers
 * for things lying about. Above the terrain (< 1) and the ground-level effect decals
 * (14–19), below the flashes thrown over the cast (29+) and every bubble and chip.
 */
export const SORT_BASE = 20;
export const SORT_SPAN = 4;

/**
 * Opacity when the art in front hides a figure's ground. Low enough to read as "behind",
 * not a glitch; high enough that body, heading and nameplate stay legible, since for long
 * stretches of a walk this is the only way the figure is seen.
 */
export const HIDDEN_ALPHA = 0.42;
/**
 * The hidden test is per CELL, so without a fade a figure snaps between opacities at every
 * canopy edge. Short enough to still coincide with the edge it reports.
 */
export const HIDDEN_FADE_MS = 220;

/**
 * Depth for something standing at screen y — further "south" draws in front. Normalized
 * by the world's height: don't use a fixed scale with a cap, it saturates partway down a
 * tall map and everyone beyond it stops sorting against each other.
 */
export function sortDepth(worldY: number, g: IsoGeometry): number {
  const t = Math.min(1, Math.max(0, worldY / Math.max(1, g.worldH)));
  return SORT_BASE + t * SORT_SPAN;
}
