/**
 * The garment dye rule, shared by the map's GPU recolor (phaser/recolorPipeline.ts, which
 * explains the rule) and the DOM figures drawn outside Phaser. One set of thresholds, so a
 * figure wears the same clothes on and off the map. No Phaser here: the DOM side must not pull
 * the renderer bundle in.
 */

/** Hue at or below this (as a 0..1 fraction) is skin / brown hair / leather: kept. */
export const WARM_MAX = 60 / 360;
/** Value at or below this is an outline: kept. */
export const LINE_VAL = 0.08;
/** The value the artist painted a lit garment at; a pixel's value relative to it carries the folds. */
export const ART_LIT = 0.7;

/** A packed 0xRRGGBB identity color as HSV (each 0..1) — all three channels are used. */
export function hsvOf(color: number): [number, number, number] {
  return rgbToHsv(((color >> 16) & 0xff) / 255, ((color >> 8) & 0xff) / 255, (color & 0xff) / 255);
}

function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h /= 6;
    if (h < 0) h += 1;
  }
  return [h, max === 0 ? 0 : d / max, max];
}

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const channel = (n: number) => {
    const k = (n + h * 6) % 6;
    return v - v * s * Math.max(0, Math.min(k, 4 - k, 1));
  };
  return [channel(5), channel(3), channel(1)];
}

/**
 * Dye RGBA pixels in place (straight alpha, as a 2D canvas hands them out), by the rule the
 * map's shader applies per fragment.
 */
export function dyePixels(data: Uint8ClampedArray, dye: [number, number, number]): void {
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] === 0) continue;
    const [h, , v] = rgbToHsv(data[i] / 255, data[i + 1] / 255, data[i + 2] / 255);
    if (v <= LINE_VAL || h <= WARM_MAX) continue;
    const lit = Math.min(1, Math.max(0, (v / ART_LIT) * dye[2]));
    const [r, g, b] = hsvToRgb(dye[0], dye[1], lit);
    data[i] = Math.round(r * 255);
    data[i + 1] = Math.round(g * 255);
    data[i + 2] = Math.round(b * 255);
  }
}
