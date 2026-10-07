/**
 * Where each frame sits on its sheet, read from the Starling atlas beside it, for drawing one frame
 * outside Phaser (the character chart, the cast bench). Phaser's own loader reads the atlas for the
 * map.
 *
 * Regexes, not `DOMParser`: unit tests run under Node with no DOM, and the format is a flat list of
 * self-closing elements. Attributes are scanned separately so their order doesn't matter.
 */

export interface FrameRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

const SUB_TEXTURE = /<SubTexture\b([^>]*?)\/?>/g;
const ATTRIBUTE = /([A-Za-z]+)\s*=\s*"([^"]*)"/g;

/** Frame name → its rectangle on the sheet. Unreadable input yields no frames. */
export function parseAtlasFrames(xml: string): Record<string, FrameRect> {
  const frames: Record<string, FrameRect> = {};
  for (const [, attributes] of xml.matchAll(SUB_TEXTURE)) {
    const found: Record<string, string> = {};
    for (const [, key, value] of attributes.matchAll(ATTRIBUTE)) found[key] = value;
    const rect = {
      x: Number(found.x),
      y: Number(found.y),
      width: Number(found.width),
      height: Number(found.height),
    };
    // Drop a named frame with no usable rectangle: it would draw a silent empty box.
    if (found.name && Object.values(rect).every(Number.isFinite)) frames[found.name] = rect;
  }
  return frames;
}

/** One body's art: where every frame is, and how big the sheet holding them is. */
export interface Sheet {
  frames: Record<string, FrameRect>;
  width: number;
  height: number;
}

/**
 * Fetch a body's sheet, with its pixel size: drawing one frame scales the whole sheet behind a
 * window (see `castSheet.frameStyle`).
 */
export async function loadSheet(imageUrl: string, atlasUrl: string): Promise<Sheet> {
  const [frames, size] = await Promise.all([
    loadAtlasFrames(atlasUrl),
    new Promise<{ width: number; height: number }>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
      img.onerror = () => reject(new Error(imageUrl));
      img.src = imageUrl;
    }),
  ]);
  return { frames, ...size };
}

/** Fetch an atlas and read its frames; a failed fetch throws rather than read as no frames. */
export async function loadAtlasFrames(atlasUrl: string): Promise<Record<string, FrameRect>> {
  const r = await fetch(atlasUrl);
  if (!r.ok) throw new Error(`${atlasUrl} (${r.status})`);
  return parseAtlasFrames(await r.text());
}
