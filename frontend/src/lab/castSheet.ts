/**
 * The cast's art laid out as a table: every body, every pose, every heading.
 *
 * Which frames a pose uses, and whether mirrored, is always asked of `CharacterSet.poseArt`, never
 * worked out here: a second reader of the manifest could certify art the renderer draws differently.
 * This only pairs each frame name with its rectangle and reports where the art has nothing.
 */

import type { FrameRect } from "../lib/atlasFrames";
import {
  AGE_BRACKETS,
  type CharacterSet,
  type Dir,
  DIRECTIONS,
  GENDERS,
  POSE_NAMES,
} from "../phaser/skins";

/** One body, one pose, one heading. */
export type Cell =
  /** Drawable: the frames in order, and whether the art is drawn facing the other way. */
  | { kind: "frames"; frames: { name: string; rect: FrameRect }[]; flip: boolean }
  /** The manifest maps no art here — the renderer would silently draw nothing. */
  | { kind: "unmapped" }
  /** The manifest names frames the sheet does not carry. */
  | { kind: "missing"; names: string[] };

export type Rects = Record<string, FrameRect>;

/** One (gender, bracket) cell of the body table, and the sheet it resolves to. */
export interface BodyCell {
  gender: string;
  bracket: string;
  atlas: string | null;
}

/**
 * The eight cells an agent can fall into. Two of them resolving to the SAME sheet is
 * a legitimate answer — it says the art draws no difference between those brackets —
 * so the sheet is reported rather than judged.
 */
export function bodyCells(cast: CharacterSet): BodyCell[] {
  return GENDERS.flatMap((gender) =>
    AGE_BRACKETS.map((bracket) => ({
      gender,
      bracket,
      atlas: cast.bodyAt(gender, bracket),
    })),
  );
}

/** One pose across all four headings. */
export function poseRow(
  cast: CharacterSet,
  body: string,
  pose: string,
  rects: Rects,
): Record<Dir, Cell> {
  const row = {} as Record<Dir, Cell>;
  for (const dir of DIRECTIONS) {
    const art = cast.poseArt(body, pose, dir);
    if (!art) {
      row[dir] = { kind: "unmapped" };
      continue;
    }
    const names = art.frames.map(([, frame]) => frame);
    const absent = names.filter((name) => !rects[name]);
    row[dir] = absent.length
      ? { kind: "missing", names: absent }
      : { kind: "frames", frames: names.map((name) => ({ name, rect: rects[name] })), flip: art.flip };
  }
  return row;
}

/**
 * Frames the sheet carries that no pose asks for: art drawn but forgotten in the manifest (the
 * shipped sets draw face-up and face-down falls per heading and map one of each pair).
 */
export function unusedFrames(cast: CharacterSet, body: string, rects: Rects): string[] {
  const named = new Set<string>();
  for (const pose of POSE_NAMES) {
    for (const dir of DIRECTIONS) {
      for (const [, frame] of cast.poseArt(body, pose, dir)?.frames ?? []) named.add(frame);
    }
  }
  return Object.keys(rects).filter((name) => !named.has(name));
}

/**
 * How to draw one frame at a chosen zoom: a box of the final size, showing a window onto the whole
 * sheet scaled to match.
 *
 * Scale the sheet, never the box: scaling the box resamples each frame from a different sub-pixel
 * offset (0, 53.76, 107.52… at 0.28×), so edges shimmer through the cycle. Scaling the sheet
 * resamples once, as the renderer does (see MapStage).
 *
 * The scale is derived from the rounded on-screen size so offsets land on whole pixels for a
 * grid-laid sheet.
 */
export function frameStyle(
  rect: FrameRect,
  sheet: { width: number; height: number },
  zoom: number,
  flip: boolean,
  imageUrl: string,
) {
  const width = Math.max(1, Math.round(rect.width * zoom));
  const height = Math.max(1, Math.round(rect.height * zoom));
  const sx = width / rect.width;
  const sy = height / rect.height;
  return {
    box: { width, height, overflow: "hidden" as const },
    clip: {
      width,
      height,
      backgroundImage: `url(${imageUrl})`,
      backgroundSize: `${sheet.width * sx}px ${sheet.height * sy}px`,
      backgroundPosition: `-${rect.x * sx}px -${rect.y * sy}px`,
      transform: flip ? "scaleX(-1)" : undefined,
      // Pixelated only when magnifying: shrunk figures stay smoothed, as the game draws them.
      imageRendering: (zoom >= 1 ? "pixelated" : "auto") as "pixelated" | "auto",
    },
  };
}
