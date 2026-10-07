/**
 * Feed ONE step to the renderer — the single way a step reaches a scene, for both the live
 * world and the render lab. A lab that drove the scene its own way would certify a build
 * nobody ships.
 *
 * Errors are returned, never raised: a driver gated on this promise would otherwise be
 * wedged for good by one bad step. The caller decides what to do (live world carries on,
 * scene bench stops).
 */

import type { StepEvent } from "../types";
import type { TiledWorldScene } from "./TiledWorldScene";

export async function renderStepInto(
  scene: TiledWorldScene,
  step: StepEvent,
): Promise<Error | null> {
  // The code-layer hour, not the narrative label: the label's shape is the theme's. (See AmbientLight.setHour.)
  scene.setWorldTime(step.world_time.hour);
  try {
    await scene.renderStep(step);
    return null;
  } catch (err) {
    return err instanceof Error ? err : new Error(String(err));
  }
}
