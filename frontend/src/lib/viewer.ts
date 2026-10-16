/**
 * Who is looking, and how much of an event they know.
 *
 * The backend already filtered every payload to its viewer (interaction/actor_view.py); this
 * only reads the split it recorded. Renderers never ask "which mode": they ask whether the
 * viewer is inside an event, and draw its private side if so, its public side if not. The
 * director is a viewer inside everything, so the god view is the same rule.
 */

import type { ActionSummary, WorldMeta } from "../types";

/**
 * The character the actor performs, whose view every payload of this world is filtered to;
 * null for the director (nothing filtered).
 */
export type Viewer = string | null;

export function viewerOf(meta: WorldMeta | null | undefined): Viewer {
  if (!meta?.actor_character_id || meta.actor_exited) return null;
  return meta.actor_character_id;
}

/** Whether the viewer knows an act from inside (took part, was acted on, was named to hear it). */
export function knowsInside(act: ActionSummary, viewer: Viewer): boolean {
  return viewer === null || act.private.audience.includes(viewer);
}

/**
 * Whether the step in hand is the whole world. Only then does absence mean gone: in a viewer's
 * step, a thing missing from it may just be out of sight.
 */
export function isWholeWorld(viewer: Viewer): boolean {
  return viewer === null;
}
