/**
 * Who is looking, and how much of an event they know.
 *
 * The backend already filtered every payload to its viewer (interaction/actor_view.py); this
 * only reads the tiers it recorded. Renderers never ask "which mode": they ask what the viewer
 * knows of an act — its shared side if it is inside, the participants' own parts it may read —
 * and draw the public side otherwise. The director is a viewer inside everything, so the god
 * view is the same rule.
 */

import type { ActionPrivateView, ActionSharedView, ActionSummary, WorldMeta } from "../types";

/**
 * The character the actor performs, whose view every payload of this world is filtered to;
 * null for the director (nothing filtered).
 */
export type Viewer = string | null;

export function viewerOf(meta: WorldMeta | null | undefined): Viewer {
  if (!meta?.actor_character_id || meta.actor_exited) return null;
  return meta.actor_character_id;
}

/**
 * The act's shared side when the viewer is inside it (took part, was acted on, was named to hear
 * it); null when it only saw it.
 */
export function sharedOf(act: ActionSummary, viewer: Viewer): ActionSharedView | null {
  const shared = act.shared;
  return shared && (viewer === null || shared.audience.includes(viewer)) ? shared : null;
}

/** The participants' own parts the viewer may read: all of them for the director, else its own. */
export function partsOf(act: ActionSummary, viewer: Viewer): ActionPrivateView[] {
  return act.private.filter((p) => viewer === null || p.agent_id === viewer);
}

/** The doer's `action_description`, his own words for it; "" to anyone but him (and the director). */
export function actionDescriptionOf(act: ActionSummary, viewer: Viewer): string {
  return partsOf(act, viewer).find((p) => p.agent_id === act.initiator_id)?.action_description ?? "";
}

/**
 * Whether the step in hand is the whole world. Only then does absence mean gone: in a viewer's
 * step, a thing missing from it may just be out of sight.
 */
export function isWholeWorld(viewer: Viewer): boolean {
  return viewer === null;
}
