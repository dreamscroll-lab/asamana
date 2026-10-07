/**
 * deed → body language. Pure data; no Phaser, no VFX.
 *
 * One total table keyed on the closed `Deed` enum (core.interfaces.action.Deed). Don't
 * derive a pose from an if/else chain over overlapping flags.
 *
 * The layering to protect:
 *   deed → POSE   — what the FIGURE did.        This file, applied by the scene.
 *   deed → VFX    — what the WORLD did about it. effects.ts.
 * Don't fuse them: strike and destroy are the same swing and differ only in the debris.
 *
 * Theme-agnostic: every pose is something a BODY does, never something a world contains (CLAUDE.md Rule 7).
 */

import { Deed, type DeedValue } from "../lib/contract";
import type { PoseName } from "./skins";

/** `attack` = the swing animation; anything else = a still pose held for a beat. */
export type ActorPose = "attack" | PoseName;

export interface DeedBody {
  actor: ActorPose;
  /** What a targeted AGENT's body does. null = this deed has no human on the other end. */
  target: PoseName | null;
  /**
   * The pose is a STATE he is in (work, talk, hiding), not a beat struck and dropped.
   * Not dropped on a timer: it lasts until the next deed, a walk, or an idle step replaces it.
   */
  sustained?: boolean;
  /**
   * The actor's opacity while in this pose (absent = solid). Don't make it a VFX: effects
   * fire only when an act CONCLUDES, so a multi-step concealment would dim only at its end.
   */
  shroud?: number;
}

export const DEED_BODY: Record<DeedValue, DeedBody> = {
  // — PHYSICAL's seven verbs. strike/destroy/exert share one swing; only the aftermath (VFX) differs.
  [Deed.strike]:   { actor: "attack",   target: "hurt" },  // the only deed that recoils a body
  [Deed.restrain]: { actor: "shove",    target: "duck" },  // NOT the swing: hands laid on, not a blow
  [Deed.seize]:    { actor: "hold",     target: null },
  // seize's inverse. `target: null`: the receiver is the recipient, which this renderer does
  // not pose (a deed's `target` is the agent acted UPON).
  [Deed.relinquish]: { actor: "show",   target: null },
  [Deed.operate]:  { actor: "interact", target: null },
  [Deed.destroy]:  { actor: "attack",   target: null },
  [Deed.exert]:    { actor: "attack",   target: null },
  // — the other action types, each its own deed. talk/work/covert are STATES; the rest are single gestures.
  [Deed.talk]:         { actor: "talk",     target: null, sustained: true },
  [Deed.send_message]: { actor: "show",     target: null }, // holding the thing out to be sent
  [Deed.work]:         { actor: "interact", target: null, sustained: true },
  // crouched, furtive, and withdrawn into shadow for as long as he stays down
  [Deed.covert]:       { actor: "duck",     target: null, sustained: true, shroud: 0.42 },
  // MOVE needs no pose: the walk cycle is already driven by the token actually moving.
  [Deed.move]: { actor: "idle", target: null },
  // REST: no cast ships a rest frame (see the manifest's `frames`). Don't use `down`: that is
  // the FALLEN frame, and a resting man would look like a corpse. Sharing `duck` with covert
  // is fine, since covert is drawn at `shroud` opacity.
  [Deed.rest]: { actor: "duck", target: null, sustained: true },
  // ERRAND is `talk` without the sustain: the errand costs the teller one beat, so holding
  // the pose would draw him still talking to a runner already gone. `target: null`: the
  // runner is an NPC, and this renderer poses no body without a mind — his answer is the walk.
  [Deed.errand]: { actor: "talk", target: null },
};

/**
 * The body language for one deed, or null when there is nothing to show.
 *
 * A FAILED deed keeps the actor's pose (a missed snatch is not a missed blow); a targeted
 * person ducks instead.
 *
 * An empty deed means the adjudication never happened, so nothing is drawn. It does NOT mark
 * a non-event: a foiled TALK still reports deed "talk".
 */
export function deedBody(deed: string, succeeded: boolean): DeedBody | null {
  const body = DEED_BODY[deed as DeedValue];
  if (!body || body.actor === "idle") return null;
  // Never the shroud on failure: the shroud IS the hiding succeeding.
  if (!succeeded) return { ...body, target: body.target ? "duck" : null, shroud: undefined };
  return body;
}
