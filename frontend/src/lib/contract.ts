/**
 * Named constants for backend-contract enums used as string literals across files. Source of
 * truth: interaction/models.py (ActionSummary.action_type, .phase, .deed, .target).
 */

import type { ActionTarget } from "../types";

// Longest display name a world may be renamed to. Mirrors world.catalog.MAX_WORLD_NAME_LEN
// — the input caps at it so the backend's 400 is a backstop, not the normal way to find out.
export const MAX_WORLD_NAME_LEN = 40;

// Most characters a world may have. Mirrors config.models.MAX_CAST_SIZE.
export const MAX_CAST_SIZE = 15;

// The step lengths a world may take, in whole hours. Mirrors engine.clock.MIN_HOURS_PER_STEP /
// MAX_HOURS_PER_STEP.
export const MIN_HOURS_PER_STEP = 1;
export const MAX_HOURS_PER_STEP = 24;

// The operation of an actor's decision summary that does nothing this step. Mirrors
// core.interfaces.actor.IDLE_OPERATION.
export const IDLE_OPERATION = "idle";

export const ActionType = {
  talk:         "talk",
  move:         "move",
  physical:     "physical",
  covert:       "covert",
  work:         "work",
  rest:         "rest",
  send_message: "send_message",
  errand:       "errand",
} as const;
export type ActionTypeValue = typeof ActionType[keyof typeof ActionType];

export const Phase = {
  interrupt: "interrupt",
  // Which beat of an execution a record reports, and so whether its `outcome` is an opening or a
  // result. An act spanning steps arrives as three beats, each saying only what is new; one that
  // opens and closes inside a step arrives as one, `settled`, which has a result but no opening to
  // conclude: don't label it "开始" or "结束". Terminal beats (outcome is a result): settled,
  // ongoing_complete, interrupt.
  begin: "begin",
  ongoing_tick: "ongoing_tick",
  ongoing_complete: "ongoing_complete",
  settled: "settled",
} as const;

// What a bystander saw the body do; mirrors core.interfaces.action.Deed. `action_type` is the kind
// of intent, `deed` what the figure was seen doing. They coincide for every type but PHYSICAL,
// whose verbs look nothing alike. Empty deed = the adjudication never happened: draw nothing. It
// does not mark a non-event (a foiled TALK still reports "talk"; only PHYSICAL's deed is
// adjudicated); whether the intent engaged the world is `not_executed`.
export const Deed = {
  // PHYSICAL's seven verbs
  strike:     "strike",      // violence upon a person
  restrain:   "restrain",    // hands laid on a person, drawing no blood
  seize:      "seize",       // take an object into one's possession
  relinquish: "relinquish",  // let a held object go — set it down, or into another's hands
  operate:    "operate",     // work an object where it stands (open / use / tear)
  destroy:    "destroy",     // wreck an object out of the world
  exert:      "exert",       // physical force with no target at all
  // every other type is its own deed
  talk:         "talk",
  send_message: "send_message",
  move:         "move",
  work:         "work",
  rest:         "rest",
  covert:       "covert",
  errand:       "errand",
} as const;
export type DeedValue = typeof Deed[keyof typeof Deed];

// What a target ref may name: routing vocabulary the backend derives from which slot a decision
// bound (never from an LLM), so the renderer can tell a person from a place from a thing.
export const RefKind = {
  agent:    "agent",
  npc:      "npc",
  location: "location",
  item:     "item",
  landmark: "landmark",
  object:   "object",
} as const;

const THING_KINDS: string[] = [RefKind.item, RefKind.landmark, RefKind.object];

/**
 * Who an action is done to; empty when it acts on a place or a thing (an answer, not a gap).
 *
 * Callers use these accessors rather than filtering `acts_on` by hand, so the question is answered
 * in one place and can't drift between call sites.
 */
export const actedOnAgents = (t: ActionTarget | undefined): string[] =>
  (t?.acts_on ?? []).filter((r) => r.kind === RefKind.agent).map((r) => r.id);

/**
 * Who is being ordered about: the body with no mind an errand is handed to.
 *
 * Not folded into `actedOnAgents`: that one feeds poses and relations, while an NPC has neither and
 * only answers "which figure does the teller turn toward".
 */
export const actedOnNpcs = (t: ActionTarget | undefined): string[] =>
  (t?.acts_on ?? []).filter((r) => r.kind === RefKind.npc).map((r) => r.id);

/**
 * Who it LANDS ON passively, their turn untouched: a hand-over's receiver, the holder of a thing
 * taken or worked, a conversation's listeners.
 */
export const reachedAgents = (t: ActionTarget | undefined): string[] =>
  (t?.reaches ?? []).filter((r) => r.kind === RefKind.agent).map((r) => r.id);

/** WHAT an action is done to, when it is a thing rather than a person or a place. */
export const actedOnEntities = (t: ActionTarget | undefined): string[] =>
  (t?.acts_on ?? []).filter((r) => THING_KINDS.includes(r.kind)).map((r) => r.id);

/** WHOSE TURN this spends — the bodies pulled into the execution (a carried companion). */
export const claimedAgents = (t: ActionTarget | undefined): string[] =>
  (t?.claims ?? []).filter((r) => r.kind === RefKind.agent).map((r) => r.id);


// Value ranges the backend validates amendments against; mirrors core.prompts. Sliders stay
// inside them, so the backend's 400 is a backstop.
export const EMOTION_INTENSITY_RANGE: [number, number] = [0, 1];
export const EMOTION_VALENCE_RANGE: [number, number] = [-1, 1];
export const TRUST_RANGE: [number, number] = [0, 1];
export const AFFECTION_RANGE: [number, number] = [-1, 1];
