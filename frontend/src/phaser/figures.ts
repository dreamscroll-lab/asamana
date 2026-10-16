/**
 * The map's reading of a step: bodies as figures, actions as beats.
 *
 * The map draws bodies, so it needs every body in one list whatever its tier, and every action
 * as each body's part in it. An act with N participants is one event on the wire and N beats
 * here: each participant turns, poses and wears a chip. The beats share the act's seq, so a
 * track plays them as one moment.
 */

import { ActionType } from "../lib/contract";
import type { ActionSummary, Transit } from "../types";
import type { Placed, StepIndex } from "../lib/step";
import { type Viewer, knowsInside } from "../lib/viewer";

/** One body on the map, agent or Npc alike: the part of its state the map draws. */
export interface Figure {
  agent_id: string;
  agent_name: string;
  location_id: string; // "" between two waypoints
  color: string;
  is_main_character: boolean;
  is_active: boolean;
  condition: string;
  transit: Transit | null;
  // The route walked to arrive this step (from his MOVE's closing beat).
  arrival: Transit | null;
  displaced: boolean;
  vitality: number | null;
  emotion_valence: number | null;
}

/**
 * One figure's part in an action.
 *
 * `inside`: the viewer knows the act from inside. `caption` is the words the map may put to it:
 * from inside, the doer's own description; from outside, what an onlooker saw, and only when the
 * figure's motion can't show it (see SHOWN_BY_MOTION).
 */
export type Beat = Placed<ActionSummary> & {
  agent_id: string; agent_name: string; inside: boolean; caption: string;
};

/**
 * Acts whose motion or effect already shows an onlooker what happened: a blow, a talk's facing
 * and pose, a veil closing, a letter's flight, a walk. Seen from outside these get no words;
 * the rest (a body bent over work, at rest, sending someone off) look alike and keep a caption.
 */
const SHOWN_BY_MOTION: ReadonlySet<string> = new Set([
  ActionType.physical, ActionType.talk, ActionType.covert, ActionType.send_message, ActionType.move,
]);

export function figuresOf(index: StepIndex): Figure[] {
  // A mover's walk in rides the beat his MOVE ended (`route`); the figure needs it to walk. Whoever
  // he brought along walked the same way, so every participant standing at its end takes it;
  // otherwise the one carried finds his own way there.
  const routes = new Map<string, Transit>();
  for (const act of index.actions) {
    const route = act.private.route;
    if (!route) continue;
    for (const id of act.participant_ids.length ? act.participant_ids : [act.initiator_id]) {
      if (index.agents[id]?.location_id === route.to_location_id) routes.set(id, route);
    }
  }
  const agents = Object.values(index.agents).map((a): Figure => ({
    agent_id: a.agent_id,
    agent_name: a.name,
    location_id: a.location_id,
    color: a.color,
    is_main_character: a.is_main_character ?? false,
    is_active: a.is_active,
    condition: a.public.condition,
    transit: a.private.transit,
    arrival: routes.get(a.agent_id) ?? null,
    displaced: a.public.displaced,
    vitality: a.private.vitality,
    emotion_valence: a.private.emotion_valence,
  }));
  // Liveness is pinned: an Npc has no vitality axis, and the death visuals must never claim one.
  const npcs = index.npcs.map((n): Figure => ({
    agent_id: n.npc_id,
    agent_name: n.name,
    location_id: n.location_id,
    color: n.color,
    is_main_character: false,
    is_active: true,
    condition: n.public.condition,
    // An Npc's walk is inferred from its position; it reports no route.
    transit: null,
    arrival: null,
    displaced: n.public.displaced,
    vitality: 1,
    emotion_valence: null,
  }));
  return [...agents, ...npcs];
}

export function beatsOf(index: StepIndex, viewer: Viewer = null): Beat[] {
  const nameOf = (id: string) => index.agents[id]?.name ?? "某人";
  return index.actions.flatMap((act) => {
    const inside = knowsInside(act, viewer);
    const caption = inside
      ? act.private.action_description || act.private.gist
      : SHOWN_BY_MOTION.has(act.action_type) ? "" : act.public?.text ?? "";
    return (act.participant_ids.length ? act.participant_ids : [act.initiator_id]).map((id) => ({
      ...act, agent_id: id, agent_name: nameOf(id), inside, caption,
    }));
  });
}
