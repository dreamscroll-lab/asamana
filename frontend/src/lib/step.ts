/**
 * A step read across its scopes.
 *
 * The wire groups everything by where it is (see StepView in types.ts). Most consumers ask about
 * one body or one channel wherever it is, so this indexes a step once per payload and tags each
 * thing with its scope. It derives; it never decides: what is where, and who knows what, is the
 * backend's answer.
 */

import type {
  ActionSummary,
  AgentView,
  BroadcastSummary,
  EntityView,
  GraphEdge,
  MessageSummary,
  NpcView,
  ObservationSummary,
  ScopeEvent,
  StepView,
} from "../types";

/**
 * A body's or event's scope: a location id; "" for the way between two waypoints (transit);
 * null for the world scope (only events live there). `location_name` is the place's narrative
 * name, 「途中」 on the way, "" for the world.
 */
export type Placed<T> = T & { location_id: string; location_name: string };
export type Scoped<T> = T & { location_id: string | null; location_name: string };

/**
 * Where a thing is, on one axis: on the ground of a location (`presence_ref` = location id) or
 * in someone's hands (`presence_ref` = holder id). A destroyed thing is in no step's table.
 */
export type EntityPresence = "at_location" | "held";
export type PlacedEntity = EntityView & { presence: EntityPresence; presence_ref: string };

export interface StepIndex {
  agents: Record<string, Placed<AgentView>>;
  npcs: Placed<NpcView>[];
  entities: Record<string, PlacedEntity>;
  // Each channel in seq order.
  actions: Placed<ActionSummary>[];
  observations: Placed<ObservationSummary>[];
  messages: Scoped<MessageSummary>[];
  broadcasts: Scoped<BroadcastSummary>[];
  // Every agent's relations, as they stand at the end of the step.
  relations: GraphEdge[];
}

const cache = new WeakMap<StepView, StepIndex>();

export function indexStep(step: StepView): StepIndex {
  const hit = cache.get(step);
  if (hit) return hit;

  const agents: Record<string, Placed<AgentView>> = {};
  const npcs: Placed<NpcView>[] = [];
  const entities: Record<string, PlacedEntity> = {};
  const events: Scoped<ScopeEvent>[] = [];

  const held = (holder: string, carrying: Record<string, EntityView>) => {
    for (const [id, e] of Object.entries(carrying)) {
      entities[id] = { ...e, presence: "held", presence_ref: holder };
    }
  };
  const nameOf = (id: string) => (id ? step.locations[id]?.name || "某地" : "途中");
  const scope = (
    location_id: string,
    s: { agents: AgentView[]; npcs: NpcView[]; events: ScopeEvent[] },
  ) => {
    const location_name = nameOf(location_id);
    for (const a of s.agents) {
      agents[a.agent_id] = { ...a, location_id, location_name };
      held(a.agent_id, a.public.carrying);
      held(a.agent_id, a.private.carrying);
    }
    for (const n of s.npcs) {
      npcs.push({ ...n, location_id, location_name });
      held(n.npc_id, n.public.carrying);
      held(n.npc_id, n.private.carrying);
    }
    for (const e of s.events) events.push({ ...e, location_id, location_name });
  };

  scope("", step.transit);
  for (const [id, place] of Object.entries(step.locations)) {
    scope(id, place);
    for (const [eid, e] of Object.entries(place.entities)) {
      entities[eid] = { ...e, presence: "at_location", presence_ref: id };
    }
  }
  for (const e of step.world.events) events.push({ ...e, location_id: null, location_name: "" });
  events.sort((a, b) => a.seq - b.seq);

  const index: StepIndex = {
    agents,
    npcs: npcs.sort((a, b) => a.npc_id.localeCompare(b.npc_id)),
    entities,
    actions: events.filter((e): e is Placed<ActionSummary> => e.kind === "action"),
    observations: events.filter((e): e is Placed<ObservationSummary> => e.kind === "observation"),
    messages: events.filter((e): e is Scoped<MessageSummary> => e.kind === "message"),
    broadcasts: events.filter((e): e is Scoped<BroadcastSummary> => e.kind === "broadcast"),
    relations: Object.values(agents).flatMap((a) => a.private.relations),
  };
  cache.set(step, index);
  return index;
}
