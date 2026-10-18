/**
 * The scene bench's fixtures: hand-written step payloads, one scene per thing the map can draw.
 * A world shows only the deeds its cast happens to perform, so rare ones go unchecked; here the
 * deed table (deedPose.ts) is the checklist.
 *
 * Only the payload is made up. Scenes write what the ENGINE records (the record builders in
 * `engine/execution_processor.py`, `snapshot_agent_state`): one body record per agent, one action
 * record per participant. `step` then files them into scopes the way the read model does
 * (`interaction/models.py` StepView), so the renderer gets a StepView shaped as the wire's.
 * Geometry, heading, pose and timing are left to the renderer; a fixture that positioned figures
 * would be testing itself.
 *
 * Every field the renderer reads must carry the engine's shape. Load-bearing:
 *
 *   - `world_time.hour` is the MACHINE clock the day/night wash reads; the narrative
 *     `world_time.label` is never parsed for it.
 *   - `emotion` is the canonical EmotionType value ("trust", "anger", …), never prose.
 *   - A body's place comes as a pair: `location` is the NAME (printed), `location_id`
 *     the ID (looked up). `transit.*` and `entity.presence_ref` are IDs.
 *   - A JOINT act emits one record PER PARTICIPANT sharing an `execution_id`;
 *     the initiator's carries the deed, the other's is boilerplate. `step` folds them into
 *     one event told by the initiator's record, as the read model does.
 *   - A multi-step act's beats carry different counters: the opening beat is
 *     `begin`, `1/N`, with a `duration_label`; the ticks are `k/N` with no label; and the
 *     CLOSING beat resets to `0/0`.
 *   - `succeeded` on a closing beat is the judge's verdict, defaulting to true (`social.py` /
 *     `work.py`: `coerce_bool(data.get("success"), True)`; REST hardcodes it in `simple.py`). So a
 *     completed act is normally a success; a failed close is its own scene with a `failure_reason`.
 *   - `interrupted_by` is an agent ID, not a name.
 */

import { Phase, RefKind } from "../../lib/contract";
import type { PlacedEntity } from "../../lib/step";
import type { ActionTarget } from "../../types";
import type { Cast } from "../../lib/cast";
import type { LabRoom } from "../places";
import type {
  ActionSummary,
  AgentView,
  BroadcastSummary,
  DialogueTurn,
  EntityView,
  GraphEdge,
  LocationScope,
  MessageSummary,
  NpcView,
  GoalView,
  ScopeEvent,
  StepView,
  Transit,
  InjectionSummary,
} from "../../types";

/** An agent's state as the engine records it (`snapshot_agent_state`), with the room he is in. */
export interface BodyRecord {
  agent_id: string;
  agent_name: string;
  gender: string;
  age: number | null;
  location: string;
  location_id: string; // "" between two waypoints
  emotion: string;
  activity_status: string;
  emotion_label: string;
  activity_label: string;
  dominant_need: string | null;
  dominant_need_label: string;
  is_main_character: boolean;
  color: string;
  long_term_goals: string[];
  short_term_goals: GoalView[];
  emotion_intensity: number | null;
  emotion_valence: number | null;
  vitality: number | null;
  is_active: boolean;
  condition?: string;
  transit: Transit | null;
  displaced?: boolean;
}

/** One participant's action record, as `completion_record` / the begin record write it. */
export interface ActRecord {
  agent_id: string;
  agent_name: string;
  action_description: string;
  outcome: string;
  gist: string;
  succeeded: boolean;
  failure_reason: string;
  detected: boolean;
  not_executed: boolean;
  action_type: string;
  phase: string;
  deed: string;
  execution_id: string;
  initiator_id: string;
  duration_label: string;
  elapsed_steps: number;
  total_steps: number;
  interrupted_by: string;
  target: ActionTarget;
  affected_entity_ids: string[];
  overheard_by: string[];
  inner_monologue: string;
  is_main_character: boolean;
  dialogue: DialogueTurn[];
  // The route walked, on the beat a MOVE ended.
  route?: Transit | null;
  seq: number;
}

/** A mindless body as the environment reports it, with the room it is in. */
export interface RunnerRecord {
  npc_id: string;
  name: string;
  location: string;
  location_id: string;
  color: string;
  gender: string;
  age: number | null;
  description: string;
  condition: string;
  outcome: string;
  ongoing: boolean;
  displaced?: boolean;
}

/** A delivered message. `location_id` is the room a place-scoped one was announced to. */
export type LetterRecord = Omit<MessageSummary, "kind" | "audience"> & { location_id?: string };

/** A delivered broadcast; `location_id` null = the whole world. */
export type NoticeRecord = Omit<BroadcastSummary, "kind" | "audience"> & { location_id: string | null };

/** A figure in the bench's cast. Gender + age pick the BODY (skins.bodyFor). */
export interface LabCharacter {
  id: string;
  name: string;
  color: string;
  gender: string;
  age: number;
  main: boolean;
}

/**
 * The deed scenes' cast. The first three share one body (young men) so any difference between them
 * is the deed; their colors differ to show the garment dye works on every pose. Bodies per
 * (gender, age) are the "身形" (build) scenes' question.
 *
 * Ids use the engine's `agent-<hex>` shape so nothing comes to depend on a legible id.
 */
export const CAST: LabCharacter[] = [
  { id: "agent-3f9c1ab204", name: "阿墨", color: "#e8b13d", gender: "男", age: 24, main: true },
  { id: "agent-7b21e5d0c8", name: "阿石", color: "#5fa8d3", gender: "男", age: 27, main: false },
  { id: "agent-c4d8027fa1", name: "阿霖", color: "#c96f8e", gender: "男", age: 22, main: false },
  // Three more so a room can be genuinely crowded: with three figures every layout looks correct.
  // Used only in the "站位" (standing) scenes.
  { id: "agent-a51e6c7b39", name: "阿岚", color: "#7fbf7f", gender: "女", age: 31, main: false },
  { id: "agent-d07f42a8c5", name: "阿岩", color: "#b98cd8", gender: "男", age: 45, main: false },
  { id: "agent-2ec95b13f7", name: "阿澈", color: "#d9784a", gender: "男", age: 19, main: false },
];

export const [MO, SHI, LIN, LAN, YAN, CHE] = CAST;

/**
 * One figure per (gender, age bracket): the whole body table at once. Ages sit well clear of the
 * bracket edges (skins.BRACKETS).
 *
 * Ordinary characters with ordinary demographics: the body must arrive as a real agent's does, through
 * `setCast`, never through a bench-only override.
 */
export const BODIES: LabCharacter[] = [
  { id: "agent-9a1b0c2d31", name: "男童", color: "#e8b13d", gender: "男", age: 8, main: false },
  { id: "agent-9a1b0c2d32", name: "男青", color: "#5fa8d3", gender: "男", age: 24, main: false },
  { id: "agent-9a1b0c2d33", name: "男壮", color: "#7fbf7f", gender: "男", age: 45, main: false },
  { id: "agent-9a1b0c2d34", name: "男老", color: "#b98cd8", gender: "男", age: 68, main: false },
  { id: "agent-9a1b0c2d35", name: "女童", color: "#c96f8e", gender: "女", age: 8, main: false },
  { id: "agent-9a1b0c2d36", name: "女青", color: "#d9784a", gender: "女", age: 24, main: false },
  { id: "agent-9a1b0c2d37", name: "女壮", color: "#4fb3a5", gender: "女", age: 45, main: false },
  { id: "agent-9a1b0c2d38", name: "女老", color: "#9aa5b1", gender: "女", age: 68, main: false },
];

/**
 * Every figure the bench can put on a map, as the renderer's cast feed takes it: whole, like the
 * observation view's roster from step-0 profiles.
 */
export function fixtureCast(): Cast {
  return Object.fromEntries(
    [...CAST, ...BODIES].map((a) => [a.id, {
      agent_id: a.id, name: a.name, gender: a.gender, age: a.age, color: a.color,
      role: "", background: "", appearance: "",
    }]),
  );
}

/**
 * The color the backend gives every NPC (`world/identity_color.py`), written out rather than
 * taken from the renderer, since fixtures stand in for the wire.
 */
export const MINDLESS_INK = "#8a93a6";

export interface LabRunner {
  npc_id: string;
  name: string;
  color: string;
  gender: string;
  age: number;
  description: string;
}

export function mindless(
  body: LabRunner,
  at: Room,
  over: Partial<RunnerRecord> = {},
): RunnerRecord {
  return {
    ...body, location: at.name, location_id: at.id, condition: "", outcome: "", ongoing: false, ...over,
  };
}

/**
 * A room with both its id and name, as the wire uses both. Rooms are cast from the loaded map (see
 * places.ts), so nothing below may name a place, in an id or in prose.
 */
export type Room = LabRoom;

/**
 * Canonical `EmotionType` values (agent/personality.py). Never prose: "愤怒" is a string the engine
 * cannot produce.
 */
export const Emotion = {
  neutral: "neutral",
  joy: "joy",
  anger: "anger",
  fear: "fear",
  trust: "trust",
  anticipation: "anticipation",
  frustration: "frustration",
} as const;

/** Canonical need ids, as `dominant_need` carries them. */
export const Need = {
  safety: "safety",
  social: "social",
  esteem: "esteem",
  selfActualization: "self_actualization",
} as const;

let seqCounter = 0;
export const nextSeq = (): number => (seqCounter += 1);
let execCounter = 0;

/** `talk_agent-80375bf6a1_1_bfedd5` — the engine's own execution-id shape. */
export function execId(type: string, character: LabCharacter, step: number): string {
  execCounter += 1;
  return `${type}_${character.id}_${step}_${execCounter.toString(16).padStart(6, "0")}`;
}

export function who(
  character: LabCharacter,
  room: Room,
  over: Partial<BodyRecord> = {},
): BodyRecord {
  return {
    agent_id: character.id,
    agent_name: character.name,
    gender: character.gender,
    age: character.age,
    location: room.name,
    location_id: room.id,
    emotion: Emotion.neutral,
    activity_status: "idle",
    emotion_label: "",
    activity_label: "",
    dominant_need_label: "",
    dominant_need: Need.safety,
    is_main_character: character.main,
    color: character.color,
    long_term_goals: [],
    short_term_goals: [],
    emotion_intensity: 0.35,
    emotion_valence: 0,
    // A live world's healthy agent reads 0.99-something, never exactly 1. The bar shows below 0.85.
    vitality: 0.994,
    is_active: true,
    transit: null,
    ...over,
    ...(over.transit ? whereOnRoad(over.transit) : {}),
  };
}

/** Aimed at people. A broadcast is one action with several recipients, so it takes several ids. */
export const aimAgents = (...ids: string[]): ActionTarget => ({
  acts_on: ids.map((id) => ({ kind: RefKind.agent, id })),
  claims: [],
  reaches: [],
});

/** A conversation. The other party is both acted on (spoken to) and claimed (it uses up his turn). */
export const aimTalk = (id: string): ActionTarget => ({
  acts_on: [{ kind: RefKind.agent, id }],
  claims: [{ kind: RefKind.agent, id }],
  reaches: [],
});

/** Aimed at a place: a move's destination. `claims` lists the people taken along, whose turn is used up. */
export const aimPlace = (id: string, claims: string[] = []): ActionTarget => ({
  acts_on: [{ kind: RefKind.location, id }],
  claims: claims.map((cid) => ({ kind: RefKind.agent, id: cid })),
  reaches: [],
});

/** Aimed at a thing. `reaches` lists who receives it in a hand-off. */
export const aimThing = (
  id: string, kind: string = RefKind.item, reaches: string[] = [],
): ActionTarget => ({
  acts_on: [{ kind, id }],
  claims: [],
  reaches: reaches.map((rid) => ({ kind: RefKind.agent, id: rid })),
});

/** Aimed at nothing: WORK, REST and any other action without a target. */
export const aimNothing = (): ActionTarget => ({ acts_on: [], claims: [], reaches: [] });


export function did(
  character: LabCharacter,
  over: Partial<ActRecord> & Pick<ActRecord, "action_type" | "deed">,
): ActRecord {
  return {
    agent_id: character.id,
    agent_name: character.name,
    action_description: "",
    outcome: "",
    succeeded: true,
    failure_reason: "",
    detected: false,
    not_executed: false,
    // The common case; a multi-step beat overrides it (begin / ongoing_tick / …).
    phase: Phase.settled,
    execution_id: execId(over.action_type || "act", character, 0),
    initiator_id: character.id,
    duration_label: "",
    elapsed_steps: 0,
    total_steps: 0,
    interrupted_by: "",
    target: aimNothing(),
    affected_entity_ids: [],
    inner_monologue: "",
    is_main_character: character.main,
    dialogue: [],
    seq: nextSeq(),
    ...over,
    // As the backend's ActionResult: the gist defaults to the outcome.
    gist: over.gist ?? over.outcome ?? "",
    overheard_by: over.overheard_by ?? [],
  };
}

/**
 * A move in progress. `path` names every room passed through, not just the ends, as a real trek
 * does; the renderer joins one A* leg per pair.
 */
export function walking(
  path: Room[],
  elapsed: number,
  total: number,
  // The step each room is stood on. Default: the legs split the journey evenly.
  arrivals: number[] = path.map((_, i) => Math.round((i * total) / (path.length - 1))),
): Transit {
  const transit: Transit = {
    from_location_id: path[0].id,
    to_location_id: path[path.length - 1].id,
    path: path.map((r) => r.id),
    arrivals,
    elapsed_steps: elapsed,
    total_steps: total,
  };
  roadRooms.set(transit, path);
  return transit;
}

// The rooms behind a `walking()` transit, so `who()` can say where a mover is the way the
// wire does: on a waypoint on the step he stands on it, "途中" between two rooms.
const roadRooms = new WeakMap<Transit, Room[]>();

function whereOnRoad(transit: Transit): { location: string; location_id: string } {
  const rooms = roadRooms.get(transit) ?? [];
  const i = transit.arrivals.findIndex(
    (a, k) => k > 0 && k < transit.arrivals.length - 1 && a === transit.elapsed_steps,
  );
  return i > 0 && rooms[i] ? { location: rooms[i].name, location_id: rooms[i].id } : { location: "途中", location_id: "" };
}

interface StepParts {
  states: BodyRecord[];
  npcs?: RunnerRecord[];
  actions?: ActRecord[];
  entities?: Record<string, PlacedEntity>;
  injections?: InjectionSummary[];
  messages?: LetterRecord[];
  broadcasts?: NoticeRecord[];
  /** Who is close to whom. Absent = everyone indifferent. */
  relations?: GraphEdge[];
  hour?: number;
  /** This step's number when it must be a particular one (0: the world as built). */
  at?: number;
  /**
   * Which of `entities` were made on this step (stamped with its number, which only `step` knows).
   * The rest keep step 0, so a later step shows the same thing as ordinary.
   */
  bornIds?: string[];
}

let stepCounter = 0;

const entityView = ({ presence: _p, presence_ref: _r, ...e }: PlacedEntity): EntityView => e;

export function step({
  states,
  npcs = [],
  actions = [],
  entities = {},
  injections = [],
  messages = [],
  broadcasts = [],
  relations = [],
  hour = 10,
  at,
  bornIds = [],
}: StepParts): StepView {
  stepCounter += 1;
  const n = at ?? stepCounter;
  const table = Object.fromEntries(
    Object.entries(entities).map(([id, e]) => [id, bornIds.includes(id) ? { ...e, created_step: n } : e]),
  );

  const locations: Record<string, LocationScope> = {};
  const transit: StepView["transit"] = { agents: [], npcs: [], events: [] };
  const scopeOf = (id: string): { agents: AgentView[]; npcs: NpcView[]; events: ScopeEvent[] } => {
    if (!id) return transit;
    return (locations[id] ??= {
      location_id: id, agents: [], npcs: [], entities: {}, events: [],
    });
  };
  const worldScopeEvents: ScopeEvent[] = [];
  const belongings = (holder: string, open: boolean) => Object.fromEntries(
    Object.entries(table)
      .filter(([, e]) => e.presence === "held" && e.presence_ref === holder && e.is_public === open)
      .map(([id, e]) => [id, entityView(e)]),
  );
  for (const [id, e] of Object.entries(table)) {
    if (e.presence === "at_location") {
      (scopeOf(e.presence_ref) as LocationScope).entities[id] = entityView(e);
    }
  }

  // The read model's fold: one event per (execution, beat), told by the initiator's record.
  const where = new Map(states.map((s) => [s.agent_id, s.location_id]));
  const thoughts = new Map<string, { seq: number; execution_id: string; text: string }[]>();
  const groups = new Map<string, ActRecord[]>();
  actions.forEach((r, i) => {
    const key = r.execution_id ? `${r.execution_id}|${r.phase}` : `#${i}`;
    (groups.get(key) ?? groups.set(key, []).get(key)!).push(r);
  });
  for (const records of groups.values()) {
    const initiator = records[0].initiator_id || records[0].agent_id;
    const lead = records.find((r) => r.agent_id === initiator) ?? records[0];
    const participants = [...new Set([initiator, ...records.map((r) => r.agent_id)])];
    for (const r of records) {
      if (!r.inner_monologue) continue;
      (thoughts.get(r.agent_id) ?? thoughts.set(r.agent_id, []).get(r.agent_id)!)
        .push({ seq: lead.seq, execution_id: lead.execution_id, text: r.inner_monologue });
    }
    const overheard = [...new Set(records.flatMap((r) => r.overheard_by))];
    const event: ActionSummary = {
      kind: "action",
      seq: lead.seq,
      execution_id: lead.execution_id,
      initiator_id: initiator,
      participant_ids: participants,
      action_type: lead.action_type,
      phase: lead.phase,
      deed: lead.deed,
      target: lead.target,
      affected_entity_ids: [...new Set(records.flatMap((r) => r.affected_entity_ids))],
      overheard_by: overheard,
      interrupted_by: lead.interrupted_by,
      succeeded: lead.succeeded,
      failure_reason: lead.failure_reason,
      detected: lead.detected,
      not_executed: lead.not_executed,
      public: null,
      private: {
        audience: [...participants, ...overheard],
        outcome: lead.outcome,
        gist: lead.gist,
        dialogue: records.find((r) => r.dialogue.length)?.dialogue ?? [],
        action_description: lead.action_description,
        duration_label: lead.duration_label,
        elapsed_steps: lead.elapsed_steps,
        total_steps: lead.total_steps,
        route: lead.route ?? null,
      },
    };
    scopeOf(where.get(initiator) ?? "").events.push(event);
  }

  const edges = new Map<string, GraphEdge[]>();
  for (const e of relations) (edges.get(e.from_id) ?? edges.set(e.from_id, []).get(e.from_id)!).push(e);
  for (const s of states) {
    scopeOf(s.location_id).agents.push({
      agent_id: s.agent_id,
      name: s.agent_name,
      gender: s.gender,
      age: s.age,
      is_main_character: s.is_main_character,
      is_active: s.is_active,
      public: {
        // Someone acting covertly doesn't look like it, as the read model has it.
        activity_status: s.activity_status === "covert" ? "" : s.activity_status,
        activity_label: s.activity_status === "covert" ? "" : s.activity_label,
        condition: s.condition ?? "",
        displaced: s.displaced ?? false,
        carrying: belongings(s.agent_id, true),
      },
      private: {
        emotion: s.emotion,
        emotion_label: s.emotion_label,
        emotion_intensity: s.emotion_intensity,
        emotion_valence: s.emotion_valence,
        dominant_need: s.dominant_need,
        dominant_need_label: s.dominant_need_label,
        vitality: s.vitality,
        activity_status: s.activity_status,
        activity_label: s.activity_label,
        transit: s.transit,
        long_term_goals: s.long_term_goals,
        short_term_goals: s.short_term_goals,
        relations: edges.get(s.agent_id) ?? [],
        thoughts: thoughts.get(s.agent_id) ?? [],
        carrying: belongings(s.agent_id, false),
      },
    });
  }
  for (const r of npcs) {
    scopeOf(r.location_id).npcs.push({
      npc_id: r.npc_id,
      name: r.name,
      color: r.color,
      gender: r.gender,
      age: r.age,
      description: r.description,
      public: {
        condition: r.condition,
        displaced: r.displaced ?? false,
        carrying: belongings(r.npc_id, true),
      },
      private: { outcome: r.outcome, ongoing: r.ongoing, carrying: belongings(r.npc_id, false) },
    });
  }

  for (const { location_id, ...m } of messages) {
    if (m.scope === "world") {
      worldScopeEvents.push({ ...m, kind: "message", audience: null });
    } else if (m.scope === "place") {
      scopeOf(location_id ?? "").events.push({ ...m, kind: "message", audience: m.receiver_ids });
    } else {
      // A letter reaches a man, not a room: it lands wherever each recipient stands.
      const byRoom = new Map<string, string[]>();
      for (const rid of m.receiver_ids) {
        const at = where.get(rid) ?? "";
        (byRoom.get(at) ?? byRoom.set(at, []).get(at)!).push(rid);
      }
      for (const [at, receivers] of byRoom) {
        scopeOf(at).events.push({ ...m, kind: "message", receiver_ids: receivers, audience: receivers });
      }
    }
  }
  for (const { location_id, ...b } of broadcasts) {
    if (!location_id) {
      worldScopeEvents.push({ ...b, kind: "broadcast", audience: null });
      continue;
    }
    // Heard by whoever stood there as the step began.
    const heard = [...where].filter(([, at]) => at === location_id).map(([id]) => id);
    scopeOf(location_id).events.push({ ...b, kind: "broadcast", audience: heard });
  }

  const bySeq = (a: ScopeEvent, b: ScopeEvent) => a.seq - b.seq;
  for (const l of Object.values(locations)) l.events.sort(bySeq);
  transit.events.sort(bySeq);
  return {
    world_id: "lab",
    step: n,
    world_time: {
      // Narrative clock: never parsed.
      label: `大唐武德九年，六月初四，${hour}时`,
      // Machine clock: what the day/night wash reads.
      hour,
      minute: 0,
    },
    locations,
    transit,
    world: { events: worldScopeEvents.sort(bySeq) },
    observer: { injections },
  };
}

export interface LabScene {
  id: string;
  group: string;
  title: string;
  watch: string;
  steps: StepView[];
  viewer?: string;
}

// Props for the object deeds: one takeable, one fixed. Re-declared per step because the entity
// snapshot is per-step state, and presence changes (on the ground → held → gone) are what the
// seize/destroy scenes show: a destroyed thing is simply absent from its step. `presence_ref` is a
// location id or a holder's agent id.
export const crate = (where: string, over: Partial<PlacedEntity> = {}): Record<string, PlacedEntity> => ({
  "seed_seed-91ac33be40": {
    name: "货箱",
    entity_type: "item",
    state: "intact",
    presence: "at_location",
    presence_ref: where,
    description: "",
    is_public: true,
    private: { content: "" },
    created_step: 0,
    ...over,
  },
});
export const CRATE = "seed_seed-91ac33be40";

/**
 * Things in somebody's hands, the pocket's input, built directly since a real world rarely reaches
 * a man carrying five things. `[name, state]` pairs: the pocket shows the state too.
 */
export const carried = (holder: string, items: [string, string][]): Record<string, PlacedEntity> =>
  Object.fromEntries(
    items.map(([name, state], i) => [
      // Ids are stable across a scene's steps (a changed id reads as one thing vanishing and another
      // appearing) and derived from the holder (arming two people merges two of these into one
      // record; shared ids would overwrite the first man's things).
      `seed_seed-${holder.slice(-8)}${i}`,
      {
        name,
        entity_type: "item",
        state,
        presence: "held" as const,
        presence_ref: holder,
        description: "",
        is_public: true,
        private: { content: "" },
        created_step: 0,
      },
    ]),
  );

export const gate = (where: string, over: Partial<PlacedEntity> = {}): Record<string, PlacedEntity> => ({
  "seed_seed-2b70df1c85": {
    name: "坊门",
    entity_type: "landmark",
    state: "sealed",
    presence: "at_location",
    presence_ref: where,
    description: "",
    is_public: true,
    private: { content: "" },
    created_step: 0,
    ...over,
  },
});
export const GATE = "seed_seed-2b70df1c85";

/**
 * Wind the counters back to zero.
 *
 * `step`, `did` and `execId` number things from module state on purpose, in build order; threading
 * a counter through hundreds of call sites would buy nothing. The cost: `buildScenes` must call
 * this first (and is the only caller), or a second template would continue the first's numbering.
 */
export function resetFixtures(): void {
  stepCounter = 0;
  seqCounter = 0;
  execCounter = 0;
}
