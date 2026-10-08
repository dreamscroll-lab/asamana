// Wire types mirroring interaction/models.py (StepEvent family) and the API layer.
// Render-neutral: these are semantic world state only — no coordinates.

export type RunState =
  | "idle"
  | "running"
  | "paused"
  | "stopping"
  | "completed"
  | "failed"
  | null;

export interface WorldMeta {
  world_id: string;
  theme: string;
  world_name: string;
  description: string;
  /**
   * How far the story has got, read off the snapshots: in_progress | unknown.
   * NOT whether it is running right now — that is `run_state`, and only it knows.
   */
  status: string;
  created_at: string | null;
  current_step: number;
  main_agent_names: string[];
  /**
   * World time per step, in seconds; authored by the theme analysis and frozen at build.
   * Shown on the creation-review card since it decides whether the story runs in hours or
   * days. 0 when unknown.
   */
  seconds_per_step: number;
  run_state: RunState;
  // Whether the user has reviewed and locked initialization. Unconfirmed worlds
  // sit in the creation-review stage and cannot start their narrative run.
  confirmed: boolean;
}

/** What this backend deployment offers, independent of any world. */
export interface Deployment {
  /** Whether the developer routes (/dev, the map workbench's) are mounted. */
  dev_tools: boolean;
  /**
   * Whether the backend holds its model API keys. Without them it serves stored worlds but
   * refuses anything that calls a model (build, run, step, direct, reset, model-calling dev tools).
   */
  model_keys: boolean;
}

/** A sample theme offered on the home screen, authored in the backend's content file. */
export interface ThemePreset {
  title: string;
  theme: string;
}

/** A map a world can be built on — one per worlds/templates/<name>/ directory. */
export interface MapTemplate {
  template: string;
  world_name: string;
  era_name: string;
  description: string;
  location_count: number;
}

export interface BuildJob {
  job_id: string;
  theme: string;
  template: string | null;
  status: "building" | "ready" | "failed";
  world_id: string | null;
  world_name: string | null;
  error: string | null;
}

/** One thing an action names, and what kind of thing it is. Mirrors the backend's TargetRef. */
export interface TargetRef {
  kind: string; // agent | npc | item | landmark | object | location
  id: string;
}

/**
 * An action's three relations to the world — the same three the backend files.
 *
 * `acts_on` is what the act is done to (one kind per action, guaranteed by the backend);
 * `claims` is whose turn it spends (e.g. a carried companion); `reaches` is who receives it
 * passively. All three carry kinded refs. Read them through lib/contract's accessors.
 */
export interface ActionTarget {
  acts_on: TargetRef[];
  claims: TargetRef[];
  reaches: TargetRef[];
}

export interface ActionSummary {
  agent_id: string;
  agent_name: string;
  action_description: string;
  outcome: string;
  // What happened in one sentence: `outcome` without the detail it may append (a TALK
  // transcript, an interrupter's thought).
  gist: string;
  succeeded: boolean;
  // Why it did not succeed, as one short third-person phrase; non-empty only on a genuine
  // failure. Read this rather than parsing `outcome` prose: render decisions come off
  // structured fields, as with `deed`.
  failure_reason: string;
  // Whether a COVERT act was noticed. Independent of `succeeded`: a theft can land and still
  // be seen, or fail unnoticed.
  detected: boolean;
  // True when the action never engaged the world (target absent or busy, no path): a
  // non-event, not a failure. The renderer mutes it (gray chip / "未成事" badge, no ✗).
  not_executed: boolean;
  action_type: string; // talk/move/physical/covert/work/rest/send_message/errand
  // WHICH BEAT of an execution this is — the only thing that says whether `outcome` is an
  // opening or a result. See lib/contract's Phase for the six values.
  phase: string;
  // What the body was seen doing (see lib/contract Deed). Same as `action_type` except for
  // physical, which packs six verbs into one type. Empty = nothing was done → draw nothing.
  deed: string;
  // A joint action emits one record per participant. To fold them into one deed, group by
  // execution_id and speak from the record whose agent_id === initiator_id (a conscripted
  // participant's outcome only restates the intent). Code-layer ids, never rendered.
  execution_id: string;
  initiator_id: string;
  // How long he expects it to take, as a natural duration ("约2小时") rendered by the producer
  // (a step count never reaches a narrative surface). "" unless phase is `begin`.
  duration_label: string;
  // How far along, for a progress bar. Code-layer integers a renderer may consume; never
  // render 「第N步」 as text.
  elapsed_steps: number;
  total_steps: number;
  // Who cut short an in-progress action (phase "interrupt"); "" otherwise. The why rides in
  // `outcome`, not in inner_monologue.
  interrupted_by: string;
  // The act's three relations, as the cognition layer filed them. Read them through
  // lib/contract's accessors (actedOnAgents / actedOnEntities / claimedAgents).
  target: ActionTarget;
  // AIMED at (intent) vs. CHANGED (result). Aim survives failure, so `target.acts_on` is what
  // a figure turns toward and walks to; `affected` is empty precisely when he tried and
  // nothing gave.
  affected_entity_ids: string[];
  // Present for it, party to none of it: who overheard this exchange. Not in
  // participant_ids — they spent no turn on it and the feed must not fold them in as
  // speakers.
  overheard_by: string[];
  // The first-person deliberation that produced this action, present only on the beat the
  // decision was made. The feed's "心声"; the map never shows it (it renders the world, not minds).
  inner_monologue: string;
  is_main_character: boolean;
  dialogue: DialogueTurn[];
  // Monotonic emission ordinal from the runtime — sort intra-step events by this,
  // never render it as user-visible text. 0 for initialization records.
  seq: number;
}

// One line of a conversation. `speaker_id` is the join key (never rendered); `speaker` is
// the name to print.
export interface DialogueTurn {
  speaker_id: string;
  speaker: string;
  line: string;
}

// A non-location world object (item / landmark) as the per-step environment
// snapshot carries it — drives entity markers + state-change effects on the map.
// Placement is a single discriminated axis: presence + presence_ref (a location
// id when at_location, a holder agent id when held, null when destroyed).
export type EntityPresence = "at_location" | "held" | "destroyed";
export interface EntityView {
  name: string;
  entity_type: string;
  state: string;
  presence: EntityPresence;
  presence_ref: string | null;
  description: string;
  // Whether anyone STANDING THERE can perceive it. False for a thing its owner has not
  // shown — a document just written and pocketed. The map is a god view and draws it
  // regardless; the flag is what lets it say "only he knows he has this".
  is_public: boolean;
  // What it carries — the words in a letter, the figures in a ledger. The map is a god view,
  // so it arrives whole; who can read it IN the world is the backend's question.
  content: string;
  // The step it came into the world on, so new things are read off the step's own table
  // rather than by diffing (impossible for the feed's earliest step).
  created_step: number;
}

// Movement transit: present while an agent is mid-move. The renderer walks it
// along `path` (the real waypoint sequence through intermediate rooms), so a
// multi-hop trek follows the logical route instead of a straight line. `path`
// includes both endpoints; from/to are its ends. `arrivals[i]` is the elapsed step
// on which he reaches `path[i]` — legs differ in cost, so this, not elapsed/total,
// says where he is.
export interface Transit {
  from_location_id: string | null;
  to_location_id: string | null;
  path: string[];
  arrivals: number[];
  elapsed_steps: number;
  total_steps: number;
}

// A short-term goal with its lifecycle state and where it came from.
// `origin` is the interesting half: "cognitive" is something they decided to pursue,
// "residue" is an item an action left unsettled — taken on in an exchange, a job half
// done, what they came here to do. The two carry different weight, so the card shows
// them apart. Older snapshots predate the field — treat a missing value as "cognitive".
export interface ShortTermGoalEntity {
  text: string;
  status: string;
  origin?: "cognitive" | "residue";
}

export interface AgentStateSummary {
  agent_id: string;
  agent_name: string;
  location: string; // narrative name — for text only
  // The same place as its id — for looking the location up (map rooms, grouping). While
  // moving: the waypoint room on the step he stands on one, "" between two rooms
  // (`location` is then 「途中」).
  // Mirrors MessageSummary's place/place_id pair.
  location_id: string;
  emotion: string;
  activity_status: string;
  // The backend's narrative-layer names for the two values above; "" when unknown.
  emotion_label: string;
  activity_label: string;
  dominant_need: string | null;
  dominant_need_label: string; // the same, for dominant_need
  is_main_character: boolean;
  color: string;
  long_term_goals: string[];
  short_term_goals: string[];
  short_term_goal_entities?: ShortTermGoalEntity[];
  emotion_intensity: number | null;
  emotion_valence: number | null;
  vitality: number | null;
  is_active: boolean;
  // An imposed, lasting condition on this body as one narrative phrase ("双手被反绑"), "" when
  // free. A standing fact until undone or worn off, so drawn as a quiet marker, not a beat.
  // Absent on older snapshots.
  condition?: string;
  transit: Transit | null;
  // The trip that ended this step (elapsed_steps == total_steps): the route he walked to get
  // here, which a trip crossed within one step has no transit to carry.
  arrival: Transit | null;
  // Transit's opposite: this step the director put him here and he did not walk. The renderer
  // must cut to the new position rather than draw a walk. Absent on older snapshots.
  displaced?: boolean;
}

// A body that acts but does not think: it walks, carries, delivers and reports back.
//
// `color` comes from the backend, which reserves gray for mindless bodies where the palette
// lives (world/identity_color.py); don't substitute a local constant. The body sheet is a
// plain local lookup from (gender, age) — see skins.bodyFor.
export interface NpcStateSummary {
  npc_id: string;
  name: string;
  location: string;
  location_id: string; // see AgentStateSummary.location_id
  color: string;
  gender: string;
  age: number | null;
  description: string;
  // Two sentences composed by the backend; this side only orders and prints them (no
  // category to reword locally).
  //
  //   condition  his situation right now   e.g. "pinned to the ground"
  //   outcome    what happened this beat   e.g. "delivered the letter to X"
  //
  // `outcome` names no place: it happened at `location` (the backend's same-place invariant),
  // so "name（location）" + `outcome` never names two places. An errand's message appears
  // verbatim only on the beat he says it; travel beats name only the kind of errand.
  // Entities are reported by name only.
  condition: string;
  outcome: string;
  // The outcome describes something still under way, like `phase="begin"` on `ActionSummary`.
  // A world fact, not a render instruction (the map uses it in TiledWorldScene.noteNpcOutcomes).
  ongoing: boolean;
  // He was moved to `location` this beat, not walked there; same as `AgentStateSummary.displaced`.
  displaced?: boolean;
}

// A message the recipients received this step (the sending was an action a step earlier).
// `scope` says how it was addressed:
//   direct — named recipients; the news is that it reached them.
//   place  — an announcement heard in `place`; the audience is a room.
//   world  — a proclamation to everyone.
// For place/world the sender is the subject; listing recipients would make one proclamation
// read as N private letters.
export type MessageScope = "direct" | "place" | "world";
export interface MessageSummary {
  message_id: string;
  sender_id: string;
  sender_name: string;
  receiver_ids: string[];
  perceived_summary: string;
  // What the sender actually said; the rest of `perceived_summary` came attached (a runner's
  // report of the scene). The backend draws the line; a surface picks what fits it. Equal for
  // an ordinary letter. Never re-derive it here by cutting the full text.
  spoken: string;
  scope: MessageScope;
  place: string; // location NAME (never an id); "" unless scope === "place"
  // The same place as its ID — the code layer's copy, for looking the location up rather
  // than printing it (the map hangs a place-scoped announcement on the room it names).
  // Never render it. Mirrors BroadcastSummary's location_name/location_scope pair.
  place_id: string;
  // See ActionSummary.seq — same monotonic ordinal, same channel semantics.
  seq: number;
}

// A senderless world announcement (a death, an injected event) — the same delivery queue
// shape as a message but with nobody behind it. `location_name` is resolved by the producer
// (which holds the WorldDirectory), exactly as `sender_name` is on a message; "" = world-wide.
// `location_scope` is the raw id, for the map's geometry only — never render it.
export interface BroadcastSummary {
  content: string;
  broadcast_type: string;
  severity: string;
  location_name: string;
  location_scope: string | null;
  // The visible phenomenon this change comes with (fire/rain/…), or "none"; independent of
  // severity. A closed vocabulary (core/interfaces/phenomenon.py); the renderer draws nothing
  // for a value it doesn't know.
  phenomenon: string;
  seq: number;
}

// The answer to a submitted directive. `accepted: false` is a normal answer, not an
// error: a vague instruction is something the director rewords, not a failed request.
export interface DirectiveResult {
  accepted: boolean;
  reason: string;   // why it could not be carried out — read this aloud to them
  preview: string;  // what will happen, when it was accepted
  queued: number;   // how many are waiting to land
  // No run state on purpose: an accepted directive is guaranteed to land (the backend
  // advances an idle world), so the caller has nothing to decide.
}

// One row of the index of interventions (GET /api/worlds/{id}/directives), oldest first.
//
// Deliberately thinner than the feed's 🎬 card: it carries "what I said → what the engine
// did", which reads on its own, and leaves the receipt (meaningful only beside its step) on
// the card. Not a slice of the feed, which holds only watched steps. Refusals are absent:
// they changed nothing in the world.
export interface Intervention {
  step: number;
  time_label: string;      // the world's own clock; never "第 N 步"
  directive_text: string;  // the sentence that was typed
  narrative: string;       // what the engine dispatched off it
}

// Who reached into the world. "system" = the LLM narrative editor inventing a twist;
// "director" = a human. Same card, different hand — and the reader has to be able to
// tell, or they cannot reason about their own experiment.
export type EventAuthor = "system" | "director";

// What a director's intervention actually did, once the step it landed on finished.
// Absent on system-authored events. Without it, an intervention that was delivered
// correctly but judged unimportant looks exactly like one that silently failed.
export interface EventReceipt {
  // It reached these people's senses.
  delivered_to: { agent_id: string; name: string }[];
  // …and the pressure evaluator judged it actually bore on these, this hard.
  pressure: { agent_id: string; name: string; urgency: string }[];
  // …so these were admitted to a decision loop because of it,
  decided: { agent_id: string; name: string }[];
  // …and these dropped what they were doing.
  interrupted: { agent_id: string; name: string }[];
}

export interface WorldEventSummary {
  id: string;
  authored_by: EventAuthor;
  narrative: string;
  // Narrative-layer references (names and place names, never ids) — resolved through
  // the WorldDirectory at injection time.
  affected_names: string[];
  location_label: string | null;
  is_positive: boolean | null;
  // The sentence a human typed to cause this (director events only), so the card shows the
  // loop: what I said → what the engine made of it → what the world did.
  directive_text: string;
  receipt: EventReceipt | null;
  // See ActionSummary.seq — same monotonic ordinal, same channel semantics.
  seq: number;
}

// The world's clock at a step — one fact, two layers. `label` is narrative: the world's own
// name for the moment, shown whole, never parsed. `hour` / `minute` are code layer: numbers a
// renderer may read; null when the step was recorded without them.
export interface WorldTimeView {
  label: string;
  hour: number | null;
  minute: number | null;
}

export interface StepEvent {
  world_id: string;
  step: number;
  world_time: WorldTimeView;
  actions: ActionSummary[];
  messages: MessageSummary[];
  world_events: WorldEventSummary[];
  agent_states: Record<string, AgentStateSummary>;
  // Kept out of agent_states on purpose: they are not agents, so the relation graph and
  // character cards exclude them without any filtering.
  npcs: NpcStateSummary[];
  broadcasts: BroadcastSummary[];
  // Items/landmarks and where each one is, by entity id.
  entities: Record<string, EntityView>;
  // Directed relations as they stand at the end of this step — the same edges GET /graph serves.
  relations: GraphEdge[];
}

// Relationship graph (GET /api/worlds/{id}/graph). Nodes are agents; edges are
// directed relations (A→B and B→A are separate). Render-neutral: no geometry.
export interface GraphNode {
  id: string;
  name: string;
  role: string;
  is_main_character: boolean;
  color: string;
}

export interface GraphEdge {
  from_id: string;
  to_id: string;
  trust: number;
  affection: number;
  labels: string[];
  interaction_count: number;
  history_summary: string;
}

export interface WorldGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

// Static agent identity (GET /api/worlds/{id}/agents/{agent_id}/profile).
// Distinct from AgentStateSummary, which is the mutable per-step state.
export interface AgentProfile {
  agent_id: string;
  name: string;
  role: string;
  age: number | null;
  gender: string;
  background: string;
  appearance: string;
  color: string;
  core_traits: string[];
  core_values: string[];
  self_image: string;
  life_goal: string;
  secret: string;
  is_main_character: boolean;
}

// Map focus/selection: the user can follow one or more agents and/or spotlight a
// location. Owned by WorldView (React), pushed into the Phaser scene each change.
export interface MapFocus {
  agents: string[]; // multi-select follow subjects (order = selection order)
  place: { kind: "location"; id: string } | null; // spatial focus (spotlight)
  follow: boolean; // camera auto-follows the focus
}

// A click on the map emits one of these up to React; React folds it into MapFocus.
// Entity clicks are not surfaced: entity markers are info-only; clicking one does
// not change the MapFocus.
export type MapSelect =
  | { kind: "agent"; id: string }
  | { kind: "location"; id: string }
  | { kind: "clear" };

// WebSocket envelope: the backend tags each frame with a type.
export type WsFrame =
  | { type: "status"; data: { world_id: string; status: RunState } }
  | { type: "snapshot"; data: StepEvent }
  | { type: "step"; data: StepEvent };
