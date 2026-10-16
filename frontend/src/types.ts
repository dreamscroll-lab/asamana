// Wire types mirroring interaction/models.py (StepView family) and the API layer.
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
  // The character an actor performs, chosen at confirmation; null if nobody was cast. Once the
  // actor exits, `actor_exited` is set and the world is the director's again.
  actor_character_id: string | null;
  actor_exited: boolean;
  // When the actor is asked to decide while the character is idle.
  actor_pace: ActorPace;
  // The whole cast has been shown to the user as the director; the world takes no actor after.
  director_reviewed: boolean;
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

// One line of a conversation. `speaker_id` is the join key (never rendered); `speaker` is
// the name to print.
export interface DialogueTurn {
  speaker_id: string;
  speaker: string;
  line: string;
}

// ----- one step, by scope -------------------------------------------------------------
//
// A StepView is what happened in the world this step and the state it ended in, grouped by
// where it happened (mirrors interaction/models.py):
//
//   locations  one per place: the bodies and things there at the end of the step, and what
//              happened there.
//   transit    bodies between two waypoints, and what happened to them on the way. A body
//              standing on a waypoint is in that location, not here.
//   world      what reached every place at once.
//
// `observer` sits outside them: what only the one watching from outside reads. Every event
// carries `seq`, its place on the world's emission timeline: unique across steps, so one event
// that several scopes saw keeps one id. Sort by it; never render it.

// What onlookers at one place saw of an action.
export interface ObservedView {
  text: string;
  strength: number | null;
}

// What only those inside an action know of it: its participants, whoever it landed on and the
// listeners it named (`audience`). `outcome` is the complete account (a TALK's transcript, an
// interrupter's thought); `gist` the same in one sentence.
export interface ActionPrivateView {
  audience: string[];
  outcome: string;
  gist: string;
  dialogue: DialogueTurn[];
  // The doer's own words for what he set out to do; onlookers see only `public.text`.
  action_description: string;
  // How long he expects it to take, as a natural duration ("约2小时"). "" unless phase is `begin`.
  duration_label: string;
  // How far along, for a progress bar. Code-layer integers; never render 「第N步」 as text.
  elapsed_steps: number;
  total_steps: number;
  // The way walked on the beat a MOVE ended (elapsed_steps == total_steps): the renderer walks
  // him in along the waypoints the engine reported, which a trip crossed within one step has
  // no transit to carry. null otherwise.
  route: Transit | null;
}

// One beat of one execution, every participant's part folded into one event, in the scope where
// its doer stood when the beat was recorded. Places that only saw it get an ObservationSummary.
export interface ActionSummary {
  kind: "action";
  seq: number;
  execution_id: string;
  initiator_id: string;
  // Everyone who spent their turn on it, initiator included.
  participant_ids: string[];
  action_type: string; // talk/move/physical/covert/work/rest/send_message/errand
  // WHICH BEAT of an execution this is — the only thing that says whether the outcome is an
  // opening or a result. See lib/contract's Phase for the six values.
  phase: string;
  // What the body was seen doing (see lib/contract Deed). Same as `action_type` except for
  // physical, which packs several verbs into one type. Empty = nothing was done → draw nothing.
  deed: string;
  // The act's three relations, as the cognition layer filed them. Read them through
  // lib/contract's accessors (actedOnAgents / actedOnEntities / claimedAgents).
  target: ActionTarget;
  // AIMED at (intent) vs. CHANGED (result). Aim survives failure, so `target.acts_on` is what
  // a figure turns toward and walks to; `affected` is empty precisely when he tried and
  // nothing gave.
  affected_entity_ids: string[];
  // Listeners the initiator named, standing there. Not in participant_ids: they spent no turn on it.
  overheard_by: string[];
  // Who cut short an in-progress action (phase "interrupt"); "" otherwise.
  interrupted_by: string;
  succeeded: boolean;
  // Why it did not succeed, as one short third-person phrase; non-empty only on a genuine
  // failure. Read this rather than parsing prose.
  failure_reason: string;
  // Whether a COVERT act was noticed. Independent of `succeeded`.
  detected: boolean;
  // The action never engaged the world (target absent or busy, no path): a non-event, not a
  // failure. The renderer mutes it (gray chip / "未成事" badge, no ✗).
  not_executed: boolean;
  // What onlookers here saw; null when the act showed nothing here.
  public: ObservedView | null;
  private: ActionPrivateView;
}

// Something seen at a place. Either an action seen from a place other than where it is
// recorded (the origin watching a man leave, a waypoint watching him pass; `seq` is the
// action's own), or a change an injection made, seen where it landed (`seq` is the injection's
// own; nobody in the world did it, so `initiator_id` is "" and `participant_ids` is empty).
export interface ObservationSummary {
  kind: "observation";
  seq: number;
  initiator_id: string;
  participant_ids: string[];
  text: string;
  strength: number | null;
}

// A message reaching the people in one scope. `scope` says how it was addressed:
//   direct — named recipients; private (`audience`: the receivers in this scope). A letter to
//            people in several places appears in each of their scopes, under one seq.
//   place  — an announcement to the room whose scope holds it; `audience`: whoever heard it there.
//   world  — a proclamation, in the world scope; public.
// For place/world the sender is the subject; listing recipients would make one proclamation
// read as N private letters.
export type MessageScope = "direct" | "place" | "world";
export interface MessageSummary {
  kind: "message";
  seq: number;
  message_id: string;
  sender_id: string;
  sender_name: string;
  // Who received it in this scope.
  receiver_ids: string[];
  scope: MessageScope;
  content: string;
  // What the sender actually said; the rest of `content` came attached (a runner's report of
  // the scene). The backend draws the line; a surface picks what fits it. Equal for an
  // ordinary letter. Never re-derive it here by cutting the full text.
  spoken: string;
  // null when public to its scope.
  audience: string[] | null;
}

// A senderless change everyone in its scope perceives (a death, an injected event).
export interface BroadcastSummary {
  kind: "broadcast";
  seq: number;
  content: string;
  broadcast_type: string;
  severity: string;
  // The visible phenomenon this change comes with (fire/rain/…), or "none"; independent of
  // severity. A closed vocabulary (core/interfaces/phenomenon.py); the renderer draws nothing
  // for a value it doesn't know.
  phenomenon: string;
  // Who heard a place's broadcast as the step began; null for a world broadcast (everyone).
  audience: string[] | null;
}

export type ScopeEvent = ActionSummary | ObservationSummary | MessageSummary | BroadcastSummary;

// A non-agent world object (item / landmark) at the end of the step. Where it is follows from
// where it is listed: a location or a holder's belongings. A destroyed thing is no longer in the
// world and is listed nowhere; the act that destroyed it names it in `affected_entity_ids`.
export interface EntityView {
  name: string;
  entity_type: string;
  state: string;
  description: string;
  // Whether anyone in its presence can perceive it. False for a thing its owner has not
  // shown — a document just written and pocketed. The map is a god view and draws it
  // regardless; the flag is what lets it say "only he knows he has this".
  is_public: boolean;
  // The step it came into the world on, so new things are read off the step's own table
  // rather than by diffing (impossible for the feed's earliest step).
  created_step: number;
  // What only its holder can read — the words in a letter, the figures in a ledger. Seeing a
  // thing is not reading it.
  private: { content: string };
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
// them apart.
export interface GoalView {
  text: string;
  status: string;
  origin: "cognitive" | "residue";
}

// What anyone in an agent's presence can see of them.
export interface AgentPublicView {
  // What he is visibly doing; "" when it doesn't show (someone acting covertly). His actual
  // activity is `private.activity_status`.
  activity_status: string;
  activity_label: string; // the backend's narrative-layer name; "" when unknown
  // An imposed, lasting condition on this body as one narrative phrase ("双手被反绑"), "" when
  // free. A standing fact until undone or worn off, so drawn as a quiet marker, not a beat.
  condition: string;
  // Transit's opposite: this step the director put him here and he did not walk. The renderer
  // must cut to the new position rather than draw a walk.
  displaced: boolean;
  // Belongings others can see.
  carrying: Record<string, EntityView>;
}

// The deliberation behind one of the agent's acts, set on the beat it was decided. The feed's
// "心声"; the map never shows it (it renders the world, not minds).
export interface ThoughtView {
  seq: number;
  execution_id: string;
  text: string;
}

// What only the agent knows: their mind, where they are headed, and hidden belongings.
export interface AgentPrivateView {
  emotion: string;
  emotion_label: string; // the backend's narrative-layer name; "" when unknown
  emotion_intensity: number | null;
  emotion_valence: number | null;
  dominant_need: string | null;
  dominant_need_label: string;
  vitality: number | null;
  activity_status: string;
  activity_label: string;
  // Private: a passer-by is seen, his route is not.
  transit: Transit | null;
  long_term_goals: string[];
  // The whole queue, finished goals included; `status` says which are still live.
  short_term_goals: GoalView[];
  // Their relations to others at the end of the step — the same edges GET /graph serves.
  relations: GraphEdge[];
  thoughts: ThoughtView[];
  // Belongings nobody else can perceive.
  carrying: Record<string, EntityView>;
}

export interface AgentView {
  agent_id: string;
  name: string;
  // Visible at a glance, like the name.
  gender: string;
  age: number | null;
  color: string;
  // null: not known to this viewer (the actor's view).
  is_main_character: boolean | null;
  is_active: boolean;
  public: AgentPublicView;
  private: AgentPrivateView;
}

// A body that acts but does not think: it walks, carries, delivers and reports back. Kept apart
// from agents, so the relation graph and character cards exclude it without any filtering.
//
// `color` comes from the backend, which reserves gray for mindless bodies where the palette
// lives (world/identity_color.py); don't substitute a local constant. The body sheet is a
// plain local lookup from (gender, age) — see skins.bodyFor.
export interface NpcView {
  npc_id: string;
  name: string;
  color: string;
  gender: string;
  age: number | null;
  description: string;
  public: {
    // His situation right now as one sentence composed by the backend ("pinned to the ground").
    condition: string;
    // He was moved here this beat, not walked here.
    displaced: boolean;
    carrying: Record<string, EntityView>;
  };
  // What nobody in the world reads: it has no mind and no actor performs it, so this is the
  // observer's alone.
  private: {
    // What happened to him this beat, as one sentence composed by the backend ("delivered the
    // letter to X"); "" when nothing did. It names no place: it happened where he ends the step.
    // The engine keeps an Npc's doings out of every onlooker's perception, hence private.
    outcome: string;
    // Something still under way, like `phase="begin"` on an action.
    ongoing: boolean;
    // Hidden belongings (a sealed letter it carries).
    carrying: Record<string, EntityView>;
  };
}

export interface LocationScope {
  location_id: string;
  name: string;
  agents: AgentView[];
  npcs: NpcView[];
  entities: Record<string, EntityView>;
  events: ScopeEvent[]; // in seq order
}

export interface TransitScope {
  agents: AgentView[];
  npcs: NpcView[];
  events: ScopeEvent[];
}

export interface WorldScope {
  events: ScopeEvent[];
}

// What only the one watching from outside the world reads; no character perceives it.
export interface ObserverView {
  injections: InjectionSummary[];
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
export type InjectionAuthor = "system" | "director";

// What a director's intervention actually did, once the step it landed on finished.
// Absent on system-authored events. Without it, an intervention that was delivered
// correctly but judged unimportant looks exactly like one that silently failed.
export interface InjectionReceipt {
  // It reached these people's senses.
  delivered_to: { agent_id: string; name: string }[];
  // …and the pressure evaluator judged it actually bore on these, this hard.
  pressure: { agent_id: string; name: string; urgency: string }[];
  // …so these were admitted to a decision loop because of it,
  decided: { agent_id: string; name: string }[];
  // …and these dropped what they were doing.
  interrupted: { agent_id: string; name: string }[];
}

export interface InjectionSummary {
  id: string;
  authored_by: InjectionAuthor;
  narrative: string;
  // Narrative-layer references (names and place names, never ids) — resolved through
  // the WorldDirectory at injection time.
  affected_names: string[];
  location_label: string | null;
  is_positive: boolean | null;
  // The sentence a human typed to cause this (director events only), so the card shows the
  // loop: what I said → what the engine made of it → what the world did.
  directive_text: string;
  receipt: InjectionReceipt | null;
  // The same emission ordinal as every scope event's `seq`.
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

export interface StepView {
  world_id: string;
  step: number;
  world_time: WorldTimeView;
  locations: Record<string, LocationScope>;
  transit: TransitScope;
  world: WorldScope;
  observer: ObserverView;
}

// Relationship graph (GET /api/worlds/{id}/graph). Nodes are agents; edges are
// directed relations (A→B and B→A are separate). Render-neutral: no geometry.
export interface GraphNode {
  id: string;
  name: string;
  role: string;
  // null: not known to this viewer (the actor's view).
  is_main_character: boolean | null;
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
// Distinct from AgentView, which is the mutable per-step state.
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
  // null: not known to this viewer (the actor's view).
  is_main_character: boolean | null;
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
  // The latest persisted step, sent once on connect; then `step` frames as they complete.
  | { type: "latest_step"; data: StepView }
  | { type: "step"; data: StepView }
  // The turn waiting for the actor: the whole turn when one opens, null when it closes.
  | { type: "actor_turn"; data: ActorTurn | null }
  // One new entry of the actor's thread; its history comes from `/actor/thread`.
  | { type: "actor_entry"; data: ActorEntry };

// --- Actor mode (core/interfaces/actor.py; interaction/api/actor.py) ---------------------------
// Field for field with the contracts. An answer is sent back in this same shape: the backend
// refuses unknown keys, so build answers from these types, never by adding fields.

/** A character as anyone could know them: whom the actor may choose to perform. */
export interface CastCandidate {
  agent_id: string;
  name: string;
  gender: string;
  age: number | null;
  role: string;
  background: string;
  appearance: string;
  color: string;
}

// `decide`: the character's turn, whether or not the world changed. `interrupt`: something reached
// it mid-action (what, is in the step's perception). `invite`: someone invites the free character
// into their act (compulsion is never asked). `speak`: the character's dialogue turn in a live conversation.
export type TurnKind = "decide" | "interrupt" | "invite" | "speak";

/** Someone the character can reach: present, or known and alive elsewhere. */
export interface BriefPerson {
  agent_id: string;
  name: string;
  gender: string;
  present: boolean;
}

export interface BriefNpc {
  npc_id: string;
  name: string;
}

export interface BriefPlace {
  location_id: string;
  name: string;
  travel_seconds: number;
}

export interface BriefEntity {
  entity_id: string;
  name: string;
}

export interface BriefMessage {
  sender_name: string; // empty when the message names nobody
  content: string;
  urgency: string;
}

export interface BriefRelation {
  agent_id: string;
  name: string;
  gender: string;
  trust: number;
  affection: number;
  labels: string[];
  summary: string;
  deceased: boolean;
}

/** What a decision can name: the lists an action picks its targets from. */
export interface ActorBriefing {
  people: BriefPerson[]; // everyone it can reach, present or not
  npcs: BriefNpc[];
  places: BriefPlace[];
  entities: BriefEntity[];
}

export interface ActorAction {
  action_type: string;
  action_description: string;
  inner_monologue: string;
  expected_outcome: string;
  estimated_steps: number;
  person_ids: string[];
  message_announce: boolean;
  message_content: string;
  urgency: string;
  destination_id: string | null;
  move_carry_ids: string[];
  physical_person_id: string | null;
  physical_npc_id: string | null;
  physical_entity_id: string | null;
  item_name: string;
  physical_recipient_id: string | null;
  errand_npc_id: string | null;
  errand_destination_id: string | null;
  errand_item_id: string | null;
  errand_recipient_id: string | null;
  errand_announce: boolean;
  errand_message: string;
}

export interface EmotionAmendment {
  emotion: string;
  intensity: number;
  valence: number;
}

/** How the character now regards someone it has a relation with; null keeps that value. */
export interface RelationAmendment {
  agent_id: string;
  trust: number | null;
  affection: number | null;
}

/** The actor's changes to the character's mind; whatever is left out keeps its value.
 *  `insights` are added. */
export interface ActorAmendments {
  emotion: EmotionAmendment | null;
  relations: RelationAmendment[];
  long_term_goals: string[] | null; // the list in full; null keeps it
  short_term_goals: string[] | null;
  insights: string[];
}

/** The character's mind as the actor sees it at any time. `pending`: saved, not landed yet. */
export interface ActorMind {
  emotion: string; // an EmotionType value
  emotion_intensity: number;
  emotion_valence: number;
  long_term_goals: string[];
  short_term_goals: string[];
  relations: BriefRelation[]; // everyone the character has a relation with, latest change first
  pending: ActorAmendments | null;
}

/** `/actor/amend`: `landed` false means it waits for the start of the next step. */
export interface ActorAmendResult {
  accepted: boolean;
  reason: string | null;
  landed: boolean;
}

/**
 * The answer to a DECIDE turn: exactly one of `action` or `idle` (the character does nothing
 * this step). Goals are amended in the mind.
 */
export interface ActorDecision {
  action: ActorAction | null;
  idle: boolean;
}

/** The answer to an INTERRUPT turn. What to do instead is said at the DECIDE turn it leads into. */
export interface ActorInterruptAnswer {
  interrupt: boolean;
}

/** The answer to an INVITE turn: join the inviter's act or not. */
export interface ActorInvitationAnswer {
  accept: boolean;
}

/** The answer to a SPEAK turn: the character's dialogue turn, what it says (`line`) and whether
 *  the conversation ends with it. Ending may leave `line` empty. */
export interface ActorDialogueTurn {
  line: string;
  end: boolean;
}

export type ActorAnswer = ActorDecision | ActorInterruptAnswer | ActorInvitationAnswer | ActorDialogueTurn;

/**
 * When an idle character's actor is asked to decide: every step, or only when something
 * reached it or an act of its own has just ended.
 */
export type ActorPace = "every_step" | "when_reached";

/** Someone inviting the character to join their act, named as the character perceives them. */
export interface Invitation {
  inviter: BriefPerson;
  action_type: string; // the kind of act it is invited into
}

/**
 * A live conversation (spoken turn by turn) the character is in, at the character's dialogue turn:
 * whom with, and what has been said so far. `turns_left` counts the dialogue turns left, this one
 * included; `last`: the conversation ends with this one whatever is said. `max_turn_chars` caps
 * what the actor says in it.
 */
export interface Conversation {
  other: BriefPerson;
  dialogue: DialogueTurn[];
  turns_left: number;
  last: boolean;
  max_turn_chars: number;
}

export interface ActorTurn {
  agent_id: string;
  step: number; // the step being decided: one past the last step on the map
  kind: TurnKind;
  briefing: ActorBriefing | null; // DECIDE only
  doing: string; // INTERRUPT only: the action in progress
  // INVITE only: the invitation, and the act the character decided on this step, which
  // accepting sets aside ("" when none).
  invitation: Invitation | null;
  decided: string;
  conversation: Conversation | null; // SPEAK only
  // Why the character is asked, as a sentence for the actor; empty when nothing brought it on.
  reason: string;
  turn_id: string;
}

/**
 * One entry of the actor's thread: what passed between actor and world, in order. `step` is a
 * code-layer coordinate; never render it as text. `time_label` is the world's own name for that
 * step's moment.
 */
export type ActorEntry =
  // What reached the character: letters and broadcasts as a step began, or someone else's act
  // landing on it (`acted_upon`, as its own memory keeps it).
  | { kind: "perceived"; entry_id: string; step: number; time_label: string; messages: BriefMessage[]; broadcasts: string[]; acted_upon: string[] }
  // The world asks: a turn opened.
  | { kind: "asked"; entry_id: string; step: number; time_label: string; turn: ActorTurn }
  // The actor's words at a DECIDE turn and their reading; `reason` (a refusal slug) when none.
  | { kind: "said"; entry_id: string; step: number; time_label: string; turn_id: string; words: string; reading: ActorDecision | null; reason: string | null }
  | { kind: "answered"; entry_id: string; step: number; time_label: string; turn_id: string; answer: ActorAnswer }
  // The actor asked to break off the act in progress, outside any turn.
  | { kind: "break_off"; entry_id: string; step: number; time_label: string }
  // The character set out on an act, expected to take `total_steps` steps, this one the first.
  | { kind: "began"; entry_id: string; step: number; time_label: string; description: string; total_steps: number }
  // The act in progress went on another step: `elapsed_steps` of `total_steps` are done.
  | { kind: "progressed"; entry_id: string; step: number; time_label: string; elapsed_steps: number; total_steps: number }
  // How it ended, as the character remembers it; `fact` empty when it was never adjudicated,
  // `happened` false when it never engaged the world.
  | { kind: "ended"; entry_id: string; step: number; time_label: string; fact: string; succeeded: boolean; happened: boolean }
  // A dialogue turn of a live conversation, either side's. `end`: the conversation ends with it
  // (by the speaker's choice, or the turns ran out); an ending turn's line may be empty. SPEAK
  // turns leave no `asked` or `answered` entries: these are their record.
  | { kind: "dialogue_turn"; entry_id: string; step: number; time_label: string; dialogue_turn: DialogueTurn; end: boolean };

/**
 * `/actor/say`, `/actor/answer`, `/actor/break-off`: a refusal is a normal answer, with a reason
 * slug. What was taken comes as entries of the thread.
 */
export interface ActorResult {
  accepted: boolean;
  reason: string | null;
}
