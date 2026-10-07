import Phaser from "phaser";

import type {
  ActionSummary,
  AgentStateSummary,
  BroadcastSummary,
  EntityView,
  MapFocus,
  MapSelect,
  NpcStateSummary,
  StepEvent,
} from "../types";
import { THEME } from "../lib/theme";
import { ActionType, type ActionTypeValue, Phase, actedOnAgents } from "../lib/contract";
import { ACTION_IDENTITY } from "../lib/actionIdentity";
import { type BodyHost, BodyRig } from "./body";
import { type EntityHost, EntityLayer } from "./entities";
import { WorldMap } from "./worldMap";
import {
  CUT_ACCENT, FLOAT_NOTE_MS, type HudHost, MapHud, NOTE_CHARS,
} from "./hud";
import { deedBody } from "./deedPose";
import { HIDDEN_ALPHA, HIDDEN_FADE_MS, sortDepth } from "./isoDepth";
import { RECOLOR_FX, RecolorFX } from "./recolorPipeline";
import type { MapSource } from "./mapSource";
import { type CharacterSet, type Demographics, type Dir, type PoseName, bearing, dirOf } from "./skins";
import { PocketLayer } from "./pocket";
import { type AmbientHost, AmbientLight } from "./ambient";
import { type AnnounceHost, Announcer } from "./announce";
import { type CameraHost, MapCamera } from "./camera";
import { type StagingHost, Staging, subjectEntityIds, subjectPeopleIds } from "./staging";
import { type WalkHost, Walker } from "./walking";
import { HoverTip } from "./hoverTip";
import { type PostHost, PostLayer } from "./post";
import { FONT, INITIAL_DIR, PLATE_GAP, PLATE_RISE, SPRITE_FOOT_Y, TEXT_RES, type AgentToken, type CarriedItem } from "./types";
import { reportedDead } from "../lib/agentState";
import { identityColor, toHex } from "../lib/color";
import {
  type FxHost,

  directedMessages,
  faceEntity,
  faceInteraction,
  faceOnlooker,
  stealthVeil,
  footDust,
  localAnnounce,
  physicalEffect,
} from "./effects";

// Everything the scene needs to know about WHICH map it is drawing arrives as
// data (see ./mapSource): the document itself plus where each of its tilesets'
// art is served from. Nothing about a particular map is compiled in here.

const NPC_DEED_INK = 0x8a93a6;

/**
 * Present a mindless body to the figure pipeline as one more figure. It is drawn like a
 * person but is not an agent, so the adaptation happens here rather than by widening
 * agent_states upstream. Liveness fields are pinned: NPCs have no vitality axis, and the
 * death visuals must never claim one.
 */
function npcFigures(npcs: NpcStateSummary[]): AgentStateSummary[] {
  return npcs.map((npc) => ({
    agent_id: npc.npc_id,
    agent_name: npc.name,
    location: npc.location,
    location_id: npc.location_id,
    emotion: "",
    // Pinned to idle: only the character card reads this, and NPCs have no card. The map infers
    // movement from position.
    activity_status: "idle",
    emotion_label: "",
    activity_label: "",
    dominant_need_label: "",
    dominant_need: null,
    is_main_character: false,
    color: npc.color,
    long_term_goals: [],
    short_term_goals: [],
    emotion_intensity: null,
    emotion_valence: null,
    vitality: 1,
    is_active: true,
    condition: npc.condition,
    transit: null,
    arrival: null,
    displaced: npc.displaced,
  }));
}

// The objects a tween config's `targets` names (one or an array).
function tweenTargets(targets: unknown): object[] {
  return (Array.isArray(targets) ? targets : [targets]).filter(
    (t): t is object => typeof t === "object" && t !== null,
  );
}

// Emotion is intentionally NOT rendered on the map: it's interior state that reads
// far better in the narrative feed + the roster panel (which already label it), while
// the map owns body / position / staging / facing / action / life-death.

// Level-of-detail: below this camera zoom, hide per-token labels/pip/vitality bar
// (unless the token is focused) so a zoomed-out map reads as clean colour dots.
const DETAIL_ZOOM = 0.85;
// Opacity of a figure the viewer did NOT single out; applyFocus explains the two values.
const BG_ALPHA = 0.3;
const BG_ALPHA_MARKED = 0.45;
// How far tokens may be enlarged as the camera pulls back — this buys far-view legibility.
// Don't use the cast's `scale` for that: it owns a person's size against the architecture,
// and inflating it makes a figure as tall as the house he stands in.
const MAX_LOD_SCALE = 3.5;
// Phaser cache key for the map document (the scene only ever holds one).
const MAP_KEY = "worldmap";
// Character art is served per world, like the tilesets: the cast is authored in the template
// beside the map and arrives as a CharacterSet on MapSource. Skin assignment lives in skins.ts.
// Shorter than this, a relocation is a man stepping aside in his own room as Staging.layoutRooms
// re-seats everyone, and an A* route would only repeat the straight line. About two tiles.
const WALK_ROUTE_MIN_PX = 70;
// Below this ground speed (at 1× playback) a figure is standing. Well under the slowest real
// walk, a short reposition inside a room (tens of pixels in half a second).
const WALK_MIN_PX_PER_SEC = 20;
// End-of-step tail: the driver advances the instant renderStep resolves, so a chip set as the
// last beat ends would be wiped by the next clearEphemeral unread. Held after the last beat.
const CHIP_TAIL_MS = 1400;
// A step with weather stays on screen at least this long. A floor on the whole step, not a
// tail: it only pays out when nothing else showed (e.g. every action focus-gated away).
const WEATHER_MIN_MS = 1500;
// How long the opening tableau (step 0) is held. It has no beats (see the initialization
// filter in playTrack), so this is its whole screen time.
const SETTLE_MS = 1800;

// No tile-name lookup tables here (name → colour, name → height): they would put the map's art
// in code, so every new map would need a code change. Tilesets are blitted instead.

/**
 * Renders the world's native Tiled map (the same .tmj the engine built on) and overlays
 * figures driven by the step stream. Location regions come from the tmj's "地点" object layer.
 *
 * The host contracts are declared, not satisfied by accident, so deleting a member a
 * collaborator needs fails here on the class rather than at a `new` call site.
 */
export class TiledWorldScene
  extends Phaser.Scene
  implements FxHost, BodyHost, HudHost, EntityHost, AmbientHost, AnnounceHost, PostHost, CameraHost,
    StagingHost, WalkHost
{
  private source: MapSource;
  private onReady?: () => void;
  private onSelect?: (t: MapSelect) => void;
  // Focus/selection state (mirrors React's MapFocus). Drives emphasis/dim, the
  // spotlight overlay, and camera-follow. Empty = no focus → everything normal.
  private focus: MapFocus = { agents: [], place: null, follow: false };
  private spotlight?: Phaser.GameObjects.Rectangle; // dim overlay for a location spotlight
  // agent_id → gender/age (from the step-0 profiles) — the only thing that picks a body.
  private demographics: Record<string, Demographics> = {};
  private tokens = new Map<string, AgentToken>();
  private placed = new Set<string>(); // agents already positioned (no fly-in from 0,0)
  // The step the last render drew. Only the step right after it is walked to; any other (a seek,
  // the first render) is a cut: the walk between two distant steps never happened.
  private drawnStep: number | null = null;
  // A step renders as an awaitable timeline; the driver waits for it before the
  // next step. renderGen supersedes an in-flight render on a deliberate scrub;
  // cancelers force-resolve its pending awaits so nothing hangs.
  renderGen = 0;
  private cancelers = new Set<() => void>();
  // Tween target -> the settles awaiting its tweens (see FxHost.awaitTweens).
  private awaiting = new Map<object, Set<() => void>>();
  // The render currently drawing, or a settled promise. renderStep awaits it so exactly one
  // render is alive at a time — see the barrier there.
  private inFlight: Promise<void> = Promise.resolve();
  // True while a superseded render winds down. Cancellation must be a state, not just the
  // one-shot `cancelPending()`: a dying render can open new waits after it ran (a hold, a fade)
  // and every scrub would pay for them at the barrier. While set, the scene grants no waits, so
  // the dying render exits in microtasks with no frame drawn.
  unwinding = false;
  private ephemeral: Phaser.GameObjects.GameObject[] = [];
  private timers: Phaser.Time.TimerEvent[] = [];

  /** Track a timer for step cleanup, the way pushEphemeral tracks an object (HudHost). */
  pushTimer(t: Phaser.Time.TimerEvent): void {
    this.timers.push(t);
  }

  /** HudHost: the HUD's one reach into the body — a spoken line poses the speaker. */
  holdPose(tok: AgentToken, pose: PoseName, sustained = false): void {
    this.body.holdPose(tok, pose, sustained);
  }
  private ready = false;
  private speedScale = 1; // >1 faster, <1 slower

  /** WeatherHost: the animation-speed multiplier. */
  get speed(): number {
    return this.speedScale;
  }

  /**
   * Set the animation-speed multiplier. Durations everywhere are written at 1×; the scene's tween
   * and timer clocks carry the multiplier, so a change also reaches tweens and waits already
   * running — scaling durations at creation would leave a walk under way at the old pace, legs
   * and ground speed out of step. Sprite cycles have no scene clock: `anims.timeScale` is
   * per-sprite, so ensureToken applies it to new tokens too.
   */
  setSpeed(mult: number): void {
    this.speedScale = Math.max(0.1, Math.min(4, mult));
    this.tweens.timeScale = this.speedScale;
    this.time.timeScale = this.speedScale;
    for (const tok of this.tokens.values()) tok.sprite.anims.timeScale = this.speedScale;
  }


  /**
   * Feed the cast's gender/age (from the step-0 profiles) so each figure gets the right body;
   * re-skins tokens created before it landed. Colour is not part of this — it comes from each
   * step's state (see ensureToken → RecolorFX).
   */
  setCast(cast: Record<string, Demographics>): void {
    for (const [id, who] of Object.entries(cast)) this.learnBody(id, who);
  }

  /**
   * Register one figure's (gender, age) and re-skin it if it is already standing.
   *
   * The only writer of `demographics`: the cast (once per world) and the NPCs (every step) are
   * disjoint halves, so neither may replace the whole record — the late fetch would drop every
   * NPC back to the default body.
   */
  private learnBody(id: string, who: Demographics): void {
    const known = this.demographics[id];
    if (known && known.gender === who.gender && known.age === who.age) return;
    this.demographics[id] = who;
    const tok = this.tokens.get(id);
    if (tok) this.body.assignSkin(tok, this.skinFor(id));
  }

  /** The rig that draws every figure — poses, headings, dye, falling. See body.ts. */
  private readonly body = new BodyRig(this);

  /** Every readable surface laid over the world — bubbles, chips, notes. See hud.ts. */
  private readonly hud = new MapHud(this);

  /** The map itself — projection, grids, rooms, terrain. See worldMap.ts. */
  readonly map: WorldMap;

  /** Everything in the world that is not a person — and who is holding it. See entities.ts. */
  private readonly entities = new EntityLayer(this);
  // What a person IS between beats — the condition on him, the things he carries, and the
  // one pocket panel that may be open on the map. See pocket.ts.
  private readonly pockets = new PocketLayer(this);
  /** Night over the ground, and lamps under the places people are in. See ambient.ts. */
  private readonly ambient = new AmbientLight(this);
  /** The hover card over a body with no mind, or over a thing. See hoverTip.ts. */
  private readonly hoverTip = new HoverTip(this);
  /** A broadcast's words and its weather. See announce.ts. */
  private readonly announcer = new Announcer(this, this.hud);
  /** A step's post, reaching its readers. See post.ts. */
  private readonly post = new PostLayer(this, this.hud, this.pockets, this.announcer);
  /** Where everyone rests this step, and the ground each thing takes. See staging.ts. */
  private readonly staging = new Staging(this, this.entities);
  /** Routes and walking. See walking.ts. */
  private readonly walker = new Walker(this, this.placed);
  /** Zoom, pan and follow — the viewer's eye on the map. See camera.ts. */
  private readonly camera: MapCamera;

  /** FxHost + HudHost: the markers, and the carry ledger the nameplates read. */
  get entityMarkers(): Map<string, Phaser.GameObjects.Container> {
    return this.entities.entityMarkers;
  }

  get carriedBy(): Map<string, CarriedItem[]> {
    return this.entities.carriedBy;
  }

  /** EntityHost: announce a change where it happened. */
  floatNote(x: number, y: number, text: string, accent: number): void {
    this.hud.floatNote(x, y, text, accent);
  }

  showEntityTip(x: number, y: number, entity: EntityView): void {
    this.hoverTip.showEntity(x, y, entity);
  }

  hideTip(): void {
    this.hoverTip.hide();
  }

  /** EntityHost: what people carry changed — every handle, and the open pocket, is stale. */
  refreshCarried(): void {
    this.pockets.refresh(this.tokens);
  }

  /** HudHost: a bubble is clamped inside the WORLD, not the viewport. */
  get worldW(): number {
    return this.map.worldW;
  }

  get worldH(): number {
    return this.map.worldH;
  }

  /** Display name for a location id (for the focus chips), or a description if unknown. */
  locationName(id: string): string {
    return this.map.locationName(id);
  }

  private skinFor(agentId: string): string {
    return this.cast.bodyFor(this.demographics[agentId]);
  }

  /**
   * Register the bodies that think for nobody. Their (gender, age) arrives on every step
   * rather than once per world — they can appear mid-run — so this is idempotent and cheap,
   * and it re-skins a token created before its demographics landed.
   */
  private noteNpcs(npcs: NpcStateSummary[]): void {
    this.npcById = new Map(npcs.map((npc) => [npc.npc_id, npc]));
    for (const npc of npcs) this.learnBody(npc.npc_id, { gender: npc.gender, age: npc.age });
  }

  /**
   * What happened to a mindless body this step, in one float over his head, so a figure never
   * crosses the city unexplained.
   *
   * Only lines with a result get a float: an `ongoing` line ("on his way to…") stays in the
   * feed. The float lands over him (not the room, which holds several) because the backend
   * places a body's line where he ends the step (see `NpcStateSummary.outcome`). Called when
   * his own track's walks land him, not once per step.
   */
  private noteNpcOutcomes(agentIds: string[]): void {
    for (const id of agentIds) {
      const npc = this.npcById.get(id);
      if (!npc?.outcome || npc.ongoing) continue;
      const tok = this.tokens.get(id);
      if (!tok) continue;
      this.floatNote(tok.container.x, tok.container.y - 34, npc.outcome, NPC_DEED_INK);
    }
  }

  /** This step's bodies-without-minds, by id — what the errand mark and the hover read. */
  private npcById = new Map<string, NpcStateSummary>();

  /** This world's character set — the manifest its own map shipped with. */
  get cast(): CharacterSet {
    return this.source.characters;
  }

  /** Re-fit the camera to the whole map, discarding the user's zoom/pan. */
  resetView(): void {
    this.camera.reset();
  }

  /** Zoom by `factor` about the centre of the view, within the same limits as the wheel. */
  zoomBy(factor: number): void {
    this.camera.zoomBy(factor);
  }

  // Keep every token's depth in sync with its current screen y each frame, so agents
  // interleave correctly even MID-TRANSIT (a token further "south" draws over one
  // behind it) — and fade the ones the map's art is standing in front of. See isoDepth
  // for both, and for why the art itself is never touched.
  update(_time: number, delta: number): void {
    if (!this.ready) return;
    const zoom = this.cameras.main.zoom;
    // Seconds since the last frame — see the movement test below, which is a SPEED and
    // therefore needs the clock, not a frame count. Floored: a stalled tab can hand back
    // a delta of 0 and would divide the speed to infinity.
    const dt = Math.max(delta, 1) / 1000;
    // LOD: enlarge tokens when zoomed OUT so they stay visible (no change when zoomed
    // in, so child offsets stay exact); show per-token detail only when zoomed in
    // enough OR focused, so a fit-view reads as clean colour dots.
    const lodScale = zoom < 1 ? Math.min(1 / zoom, MAX_LOD_SCALE) : 1;
    const detailZoom = zoom >= DETAIL_ZOOM;
    const focusOn = this.focusActive();
    const t = this.time.now;
    const geometry = this.map.geometry;
    // How far this frame carries a fade, as a fraction of what is left to travel.
    // Framed in TIME, not frames, so the fade lasts as long on a 30fps tab as on 120.
    const fadeStep = 1 - Math.exp((-dt * 3000) / HIDDEN_FADE_MS);
    for (const [id, tok] of this.tokens) {
      // A figure on ground the map's art hides fades — the only occlusion handling. The test is
      // the backend's `standable` grid, the same one A* tolls (HIDDEN_STEP_TOLL) and findStanding
      // refuses: one truth, no second opinion about the art.
      const want = tok.baseAlpha * (this.map.hiddenGround(tok.container.x, tok.container.y) ? HIDDEN_ALPHA : 1);
      tok.container
        .setDepth(sortDepth(tok.container.y, geometry))
        .setAlpha(tok.container.alpha + (want - tok.container.alpha) * fadeStep)
        .setScale(lodScale);
      // Idle bob on the sprite only (the container anchors bubbles/effects); the dead lie still.
      // Skipped while posed: writing sprite.y every frame would erase the vertical half of every
      // lunge (effects.ts), throwing blows sideways on an isometric map.
      if (!tok.posed) {
        tok.sprite.y = 6 - (tok.dead ? 0 : Math.abs(Math.sin(t / 620 + tok.phase)) * 1.6);
      }
      // Walk anim and heading come from real movement (transit or any reposition tween), so
      // there is no per-path bookkeeping.
      const dx = tok.container.x - tok.px;
      const dy = tok.container.y - tok.py;
      // Walking is a speed test, not a per-frame distance: a per-frame threshold depends on frame
      // rate and playback speed, and slow motion would snap a walking figure to idle so he slides.
      // The bar scales with speedScale because slow motion slows ground speed too.
      const speed = Math.hypot(dx, dy) / dt;
      const moving = !Number.isNaN(tok.px) && speed > WALK_MIN_PX_PER_SEC * this.speedScale;
      tok.px = tok.container.x;
      tok.py = tok.container.y;
      // A held pose gives way to walking. `posed` only protects a figure standing still; one
      // that has left its slot would otherwise carry a sustained pose (no timer ends it) across
      // the map.
      if (!tok.dead && tok.posed && moving) this.body.restorePose(tok);
      if (!tok.dead && !tok.posed) {
        if (moving) {
          // Both screen axes count: the vertical one tells walking away from walking toward
          // the camera, so a figure heading north is seen from behind.
          tok.dir = dirOf(dx, dy, tok.dir);
          this.body.playWalk(tok);
        } else if (tok.sprite.anims.isPlaying) {
          tok.sprite.stop();
          this.body.applyPose(tok, "idle");
        }
      }
      const foc = focusOn ? this.isFocused(id) : false;
      const detail = detailZoom || foc;
      // The nameplate shows as one unit at detail zoom or when focused. A focus does not hide
      // other plates: at background alpha the name is the last way to tell who the crowd is.
      // Only the pouch handle (see sync) and the timed beats (see playTrack) are subject-only.
      tok.plate.setVisible(detail);
      // The open pocket hangs off the nameplate and shares its gate. The head mark does not: it
      // must survive a pulled-back camera, which is why it is a mark rather than text.
      tok.pocket?.setVisible(detail);
      // A slow, shallow breath on the mark so a standing condition reads as ongoing, without
      // competing with this step's beats.
      this.pockets.pulse(tok, t);
      // The action chip shares the nameplate's gate, and stays on a token that acted then died
      // this step so the final deed is legible.
      tok.actionChip?.setVisible((!focusOn || foc) && detail);
      tok.vitTrack.setVisible(tok.hurt);
      tok.vitFill.setVisible(tok.hurt);
    }
    // Things on the ground depth-sort on the same axis as people, not at a flat depth.
    for (const marker of this.entityMarkers.values()) {
      marker.setDepth(sortDepth(marker.y, geometry));
    }
    this.announcer.pinBanner();
    this.camera.fly(this.focus, this.focusActive(), dt * this.speedScale);
    this.ambient.follow();
  }

  // --- FxHost surface ---------------------------------------------------------
  pushEphemeral(obj: Phaser.GameObjects.GameObject): void {
    this.ephemeral.push(obj);
  }

  addCanceler(fn: () => void): void {
    this.cancelers.add(fn);
  }

  removeCanceler(fn: () => void): void {
    this.cancelers.delete(fn);
  }

  killTweens(targets: unknown): void {
    this.tweens.killTweensOf(targets as object);
    for (const t of tweenTargets(targets)) {
      for (const settle of [...(this.awaiting.get(t) ?? [])]) settle();
    }
  }

  awaitTweens(targets: unknown, settle: () => void): () => void {
    const list = tweenTargets(targets);
    let done = false;
    const once = () => {
      if (done) return;
      done = true;
      for (const t of list) {
        const waiting = this.awaiting.get(t);
        waiting?.delete(once);
        if (waiting?.size === 0) this.awaiting.delete(t);
      }
      settle();
    };
    for (const t of list) {
      const waiting = this.awaiting.get(t) ?? new Set<() => void>();
      waiting.add(once);
      this.awaiting.set(t, waiting);
    }
    return once;
  }

  token(id: string): AgentToken | undefined {
    return this.tokens.get(id);
  }

  /**
   * Turn to what the beat is about — the one place that decides a heading from an action.
   *
   * Heading sources rank: 1. travel (update() re-faces a moving figure every frame);
   * 2. the subject of the deed (here, for a figure standing still); 3. the previous heading
   * (hysteresis in dirOf). Within 2, a person outranks a thing — the thing is the instrument —
   * and the thing is used when the person is off the map. Kept as one function so no beat
   * can act without facing its subject.
   *
   * The dialogue loop does not come here: it turns the speaker to each utterance's listeners.
   */
  private faceSubject(tok: AgentToken, act: ActionSummary): void {
    if (faceInteraction(this, tok, subjectPeopleIds(act))) return;
    faceEntity(this, tok, subjectEntityIds(act));
  }

  /**
   * FxHost: which drawn direction points at a screen offset. Lives here because inverting the
   * isometric projection needs the map's tile shape; the rule is skins.ts `bearing`.
   */
  bearing(dx: number, dy: number, previous: Dir): Dir {
    return bearing(dx, dy, this.map.tileW, this.map.tileH, previous);
  }

  /**
   * Redraw a standing figure on its current heading (see effects.ts `turnToward`). A walking
   * or posed figure is left alone: the walk cycle re-aims itself, and a pose is held on purpose.
   */
  reface(tok: AgentToken): void {
    if (tok.dead || tok.posed || tok.sprite.anims.isPlaying) return;
    this.body.applyPose(tok, "idle");
  }

  // --- awaitable, cancellable animation primitives ---------------------------
  delayP(ms: number): Promise<void> {
    if (this.unwinding) return Promise.resolve(); // winding down: nothing may take time
    return new Promise((resolve) => {
      let timer: Phaser.Time.TimerEvent;
      const cancel = () => {
        timer?.remove(false);
        resolve();
      };
      this.cancelers.add(cancel);
      timer = this.time.delayedCall(Math.max(1, ms), () => {
        this.cancelers.delete(cancel);
        resolve();
      });
      this.timers.push(timer);
    });
  }

  tweenP(config: Phaser.Types.Tweens.TweenBuilderConfig): Promise<void> {
    if (this.unwinding) return Promise.resolve(); // …and nothing may be drawn, either
    return new Promise((resolve) => {
      const settle = this.awaitTweens(config.targets, () => {
        this.cancelers.delete(cancel);
        resolve();
      });
      const cancel = () => this.killTweens(config.targets);
      this.cancelers.add(cancel);
      // onStop for a tween stopped with stop(); a kill settles through awaitTweens instead.
      this.tweens.add({ ...config, onComplete: settle, onStop: settle });
    });
  }

  // Resolve every pending await (a superseding render calls this so the old
  // render's timeline unblocks; its gen guard then bails without drawing more).
  private cancelPending(): void {
    const pending = [...this.cancelers];
    this.cancelers.clear();
    pending.forEach((c) => c());
  }

  constructor(
    source: MapSource,
    onReady?: () => void,
    onSelect?: (t: MapSelect) => void,
    onGrabCamera?: () => void,
  ) {
    super("tiled-world");
    this.source = source;
    this.map = new WorldMap(this, source);
    this.onReady = onReady;
    this.onSelect = onSelect;
    this.camera = new MapCamera(this, onGrabCamera);
  }

  preload(): void {
    for (const ts of this.source.tilesets) {
      this.load.image(ts.key, ts.url);
    }
    for (const { key, image, atlas } of this.cast.atlases) {
      this.load.atlasXML(key, image, atlas);
    }
  }

  create(): void {
    // The document was already fetched (to resolve its asset URLs), so hand it
    // straight to Phaser instead of making it fetch the same map a second time.
    this.cache.tilemap.add(MAP_KEY, {
      format: Phaser.Tilemaps.Formats.TILED_JSON,
      data: this.source.doc,
    });
    const map = this.make.tilemap({ key: MAP_KEY });
    // Register the garment recolour shader before any token asks for it. WebGL only; under
    // canvas, figures keep the art's own colours.
    const pipelines = (this.renderer as Phaser.Renderer.WebGL.WebGLRenderer).pipelines;
    if (pipelines && !pipelines.getPostPipeline(RECOLOR_FX)) {
      pipelines.addPostPipeline(RECOLOR_FX, RecolorFX);
    }
    this.body.buildCharacterAnims();
    this.map.build(map);
    this.walker.mount();
    this.map.drawLabels();
    this.ambient.mount();
    this.camera.mount(() => this.hoverTip.hide(), (p) => this.handleClick(p));
    // `pointerout` does not fire when the pointer leaves the canvas entirely, so the hover card
    // would stay up indefinitely; `gameout` covers that. (A press hides it too, via camera.mount.)
    this.input.on("gameout", () => this.hoverTip.hide());
    this.ready = true;
    this.onReady?.();
  }

  /** Day/night colour wash from the world's hour — see AmbientLight.setHour. */
  setWorldTime(hour: number | null | undefined): void {
    this.ambient.setHour(hour);
  }

  // Resolve a click into a selection: an agent token, a location region, or
  // empty → clear. Entity markers are info-only — clicking one does not change
  // the MapFocus (WorldView ignores the entity variant; MapSelect has none).
  private handleClick(p: Phaser.Input.Pointer): void {
    if (!this.onSelect) return;
    const hits = this.input.hitTestPointer(p) as Phaser.GameObjects.GameObject[];
    // The pocket is checked before the man: both live in the token container, so a single pass
    // in hit order could select the owner on a click meant for the handle.
    for (const obj of hits) {
      const owner = obj.getData?.("pocketOf");
      if (owner) return this.pockets.toggle(String(owner));
      if (obj.getData?.("pocketPanel")) return; // reading is not selecting
    }
    for (const obj of hits) {
      const agentId = obj.getData?.("agentId");
      if (agentId) return this.onSelect({ kind: "agent", id: String(agentId) });
    }
    const world = this.cameras.main.getWorldPoint(p.x, p.y);
    const room = this.map.roomAt(world.x, world.y);
    this.onSelect(room ? { kind: "location", id: room.id } : { kind: "clear" });
  }

  // --- focus / selection -----------------------------------------------------
  /** Push the current focus state in (from React); re-applies emphasis + follow. */
  setFocus(focus: MapFocus): void {
    this.focus = focus;
    if (!this.ready) return;
    this.applyFocus();
    this.camera.followFocus(this.focus, this.focusActive());
  }

  // Public for the pocket layer: "not the subject" differs from "no focus at all".
  focusActive(): boolean {
    return this.focus.agents.length > 0 || this.focus.place !== null;
  }

  // A focus subject = directly selected, or standing in the spotlighted location.
  isFocused(agentId: string): boolean {
    if (this.focus.agents.includes(agentId)) return true;
    if (this.focus.place) {
      const tok = this.tokens.get(agentId);
      return !!tok && tok.locationId === this.focus.place.id;
    }
    return false;
  }

  // Emphasis/dim pass: focused subjects get full opacity and the focus ring, the rest are
  // dimmed. A location focus also dims everything outside it.
  private applyFocus(): void {
    const active = this.focusActive();
    // A panel belonging to somebody the viewer just stopped watching stands down.
    this.pockets.onFocusChanged(active);
    if (this.focus.place) {
      if (!this.spotlight) {
        this.spotlight = this.add
          .rectangle(0, 0, this.map.worldW, this.map.worldH, 0x0a0812, 0.55)
          .setOrigin(0, 0)
          .setDepth(6);
      }
      this.spotlight.setVisible(true);
    } else {
      this.spotlight?.setVisible(false);
    }
    for (const [id, tok] of this.tokens) {
      if (!active) {
        tok.baseAlpha = tok.dead ? 0.5 : 1;
        tok.focusRing.setVisible(false);
        this.hud.renderChip(tok); // focus cleared → every chip back to compact
        this.pockets.sync(id, tok); // …and every condition back to its bare mark
        continue;
      }
      const on = this.isFocused(id);
      tok.focusRing.setVisible(on);
      // A figure under a standing condition keeps a higher floor: the mark is a child of this
      // container, so the dim multiplies into it and at BG_ALPHA it would be a smudge.
      tok.baseAlpha = on ? 1 : tok.condText ? BG_ALPHA_MARKED : BG_ALPHA;
      this.hud.renderChip(tok); // focused subject → full text, others → compact (no-op if no chip)
      this.pockets.sync(id, tok); // words for the subject, the bare mark for everyone else
    }
  }

  /** EntityHost: deal this step's things onto the ground their rooms can spare. */
  layoutFixtures(entities: Record<string, EntityView>): void {
    this.staging.layoutFixtures(entities);
  }

  private ensureToken(state: AgentStateSummary): AgentToken {
    let tok = this.tokens.get(state.agent_id);
    if (tok) return tok;
    // The fixed identity colour (from the backend) dyes garment, name tag and ground ring.
    const color = identityColor(state.color, state.agent_id);
    // Ground decals at the feet. Each question gets its own visual property:
    //   who is this        → ring colour = identity colour; the one identifier that survives
    //                        zooming out, matching the relation graph, card and feed.
    //   main or background → ring form (heavier stroke + outer ring). Never by hue: a rank
    //                        colour collides with identity colours (e.g. a gold agent).
    //   am I selected      → the purple focus ring, outside the identity palette, outermost.
    const shadow = this.add.ellipse(0, 6, 22, 9, 0x000000, 0.3);
    const focusRing = this.add.ellipse(0, 6, 36, 17).setStrokeStyle(3, THEME.toInt(THEME.accentSoft)).setVisible(false);
    const parts: Phaser.GameObjects.GameObject[] = [shadow, focusRing];
    // Narrative tier on one channel: double ring for a main character, plain for the cast,
    // thin and faint for an NPC.
    const mindless = this.npcById.has(state.agent_id);
    parts.push(this.add
      .ellipse(0, 6, 26, 12)
      .setStrokeStyle(state.is_main_character ? 3 : mindless ? 1 : 2, color, mindless ? 0.55 : 1));
    if (state.is_main_character) {
      parts.push(this.add.ellipse(0, 6, 31, 14.5).setStrokeStyle(1.5, color, 0.7));
    }
    // The body its gender+age picks, feet at the container origin; the garment is dyed with
    // the identity colour by RecolorFX. Skin is never touched (see recolorPipeline.ts).
    const skinId = this.skinFor(state.agent_id);
    // Faces the camera until the figure first travels.
    const idle = this.cast.poseArt(skinId, "idle", INITIAL_DIR);
    const sprite = this.add
      .sprite(0, SPRITE_FOOT_Y, idle?.frames[0][0] ?? skinId, idle?.frames[0][1])
      .setOrigin(0.5, 1)
      .setScale(this.cast.scale)
      .setFlipX(idle?.flip ?? false);
    // setSpeed only reaches existing tokens, so a new one is born at the current speed.
    sprite.anims.timeScale = this.speedScale;
    this.body.dyeGarment(sprite, color);
    parts.push(sprite);
    const label = this.add
      .text(0, 0, state.agent_name, {
        fontFamily: FONT,
        fontSize: "10px",
        color: toHex(color),
      })
      .setOrigin(0.5)
      .setResolution(TEXT_RES)
      .setStroke("#0a0e1a", 3);
    const barW = Math.max(14, label.width);
    const vitTrack = this.add.rectangle(0, -8, barW, 2.5, 0x0a0e1a, 0.75).setVisible(false);
    const vitFill = this.add
      .rectangle(-barW / 2, -8, barW, 2, 0x6ab04c)
      .setOrigin(0, 0.5)
      .setVisible(false);
    // Condition mark and pouch are built by pocket.ts and placed here, where the plate's
    // anatomy is known: the mark rides the body (it must outlive the plate's LOD gate).
    const { pip, pouch } = this.pockets.mount(
      state.agent_id, this.body.measureHeadY(skinId), label.width / 2,
    );
    parts.push(pip);
    // Nameplate above the head: health when hurt, then the name with its pouch handle. One
    // line of type, never prose, so a crowded room stays readable.
    const plate = this.add.container(
      0, this.body.measureHeadY(skinId) - PLATE_GAP, [vitTrack, vitFill, label, pouch],
    );
    parts.push(plate);
    const container = this.add.container(0, 0, parts).setDepth(20);
    // Hit area is a circle over the figure, independent of the container's child bounds.
    container.setInteractive(new Phaser.Geom.Circle(0, -16, 22), Phaser.Geom.Circle.Contains);
    container.setData("agentId", state.agent_id);
    // The hover tip is for NPCs only (agents have a character card). Check `npcById` on every
    // hover: a tier cached on the token goes stale when an id changes identity (restore, world
    // switch).
    container.on("pointerover", () => {
      const npc = this.npcById.get(state.agent_id);
      const tok = this.tokens.get(state.agent_id);
      if (npc && tok) this.hoverTip.show(tok, npc);
    });
    container.on("pointerout", () => this.hoverTip.hide());
    tok = {
      container, sprite, skinId, posed: false, headY: this.body.measureHeadY(skinId),
      plate, focusRing, label, color, baseAlpha: 1, floor: 0, vitTrack, vitFill,
      condPip: pip, pouch, condText: "",
      locationId: state.location_id, dead: false, hurt: false, vit: 1, phase: Math.random() * Math.PI * 2,
      dir: INITIAL_DIR,
      px: NaN, py: NaN,
    };
    this.tokens.set(state.agent_id, tok);
    return tok;
  }

  /**
   * Render a step as an awaitable timeline that resolves when everything it shows has finished;
   * the driver awaits it before the next step.
   *
   * A scrub replaces the render in flight: the old one is told to stop and unwind before the
   * new one begins, so at most one render is ever alive. Correctness rests on that barrier;
   * the `renderGen !== gen` checks only make the unwind prompt (without the barrier, one missed
   * check lets a stale render kill the new one's tweens and hang the player).
   *
   * Takes the whole `StepEvent` so how its channels degrade is decided here, not per caller.
   */
  async renderStep(step: StepEvent): Promise<void> {
    if (!this.ready) return;
    const gen = ++this.renderGen;
    this.unwinding = true; // no beat may take time until the old render is out
    this.cancelPending(); // wake the render in flight so it can see it is stale
    await this.inFlight; // …and let it actually unwind before anything is touched
    // Scrubbed again while we waited: that newer call owns the map now, not this one — and
    // owns `unwinding` with it, so leave the flag alone. Whichever call gets to draw is the
    // one that clears it, and exactly one of them does.
    if (this.renderGen !== gen) return;
    this.unwinding = false;
    const running = this.draw(step, gen);
    // The barrier answers "is the previous render finished", not "did it succeed" — so it
    // never rejects. A thrown render must not poison every render after it; the caller
    // still sees the failure through `running`.
    this.inFlight = running.catch(() => {});
    await running;
  }

  private async draw(step: StepEvent, gen: number): Promise<void> {
    const cut = this.drawnStep === null || step.step !== this.drawnStep + 1;
    this.drawnStep = step.step;
    this.noteNpcs(step.npcs ?? []);
    this.staging.setRelations(step.relations ?? []);
    const states = [...Object.values(step.agent_states), ...npcFigures(step.npcs ?? [])];
    const actions = step.actions;
    const entities = step.entities ?? {};
    // `step.world_events` is deliberately not read: a WorldEvent is the editor's summary of a
    // broadcast or message this same step already carries, with no place or cast by contract.
    // Drawing it would show one happening twice. The map draws the channels instead.
    const messages = step.messages ?? [];
    const broadcasts = step.broadcasts ?? [];
    // The hover card is not ephemeral but is pinned to a token about to move, so hide it here.
    this.hoverTip.hide();
    // Safe only because the previous render has unwound (see the barrier in renderStep): timers
    // are removed with `remove(false)`, which skips their callback and would hang a waiting delayP.
    this.clearEphemeral();
    // The previous weather is retired, not deleted: it finishes falling while this step runs,
    // since a shower that ends between two frames never looks like weather ending.
    this.announcer.retireWeather();
    // A sustained pose (see holdPose) outlives its step and ends when the figure has nothing
    // to do; whoever will pose again this step keeps it until playDeed replaces it, so a long
    // job does not flicker to idle. The test is "will he be given a pose" (ask deedBody), not
    // "does he have an action": MOVE and REST have no pose, so the latter would keep it forever.
    this.body.releaseIdlePoses(
      this.tokens,
      new Set(
        actions
          .filter((a) => !a.not_executed && a.phase !== Phase.interrupt)
          .filter((a) => deedBody(a.deed, a.succeeded !== false) !== null)
          .map((a) => a.agent_id),
      ),
    );
    // Withhold every entity change a deed this step will cause, so the map shows cause before
    // effect: a step reports the world after the step, so applying it up front opens the door
    // before the hand touches it. The test is general — it changed visibly and an action names it
    // in affected_entity_ids — not a `destroyed` special case.
    //
    // Withheld means the previous snapshot keeps drawing, so layout and marker code need not know.
    // playTrack reveals each as its beat ends; the safety net below catches the rest. Computed
    // before layout so a withheld item still on the ground gets a cell from layoutFixtures.
    const causedEntityIds = new Set<string>();
    for (const a of actions) for (const eid of a.affected_entity_ids ?? []) causedEntityIds.add(eid);
    const shownEntities = this.entities.beginStep(entities, causedEntityIds);

    // Clear corpses shown dead in a prior step (a backward scrub re-creates them below). Runs
    // before layout so "has a token" means "has a body on this map" for the rest of the render
    // — see gone().
    for (const [id, tok] of [...this.tokens]) {
      if (!tok.dead) continue;
      this.body.removeToken(tok);
      this.tokens.delete(id);
      this.placed.delete(id);
    }

    // Resting slots, relationship- and interaction-aware: figures about to deal with each other
    // stand together, and playTrack awaits the walk before any deed. Lamps are lit from the
    // layout's own per-room count so the two never disagree.
    this.ambient.lightRooms(this.staging.layoutRooms(states, actions, shownEntities));

    // Withhold the reported body (vitality and death together, see applyBodyState) of anyone a
    // deed touches, by the same rule as entities. Visible cause: he acts, or a physical blow
    // targets him (victims have no action of their own). Without a cause the change lands now.
    const agentIdsWithActions = new Set(actions.map((a) => a.agent_id));
    const physicalTargetIds = new Set<string>();
    for (const a of actions) {
      if (a.action_type === ActionType.physical) {
        for (const t of actedOnAgents(a.target)) physicalTargetIds.add(t);
      }
    }
    // Reads the previous step's token (ensureToken runs below); an unplaced body simply appears.
    const withheldAgents = new Set<string>();
    states.forEach((state) => {
      const tok = this.tokens.get(state.agent_id);
      if (!tok || !this.placed.has(state.agent_id)) return;
      const hasVisibleCause = agentIdsWithActions.has(state.agent_id) || physicalTargetIds.has(state.agent_id);
      if (hasVisibleCause && this.body.bodyChanged(tok, state)) withheldAgents.add(state.agent_id);
    });

    // Instant state (no animation): body, chip slots, entity markers.
    const stateById = new Map<string, AgentStateSummary>();
    states.forEach((state) => {
      // Creating a token here would resurrect a corpse every other step. See gone().
      if (this.gone(state)) return;
      const tok = this.ensureToken(state);
      tok.locationId = state.location_id;
      // A standing condition is state, not a beat: written every step from the report, never
      // withheld, so it also vanishes when it wears off unannounced.
      this.pockets.sync(state.agent_id, tok, state.condition ?? "");
      // A dead man's pocket closes: an open panel over a corpse reads as a live control.
      if (reportedDead(state)) this.pockets.close(state.agent_id);
      // Withheld → leave the body exactly as it stands; playTrack lands it after the beat.
      if (!withheldAgents.has(state.agent_id)) this.body.applyBodyState(tok, state, this.placed.has(state.agent_id));
      stateById.set(state.agent_id, state);
    });
    this.staging.layoutLabels(states);
    this.entities.applyEntities(shownEntities);
    // Runs after layoutLabels so focus wins on label visibility/alpha.
    this.applyFocus();
    this.camera.followFocus(this.focus, this.focusActive());

    // Weather first, lasting through the step: a broadcast is injected at the step head (at
    // `poll_event`, before cognition) and the cast perceived it this step, so people act in the
    // rain. Played here, not with the announcements, so it gets the full step weather.ts promises.
    // `depicted`: broadcasts the map has drawn a picture of.
    const depicted = new Set<BroadcastSummary>();
    for (const b of broadcasts) {
      if (this.announcer.playPhenomenon(b, this.map.rooms.get(String(b.location_scope ?? "")))) depicted.add(b);
    }
    const weatherAt = depicted.size > 0 ? this.time.now : null;

    // What reached them plays before what they did: the engine perceives before it decides, so
    // anything delivered on step N is an input to step N's decisions. Nothing delivered can
    // narrate a deed of this step (`deliver_step` means "when perceived"; see engine/broadcast.py).
    // Don't run these alongside the tracks: a death notice could show while the victim still stands.
    if (this.renderGen === gen) {
      for (const b of broadcasts) {
        if (this.renderGen !== gen) break;
        // A broadcast whose weather drew skips its words (they stay in the feed); one whose
        // phenomenon was unrecognised still gets them.
        if (depicted.has(b)) continue;
        await this.announcer.playBroadcast(b);
      }
      // One pass over the step's whole post: who opens what is decided with all the mail in hand.
      await this.post.playDeliveries(messages, gen);
    }

    // Concurrent tracks partitioned by shared agents: sequential within a track, parallel across.
    const tracks = this.partition(states, actions).map((p, i) =>
      this.playTrack(p, stateById, states, gen, i, withheldAgents, cut),
    );
    await Promise.all(tracks);
    // Past this line everything writes to the map; a superseded render must not.
    if (this.renderGen !== gen) return;

    // Safety net: land any withheld body whose causing beat never played (e.g. the killing
    // action was filtered out by focus-gating), so no token keeps a look the step disowns.
    // applyBodyState is idempotent, so re-landing one that already played costs nothing.
    for (const id of withheldAgents) {
      const tok = this.tokens.get(id);
      const state = stateById.get(id);
      if (tok && state) this.body.applyBodyState(tok, state, this.placed.has(state.agent_id));
    }
    // Same net for withheld entity changes: if the causing beat was focus-gated out, land
    // the change now rather than leave the map showing a stale world.
    this.entities.revealWithheld();
    // Weather floor (see WEATHER_MIN_MS), measured from when it started.
    if (weatherAt !== null && this.renderGen === gen) {
      // `time.now` is wall-clock; the floor and delayP are in playback ms.
      const left = WEATHER_MIN_MS - (this.time.now - weatherAt) * this.speedScale;
      if (left > 0) await this.delayP(left);
    }
    // Tail dwell so the last result chip is readable (see CHIP_TAIL_MS).
    if (this.renderGen === gen && [...this.tokens.values()].some((t) => t.actionChip)) {
      await this.delayP(CHIP_TAIL_MS);
    }
    // Hold the establishing shot (see SETTLE_MS); otherwise step 0 would never be seen.
    if (this.renderGen === gen && actions.some((a) => a.phase === Phase.initialization)) {
      await this.delayP(SETTLE_MS);
    }
  }

  // Partition agents+actions into groups connected by shared participants
  // (actor, targets, dialogue speakers). Disjoint groups render concurrently.
  private partition(
    states: AgentStateSummary[],
    actions: ActionSummary[],
  ): Array<{ agentIds: string[]; actions: ActionSummary[] }> {
    const ids = new Set(states.map((s) => s.agent_id));
    const parent = new Map<string, string>();
    const find = (x: string): string => {
      if (!parent.has(x)) parent.set(x, x);
      let r = x;
      while (parent.get(r) !== r) r = parent.get(r)!;
      parent.set(x, r);
      return r;
    };
    const union = (a: string, b: string) => parent.set(find(a), find(b));
    for (const s of states) find(s.agent_id);
    for (const act of actions) {
      find(act.agent_id);
      const linked = [
        ...actedOnAgents(act.target),
        ...(act.dialogue ?? []).map((t) => t.speaker_id).filter(Boolean),
      ];
      for (const id of linked) if (ids.has(id)) union(act.agent_id, id);
    }
    // Co-located agents share a track too, so their bubbles play one at a time instead of
    // piling onto the same spot.
    const byLoc = new Map<string, string[]>();
    for (const s of states) {
      if (!s.location_id) continue;
      const list = byLoc.get(s.location_id) ?? [];
      list.push(s.agent_id);
      byLoc.set(s.location_id, list);
    }
    for (const members of byLoc.values()) {
      for (let i = 1; i < members.length; i++) union(members[0], members[i]);
    }
    const groups = new Map<string, { agentIds: string[]; actions: ActionSummary[] }>();
    const groupOf = (id: string) => {
      const r = find(id);
      if (!groups.has(r)) groups.set(r, { agentIds: [], actions: [] });
      return groups.get(r)!;
    };
    for (const s of states) groupOf(s.agent_id).agentIds.push(s.agent_id);
    for (const act of actions) groupOf(act.agent_id).actions.push(act);
    return [...groups.values()];
  }

  // One partition's timeline: members move concurrently (different agents), then
  // the group's interactions play in narrative order, sequentially.
  private async playTrack(
    group: { agentIds: string[]; actions: ActionSummary[] },
    stateById: Map<string, AgentStateSummary>,
    states: AgentStateSummary[],
    gen: number,
    order: number,
    withheldAgents: Set<string>,
    cut: boolean,
  ): Promise<void> {
    // A superseded render moves nobody. Don't remove: cancelPending unblocks the old render
    // asynchronously, and its moveAgent (killTweensOf) would kill the new render's walk and
    // hang the player.
    if (this.renderGen !== gen) return;
    await Promise.all(
      group.agentIds.map((id) => {
        const st = stateById.get(id);
        if (!st) return Promise.resolve();
        if (cut) {
          this.placeAgent(st, states);
          return Promise.resolve();
        }
        return this.moveAgent(st, states);
      }),
    );
    if (this.renderGen !== gen) return;
    // Exactly when movement lands: earlier, the note hangs over last beat's room; later, it
    // waits behind dialogue and deeds.
    this.noteNpcOutcomes(group.agentIds);
    // Stagger so concurrent locations' bubbles cascade in rather than pop at once.
    if (order > 0) {
      await this.delayP(Math.min(order, 5) * 160);
      if (this.renderGen !== gen) return;
    }

    // The deed's icon + label, worn on the chip for its whole life on the map ("开始" → "进行中"
    // → result) and shared with the feed via lib/actionIdentity.
    const chipOf = (type: string): string | undefined => {
      const identity = ACTION_IDENTITY[type as ActionTypeValue];
      return identity && `${identity.icon} ${identity.label}`;
    };
    const acts = [...group.actions]
      .filter((a) => a.action_type !== ActionType.move)
      // Step-0 seeding is not a deed and gets no beat, like move: the map already shows where
      // everyone stands, and the record has no type, deed or target to play. Six co-located
      // agents would otherwise open the world with six queued identical bubbles. The feed keeps it.
      .filter((a) => a.phase !== Phase.initialization)
      // Under a focus only the subjects' actions play; everyone still moves.
      .filter((a) => !this.focusActive() || this.isFocused(a.agent_id))
      // Backend emission ordinal = causal order; records without seq sort stably to the end.
      .sort((a, b) => (a.seq ?? Number.MAX_SAFE_INTEGER) - (b.seq ?? Number.MAX_SAFE_INTEGER));
    // For each withheld agent/entity, the index of the last beat involving it; afterBeat lands
    // the change as that beat ends, so the cause always plays first.
    //   body:   the agent's own last action, OR the last physical strike that targets them.
    //   entity: the last action that names it in affected_entity_ids.
    const bodyBeat = new Map<string, number>();
    const entityBeat = new Map<string, number>();
    if (withheldAgents.size > 0 || this.entities.withheld.size > 0) {
      acts.forEach((a, idx) => {
        if (withheldAgents.has(a.agent_id)) bodyBeat.set(a.agent_id, Math.max(bodyBeat.get(a.agent_id) ?? -1, idx));
        if (a.action_type === ActionType.physical) {
          for (const t of actedOnAgents(a.target)) {
            if (withheldAgents.has(t)) bodyBeat.set(t, Math.max(bodyBeat.get(t) ?? -1, idx));
          }
        }
        for (const eid of a.affected_entity_ids ?? []) {
          if (this.entities.withheld.has(eid)) entityBeat.set(eid, Math.max(entityBeat.get(eid) ?? -1, idx));
        }
      });
    }
    // One beat's entity reveals go in one call, so things changed by the same deed (a key and
    // its door) land together.
    const afterBeat = (actIdx: number): void => {
      for (const [id, beat] of bodyBeat) {
        if (beat !== actIdx) continue;
        const tok = this.tokens.get(id);
        const state = stateById.get(id);
        if (tok && state) this.body.applyBodyState(tok, state, this.placed.has(state.agent_id));
      }
      this.entities.revealEntities([...entityBeat].filter(([, beat]) => beat === actIdx).map(([eid]) => eid));
    };

    const seenDialogue = new Set<string>();
    for (let actIdx = 0; actIdx < acts.length; actIdx++) {
      const act = acts[actIdx];
      if (this.renderGen !== gen) return;
      const tok = this.tokens.get(act.agent_id);
      if (!tok) continue;
      // An interrupt shows the cut, never the deed it cut (no effect, no intent caption). It is a
      // float note, not a result chip: he did nothing this step, and a note also stays out of the
      // end-of-step chip dwell. The gist only — the reason is in the feed as "心声" — and the
      // fallback must not be act.action_description, which is the deed again.
      if (act.phase === Phase.interrupt) {
        const said = act.gist || "行动被打断";
        const chars = [...said];
        this.hud.floatNote(
          tok.container.x,
          tok.container.y + tok.headY - PLATE_GAP - PLATE_RISE - 12,
          `✕ ${chars.length > NOTE_CHARS ? chars.slice(0, NOTE_CHARS).join("") + "…" : said}`,
          CUT_ACCENT,
        );
        await this.delayP(FLOAT_NOTE_MS);
        if (this.renderGen !== gen) return;
        afterBeat(actIdx);
        continue;
      }
      // not_executed never reaches the stage: the intent never engaged the world, so there is
      // nothing to pose or fire, and being foiled is a cognitive event the feed keeps. Skipping it
      // also keeps a non-event from delaying the real deeds queued behind it in the track.
      if (act.not_executed) {
        afterBeat(actIdx); // …but if he died on the step his plan failed, he still greys out
        continue;
      }
      // Opening beat of a multi-step act: nothing has happened yet, so only the pose and the
      // intent. The result belongs to the closing beat; playing it here would stage it twice.
      if (act.phase === Phase.begin) {
        this.faceSubject(tok, act);
        this.body.playDeed(tok, act, this.tokens);
        // The veil is not a result: he is hidden from this beat on. Firing it only with results
        // would make him vanish steps after he crouched.
        if (act.action_type === ActionType.covert) stealthVeil(this, tok, act.detected);
        // Its await also covers the veil above: without a wait, the next clearEphemeral would wipe
        // it mid-draw.
        await this.hud.playActionCaption(tok, act, chipOf(act.action_type) ?? "•", gen);
        // The deed's own icon, not ⏳, so the glyph never switches mid-act; "开始 · 预计X" carries
        // the temporal state.
        this.hud.setActionChip(
          tok, chipOf(act.action_type) ?? "⏳",
          act.duration_label ? `开始 · 预计${act.duration_label}` : "开始",
          "ok", undefined, act.action_description,
        );
        continue;
      }
      // Middle beat of a multi-step act: hold the pose and show progress on the chip. Never
      // replay the deed (caption, dialogue, strike, letter), or a long act stutters.
      if (act.phase === Phase.ongoing_tick) {
        // Re-aim: over a long act his subject may have crossed the room.
        this.faceSubject(tok, act);
        this.body.playDeed(tok, act, this.tokens);
        // "进行中" + a progress bar; same identity glyph as the other beats, not ⏳.
        this.hud.setActionChip(
          tok, chipOf(act.action_type) ?? "⏳", "进行中", "ok",
          act.total_steps > 0 ? act.elapsed_steps / act.total_steps : undefined,
          act.action_description,
        );
        continue;
      }
      this.faceSubject(tok, act);
      // The body acts for every kind of action — one table over the deed (deedPose.ts).
      this.body.playDeed(tok, act, this.tokens);
      if (act.action_type === ActionType.talk) {
        // No connector line: facing, the talk pose and alternating bubbles already say it.
        // An overhearer only turns to watch — a bubble or talk pose would make him a party to it.
        for (const oid of act.overheard_by ?? []) {
          const ot = this.token(oid);
          if (ot && ot !== tok) faceOnlooker(this, ot, [act.agent_id]);
        }
        const sig = (act.dialogue ?? []).map((t) => `${t.speaker}:${t.line}`).join("|");
        if (act.dialogue?.length && !seenDialogue.has(sig)) {
          seenDialogue.add(sig);
          await this.hud.playDialogue(act.dialogue, tok, [act.agent_id, ...subjectPeopleIds(act)], gen);
        } else {
          await this.delayP(650);
        }
        // No result chip for talk: the bubbles already render the exchange.
        afterBeat(actIdx);
        continue;
      }
      // Non-talk: transient effect, then the intent caption, then the persistent result chip.
      if (act.action_type === ActionType.physical) physicalEffect(this, tok, act);
      // Covert: a shadow closes in around him (red if spotted) — about the actor, since COVERT has
      // no structured target. Only where he enters concealment; the closing beat is him leaving it.
      else if (act.action_type === ActionType.covert && act.phase !== Phase.ongoing_complete)
        stealthVeil(this, tok, act.detected);
      // The closing beat gets no intent bubble (it was said when he set out; the chip carries it).
      // The wait stays: it covers the fire-and-forget effects above, which the next clearEphemeral
      // would otherwise wipe mid-swing.
      if (act.phase === Phase.ongoing_complete) {
        await this.delayP(1500);
      } else {
        await this.hud.playActionCaption(tok, act, chipOf(act.action_type) ?? "•", gen);
      }
      // send_message fires after the caption: arcs to named recipients, else a local announcement.
      // Awaited, unlike other effects: nothing after it covers its flight, so the next
      // clearEphemeral would delete the letter mid-air.
      if (act.action_type === ActionType.send_message) {
        const addressees = actedOnAgents(act.target);
        if (addressees.length) await directedMessages(this, tok, addressees);
        else await localAnnounce(this, tok);
      }
      // On a failure the chip says why: `failure_reason` is a short phrase written for this spot,
      // never parsed out of `outcome`. Falls back to the gist, then the intent.
      this.hud.setActionChip(
        tok,
        chipOf(act.action_type) ?? "•",
        act.failure_reason || act.gist || act.action_description,
        act.succeeded === false ? "fail" : "ok",
        undefined,
        act.action_description,
      );
      afterBeat(actIdx);
    }
  }

  /** Put a figure where this step says he is, at once: the step before it was not drawn. */
  private placeAgent(state: AgentStateSummary, states: AgentStateSummary[]): void {
    const tok = this.tokens.get(state.agent_id);
    if (!tok || tok.dead) return;
    const pos = this.staging.positionFor(state, states);
    const trip = state.transit && state.transit.total_steps > 0 ? state.transit : null;
    if (trip) {
      this.walker.placeOnTransit(tok, state.agent_id, trip);
    } else {
      this.walker.clearTransit(state.agent_id);
      if (!pos) return;
      this.killTweens(tok.container);
      tok.container.setPosition(pos.x, pos.y);
      this.placed.add(state.agent_id);
    }
    // The follow pan would ease in from where he stood before the cut.
    if (this.focus.agents.includes(state.agent_id)) this.camera.cut();
  }

  /**
   * Put a figure where this step says he is, on foot over walkable ground; resolves when the
   * walk finishes. Only for the step right after the one drawn before (see placeAgent).
   *
   * Don't add a straight-line branch for moves without a transit: an adjacent-room move
   * arrives with `transit: null` and would slide through the palace wall. The transit only
   * supplies waypoints and this step's portion of the route.
   */
  private moveAgent(state: AgentStateSummary, states: AgentStateSummary[]): Promise<void> {
    const tok = this.tokens.get(state.agent_id);
    if (!tok) return Promise.resolve();
    // A body already drawn down does not travel. Keyed on the token, not the report: the report
    // says "dead" for the whole step he dies in, and a withheld death must still walk to the fight.
    if (tok.dead) return Promise.resolve();
    if (state.transit && state.transit.total_steps > 0) {
      return this.walker.walkTransit(tok, state.agent_id, state.transit);
    }
    const pos = this.staging.positionFor(state, states);
    if (state.arrival && state.arrival.total_steps > 0) {
      // Walk the engine's route (through the waypoints it reported him passing) into his place in
      // the room. A ground route here could miss those waypoints.
      const walked = this.walker.walkTransit(tok, state.agent_id, state.arrival, pos);
      this.walker.clearTransit(state.agent_id);
      return walked;
    }
    this.walker.clearTransit(state.agent_id);
    if (!pos) return Promise.resolve();
    this.killTweens(tok.container);
    if (!this.placed.has(state.agent_id)) {
      tok.container.setPosition(pos.x, pos.y); // first sighting: snap, don't fly from (0,0)
      this.placed.add(state.agent_id);
      return Promise.resolve();
    }
    // Displaced by the director, not walked: a cut. Don't route him on foot — that would show a
    // teleport as a stroll through the palace and tell a story that isn't true.
    if (state.displaced) {
      // A puff at each end, as a walk leaves; otherwise the cut looks like a dropped frame.
      const leftFrom = { x: tok.container.x, y: tok.container.y };
      tok.container.setPosition(pos.x, pos.y);
      footDust(this, leftFrom.x, leftFrom.y + 8);
      footDust(this, pos.x, pos.y + 8);
      // The camera cuts with a followed subject: the follow pan's ease would leave him off-frame
      // for about a second.
      if (this.focus.agents.includes(state.agent_id)) this.camera.cut();
      return Promise.resolve();
    }
    const from = { x: tok.container.x, y: tok.container.y };
    // A shuffle within his room is tweened straight (see WALK_ROUTE_MIN_PX).
    if (Phaser.Math.Distance.Between(from.x, from.y, pos.x, pos.y) < WALK_ROUTE_MIN_PX) {
      return this.tweenP({ targets: tok.container, x: pos.x, y: pos.y, duration: 500, ease: "Sine.easeInOut" });
    }
    const route = this.walker.groundRoute(from, pos);
    if (!route) {
      // No route: place him as a cut. Don't slide him through the wall instead.
      tok.container.setPosition(pos.x, pos.y);
      return Promise.resolve();
    }
    footDust(this, from.x, from.y + 8);
    return this.walker.walkAlong(tok, route).then(() => footDust(this, pos.x, pos.y + 8));
  }

  /** Reported dead and no token left on the map: a body an earlier step already showed falling. */
  gone(state: AgentStateSummary): boolean {
    return reportedDead(state) && !this.tokens.has(state.agent_id);
  }

  // FxHost: a point outward along the ray from the world's centre through (x,y), off the map.
  beyondTheMap(x: number, y: number): { x: number; y: number } {
    let dx = x - this.map.worldW / 2;
    let dy = y - this.map.worldH / 2;
    const len = Math.hypot(dx, dy);
    if (len < 1) {
      dx = 0; dy = -1; // dead centre: no outward direction — send it skyward
    } else {
      dx /= len; dy /= len;
    }
    const reach = Math.max(this.map.worldW, this.map.worldH) * 0.35;
    return { x: x + dx * reach, y: y + dy * reach };
  }

  private clearEphemeral(): void {
    this.timers.forEach((t) => t.remove(false));
    this.timers = [];
    this.ephemeral.forEach((o) => o.destroy());
    this.ephemeral = [];
    // Action chips and ✉ badges are step-scoped; cross-step history lives in the feed.
    for (const tok of this.tokens.values()) {
      tok.actionChip?.destroy();
      tok.actionChip = undefined;
      tok.chipData = undefined;
      tok.postBadge?.destroy();
      tok.postBadge = undefined;
      tok.postData = undefined;
    }
  }
}
