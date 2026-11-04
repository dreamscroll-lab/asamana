import { describe, expect, it, vi } from "vitest";

import type { AgentToken } from "./types";
import type { Figure } from "./figures";
import type { MapSource } from "./mapSource";
import type { Presence } from "./presence";
import type { Walker } from "./walking";
import type { BodyRig } from "./body";
import type { Staging } from "./staging";
import type { EntityLayer } from "./entities";
import type { PocketLayer } from "./pocket";
import type { MapCamera } from "./camera";
import type { AmbientLight } from "./ambient";
import type { Announcer } from "./announce";
import type { PostLayer } from "./post";
import type { HoverTip } from "./hoverTip";
import { LIN, MO, SHI, WANG, aimAgents, did, fixtureNpcs, mindless, step, walking, who } from "../lab/scenes/fixtures";
import { ActionType, Deed } from "../lib/contract";
import type { StepView } from "../types";
import { HIDDEN_ALPHA, HIDDEN_FADE_MS } from "./isoDepth";

vi.mock("phaser", () => ({ default: {
  Scene: class {},
  Math: {
    Clamp: (n: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, n)),
    Distance: {
      Between: (x: number, y: number, xx: number, yy: number) => Math.hypot(xx - x, yy - y),
      BetweenPoints: (a: { x: number; y: number }, b: { x: number; y: number }) => Math.hypot(b.x - a.x, b.y - a.y),
    },
  },
} }));
vi.mock("./recolorPipeline", () => ({ RECOLOR_FX: "recolor", RecolorFX: class {} }));
vi.mock("./effects", async (original) => ({ ...await original<object>(), footDust: vi.fn() }));
Object.assign(globalThis, { window: { devicePixelRatio: 2 } });
const { TiledWorldScene } = await import("./TiledWorldScene");

interface Internals {
  ready: boolean;
  drawnStep: number | null;
  tokens: Map<string, AgentToken>;
  placed: Set<string>;
  presence: Presence;
  walker: Walker;
  body: BodyRig;
  staging: Staging;
  entities: EntityLayer;
  pockets: PocketLayer;
  camera: MapCamera;
  ambient: AmbientLight;
  announcer: Announcer;
  post: PostLayer;
  hoverTip: HoverTip;
  applyFocus(): void;
  ensureToken(figure: Figure): AgentToken;
  skinFor(id: string): string;
}

function token(locationId: string): AgentToken {
  const container = {
    x: 0, y: 0, alpha: 0.8,
    setPosition: vi.fn(function (this: { x: number; y: number }, x: number, y: number) { this.x = x; this.y = y; return this; }),
    setDepth: vi.fn().mockReturnThis(), setAlpha: vi.fn().mockReturnThis(), setScale: vi.fn().mockReturnThis(),
    disableInteractive: vi.fn(), destroy: vi.fn(),
  };
  return {
    container, locationId, baseAlpha: 1, presenceAlpha: 1, dead: false,
    sprite: { anims: { isPlaying: false } }, px: NaN, py: NaN,
    plate: { setVisible: vi.fn() }, vitTrack: { setVisible: vi.fn() }, vitFill: { setVisible: vi.fn() },
  } as unknown as AgentToken;
}

function setup() {
  const scene = new TiledWorldScene({} as MapSource);
  const internal = scene as unknown as Internals;
  internal.ready = true;
  vi.spyOn(internal, "ensureToken").mockImplementation((figure) => {
    let tok = internal.tokens.get(figure.agent_id);
    if (!tok) {
      tok = token(figure.location_id);
      internal.tokens.set(figure.agent_id, tok);
    }
    return tok;
  });
  vi.spyOn(internal, "applyFocus").mockImplementation(() => {});
  vi.spyOn(internal.body, "releaseIdlePoses").mockImplementation(() => {});
  vi.spyOn(internal.body, "applyBodyState").mockImplementation(() => {});
  vi.spyOn(internal.body, "removeToken").mockImplementation(() => {});
  vi.spyOn(internal.body, "bodyChanged").mockReturnValue(false);
  vi.spyOn(internal.staging, "layoutRooms").mockReturnValue(new Map());
  vi.spyOn(internal.staging, "layoutLabels").mockImplementation(() => {});
  vi.spyOn(internal.staging, "positionFor").mockImplementation((figure) => ({ x: figure.location_id === "near" ? 100 : 300, y: 0 }));
  vi.spyOn(internal.entities, "beginStep").mockImplementation((entities) => entities);
  vi.spyOn(internal.entities, "applyEntities").mockImplementation(() => {});
  vi.spyOn(internal.entities, "revealWithheld").mockImplementation(() => {});
  vi.spyOn(internal.pockets, "sync").mockImplementation(() => {});
  vi.spyOn(internal.pockets, "close").mockImplementation(() => {});
  vi.spyOn(internal.pockets, "pulse").mockImplementation(() => {});
  vi.spyOn(internal.camera, "followFocus").mockImplementation(() => {});
  vi.spyOn(internal.camera, "fly").mockImplementation(() => {});
  vi.spyOn(internal.ambient, "lightRooms").mockImplementation(() => {});
  vi.spyOn(internal.ambient, "follow").mockImplementation(() => {});
  vi.spyOn(internal.announcer, "retireWeather").mockImplementation(() => {});
  vi.spyOn(internal.announcer, "pinBanner").mockImplementation(() => {});
  vi.spyOn(internal.post, "playDeliveries").mockResolvedValue(undefined);
  vi.spyOn(internal.hoverTip, "hide").mockImplementation(() => {});
  vi.spyOn(scene, "killTweens").mockImplementation(() => {});
  vi.spyOn(scene, "delayP").mockResolvedValue(undefined);
  vi.spyOn(scene, "tweenP").mockResolvedValue(undefined);
  const enter = vi.spyOn(internal.presence, "enter").mockResolvedValue(undefined);
  const leave = vi.spyOn(internal.presence, "leave").mockResolvedValue(undefined);
  return { scene, internal, enter, leave };
}

const near = { id: "near", name: "Near" };
const far = { id: "far", name: "Far" };

describe("director rendering isolation", () => {
  it.each(["agent", "npc"] as const)("updates an existing %s token when its missing identity is recovered", async (kind) => {
    const { scene, internal, enter, leave } = setup();
    const id = kind === "agent" ? MO.id : WANG.npc_id;
    await scene.renderStep(step({
      at: 1,
      states: kind === "agent" ? [who(MO, near)] : [],
      npcs: kind === "npc" ? [mindless(WANG, near)] : [],
    }));
    const existing = scene.token(id)!;
    const label = {
      text: "某人", width: 20,
      setColor: vi.fn(),
      setText: vi.fn(function (this: { text: string; width: number }, text: string) {
        this.text = text;
        this.width = text.length * 10;
        return this;
      }),
    };
    const pouch = {
      x: 16,
      setX: vi.fn(function (this: { x: number }, x: number) { this.x = x; return this; }),
    };
    const vitTrack = { height: 2.5, setSize: vi.fn().mockReturnThis() };
    const vitFill = { height: 2.5, scaleX: 0.4, setSize: vi.fn().mockReturnThis(), setX: vi.fn().mockReturnThis() };
    const rings = (kind === "agent" ? [[3, 1], [1.5, 0.7]] : [[1, 0.55]])
      .map(([lineWidth, strokeAlpha]) => ({ lineWidth, strokeAlpha, setStrokeStyle: vi.fn() }));
    Object.assign(existing, { color: 0xabcdef, label, pouch, vitTrack, vitFill, identityRings: rings });
    vi.spyOn(internal, "skinFor").mockReturnValue("recovered-skin");
    const assignSkin = vi.spyOn(internal.body, "assignSkin").mockImplementation(() => {});
    const dye = vi.spyOn(internal.body, "dyeGarment").mockImplementation(() => {});
    const render = vi.spyOn(scene, "renderStep");
    vi.mocked(internal.ensureToken).mockClear();
    vi.mocked(internal.post.playDeliveries).mockClear();
    vi.mocked(internal.announcer.pinBanner).mockClear();
    const name = kind === "agent" ? "身份恢复后的角色" : "身份恢复后的门房";
    const color = kind === "agent" ? "#123456" : "#8a93a6";
    const numericColor = Number.parseInt(color.slice(1), 16);

    if (kind === "agent") {
      scene.setCast({ [id]: {
        agent_id: id, name, color, gender: "女", age: 30, role: "旅人", background: "", appearance: "",
      } });
    } else {
      scene.setNpcs({ [id]: { npc_id: id, name, color, gender: "男", age: 40, description: "门房" } });
    }

    expect(scene.token(id)).toBe(existing);
    expect(existing.color).toBe(numericColor);
    expect(label.text).toBe(name);
    expect(label.setColor).toHaveBeenCalledWith(color);
    expect(dye).toHaveBeenCalledWith(existing.sprite, numericColor);
    expect(assignSkin).toHaveBeenCalledWith(existing, "recovered-skin");
    for (const ring of rings) {
      expect(ring.setStrokeStyle).toHaveBeenCalledWith(ring.lineWidth, numericColor, ring.strokeAlpha);
    }
    expect(pouch.x - label.width / 2).toBe(6);
    expect(vitTrack.setSize).toHaveBeenCalledWith(label.width, 2.5);
    expect(vitFill.setSize).toHaveBeenCalledWith(label.width, 2.5);
    expect(vitFill.setX).toHaveBeenCalledWith(-label.width / 2);
    expect(vitFill.scaleX).toBe(0.4);
    expect(internal.ensureToken).not.toHaveBeenCalled();
    expect(render).not.toHaveBeenCalled();
    expect(enter).not.toHaveBeenCalled();
    expect(leave).not.toHaveBeenCalled();
    expect(internal.post.playDeliveries).not.toHaveBeenCalled();
    expect(internal.announcer.pinBanner).not.toHaveBeenCalled();
  });

  it("renders NPC figures from the global identity roster", async () => {
    const { scene, internal } = setup();
    scene.setNpcs(fixtureNpcs());
    await scene.renderStep(step({ at: 1, states: [], npcs: [mindless(WANG, near)] }));
    expect(internal.ensureToken).toHaveBeenCalledWith(expect.objectContaining({
      agent_id: WANG.npc_id, agent_name: WANG.name, color: WANG.color,
    }));
  });

  it.each(["continuous", "seek", "relocate", "viewpoint"] as const)("avoids corpse shrink cleanup for a living actor peer on %s", async (change) => {
    const { scene, internal, leave } = setup();
    scene.setViewer(MO.id);
    await scene.renderStep(step({ at: 1, states: [who(MO, near), who(SHI, near)] }));
    const guest = scene.token(SHI.id)!;
    if (change === "viewpoint") {
      scene.setViewer(null);
      scene.setViewer(MO.id);
    }
    await scene.renderStep(step({
      at: change === "seek" ? 7 : 2,
      states: [who(MO, change === "relocate" ? far : near)],
    }));
    expect(leave).toHaveBeenCalledWith(SHI.id, guest, change === "continuous" ? "walk" : "fade");
    expect(internal.body.removeToken).not.toHaveBeenCalled();
    expect(scene.token(SHI.id)).toBeUndefined();
    expect(scene.canSelect(SHI.id)).toBe(false);
  });

  it("preserves corpse cleanup in actor mode", async () => {
    const { scene, internal, leave } = setup();
    scene.setViewer(MO.id);
    await scene.renderStep(step({ at: 1, states: [who(MO, near), who(SHI, near)] }));
    const guest = scene.token(SHI.id)!;
    guest.dead = true;
    await scene.renderStep(step({ at: 2, states: [who(MO, near)] }));
    expect(internal.body.removeToken).toHaveBeenCalledExactlyOnceWith(guest);
    expect(leave).not.toHaveBeenCalled();
  });

  it("keeps new and missing director figures on the existing placement/removal paths", async () => {
    const { scene, internal, enter, leave } = setup();
    await scene.renderStep(step({ at: 1, states: [who(MO, near)] }));
    await scene.renderStep(step({ at: 2, states: [who(MO, near), who(SHI, near)] }));
    const guest = scene.token(SHI.id)!;
    expect(guest.container.setPosition).toHaveBeenCalledWith(100, 0);
    expect(scene.canSelect(SHI.id)).toBe(true);
    await scene.renderStep(step({ at: 3, states: [who(MO, near)] }));
    expect(internal.body.removeToken).toHaveBeenCalledWith(guest);
    expect(scene.token(SHI.id)).toBeUndefined();
    expect(enter).not.toHaveBeenCalled();
    expect(leave).not.toHaveBeenCalled();
  });

  it("retains director road routes, transit and displacement behavior", async () => {
    const { scene, internal, enter, leave } = setup();
    const route = [{ x: 100, y: 0 }, { x: 300, y: 0 }];
    vi.spyOn(internal.walker, "groundRoute").mockReturnValue(route);
    const along = vi.spyOn(internal.walker, "walkAlong").mockResolvedValue(undefined);
    const transit = vi.spyOn(internal.walker, "walkTransit").mockResolvedValue(undefined);
    await scene.renderStep(step({ at: 1, states: [who(MO, near)] }));
    await scene.renderStep(step({ at: 2, states: [who(MO, far)] }));
    expect(along).toHaveBeenCalledWith(scene.token(MO.id), route);
    const trip = walking([far, near], 1, 2);
    await scene.renderStep(step({ at: 3, states: [who(MO, far, { transit: trip })] }));
    expect(transit).toHaveBeenCalledWith(scene.token(MO.id), MO.id, trip);
    await scene.renderStep(step({ at: 4, states: [who(MO, near, { displaced: true })] }));
    expect(along).toHaveBeenCalledTimes(1);
    expect(transit).toHaveBeenCalledTimes(1);
    expect(enter).not.toHaveBeenCalled();
    expect(leave).not.toHaveBeenCalled();
  });

  it.each([1, 0.3, 0.5])("preserves focus/death opacity %s and ground occlusion", async (baseAlpha) => {
    const { scene, internal } = setup();
    await scene.renderStep(step({ at: 1, states: [who(MO, near)] }));
    const tok = scene.token(MO.id)!;
    tok.baseAlpha = baseAlpha;
    Object.assign(scene, { cameras: { main: { zoom: 1 } }, time: { now: 0 } });
    vi.spyOn(scene.map, "hiddenGround").mockReturnValue(true);
    scene.update(0, 16);
    const fade = 1 - Math.exp((-0.016 * 3000) / HIDDEN_FADE_MS);
    expect(tok.container.setAlpha).toHaveBeenCalledWith(0.8 + (Math.min(baseAlpha, HIDDEN_ALPHA) - 0.8) * fade);
    expect(internal.presence.departing.size).toBe(0);
  });

  it("restores a pending actor figure's opacity when a director render starts", async () => {
    const { scene, internal, enter, leave } = setup();
    scene.setViewer(MO.id);
    await scene.renderStep(step({ at: 1, states: [who(MO, near)] }));
    await scene.renderStep(step({ at: 2, states: [who(MO, near), who(SHI, near)] }));
    expect(enter).toHaveBeenCalledOnce();
    const guest = scene.token(SHI.id)!;
    guest.presenceAlpha = 0;
    scene.setViewer(null);
    await scene.renderStep(step({ at: 3, states: [who(MO, near), who(SHI, near)] }));
    expect(guest.presenceAlpha).toBe(1);
    expect(scene.canSelect(SHI.id)).toBe(true);
    expect(enter).toHaveBeenCalledOnce();
    expect(leave).not.toHaveBeenCalled();
    expect(internal.presence.departing.size).toBe(0);
  });

  it.each(["enter", "leave"] as const)("cancels an actual actor %s animation before drawing the director's world", async (transition) => {
    const { scene, internal, enter, leave } = setup();
    enter.mockRestore();
    leave.mockRestore();
    vi.mocked(scene.killTweens).mockRestore();
    vi.mocked(scene.tweenP).mockRestore();
    const tweens = { add: vi.fn(), chain: vi.fn(), killTweensOf: vi.fn() };
    Object.assign(scene, { tweens });
    vi.spyOn(internal.walker, "presenceRoute").mockReturnValue([{ x: 100, y: 0 }, { x: 180, y: 0 }]);
    scene.setViewer(MO.id);
    const actorOnly = [who(MO, near)];
    const together = [who(MO, near), who(SHI, near)];
    await scene.renderStep(step({ at: 1, states: transition === "enter" ? actorOnly : together }));
    const oldGuest = scene.token(SHI.id);
    const pending = scene.renderStep(step({ at: 2, states: transition === "enter" ? together : actorOnly }));
    await vi.waitFor(() => {
      expect(transition === "enter" ? internal.presence.entering.size : internal.presence.departing.size).toBe(1);
    });
    const arrivingGuest = scene.token(SHI.id);
    expect(scene.canSelect(SHI.id)).toBe(false);
    // The director's own reposition can complete; only the actor's already-open waits are canceled.
    vi.spyOn(scene, "tweenP").mockResolvedValue(undefined);
    scene.setViewer(null);
    await scene.renderStep(step({ at: 3, states: together }));
    await pending;
    expect(internal.presence.entering.size).toBe(0);
    expect(internal.presence.departing.size).toBe(0);
    expect(scene.canSelect(SHI.id)).toBe(true);
    expect(scene.token(SHI.id)?.presenceAlpha).toBe(1);
    if (transition === "leave") {
      expect(oldGuest!.container.destroy).toHaveBeenCalledOnce();
      expect(scene.token(SHI.id)).not.toBe(oldGuest);
      expect(scene.token(SHI.id)!.container.destroy).not.toHaveBeenCalled();
    } else {
      expect(scene.token(SHI.id)).toBe(arrivingGuest);
      expect(arrivingGuest!.container.destroy).not.toHaveBeenCalled();
    }
  });
});

function landings(internal: Internals, tok: AgentToken): { fallsNow: boolean; order: number }[] {
  const { calls, invocationCallOrder } = vi.mocked(internal.body.applyBodyState).mock;
  return calls.flatMap(([t, , fallsNow], i) => t === tok ? [{ fallsNow, order: invocationCallOrder[i] }] : []);
}

// The fixtures date a death from consecutive steps; a viewer away meanwhile needs it set.
function diedAt(view: StepView, agentId: string, deathStep: number): StepView {
  for (const scope of Object.values(view.locations)) {
    for (const a of scope.agents) if (a.agent_id === agentId) a.death_step = deathStep;
  }
  return view;
}

describe("a body is drawn only on the step it falls", () => {
  const dead = { vitality: 0, is_active: false };

  it("shows someone first seen on the step he dies standing, then falling to the blow", async () => {
    const { scene, internal } = setup();
    scene.setViewer(MO.id);
    // The blow doesn't play under this focus, so the withheld fall lands in the safety net.
    scene.setFocus({ agents: [MO.id], place: null, follow: false });
    await scene.renderStep(step({ at: 1, states: [who(MO, far)] }));
    await scene.renderStep(step({
      at: 2,
      states: [who(MO, near), who(LIN, near), who(SHI, near, dead)],
      actions: [did(LIN, { action_type: ActionType.physical, deed: Deed.strike, target: aimAgents(SHI.id) })],
    }));
    const victim = scene.token(SHI.id)!;
    expect(victim).toBeDefined();
    expect(victim.container.setPosition).toHaveBeenCalledWith(100, 0);
    // Withheld: the fall lands once, after he was put in place, as this step's death.
    const landed = landings(internal, victim);
    expect(landed).toHaveLength(1);
    expect(landed[0].fallsNow).toBe(true);
    expect(landed[0].order).toBeGreaterThan(vi.mocked(victim.container.setPosition).mock.invocationCallOrder[0]);
  });

  it("puts someone first seen already down, with no visible cause, where he lies", async () => {
    const { scene, internal } = setup();
    vi.mocked(internal.body.applyBodyState).mockImplementation((tok, state) => {
      tok.dead = state.vitality === 0 || !state.is_active;
    });
    scene.setViewer(MO.id);
    await scene.renderStep(step({ at: 1, states: [who(MO, far)] }));
    await scene.renderStep(step({ at: 2, states: [who(MO, near), who(SHI, near, dead)] }));
    const body = scene.token(SHI.id)!;
    expect(body.dead).toBe(true);
    expect(body.container.setPosition).toHaveBeenCalledWith(100, 0);
    // He falls where he lies, with the death's visual like anyone dying on this step.
    const landed = landings(internal, body);
    expect(landed[0].fallsNow).toBe(true);
    expect(landed[0].order).toBeGreaterThan(vi.mocked(body.container.setPosition).mock.invocationCallOrder[0]);
    await scene.renderStep(step({ at: 3, states: [who(MO, near), who(SHI, near, dead)] }));
    expect(scene.token(SHI.id)).toBeUndefined();
  });

  it("gives a death the one-time visual on its step only, to someone long in view as well", async () => {
    const { scene, internal } = setup();
    vi.mocked(internal.body.applyBodyState).mockImplementation((tok, state) => {
      tok.dead = state.vitality === 0 || !state.is_active;
    });
    await scene.renderStep(step({ at: 1, states: [who(MO, near), who(SHI, near)] }));
    const tok = scene.token(SHI.id)!;
    await scene.renderStep(step({ at: 2, states: [who(MO, near), who(SHI, near, dead)] }));
    await scene.renderStep(step({ at: 3, states: [who(MO, near), who(SHI, near, dead)] }));
    expect(landings(internal, tok).map((l) => l.fallsNow)).toEqual([false, true]);
    expect(scene.token(SHI.id)).toBeUndefined();
  });

  it("removes someone a seek carried past his death, not leaving him standing", async () => {
    const { scene, internal } = setup();
    await scene.renderStep(step({ at: 1, states: [who(MO, near), who(SHI, near)] }));
    const tok = scene.token(SHI.id)!;
    await scene.renderStep(diedAt(step({ at: 9, states: [who(MO, near), who(SHI, near, dead)] }), SHI.id, 5));
    expect(internal.body.removeToken).toHaveBeenCalledExactlyOnceWith(tok);
    expect(scene.token(SHI.id)).toBeUndefined();
    expect(landings(internal, tok).map((l) => l.fallsNow)).toEqual([false]);
  });

  it("never draws the long dead, even to a viewer who never saw them", async () => {
    const { scene } = setup();
    scene.setViewer(MO.id);
    await scene.renderStep(step({ at: 4, states: [who(MO, far)] }));
    await scene.renderStep(diedAt(step({ at: 5, states: [who(MO, near), who(SHI, near, dead)] }), SHI.id, 2));
    expect(scene.token(SHI.id)).toBeUndefined();
  });
});

describe("a step plays from where the step before it ended", () => {
  const moved = (route: ReturnType<typeof walking>) =>
    did(MO, { action_type: ActionType.move, deed: Deed.move, route });

  it("establishes step N-1 on a seek, then plays step N as continuous playback does", async () => {
    const { scene, internal } = setup();
    const transit = vi.spyOn(internal.walker, "walkTransit").mockResolvedValue(undefined);
    const deed = vi.spyOn(internal.body, "playDeed").mockImplementation(() => {});
    await scene.renderStep(step({ at: 9, states: [who(MO, far)] }));
    vi.mocked(internal.post.playDeliveries).mockClear();
    const seven = step({
      at: 7, states: [who(MO, near)], actions: [did(MO, { action_type: ActionType.work, deed: Deed.work })],
    });
    const previous = vi.fn().mockResolvedValue(seven);
    const route = walking([near, far], 1, 1);
    await scene.renderStep(step({ at: 8, states: [who(MO, far)], actions: [moved(route)] }), previous);
    expect(previous).toHaveBeenCalledExactlyOnceWith(7);
    // Step 7 is only landed: its deeds and post never play.
    expect(deed).not.toHaveBeenCalled();
    expect(internal.post.playDeliveries).toHaveBeenCalledTimes(1);
    // …so step 8's walk is the one continuous playback draws.
    expect(transit).toHaveBeenCalledWith(scene.token(MO.id), MO.id, route, { x: 300, y: 0 });
  });

  it("lands step N-1 at once even when the map stands where N-2 ended", async () => {
    const { scene, internal } = setup();
    const transit = vi.spyOn(internal.walker, "walkTransit").mockResolvedValue(undefined);
    await scene.renderStep(step({ at: 6, states: [who(MO, near)] }));
    const route = walking([near, far], 1, 1);
    const seven = step({ at: 7, states: [who(MO, far)], actions: [moved(route)] });
    await scene.renderStep(step({ at: 8, states: [who(MO, far)] }), vi.fn().mockResolvedValue(seven));
    // Step 7 continues from 6, but it is only established: its walk never plays.
    expect(transit).not.toHaveBeenCalled();
  });

  it("lets a newer render through while the step-before fetch stalls", async () => {
    const { scene, internal } = setup();
    await scene.renderStep(step({ at: 9, states: [who(MO, far)] }));
    const stalled = scene.renderStep(
      step({ at: 4, states: [who(MO, near)] }), () => new Promise<never>(() => {}),
    );
    await Promise.resolve();
    await scene.renderStep(step({ at: 0, states: [who(MO, far)] }));
    await stalled;
    expect(internal.drawnStep).toBe(0);
  });

  it("asks for no earlier step in continuous playback, or for step 0", async () => {
    const { scene } = setup();
    const previous = vi.fn();
    await scene.renderStep(step({ at: 0, states: [who(MO, near)] }), previous);
    await scene.renderStep(step({ at: 1, states: [who(MO, near)] }), previous);
    expect(previous).not.toHaveBeenCalled();
  });

  it("draws the step as a cut when step N-1 can't be had", async () => {
    const { scene, internal } = setup();
    const transit = vi.spyOn(internal.walker, "walkTransit").mockResolvedValue(undefined);
    const route = walking([near, far], 1, 1);
    await scene.renderStep(
      step({ at: 8, states: [who(MO, far)], actions: [moved(route)] }),
      vi.fn().mockRejectedValue(new Error("offline")),
    );
    expect(transit).not.toHaveBeenCalled();
    expect(scene.token(MO.id)!.container.x).toBe(300);
  });

  it("treats the step after a superseded render as a cut, not as continuing from where it stopped", async () => {
    const { scene, internal } = setup();
    vi.spyOn(internal.walker, "groundRoute").mockReturnValue([{ x: 100, y: 0 }, { x: 300, y: 0 }]);
    let release = () => {};
    const along = vi.spyOn(internal.walker, "walkAlong").mockImplementation(
      () => new Promise<void>((resolve) => { release = resolve; }),
    );
    await scene.renderStep(step({ at: 1, states: [who(MO, near)] }));
    const second = scene.renderStep(step({ at: 2, states: [who(MO, far)] }));
    await vi.waitFor(() => expect(along).toHaveBeenCalledTimes(1));
    const third = scene.renderStep(step({ at: 3, states: [who(MO, near)] }));
    release();
    await Promise.all([second, third]);
    expect(along).toHaveBeenCalledTimes(1);
    expect(scene.token(MO.id)!.container.x).toBe(100);
  });

  it("doesn't hold a body back by comparing it with another step's", async () => {
    const { scene, internal } = setup();
    await scene.renderStep(step({ at: 9, states: [who(MO, near)] }));
    vi.spyOn(internal.body, "playDeed").mockImplementation(() => {});
    vi.mocked(internal.body.bodyChanged).mockClear();
    await scene.renderStep(step({
      at: 4, states: [who(MO, near)], actions: [did(MO, { action_type: ActionType.work, deed: Deed.work })],
    }));
    expect(internal.body.bodyChanged).not.toHaveBeenCalled();
  });
});
