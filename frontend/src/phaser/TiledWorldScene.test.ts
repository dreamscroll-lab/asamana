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
import { MO, SHI, step, walking, who } from "../lab/scenes/fixtures";
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
    expect(tok.container.setAlpha).toHaveBeenCalledWith(0.8 + (baseAlpha * HIDDEN_ALPHA - 0.8) * fade);
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
