import { describe, expect, it, vi } from "vitest";

import { Presence, continuesPresence } from "./presence";
import type { AgentToken } from "./types";
import type { Walker, WalkHost } from "./walking";

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve };
}

function setup(route: { x: number; y: number }[] | null = [{ x: 0, y: 0 }, { x: 80, y: 20 }]) {
  const walk = deferred();
  const fade = deferred();
  const container = {
    x: 0, y: 0, setPosition: vi.fn().mockReturnThis(), setAlpha: vi.fn().mockReturnThis(),
    disableInteractive: vi.fn(), destroy: vi.fn(),
  };
  const tok = { container, locationId: "room", presenceAlpha: 1, px: 0, py: 0 } as unknown as AgentToken;
  const host = { tweenP: vi.fn(() => fade.promise), killTweens: vi.fn() } as unknown as WalkHost;
  const walker = {
    presenceRoute: vi.fn(() => route?.map((p) => ({ ...p })) ?? null),
    walkAlong: vi.fn(() => walk.promise),
  } as unknown as Walker;
  return { layer: new Presence(host, walker), tok, host, walker, walk, fade, container };
}

describe("actor presence", () => {
  it("animates only consecutive steps in the same actor's nonempty place", () => {
    const view = { viewer: "self", location: "room" };
    expect(continuesPresence(view, view, false)).toBe(true);
    expect(continuesPresence(null, view, false)).toBe(false);
    expect(continuesPresence(view, view, true)).toBe(false);
    expect(continuesPresence(view, { ...view, viewer: "other" }, false)).toBe(false);
    expect(continuesPresence(view, { ...view, location: "elsewhere" }, false)).toBe(false);
    expect(continuesPresence({ viewer: "self", location: "" }, { viewer: "self", location: "" }, false)).toBe(false);
    expect(continuesPresence({ viewer: null, location: "room" }, { viewer: null, location: "room" }, false)).toBe(false);
  });

  it("walks in from the local boundary and stays unselectable until the walk ends", async () => {
    const { layer, tok, container, host, walker, fade, walk } = setup();
    const running = layer.enter(tok, { x: 0, y: 0 });
    expect(container.setPosition).toHaveBeenCalledWith(80, 20);
    expect(tok.presenceAlpha).toBe(0);
    expect(layer.entering.has(tok)).toBe(true);
    expect(walker.walkAlong).toHaveBeenCalledWith(tok, [{ x: 80, y: 20 }, { x: 0, y: 0 }], 2000);
    expect(host.tweenP).toHaveBeenCalledWith({ targets: tok, presenceAlpha: 1, duration: 600 });
    fade.resolve();
    await Promise.resolve();
    expect(layer.entering.has(tok)).toBe(true);
    walk.resolve();
    await running;
    expect(layer.entering.has(tok)).toBe(false);
    expect(tok.presenceAlpha).toBe(1);
  });

  it("retains only a departing drawing and cleans it when canceled waits settle", async () => {
    const { layer, tok, container, host, walker, fade, walk } = setup();
    const running = layer.leave("guest", tok);
    expect(layer.departing.get("guest")).toBe(tok);
    expect(container.disableInteractive).toHaveBeenCalledOnce();
    expect(walker.walkAlong).toHaveBeenCalledWith(tok, [{ x: 0, y: 0 }, { x: 80, y: 20 }], 2000);
    expect(host.tweenP).toHaveBeenCalledWith({ targets: tok, presenceAlpha: 0, duration: 600, delay: 1400 });
    walk.resolve();
    fade.resolve();
    await running;
    expect(layer.departing.size).toBe(0);
    expect(container.destroy).toHaveBeenCalledOnce();
    expect(host.killTweens).toHaveBeenCalledWith(tok);
    expect(host.killTweens).toHaveBeenCalledWith(container);
    const replacement = setup();
    const next = layer.enter(replacement.tok, { x: 0, y: 0 });
    fade.resolve();
    walk.resolve();
    await next;
    expect(container.destroy).toHaveBeenCalledOnce();
    expect(replacement.container.destroy).not.toHaveBeenCalled();
  });

  it("fades in place without inventing a path across unreachable ground", async () => {
    const { layer, tok, container, walker, fade } = setup(null);
    const running = layer.enter(tok, { x: 12, y: 34 });
    expect(container.setPosition).toHaveBeenCalledWith(12, 34);
    expect(walker.walkAlong).not.toHaveBeenCalled();
    fade.resolve();
    await running;
  });

  it("fades out without changing scale or walking when the viewpoint jumps", async () => {
    const { layer, tok, container, host, walker, fade } = setup();
    const running = layer.leave("guest", tok, "fade");
    expect(walker.presenceRoute).not.toHaveBeenCalled();
    expect(walker.walkAlong).not.toHaveBeenCalled();
    expect(host.tweenP).toHaveBeenCalledExactlyOnceWith({
      targets: tok, presenceAlpha: 0, duration: 600, delay: 0,
    });
    fade.resolve();
    await running;
    expect(container.destroy).toHaveBeenCalledOnce();
    expect(layer.departing.size).toBe(0);
  });
});
