import { describe, expect, it, vi } from "vitest";

import { ActionType, Deed, RefKind } from "../lib/contract";
import type { Beat } from "./figures";
import type { FxHost } from "./effects";
import type { AgentToken } from "./types";

vi.mock("phaser", () => ({ default: {} }));

Object.assign(globalThis, { window: { devicePixelRatio: 2 } });

const { handOff, physicalEffect } = await import("./effects");
const { subjectPeopleIds } = await import("./staging");

// A game object that answers every builder call with itself.
function gameObject(x: number, y: number): Record<string, unknown> {
  const self: Record<string, unknown> = new Proxy({ x, y, visible: true } as Record<string, unknown>, {
    get: (target, key: string) => (key in target ? target[key] : () => self),
  });
  return self;
}

function token(x: number, y: number): AgentToken {
  return { container: { x, y }, sprite: { x: 0, y: 0 }, dir: "s" } as unknown as AgentToken;
}

function setup(markers: Record<string, { x: number; y: number }>) {
  const circles: { x: number; y: number }[] = [];
  const texts: { x: number; y: number }[] = [];
  const tokens = new Map([["doer", token(0, 0)], ["holder", token(100, 50)]]);
  const host = {
    add: {
      circle: (x: number, y: number) => { circles.push({ x, y }); return chain(); },
      rectangle: () => chain(),
      graphics: () => chain(),
      text: (x: number, y: number) => { texts.push({ x, y }); return chain(); },
    },
    tweens: { add: vi.fn() },
    pushEphemeral: vi.fn(),
    addCanceler: vi.fn(),
    removeCanceler: vi.fn(),
    delayP: () => Promise.resolve(),
    token: (id: string) => tokens.get(id),
    entityMarkers: new Map(Object.entries(markers).map(([id, p]) => [id, gameObject(p.x, p.y)])),
    bearing: () => "s",
    reface: vi.fn(),
  } as unknown as FxHost;
  return { host, circles, texts, doer: tokens.get("doer")! };
}

function chain(): unknown {
  const self: unknown = new Proxy({}, { get: () => () => self });
  return self;
}

const seizeHeld = (succeeded: boolean | null): Beat => ({
  action_type: ActionType.physical,
  deed: Deed.seize,
  agent_id: "doer",
  target: {
    acts_on: [{ kind: RefKind.item, id: "edict" }],
    claims: [],
    reaches: [{ kind: RefKind.agent, id: "holder" }],
  },
  affected_entity_ids: succeeded ? ["edict"] : [],
  shared: succeeded === null ? null : { succeeded },
} as unknown as Beat);

describe("an act on a thing someone holds", () => {
  it("is about its holder: the doer turns toward and stands beside him", () => {
    expect(subjectPeopleIds(seizeHeld(true))).toEqual(["holder"]);
  });

  it("flies the thing from the holder to the doer instead of gleaming it off the holder", () => {
    const { host, circles, texts, doer } = setup({});
    physicalEffect(host, doer, seizeHeld(true));
    expect(circles).toEqual([]);
    expect(texts).toEqual([{ x: 100, y: 50 }]); // the star sets off from the holder
  });

  it("still draws from the marker when the thing has one", () => {
    const { host, circles, doer } = setup({ edict: { x: 30, y: 40 } });
    physicalEffect(host, doer, seizeHeld(true));
    expect(circles[0]).toEqual({ x: 30, y: 40 });
  });

  it("rings a failed snatch off the holder", () => {
    const { host, circles, doer } = setup({});
    physicalEffect(host, doer, seizeHeld(false));
    expect(circles).toContainEqual({ x: 100, y: 48 });
  });
});

const relinquish = (over: Partial<Beat> = {}): Beat => ({
  action_type: ActionType.physical,
  deed: Deed.relinquish,
  agent_id: "doer",
  target: {
    acts_on: [{ kind: RefKind.item, id: "note" }],
    claims: [],
    reaches: [{ kind: RefKind.agent, id: "holder" }],
  },
  affected_entity_ids: ["note"],
  shared: { succeeded: true },
  ...over,
} as unknown as Beat);

describe("a thing handed over", () => {
  it("flies from the doer to the one given it, from the act alone: the view may not show it in his hands", () => {
    const { host, texts, doer } = setup({});
    physicalEffect(host, doer, relinquish());
    expect(texts).toEqual([{ x: 0, y: 0 }]);
  });

  it("also flies for an onlooker, who sees the hand-off but not how the act was judged", () => {
    const { host, texts, doer } = setup({});
    physicalEffect(host, doer, relinquish({ shared: null }));
    expect(texts).toEqual([{ x: 0, y: 0 }]);
  });

  it("names the hands for each direction and nothing for a thing merely set down", () => {
    expect(handOff(relinquish())).toEqual({ from: "doer", to: "holder" });
    expect(handOff(seizeHeld(true))).toEqual({ from: "holder", to: "doer" });
    expect(handOff(seizeHeld(false))).toBeNull();
    expect(handOff(relinquish({ target: { acts_on: [{ kind: RefKind.item, id: "note" }], claims: [], reaches: [] } }))).toBeNull();
    expect(handOff(relinquish({ deed: Deed.operate }))).toBeNull();
  });
});
