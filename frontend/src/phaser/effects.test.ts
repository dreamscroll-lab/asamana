import { describe, expect, it, vi } from "vitest";

import { ActionType, Deed, RefKind } from "../lib/contract";
import type { Beat } from "./figures";
import type { FxHost } from "./effects";
import type { AgentToken } from "./types";

vi.mock("phaser", () => ({ default: {} }));

Object.assign(globalThis, { window: { devicePixelRatio: 2 } });

const { physicalEffect } = await import("./effects");
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
  const tokens = new Map([["doer", token(0, 0)], ["holder", token(100, 50)]]);
  const host = {
    add: {
      circle: (x: number, y: number) => { circles.push({ x, y }); return chain(); },
      rectangle: () => chain(),
      graphics: () => chain(),
    },
    tweens: { add: vi.fn() },
    pushEphemeral: vi.fn(),
    delayP: () => Promise.resolve(),
    token: (id: string) => tokens.get(id),
    entityMarkers: new Map(Object.entries(markers).map(([id, p]) => [id, gameObject(p.x, p.y)])),
    bearing: () => "s",
    reface: vi.fn(),
  } as unknown as FxHost;
  return { host, circles, doer: tokens.get("doer")! };
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

  it("draws no seize gleam from the holder: the hand-off arc already shows the thing changing hands", () => {
    const { host, circles, doer } = setup({});
    physicalEffect(host, doer, seizeHeld(true));
    expect(circles).toEqual([]);
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
