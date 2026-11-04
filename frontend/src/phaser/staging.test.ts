/**
 * Where the resting layout seats people. A room drawn mostly as roof keeps its open ground on
 * one side, while the street or the next place over can be nearer the spot the force sim
 * wanted. Ground outside the room is only for when the room's own is full.
 */

import { describe, expect, it, vi } from "vitest";

import type { Figure } from "./figures";
import type { Room } from "./worldMap";

vi.mock("phaser", () => ({ default: {
  Math: {
    Clamp: (n: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, n)),
    Distance: {
      BetweenPoints: (a: { x: number; y: number }, b: { x: number; y: number }) => Math.hypot(b.x - a.x, b.y - a.y),
    },
  },
} }));
Object.assign(globalThis, { window: { devicePixelRatio: 2 } });
const { Staging } = await import("./staging");

// Screen space equals grid space here, so a slot reads back as the cell it sits on.
const iso = (gx: number, gy: number) => ({ x: gx, y: gy });

// A hall over y4-9: open ground only in its south-east corner. The cells north of it, just
// outside its rectangle, lie nearer the spots the sim wants for a crowd.
const hall: Room = {
  id: "hall", name: "hall", gx0: 0, gy0: 4, gw: 6, gh: 6, gx: 3, gy: 7,
  sx: 0, sy: 0, atx: 0, aty: 0,
  stand: [
    { x: 5, y: 9, rank: 0 }, { x: 4, y: 9, rank: 0 }, { x: 5, y: 8, rank: 0 },
    { x: 1, y: 3, rank: 1 }, { x: 2, y: 3, rank: 1 }, { x: 0, y: 3, rank: 1 },
  ],
  inRect: 3,
};

function figure(id: string): Figure {
  return {
    agent_id: id, agent_name: id, location_id: "hall", color: "", is_main_character: false,
    is_active: true, death_step: null, condition: "", transit: null, arrival: null,
    displaced: false, vitality: 1, emotion_valence: null,
  };
}

function seat(ids: string[]): Map<string, { x: number; y: number }> {
  const staging = new Staging(
    { map: { rooms: new Map([["hall", hall]]), iso, tileW: 2, tileH: 1 } as never, token: () => undefined, gone: () => false },
    { entitySlotOf: new Map() } as never,
  );
  const states = ids.map(figure);
  staging.layoutRooms(states, [], {});
  return new Map(ids.map((id) => [id, staging.positionFor(states.find((s) => s.agent_id === id)!, states)!]));
}

const inHall = (p: { x: number; y: number }): boolean =>
  p.x - 0.5 >= hall.gx0 && p.x - 0.5 < hall.gx0 + hall.gw && p.y - 0.5 >= hall.gy0 && p.y - 0.5 < hall.gy0 + hall.gh;

describe("assignSlots", () => {
  it("把人安在本地点矩形内的地面上,哪怕矩形外有离想要位置更近的空地", () => {
    const slots = seat(["a", "b"]);
    expect([...slots.values()].every(inHall)).toBe(true);
  });

  it("本地点的地面坐满了,才用矩形外的地面", () => {
    const slots = seat(["a", "b", "c", "d"]);
    expect([...slots.values()].filter(inHall)).toHaveLength(3);
  });

  it("可达的地优先于不可达的地,哪怕后者在本地点矩形内", () => {
    // A gate whose own open cells are walled off, with reachable ground only in the next place.
    const own = hall.stand;
    hall.stand = [
      { x: 1, y: 3, rank: 2 }, { x: 2, y: 3, rank: 2 },
      { x: 2, y: 7, rank: 3 }, { x: 3, y: 7, rank: 3 },
    ];
    try {
      const slots = seat(["a", "b"]);
      expect([...slots.values()].every((p) => p.y === 3.5)).toBe(true);
    } finally {
      hall.stand = own;
    }
  });
});
