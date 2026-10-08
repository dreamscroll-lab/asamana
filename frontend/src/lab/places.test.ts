/**
 * Casting the fixtures' roles out of a map.
 *
 * The map is synthetic: the frontend's build context is `frontend/` alone
 * (deploy/frontend.Dockerfile), so a test reading `worlds/templates` would break the image build.
 */

import { describe, expect, it } from "vitest";

import type { GroundGrids, MapSource } from "../phaser/mapSource";
import { resolvePlaces } from "./places";

const UNIT = 64; // tileheight: the unit of Tiled's object space on an isometric map

interface Spec {
  id: string;
  gx0: number;
  gy0: number;
  gw: number;
  gh: number;
  connections?: string[];
}

/** A map document carrying nothing but its locations — all `resolvePlaces` reads. */
function mapOf(specs: Spec[]): MapSource {
  const objects = specs.map((s, i) => ({
    id: i + 1,
    name: `地点${s.id}`,
    type: "location",
    x: s.gx0 * UNIT,
    y: s.gy0 * UNIT,
    width: s.gw * UNIT,
    height: s.gh * UNIT,
    properties: [
      { name: "location_id", type: "string", value: s.id },
      { name: "connections", type: "string", value: (s.connections ?? []).join(",") },
    ],
  }));
  return {
    // Nested in a group, as Tiled writes it and as both shipped templates carry it.
    doc: {
      tilewidth: 128,
      tileheight: UNIT,
      layers: [{ type: "group", layers: [{ type: "objectgroup", name: "place", objects }] }],
    },
    tilesets: [],
    ground: null,
    characters: null as never,
  };
}

/**
 * A hub with one clean north-east neighbor and one clean south-east one, plus a
 * north-east candidate that is nearer but drifts badly across the axis — the case the
 * rule exists for, since distance alone would pick the drifting one.
 */
const CITY: Spec[] = [
  { id: "hub", gx0: 9, gy0: 9, gw: 2, gh: 2 },
  { id: "clean-ne", gx0: 9, gy0: 3, gw: 2, gh: 2 },
  { id: "drifting-ne", gx0: 13, gy0: 4, gw: 2, gh: 2 },
  { id: "clean-se", gx0: 16, gy0: 9, gw: 2, gh: 2 },
  { id: "square", gx0: 20, gy0: 20, gw: 6, gh: 6 }, // the roomiest rectangle
  { id: "cell", gx0: 30, gy0: 4, gw: 1, gh: 1, connections: ["vault"] },
  { id: "vault", gx0: 32, gy0: 4, gw: 1, gh: 1, connections: ["cell"] },
  { id: "outpost", gx0: 40, gy0: 40, gw: 2, gh: 2 },
];

describe("resolvePlaces", () => {
  const places = resolvePlaces(mapOf(CITY))!;

  it("prefers the clean heading over the nearer one that drifts across the axis", () => {
    expect(places.pivot.id).toBe("hub");
    expect(places.ne.id).toBe("clean-ne");
    expect(places.se.id).toBe("clean-se");
  });

  it("sends the long trek somewhere genuinely far from both legs", () => {
    expect(places.far.id).toBe("outpost");
  });

  it("stages the deeds on the most open ground", () => {
    expect(places.open.id).toBe("square");
  });

  it("crowds a tight place the map's own graph joins to another", () => {
    expect(places.tight.id).toBe("cell");
    expect(places.next.id).toBe("vault");
  });

  it("reads openness off the standable grid when the backend serves one", () => {
    // The big rectangle is built over: same area, but no longer the roomiest by standable ground.
    const source = mapOf(CITY);
    const cols = 48;
    const rows = 48;
    const standable = Array.from({ length: rows }, (_, y) =>
      Array.from({ length: cols }, (_, x) => !(x >= 20 && x < 26 && y >= 20 && y < 26)),
    );
    const ground: GroundGrids = { cols, rows, walkable: standable, standable };
    const built = resolvePlaces({ ...source, ground })!;
    expect(built.open.id).not.toBe("square");
  });

  it("says so rather than guessing when the map cannot host the fixtures", () => {
    expect(resolvePlaces(mapOf(CITY.slice(0, 2)))).toBeNull();
  });
});
