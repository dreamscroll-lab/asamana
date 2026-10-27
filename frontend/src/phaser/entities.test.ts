import type Phaser from "phaser";
import { describe, expect, it, vi } from "vitest";

import type { PlacedEntity } from "../lib/step";
import type { EntityHost } from "./entities";

vi.mock("phaser", () => ({ default: {} }));
vi.mock("./effects", () => ({ flyMessageArc: vi.fn() }));
vi.mock("./hud", () => ({ inkOn: vi.fn() }));

Object.assign(globalThis, { window: { devicePixelRatio: 2 } });
const { EntityLayer } = await import("./entities");
const { flyMessageArc } = await import("./effects");

function setup(entityType: string) {
  const floatNote = vi.fn();
  const host = {
    map: { rooms: new Map([["hall", { sx: 100, sy: 200 }]]) },
    floatNote,
    hideTip: vi.fn(),
    refreshCarried: vi.fn(),
    layoutFixtures: vi.fn(),
    tweens: { add: vi.fn() },
  } as unknown as EntityHost;
  const layer = new EntityLayer(host);
  const marker = {
    x: 0, y: 0,
    setData: vi.fn(),
    setVisible() { return this; },
    setScale() { return this; },
    setPosition(this: { x: number; y: number }, x: number, y: number) { this.x = x; this.y = y; return this; },
  } as unknown as Phaser.GameObjects.Container;
  layer.entityMarkers.set("letter", marker);
  layer.entitySlotOf.set("letter", { x: 420, y: 580, gx: 0, gy: 0 });
  vi.spyOn(layer, "layoutEntityLabel").mockImplementation(() => {});
  vi.spyOn(layer, "flashFixture").mockImplementation(() => {});
  const entity: PlacedEntity = {
    name: "Letter", entity_type: entityType, state: "sealed", description: "",
    is_public: true, created_step: 0, private: { content: "" },
    presence: "at_location", presence_ref: "hall",
  };
  return { layer, floatNote, entity };
}

describe("entity note positions", () => {
  it.each(["item", "landmark"])("anchors a deferred %s state change above its placed marker", (kind) => {
    const { layer, floatNote, entity } = setup(kind);
    layer.applyEntities(layer.beginStep({ letter: entity }, new Set(), new Set(), true));
    const held = layer.beginStep({ letter: { ...entity, state: "opened" } }, new Set(["letter"]), new Set(), false);
    layer.applyEntities(held);
    expect(floatNote).not.toHaveBeenCalled();

    layer.revealEntities(["letter"], new Set());

    expect(floatNote).toHaveBeenCalledExactlyOnceWith(420, 554, "Letter：sealed→opened", 0xd9a441);
  });

  it("anchors a newly revealed object above its placed marker", () => {
    const { layer, floatNote, entity } = setup("item");
    layer.applyEntities(layer.beginStep({ letter: entity }, new Set(["letter"]), new Set(["letter"]), false));
    expect(floatNote).not.toHaveBeenCalled();

    layer.revealEntities(["letter"], new Set(["letter"]));

    expect(floatNote).toHaveBeenCalledExactlyOnceWith(420, 554, "Letter 已出现", 0xe8c06a);
  });

  it("does not announce creation when a newly injected object is touched", () => {
    const { layer, floatNote, entity } = setup("item");
    const shown = layer.beginStep({ letter: { ...entity, created_step: 13 } }, new Set(["letter"]), new Set(), false);

    expect(shown.letter).toBeDefined();
    expect(layer.withheld.size).toBe(0);
    layer.applyEntities(shown);
    layer.revealWithheld();

    expect(floatNote).not.toHaveBeenCalled();
  });

  it("does not replay an arrival when the same creation beat is drawn again", () => {
    const { layer, floatNote, entity } = setup("item");
    for (let draw = 0; draw < 2; draw++) {
      layer.applyEntities(layer.beginStep({ letter: entity }, new Set(["letter"]), new Set(["letter"]), true));
      layer.revealEntities(["letter"], new Set(["letter"]));
    }

    expect(floatNote).toHaveBeenCalledTimes(1);
  });

  it("quietly reveals a product whose final state lands with a later unrelated deed", () => {
    const { layer, floatNote, entity } = setup("item");
    layer.applyEntities(layer.beginStep({ letter: entity }, new Set(["letter"]), new Set(["letter"]), false));

    layer.revealEntities(["letter"], new Set());

    expect(layer.withheld.size).toBe(0);
    expect(layer.effectiveEntities().letter).toEqual(entity);
    expect(floatNote).not.toHaveBeenCalled();
  });
});

describe("a thing changing hands", () => {
  it("flies as a gold star from the old holder to the new, never as a letter", () => {
    const { layer, entity } = setup("item");
    const tokens: Record<string, unknown> = {
      a: { container: { x: 10, y: 20 } },
      b: { container: { x: 70, y: 80 } },
    };
    Object.assign((layer as unknown as { host: object }).host, { token: (id: string) => tokens[id] });
    layer.entityMarkers.delete("letter"); // held things have no marker
    const held = (by: string): PlacedEntity => ({ ...entity, presence: "held", presence_ref: by });
    layer.applyEntities(layer.beginStep({ letter: held("a") }, new Set(), new Set(), true));
    layer.applyEntities(layer.beginStep({ letter: held("b") }, new Set(), new Set(), false));

    expect(flyMessageArc).toHaveBeenCalledExactlyOnceWith(
      expect.anything(), 10, 20, 70, 80, { color: 0xd9a441, glyph: "✦", ripple: 0xd9a441 },
    );
  });
});
