import { describe, expect, it } from "vitest";

import { indexStep } from "../../lib/step";
import { buildScenes } from "./index";
import { SHI, fixtureCast, fixtureNpcs } from "./fixtures";
import { DEED_BODY } from "../../phaser/deedPose";
import type { LabPlaces } from "../places";

const room = (id: string) => ({ id, name: `地点-${id}` });
const PLACES: LabPlaces = {
  pivot: room("pivot"),
  ne: room("ne"),
  se: room("se"),
  far: room("far"),
  open: room("open"),
  tight: room("tight"),
  next: room("next"),
};

describe("lab scenes", () => {
  const scenes = buildScenes(PLACES);

  it("ids are unique", () => {
    // The id keys the checklist's marks and the address bar's deep link.
    const ids = scenes.map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("stages actor entrances and departures with no private route to follow", () => {
    const scene = scenes.find((s) => s.id === "actor-presence")!;
    expect(scene.viewer).toBeTruthy();
    const indexes = scene.steps.map(indexStep);
    expect(indexes.map((s) => Object.keys(s.agents).length)).toEqual([1, 2, 1, 2, 2]);
    const visitor = Object.values(indexes[1].agents).find((a) => a.agent_id !== scene.viewer)!;
    expect(visitor.private.transit).toBeNull();
    expect(visitor.is_main_character).toBeNull();
    expect(visitor.private.vitality).toBeNull();
    expect(indexes[3].agents[scene.viewer!].location_id).not.toBe(indexes[4].agents[scene.viewer!].location_id);
  });

  it("a scene's beats are consecutive steps", () => {
    // The map walks a figure only to the step right after the one it drew (TiledWorldScene's
    // drawnStep); any other is a cut. Beats out of sequence would turn every walk into a cut.
    for (const s of scenes) {
      const steps = s.steps.map((st) => st.step);
      expect(steps, s.id).toEqual(steps.map((_, i) => steps[0] + i));
    }
  });

  it("every scene has a title, something to watch for, and at least one beat", () => {
    for (const s of scenes) {
      expect(s.steps.length, s.id).toBeGreaterThan(0);
      expect(s.title.length, s.id).toBeGreaterThan(0);
      expect(s.watch.length, s.id).toBeGreaterThan(0);
    }
  });

  it("every figure staged has a registered gender and age", () => {
    // A character missing from the setCast table still draws, on the fallback body, so the bench
    // would quietly examine a figure no world produces.
    const known = fixtureCast();
    for (const s of scenes) {
      for (const st of s.steps) {
        for (const id of Object.keys(indexStep(st).agents)) {
          expect(known[id], `${s.id} stages ${id}`).toBeDefined();
        }
      }
    }
  });

  it("keeps NPC identity in a full roster and only dynamic state in step bodies", () => {
    const npcs = fixtureNpcs();
    expect(Object.keys(npcs)).toHaveLength(4);
    expect(npcs["npc_wang-er"].name).toBe("王二");
    expect(indexStep(scenes[0].steps[0]).npcs).toEqual([]);
    for (const scene of scenes) {
      for (const st of scene.steps) {
        for (const npc of indexStep(st).npcs) {
          expect(npcs[npc.npc_id], `${scene.id} stages ${npc.npc_id}`).toBeDefined();
          for (const field of ["name", "color", "gender", "age", "description"]) {
            expect(npc).not.toHaveProperty(field);
          }
        }
      }
    }
  });

  it("every deed the renderer can draw has a scene", () => {
    // The fixtures' premise: the deed table is the checklist. Nothing but this test catches a
    // new deed pose with no scene staging it.
    const staged = new Set(scenes.flatMap((s) => s.steps.flatMap((st) => indexStep(st).actions.map((a) => a.deed))));
    for (const deed of Object.keys(DEED_BODY)) {
      expect(staged.has(deed), `no scene stages deed "${deed}"`).toBe(true);
    }
  });

  it.each(["death", "death-offscreen", "death-on-arrival"])("dates %s's death to the step he falls on", (id) => {
    // The map draws a body only on its death step; a death re-dated each step would draw him again.
    const { steps } = scenes.find((s) => s.id === id)!;
    const shi = steps.map((st) => indexStep(st).agents[SHI.id]);
    const falls = steps.findIndex((_st, i) => shi[i]?.is_active === false);
    for (const [i, a] of shi.entries()) {
      expect(a?.death_step ?? null).toBe(i < falls ? null : steps[falls].step);
    }
  });
});
