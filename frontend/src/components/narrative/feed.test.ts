import { describe, expect, it } from "vitest";

import { stepCast } from "./cast";
import { identityColor, toHex } from "../../lib/color";
import { indexStep } from "../../lib/step";
import type { Cast } from "../../lib/cast";
import type { Places } from "../../lib/places";
import { bornInStep, buildFeed } from "./feed";
import type {
  ActionSummary, AgentView, EntityView, LocationScope, MessageSummary, ObservationSummary, StepView,
} from "../../types";

const thing = (createdStep: number): EntityView => ({
  name: "横刀",
  entity_type: "item",
  state: "intact",
  description: "",
  is_public: true,
  created_step: createdStep,
  private: { content: "" },
});

const place = (location_id: string, over: Partial<LocationScope> = {}): LocationScope => ({
  location_id, agents: [], npcs: [], entities: {}, events: [], ...over,
});

const at = (step: number, locations: LocationScope[]): StepView => ({
  world_id: "w",
  step,
  world_time: { label: "", hour: null, minute: null },
  locations: Object.fromEntries(locations.map((l) => [l.location_id, l])),
  transit: { agents: [], npcs: [], events: [] },
  world: { events: [] },
  observer: { injections: [] },
});

const agent = (agent_id: string, thoughts: AgentView["private"]["thoughts"] = []): AgentView => ({
  agent_id, name: agent_id, gender: "", age: null, is_main_character: false, is_active: true,
  public: { activity_status: "", activity_label: "", condition: "", displaced: false, carrying: {} },
  private: {
    emotion: "", emotion_label: "", emotion_intensity: null, emotion_valence: null, dominant_need: null,
    dominant_need_label: "", vitality: 1, activity_status: "", activity_label: "", transit: null,
    long_term_goals: [], short_term_goals: [],
    relations: [], thoughts, carrying: {},
  },
});

const act = (seq: number, over: Partial<ActionSummary> = {}): ActionSummary => ({
  kind: "action", seq, execution_id: `x${seq}`, initiator_id: "a", participant_ids: ["a"], action_type: "move",
  phase: "settled", deed: "move", target: { acts_on: [], claims: [], reaches: [] },
  affected_entity_ids: [], overheard_by: [], interrupted_by: "", succeeded: true, failure_reason: "",
  detected: false, not_executed: false, public: null,
  private: {
    audience: ["a"], outcome: "", gist: "", dialogue: [], action_description: "", duration_label: "",
    elapsed_steps: 0, total_steps: 0, route: null,
  },
  ...over,
});

const letter = (receiver: string): MessageSummary => ({
  kind: "message", seq: 3, message_id: "m1", sender_id: "s", sender_name: "某人", receiver_ids: [receiver],
  scope: "direct", content: "来", spoken: "来", audience: [receiver],
});

describe("bornInStep", () => {
  it("只认出生步等于本步的物件", () => {
    const born = bornInStep(at(41, [place("loc-1", { entities: { seed: thing(0), made: thing(41) } })]));
    expect([...born]).toEqual(["made"]);
  });

  it("叙事流最早的一步不会把世界原有之物算成新造的", () => {
    // This is the edge of the 60-step sliding window: there's no step 40 to compare against.
    // Diffing against the previous step would mark every entity in this step's table as made
    // this beat, so an action touching an age-old knife would be shown as having made it.
    expect([...bornInStep(at(41, [place("loc-1", { entities: { seed: thing(0), old: thing(7) } })]))]).toEqual([]);
  });

  it("造物的下一步它就是寻常物件", () => {
    expect([...bornInStep(at(42, [place("loc-1", { entities: { made: thing(41) } })]))]).toEqual([]);
  });
});

describe("buildFeed", () => {
  it("一件事只讲一次:别处的目击不另起一行", () => {
    const seen: ObservationSummary = {
      kind: "observation", seq: 5, initiator_id: "a", participant_ids: ["a"], text: "甲离开了", strength: null,
    };
    const feed = buildFeed(at(1, [
      place("loc-gate", { events: [seen] }),
      place("loc-hall", { events: [act(5)] }),
    ]));
    expect(feed.map((i) => i.kind)).toEqual(["action"]);
  });

  it("心声跟着它促成的那件事", () => {
    const feed = buildFeed(at(1, [place("loc-hall", {
      agents: [agent("a", [{ seq: 5, execution_id: "x5", text: "我得走" }])],
      events: [act(5)],
    })]));
    const [item] = feed;
    expect(item.kind === "action" && item.thoughts).toEqual([{ agent_id: "a", text: "我得走" }]);
  });

  it("角色视角:事发在视野外时,所见的目击自成一行", () => {
    const seen: ObservationSummary = {
      kind: "observation", seq: 5, initiator_id: "a", participant_ids: ["a"], text: "甲离开了", strength: null,
    };
    const feed = buildFeed(at(1, [place("loc-gate", { events: [seen] })]), "me");
    expect(feed.map((i) => i.kind)).toEqual(["observation"]);
  });

  it("角色视角:不在其中的事只算旁观,身在其中的才算亲历", () => {
    const outside = act(5, { public: { text: "甲在打点行装", strength: null }, private: { ...act(5).private, audience: [] } });
    const mine = act(6, { private: { ...act(6).private, audience: ["a", "me"] } });
    const feed = buildFeed(at(1, [place("loc-hall", { events: [outside, mine] })]), "me");
    expect(feed.map((i) => i.kind === "action" && i.inside)).toEqual([false, true]);
  });

  it("导演视角什么都算亲历", () => {
    const feed = buildFeed(at(1, [place("loc-hall", { events: [act(5)] })]));
    expect(feed[0].kind === "action" && feed[0].inside).toBe(true);
  });

  it("收信人分在两处的一封信仍是一行", () => {
    const feed = buildFeed(at(1, [
      place("loc-gate", { events: [letter("b")] }),
      place("loc-hall", { events: [letter("c")] }),
    ]));
    expect(feed).toHaveLength(1);
    expect(feed[0].kind === "message" && feed[0].data.receiver_ids.sort()).toEqual(["b", "c"]);
  });
});


describe("stepCast public identity", () => {
  const cast: Cast = {
    absent: { agent_id: "absent", name: "Public name", gender: "", age: null, role: "", background: "", appearance: "", color: "#336699" },
    malformed: { agent_id: "malformed", name: "Other", gender: "", age: null, role: "", background: "", appearance: "", color: "invalid" },
  };
  const places: Places = { here: { location_id: "here", name: "大殿", description: "" } };

  it("resolves identities in dialogue even when their bodies are absent from the step", () => {
    const current = at(1, [place("here", { agents: [agent("present")] })]);
    const lookup = stepCast(current, cast, places);
    expect(lookup.nameOf("absent")).toBe("Public name");
    expect(lookup.inkOf("absent")).toBe("#336699");
    expect(Object.keys(indexStep(current).agents)).toEqual(["present"]);
    expect(lookup.isMain("absent")).toBe(false);
    expect(lookup.conditionOf("absent")).toBe("");
  });

  it("names a scope from the world's place table, never by its id", () => {
    const lookup = stepCast(at(1, []), cast, places);
    expect([lookup.placeOf("here"), lookup.placeOf("gone"), lookup.placeOf(""), lookup.placeOf(null)])
      .toEqual(["大殿", "某地", "途中", ""]);
  });

  it("uses the shared identity fallback for missing or malformed colors", () => {
    const lookup = stepCast(at(1, []), cast, places);
    for (const id of ["missing", "malformed"]) {
      expect(lookup.inkOf(id)).toBe(toHex(identityColor(cast[id]?.color, id)));
    }
  });
});
