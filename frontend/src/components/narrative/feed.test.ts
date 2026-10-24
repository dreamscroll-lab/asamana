import { describe, expect, it } from "vitest";

import { stepCast } from "./cast";
import { identityColor, toHex } from "../../lib/color";
import { indexStep } from "../../lib/step";
import type { Npcs } from "../../lib/npcs";
import type { Cast } from "../../lib/cast";
import type { Places } from "../../lib/places";
import { buildFeed } from "./feed";
import type {
  ActionSummary, AgentView, LocationScope, MessageSummary, ObservationSummary, StepView,
} from "../../types";

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

const agent = (agent_id: string): AgentView => ({
  agent_id, is_main_character: false, is_active: true,
  public: { activity_status: "", activity_label: "", condition: "", displaced: false, carrying: {} },
  private: {
    emotion: "", emotion_label: "", emotion_intensity: null, emotion_valence: null, dominant_need: null,
    dominant_need_label: "", vitality: 1, activity_status: "", activity_label: "", transit: null,
    long_term_goals: [], short_term_goals: [],
    relations: [], carrying: {},
  },
});

const act = (seq: number, over: Partial<ActionSummary> = {}): ActionSummary => ({
  kind: "action", seq, execution_id: `x${seq}`, initiator_id: "a", participant_ids: ["a"], action_type: "move",
  phase: "settled", deed: "move", target: { acts_on: [], claims: [], reaches: [] },
  affected_entity_ids: [], created_entity_ids: [], overheard_by: [], interrupted_by: "",
  detected: false, not_executed: false, public: null,
  shared: {
    audience: ["a"], outcome: "", gist: "", succeeded: true, failure_reason: "", dialogue: [],
    talk_topic: "", duration_label: "", elapsed_steps: 0, total_steps: 0, route: null,
  },
  private: [],
  ...over,
});

const letter = (receiver: string): MessageSummary => ({
  kind: "message", seq: 3, message_id: "m1", sender_id: "s", sender_name: "某人", receiver_ids: [receiver],
  scope: "direct", content: "来", spoken: "来", audience: [receiver],
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

  it("导演读得到每个参与者自己的那一份", () => {
    const parts = [
      { agent_id: "a", action_description: "甲的打算", inner_monologue: "我得走" },
      { agent_id: "b", action_description: "", inner_monologue: "" },
    ];
    const [item] = buildFeed(at(1, [place("loc-hall", { events: [act(5, { private: parts })] })]));
    expect(item.kind === "action" && [item.action_description, item.private_parts]).toEqual(["甲的打算", parts]);
  });

  it("角色视角:同一件事里只读得到自己的心声和打算", () => {
    const joint = act(5, {
      participant_ids: ["a", "me"],
      shared: { ...act(5).shared!, audience: ["a", "me"] },
      private: [
        { agent_id: "a", action_description: "甲的打算", inner_monologue: "甲的念头" },
        { agent_id: "me", action_description: "", inner_monologue: "我的念头" },
      ],
    });
    const [item] = buildFeed(at(1, [place("loc-hall", { events: [joint] })]), "me");
    expect(item.kind === "action" && [item.action_description, item.private_parts.map((t) => t.inner_monologue)])
      .toEqual(["", ["我的念头"]]);
  });

  it("角色视角:事发在视野外时,所见的目击自成一行", () => {
    const seen: ObservationSummary = {
      kind: "observation", seq: 5, initiator_id: "a", participant_ids: ["a"], text: "甲离开了", strength: null,
    };
    const feed = buildFeed(at(1, [place("loc-gate", { events: [seen] })]), "me");
    expect(feed.map((i) => i.kind)).toEqual(["observation"]);
  });

  it("角色视角:不在其中的事只算旁观,身在其中的才算亲历", () => {
    const outside = act(5, { public: { text: "甲在打点行装", strength: null }, shared: null });
    const mine = act(6, { shared: { ...act(6).shared!, audience: ["a", "me"] } });
    const feed = buildFeed(at(1, [place("loc-hall", { events: [outside, mine] })]), "me");
    expect(feed.map((i) => i.kind === "action" && i.shared !== null)).toEqual([false, true]);
  });

  it("导演视角什么都算亲历", () => {
    const feed = buildFeed(at(1, [place("loc-hall", { events: [act(5)] })]));
    expect(feed[0].kind === "action" && feed[0].shared !== null).toBe(true);
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
  const npcs: Npcs = {
    runner: { npc_id: "runner", name: "王二", gender: "男", age: 34, description: "跑腿", color: "#8a93a6" },
  };
  const places: Places = { here: { location_id: "here", name: "大殿", description: "" } };

  it("resolves identities in dialogue even when their bodies are absent from the step", () => {
    const current = at(1, [place("here", { agents: [agent("present")] })]);
    const lookup = stepCast(current, cast, places, npcs);
    expect(lookup.nameOf("absent")).toBe("Public name");
    expect(lookup.inkOf("absent")).toBe("#336699");
    expect(Object.keys(indexStep(current).agents)).toEqual(["present"]);
    expect(lookup.isMain("absent")).toBe(false);
    expect(lookup.conditionOf("absent")).toBe("");
  });

  it("resolves NPC names and colors when their bodies are absent from the current step", () => {
    const current = at(1, []);
    const lookup = stepCast(current, cast, places, npcs);
    expect(indexStep(current).npcs).toEqual([]);
    expect(lookup.nameOf("runner")).toBe("王二");
    expect(lookup.inkOf("runner")).toBe("#8a93a6");
    expect(lookup.isMain("runner")).toBe(false);
  });

  it("names a scope from the world's place table, never by its id", () => {
    const lookup = stepCast(at(1, []), cast, places, npcs);
    expect([lookup.placeOf("here"), lookup.placeOf("gone"), lookup.placeOf(""), lookup.placeOf(null)])
      .toEqual(["大殿", "某地", "途中", ""]);
  });

  it("uses the shared identity fallback for missing or malformed colors", () => {
    const lookup = stepCast(at(1, []), cast, places, npcs);
    for (const id of ["missing", "malformed"]) {
      expect(lookup.inkOf(id)).toBe(toHex(identityColor(cast[id]?.color, id)));
    }
  });
});
