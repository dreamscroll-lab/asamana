import { describe, expect, it } from "vitest";

import { blankDraft, draftOf, missingForNew, newCharacterOf, revisionOf } from "./CharacterEditor";
import { chronological } from "./HistoryPanel";
import { draftProblem } from "./RelationPanel";
import type { AgentProfile, CharacterSetupView, Vocabulary } from "../../types";

const vocabulary: Vocabulary = {
  emotions: [{ value: "neutral", label: "平静" }, { value: "fear", label: "恐惧" }],
  needs: [{ value: "safety", label: "安全" }, { value: "esteem", label: "尊重" }],
};

const profile: AgentProfile = {
  agent_id: "a", name: "甲", role: "皇子", age: 28, gender: "男", background: "久经沙场",
  appearance: "", color: "#123456", core_traits: ["果决", "隐忍"], core_values: ["社稷"],
  self_image: "", life_goal: "", secret: "", is_main_character: true,
};

const setup: CharacterSetupView = {
  ...profile,
  agent_id: "a", is_main_character: true, initial_location: "palace",
  initial_emotion: { emotion: "neutral", intensity: 0.2, valence: 0 },
  long_term_goals: ["守住家业"], hard_constraints: [],
  needs: [{ type: "safety", label: "求安稳", intensity: 0.5, weight: 1, is_hidden: false }],
};

const before = draftOf(setup, vocabulary);

describe("revisionOf", () => {
  it("sends nothing for an untouched character", () => {
    expect(revisionOf(before, draftOf(setup, vocabulary))).toEqual({});
  });

  it("sends only the fields that changed", () => {
    const draft = { ...before, gender: "女", age: "30", core_traits: "果决、多疑", initial_location: "garden" };
    expect(revisionOf(before, draft)).toEqual({
      gender: "女", age: 30, core_traits: ["果决", "多疑"], initial_location: "garden",
    });
  });

  it("never sends a cleared age, which has no value to send", () => {
    expect(revisionOf(before, { ...before, age: "" })).toEqual({});
  });

  it("sends goals a line each, and the needs that have a label", () => {
    const needs = before.needs.map((n) => (n.type === "esteem" ? { ...n, weight: 1.5 } : n));
    const revision = revisionOf(before, { ...before, long_term_goals: "守住家业\n\n 寻回旧友 ", needs });
    expect(revision.long_term_goals).toEqual(["守住家业", "寻回旧友"]);
    expect(revision.needs?.map((n) => n.type)).toEqual(["safety"]);
  });
});

describe("a new character", () => {
  const blank = blankDraft(vocabulary);

  it("names what it still lacks", () => {
    expect(missingForNew(blank)).toContain("名字");
    expect(missingForNew({ ...blank, name: "丁", gender: "女", age: "40", role: "商人", initial_location: "garden" })).toBeNull();
  });

  it("always carries the required fields", () => {
    const draft = { ...blank, name: " 丁 ", gender: "女", age: "40", role: "商人", initial_location: "garden" };
    expect(newCharacterOf(draft, blank)).toEqual({
      name: "丁", role: "商人", gender: "女", age: 40, is_main_character: false, initial_location: "garden",
    });
  });
});

describe("chronological", () => {
  it("puts the earliest event first", () => {
    const events = [2, 10, 5].map((hours_before_start) => ({ event: "", hours_before_start, figure_ids: [], importance: 0.5 }));
    expect(chronological(events).map((e) => e.hours_before_start)).toEqual([10, 5, 2]);
  });
});

describe("draftProblem", () => {
  const relation = (source_id: string, target_id: string, labels = "", reverse_labels = "") => ({
    source_id, target_id, labels, reverse_labels,
  });

  it("accepts distinct pairs with labels in form", () => {
    expect(draftProblem([relation("a", "b", "父子：父亲、政敌", "父子:儿子"), relation("a", "c")])).toBeNull();
  });

  it.each([
    [[relation("a", "")]],
    [[relation("a", "a")]],
    [[relation("a", "b"), relation("b", "a")]],
    [[relation("a", "b", "父子:")]],
    [[relation("a", "b", "", ":父亲")]],
    [[relation("a", "b", "父子:父亲:长子")]],
    [[relation("a", "b", "父子：父亲、父子:父亲")]],
  ])("refuses %j", (drafts) => {
    expect(draftProblem(drafts)).not.toBeNull();
  });
});
