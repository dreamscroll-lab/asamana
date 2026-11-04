import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { ActorMind, BriefGoal } from "../../types";
import GoalPanel from "./GoalPanel";
import { goalChoices, goalRows, goalsChanged, shortTermAmendments } from "./goals";

const goal = (id: string, over: Partial<BriefGoal> = {}): BriefGoal => ({
  id, text: id, status: "active", origin: "cognitive", created_step: 0, due_step: null,
  progress_summary: "", time_hint: "", attention: false, ...over,
});
const mind = (goals: BriefGoal[]): ActorMind => ({
  emotion: "calm", emotion_intensity: 0.2, emotion_valence: 0,
  long_term_goals: [], short_term_goals: goals, relations: [], pending: null,
});

describe("actor goal choices", () => {
  it("selects at most five, putting due agreements before residue and ordinary goals", () => {
    const m = mind([
      goal("ordinary", { created_step: 1 }),
      goal("future", { due_step: 20 }),
      goal("owed", { origin: "residue" }),
      goal("due", { due_step: 5, attention: true }),
      goal("late", { due_step: 2, attention: true }),
      goal("old", { attention: true }),
      goal("done", { status: "completed" }),
      goal("failed", { status: "failed" }),
    ]);
    expect(goalChoices(m).map((g) => g.id)).toEqual(["late", "due", "owed", "future", "old"]);
  });

  it("keeps interrupted goals selectable and returns no choices for terminal goals", () => {
    expect(goalChoices(mind([goal("blocked", { status: "interrupted" })]))[0].id).toBe("blocked");
    expect(goalChoices(mind([goal("done", { status: "completed" })]))).toEqual([]);
  });

  it("uses saved text and deletions while preserving metadata by id, including duplicate texts", () => {
    const m = mind([
      goal("a", { text: "相同目标", origin: "residue", due_step: 5 }),
      goal("b", { text: "相同目标" }),
      goal("removed"),
    ]);
    m.pending = {
      emotion: null, relations: [], long_term_goals: null, insights: [],
      removed_short_term_goal_ids: ["removed"],
      short_term_goals: [{ id: "a", text: "改后的目标" }, { id: "b", text: "相同目标" }, { id: null, text: "新增" }],
    };
    const rows = goalRows(m);
    expect(rows.map((g) => g.text)).toEqual(["改后的目标", "相同目标", "新增"]);
    expect(rows[0].detail?.origin).toBe("residue");
    expect(rows[0].detail?.due_step).toBe(5);
    expect(rows[2].id).toBeNull();
    expect(rows[2].detail).toBeNull();
    expect(goalRows(m, [], ["a", "b"])).toEqual([]);
  });

  it("rebases pending edits on the live queue without resurrecting closed goals or hiding new goals", () => {
    const m = mind([
      goal("finished", { status: "completed" }), goal("fresh", { origin: "residue" }), goal("kept"),
    ]);
    m.pending = {
      emotion: null, relations: [], long_term_goals: null, insights: [], removed_short_term_goal_ids: [],
      short_term_goals: [{ id: "finished", text: "不能复活" }, { id: "kept", text: "已保存文本" }],
    };
    expect(goalRows(m).map((g) => g.text)).toEqual(["fresh", "已保存文本"]);
    expect(goalChoices(m).map((g) => g.id)).toEqual(["fresh", "kept"]);
    expect(goalRows(m, [{ id: "kept", text: "草稿" }]).map((g) => g.text)).toEqual(["fresh", "草稿"]);
  });

  it("distinguishes changed goal state from elapsed-time hints", () => {
    const before = mind([goal("a", { time_hint: "约定在约一小时之后" })]);
    const after = mind([goal("a", { time_hint: "约定就在此刻", attention: true })]);
    expect(goalsChanged(before, after)).toBe(false);
    after.short_term_goals[0].progress_summary = "已经抵达";
    expect(goalsChanged(before, after)).toBe(true);
    expect(goalsChanged(before, mind([goal("a", { status: "completed" })]))).toBe(true);
  });

  it("saves only live edits and explicit live removals after feedback changes the queue", () => {
    const m = mind([goal("finished", { status: "completed" }), goal("kept"), goal("drop"), goal("new")]);
    const edits = shortTermAmendments(m, [
      { id: "finished", text: "过时的编辑" }, { id: "kept", text: "  修改  " }, { id: null, text: "新增" },
    ], ["finished", "drop"]);
    expect(edits).toEqual({
      short_term_goals: [{ id: "kept", text: "修改" }, { id: "new", text: "new" }, { id: null, text: "新增" }],
      removed_short_term_goal_ids: ["drop"],
    });
  });
});

describe("actor goal panel", () => {
  it("renders full text, due reminders, progress, residue, and read-only recent history", () => {
    const text = "完整目标内容".repeat(60);
    const m = mind([
      goal("owed", { text, origin: "residue", attention: true, due_step: 2,
        time_hint: "约定时刻已过约两小时", progress_summary: "尚未收到回信" }),
      goal("old", { attention: true, time_hint: "已历约一天仍未了结" }),
      goal("done", { text: "已经送出的信", status: "completed" }),
    ]);
    const html = renderToStaticMarkup(<GoalPanel mind={m} draft={null} removedIds={[]} onChange={() => {}} />);
    expect(html).toContain(text);
    for (const label of ["短期目标", "约定时刻已过约两小时", "尚未收到回信", "该目标已长期未完成，是否继续？", "近期记录", "已经送出的信", "已完成"])
      expect(html).toContain(label);
    expect(html).not.toContain("待办");
    expect(html).not.toContain("<textarea");
    expect(html).not.toContain("<select");
  });
});
