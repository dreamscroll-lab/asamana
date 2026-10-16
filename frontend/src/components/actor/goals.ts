import type { ActorAmendments, ActorMind, BriefGoal, GoalAmendment } from "../../types";

export const GOAL_STATUS: Record<BriefGoal["status"], string> = {
  active: "进行中", interrupted: "受阻", completed: "已完成", failed: "已放下 / 未完成",
};

export function liveGoal(g: BriefGoal): boolean {
  return g.status === "active" || g.status === "interrupted";
}

export interface GoalRow extends GoalAmendment {
  detail: BriefGoal | null;
}

/** Saved amendments overlay the live queue; history stays separate and read-only. */
export function goalRows(mind: ActorMind, draft?: GoalAmendment[] | null, removedIds: string[] = []): GoalRow[] {
  const live = mind.short_term_goals.filter(liveGoal);
  const values = draft ?? mind.pending?.short_term_goals ?? [];
  const edits = new Map(values.filter((g) => g.id !== null).map((g) => [g.id, g.text]));
  const removed = new Set([...(mind.pending?.removed_short_term_goal_ids ?? []), ...removedIds]);
  return [
    ...live.filter((g) => !removed.has(g.id)).map((g) => ({
      id: g.id, text: edits.get(g.id) ?? g.text, detail: g,
    })),
    ...values.filter((g) => g.id === null).map((g) => ({ ...g, detail: null })),
  ];
}

export function shortTermAmendments(
  mind: ActorMind, draft: GoalAmendment[] | null, removedIds: string[],
): Pick<ActorAmendments, "short_term_goals" | "removed_short_term_goal_ids"> {
  if (draft === null) return { short_term_goals: null, removed_short_term_goal_ids: [] };
  const liveIds = new Set(mind.short_term_goals.filter(liveGoal).map((g) => g.id));
  return {
    short_term_goals: goalRows(mind, draft, removedIds)
      .map((g) => ({ id: g.id, text: g.text.trim() })).filter((g) => g.text),
    removed_short_term_goal_ids: [...new Set([
      ...(mind.pending?.removed_short_term_goal_ids ?? []), ...removedIds,
    ])].filter((id) => liveIds.has(id)),
  };
}

/** Time passing changes hints, but does not itself change a goal. */
export function goalsChanged(before: ActorMind, after: ActorMind): boolean {
  if (before.long_term_goals.length !== after.long_term_goals.length ||
      before.long_term_goals.some((g, i) => g !== after.long_term_goals[i])) return true;
  if (before.short_term_goals.length !== after.short_term_goals.length) return true;
  return before.short_term_goals.some((g, i) => {
    const next = after.short_term_goals[i];
    return g.id !== next.id || g.text !== next.text || g.status !== next.status ||
      g.progress_summary !== next.progress_summary || g.due_step !== next.due_step || g.origin !== next.origin;
  });
}

/** A display order, not a judgment of the character's priorities. */
export function goalChoices(mind: ActorMind): GoalRow[] {
  return goalRows(mind).filter((g) => g.text.trim()).sort((a, b) => {
    const rank = (g: GoalRow) => g.detail?.attention && g.detail.due_step !== null
      ? 0 : g.detail?.origin === "residue" ? 1 : 2;
    const category = rank(a) - rank(b);
    if (category) return category;
    if (rank(a) === 0) return a.detail!.due_step! - b.detail!.due_step!;
    return (a.detail?.created_step ?? Infinity) - (b.detail?.created_step ?? Infinity);
  }).slice(0, 5);
}
