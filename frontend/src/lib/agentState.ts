import type { AgentStateSummary } from "../types";

/**
 * The frontend's one definition of death in a step report.
 *
 * Both checks are needed: `vitality` is the source of truth, but `is_active` is flipped separately at
 * death (agent.py `_trigger_death`), so a report can carry one before the other.
 *
 * Says nothing about the drawn token: every step after a death still reports him dead, so conflating
 * "reported dead" with "what is drawn" brings corpses back to life.
 */
export function reportedDead(state: AgentStateSummary): boolean {
  return state.vitality === 0 || state.is_active === false;
}

/**
 * The unfinished short-term queue split by origin (see ShortTermGoalEntity): what they planned,
 * and what an action left them owing. Snapshots without entities fall back to the flat text
 * list, read as all-planned.
 */
export function unfinishedGoals(state: AgentStateSummary | undefined): { planned: string[]; owed: string[] } {
  const unfinished = (state?.short_term_goal_entities ?? []).filter(
    (g) => g.status === "active" || g.status === "interrupted",
  );
  if (unfinished.length === 0) return { planned: state?.short_term_goals ?? [], owed: [] };
  return {
    planned: unfinished.filter((g) => g.origin !== "residue").map((g) => g.text),
    owed: unfinished.filter((g) => g.origin === "residue").map((g) => g.text),
  };
}
