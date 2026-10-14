import type { AgentPrivateView } from "../types";

/**
 * The frontend's one definition of death in a step report.
 *
 * Both checks are needed: `vitality` is the source of truth, but `is_active` is flipped separately at
 * death (agent.py `_trigger_death`), so a report can carry one before the other.
 *
 * Says nothing about the drawn token: every step after a death still reports him dead, so conflating
 * "reported dead" with "what is drawn" brings corpses back to life.
 */
export function reportedDead(isActive: boolean, vitality: number | null): boolean {
  return vitality === 0 || isActive === false;
}

/**
 * The unfinished short-term queue split by origin (see GoalView): what they planned, and what an
 * action left them owing.
 */
export function unfinishedGoals(state: AgentPrivateView | undefined): { planned: string[]; owed: string[] } {
  const unfinished = (state?.short_term_goals ?? []).filter(
    (g) => g.status === "active" || g.status === "interrupted",
  );
  return {
    planned: unfinished.filter((g) => g.origin !== "residue").map((g) => g.text),
    owed: unfinished.filter((g) => g.origin === "residue").map((g) => g.text),
  };
}
