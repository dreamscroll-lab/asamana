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

/** All unfinished short-term goals, regardless of origin, in queue order. */
export function unfinishedGoals(state: AgentPrivateView | undefined): string[] {
  return (state?.short_term_goals ?? [])
    .filter((g) => g.status === "active" || g.status === "interrupted")
    .map((g) => g.text);
}
