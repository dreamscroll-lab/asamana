import type { SetupView, WorldGraph } from "../types";

/**
 * The relationship graph a world's setup describes, before the world is confirmed: nothing is
 * materialized yet, so there is no step whose relations the backend could serve. Each relation
 * of the setup is a pair, and the graph shows it as its two directed edges.
 */
export function setupGraph(setup: SetupView): WorldGraph {
  return {
    nodes: setup.characters.map((c) => ({
      id: c.agent_id,
      is_main_character: c.is_main_character,
    })),
    edges: setup.relations.flatMap((r) => [
      {
        from_id: r.source_id, to_id: r.target_id, trust: r.trust, affection: r.affection,
        labels: r.labels, interaction_count: 0, history_summary: "",
      },
      {
        from_id: r.target_id, to_id: r.source_id, trust: r.reverse_trust, affection: r.reverse_affection,
        labels: r.reverse_labels, interaction_count: 0, history_summary: "",
      },
    ]),
  };
}
