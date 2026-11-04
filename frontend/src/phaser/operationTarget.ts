import type { ActorTarget } from "../types";

export interface OperationHit {
  entityId?: string;
  entityKind?: string;
  agentId?: string;
  pocketPanel?: boolean;
}

export function operationTarget(
  hits: OperationHit[], locationId: string | null,
  canSelect: (id: string) => boolean, isNpc: (id: string) => boolean,
): ActorTarget | null {
  // Held-item rows overlap both the pocket panel and its owner's token.
  const entity = hits.find((hit) => hit.entityId);
  if (entity?.entityId) return { kind: entity.entityKind === "landmark" ? "landmark" : "item", id: entity.entityId };
  if (hits.some((hit) => hit.pocketPanel)) return null;
  const person = hits.find((hit) => hit.agentId && canSelect(hit.agentId));
  if (person?.agentId) return { kind: isNpc(person.agentId) ? "npc" : "agent", id: person.agentId };
  return locationId ? { kind: "location", id: locationId } : null;
}
