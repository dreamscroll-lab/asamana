// REST client for the Asamana backend. Paths are relative so the Vite dev
// proxy (dev) or the backend static mount (prod) both work without config.

import type {
  ActorIntent,
  ActorIntentResult,
  ActorOperations,
  ActorTarget,
  ActorAmendResult,
  ActorAmendments,
  ActorResult,
  ActorPublicAnswer,
  ActorEntry,
  ActorMind,
  ActorPace,
  AgentProfile,
  CharacterIdentity,
  NpcIdentity,
  CharacterRevision,
  NewCharacter,
  Vocabulary,
  SetupHistoricalEventView,
  PlaceIdentity,
  SetupRelationView,
  SetupView,
  EntityRevision,
  NewEntity,
  NewNpc,
  NpcRevision,
  BuildJob,
  Deployment,
  DirectiveResult,
  Intervention,
  MapTemplate,
  ThemePreset,
  StepView,
  WorldGraph,
  WorldMeta,
} from "../types";

// Empty by default → same-origin relative paths (works behind the nginx proxy).
// Set VITE_API_BASE at build time to point at a backend on another origin.
export const API_BASE: string = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

// A world's map artifact (document, art, ground, cast) is addressed by BASE alone and
// fetched by phaser/mapSource.loadMapSource — which is why no per-route URL builders
// live here. See LiveWorldMap for the call.

export async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API_BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const api = {
  getDeployment: () => req<Deployment>("/api/deployment"),
  // Every emotion and need a client may name, the same for every world.
  vocabulary: () => req<Vocabulary>("/api/vocabulary"),
  listWorlds: () => req<WorldMeta[]>("/api/worlds"),
  getWorld: (id: string) => req<WorldMeta>(`/api/worlds/${id}`),
  // Editable interface text (sample themes), re-read by the backend per request.
  listPresets: () =>
    req<{ presets: ThemePreset[] }>("/api/content").then((c) => c.presets),
  // The backend scans its template directories, so a new map appears here with no
  // frontend change.
  listTemplates: () => req<MapTemplate[]>("/api/templates"),
  // `template` omitted → the backend reads the theme and picks one of the installed maps.
  createWorld: (theme: string, template?: string | null) =>
    req<BuildJob>("/api/worlds", {
      method: "POST",
      body: JSON.stringify(template ? { theme, template } : { theme }),
    }),
  getBuildJob: (jobId: string) => req<BuildJob>(`/api/worlds/jobs/${jobId}`),
  // Renames the label in the world list only; nothing the simulation reads changes.
  renameWorld: (id: string, worldName: string) =>
    req<WorldMeta>(`/api/worlds/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ world_name: worldName }),
    }),
  // Shows the whole cast as the director. One way: the world takes no actor afterwards.
  openReview: (id: string) => req<WorldMeta>(`/api/worlds/${id}/review`, { method: "POST" }),
  // `characterId` casts the actor in the same act; without it the world is the director's.
  confirmWorld: (id: string, characterId?: string) =>
    req<WorldMeta>(`/api/worlds/${id}/confirm`, {
      method: "POST",
      body: JSON.stringify(characterId ? { character_id: characterId } : {}),
    }),
  // The world's setup, revisable until the world is confirmed and read-only after.
  getSetup: (id: string) => req<SetupView>(`/api/worlds/${id}/setup`),
  addCharacter: (id: string, character: NewCharacter) =>
    req<{ agent_id: string; setup: SetupView }>(`/api/worlds/${id}/setup/characters`, {
      method: "POST",
      body: JSON.stringify(character),
    }),
  removeCharacter: (id: string, agentId: string) =>
    req<SetupView>(`/api/worlds/${id}/setup/characters/${agentId}`, { method: "DELETE" }),
  reviseCharacter: (id: string, agentId: string, revision: CharacterRevision) =>
    req<SetupView>(`/api/worlds/${id}/setup/characters/${agentId}`, {
      method: "PATCH",
      body: JSON.stringify(revision),
    }),
  reviseLore: (id: string, lore: Pick<SetupView, "narrative_theme" | "opening_time" | "seconds_per_step">) =>
    req<SetupView>(`/api/worlds/${id}/setup/lore`, {
      method: "PUT", body: JSON.stringify(lore),
    }),
  addEntity: (id: string, entity: NewEntity) =>
    req<{ entity_id: string; setup: SetupView }>(`/api/worlds/${id}/setup/entities`, { method: "POST", body: JSON.stringify(entity) }),
  removeEntity: (id: string, entityId: string) =>
    req<SetupView>(`/api/worlds/${id}/setup/entities/${entityId}`, { method: "DELETE" }),
  addNpc: (id: string, npc: NewNpc) =>
    req<{ npc_id: string; setup: SetupView }>(`/api/worlds/${id}/setup/npcs`, { method: "POST", body: JSON.stringify(npc) }),
  removeNpc: (id: string, npcId: string) =>
    req<SetupView>(`/api/worlds/${id}/setup/npcs/${npcId}`, { method: "DELETE" }),
  reviseEntity: (id: string, entityId: string, revision: EntityRevision) =>
    req<SetupView>(`/api/worlds/${id}/setup/entities/${entityId}`, {
      method: "PATCH", body: JSON.stringify(revision),
    }),
  reviseNpc: (id: string, npcId: string, revision: NpcRevision) =>
    req<SetupView>(`/api/worlds/${id}/setup/npcs/${npcId}`, {
      method: "PATCH", body: JSON.stringify(revision),
    }),
  // Both replace the whole list: an event or a relation left out is removed.
  replaceHistory: (id: string, events: SetupHistoricalEventView[]) =>
    req<SetupView>(`/api/worlds/${id}/setup/history`, {
      method: "PUT",
      body: JSON.stringify({ historical_events: events }),
    }),
  replaceRelations: (id: string, relations: SetupRelationView[]) =>
    req<SetupView>(`/api/worlds/${id}/setup/relations`, {
      method: "PUT",
      body: JSON.stringify({ relations }),
    }),
  deleteWorld: (id: string) =>
    req<{ deleted: boolean }>(`/api/worlds/${id}`, { method: "DELETE" }),

  // --- Director: the one surface that reaches INTO a running world -----------
  // `accepted: false` is a normal answer, not an error: this resolves on a refusal
  // and `reason` says what the director should reword.
  direct: (id: string, text: string) =>
    req<DirectiveResult>(`/api/worlds/${id}/direct`, {
      method: "POST",
      body: JSON.stringify({ text }),
    }),
  // Advance exactly one step, then leave the world parked: pause → inject → step → look.
  stepOnce: (id: string) => req(`/api/worlds/${id}/step`, { method: "POST" }),
  // Everything this world has been told to do, oldest first. Read from the record because
  // the feed only holds steps you watched.
  listDirectives: (id: string) =>
    req<Intervention[]>(`/api/worlds/${id}/directives`),

  // Omitting `steps` means ONE step, not "run forever": every step bills LLM calls.
  run: (id: string, steps?: number) =>
    req(`/api/worlds/${id}/run`, {
      method: "POST",
      body: JSON.stringify(steps == null ? {} : { steps }),
    }),
  pause: (id: string) => req(`/api/worlds/${id}/pause`, { method: "POST" }),
  resume: (id: string) => req(`/api/worlds/${id}/resume`, { method: "POST" }),
  stop: (id: string) => req(`/api/worlds/${id}/stop`, { method: "POST" }),
  reset: (id: string) => req(`/api/worlds/${id}/reset`, { method: "POST" }),

  listSteps: (id: string) => req<number[]>(`/api/worlds/${id}/steps`),
  getStep: (id: string, step: number) =>
    req<StepView>(`/api/worlds/${id}/steps/${step}`),
  graph: (id: string, step?: number) =>
    req<WorldGraph>(
      `/api/worlds/${id}/graph${step != null ? `?step=${step}` : ""}`,
    ),
  // The whole cast as persisted at step 0 — stable for the world's life, unlike a
  // StepView's agents, so stable things (character-skin dealing) key off this.
  agents: (id: string) => req<CharacterIdentity[]>(`/api/worlds/${id}/agents`),
  // The world's NPCs and places, fixed by setup.
  npcs: (id: string) => req<NpcIdentity[]>(`/api/worlds/${id}/npcs`),
  places: (id: string) => req<PlaceIdentity[]>(`/api/worlds/${id}/places`),
  agentProfile: (id: string, agentId: string) =>
    req<AgentProfile>(`/api/worlds/${id}/agents/${agentId}/profile`),

  // --- Actor: the person performing one character -------------------------------
  // The actor's thread over the latest steps, in order.
  actorThread: (id: string) => req<{ entries: ActorEntry[] }>(`/api/worlds/${id}/actor/thread`),
  actorIntent: (id: string, intent: ActorIntent) =>
    req<ActorIntentResult>(`/api/worlds/${id}/actor/intent`, {
      method: "POST", body: JSON.stringify(intent),
    }),
  actorOperations: (id: string, turnId: string, target: ActorTarget) =>
    req<ActorOperations>(`/api/worlds/${id}/actor/operations?${new URLSearchParams({
      turn_id: turnId, target_kind: target.kind, target_id: target.id,
    })}`),
  // Break off what the character is doing, outside any turn.
  breakOffActor: (id: string) => req<ActorResult>(`/api/worlds/${id}/actor/break-off`, { method: "POST" }),
  answerActor: (id: string, turnId: string, answer: ActorPublicAnswer) =>
    req<ActorResult>(`/api/worlds/${id}/actor/answer`, {
      method: "POST",
      body: JSON.stringify({ turn_id: turnId, answer }),
    }),
  setActorPace: (id: string, pace: ActorPace) =>
    req<{ pace: ActorPace }>(`/api/worlds/${id}/actor/pace`, {
      method: "PUT",
      body: JSON.stringify({ pace }),
    }),
  actorMind: (id: string) => req<ActorMind>(`/api/worlds/${id}/actor/mind`),
  amendActor: (id: string, amendments: ActorAmendments) =>
    req<ActorAmendResult>(`/api/worlds/${id}/actor/amend`, {
      method: "POST",
      body: JSON.stringify(amendments),
    }),
  // For good: the character returns to AI cognition and the world to the director.
  exitActor: (id: string) => req<WorldMeta>(`/api/worlds/${id}/actor/exit`, { method: "POST" }),
};
