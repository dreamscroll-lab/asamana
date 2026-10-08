// REST client for the Asamana backend. Paths are relative so the Vite dev
// proxy (dev) or the backend static mount (prod) both work without config.

import type {
  AgentProfile,
  BuildJob,
  Deployment,
  DirectiveResult,
  Intervention,
  MapTemplate,
  ThemePreset,
  StepEvent,
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
  confirmWorld: (id: string) =>
    req<WorldMeta>(`/api/worlds/${id}/confirm`, { method: "POST" }),
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
    req<StepEvent>(`/api/worlds/${id}/steps/${step}`),
  graph: (id: string, step?: number) =>
    req<WorldGraph>(
      `/api/worlds/${id}/graph${step != null ? `?step=${step}` : ""}`,
    ),
  // The whole cast as persisted at step 0 — stable for the world's life, unlike a
  // StepEvent's agent_states, so stable things (character-skin dealing) key off this.
  agents: (id: string) => req<AgentProfile[]>(`/api/worlds/${id}/agents`),
  agentProfile: (id: string, agentId: string) =>
    req<AgentProfile>(`/api/worlds/${id}/agents/${agentId}/profile`),
};
